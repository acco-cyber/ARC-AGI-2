"""Run the full ARC-JEPA training schedule A -> B -> C -> D time-boxed by ``--hours``, then export the package.

::

    python -m arcjepa.training.train_all --config configs/debug.yaml --hours 0.03 --out runs/debug
    torchrun --nproc_per_node 4 -m arcjepa.training.train_all --config configs/l4x4.yaml --hours 9.5 --out /kaggle/working/run

* Budget: ``training.reserve_frac`` of the wall-clock budget is kept for the export; the rest is split over
  the stages still to run in proportion to ``training.time_fractions`` and re-split after every stage, so time a
  stage leaves unused rolls over to the next one. Every stage also stops at its planned step count
  (``epochs`` x steps/epoch, capped by ``max_steps``), whichever comes first.
* Checkpoints: ``<out>/checkpoints/last.pt`` every ``training.checkpoint_minutes`` and after each stage. A rerun
  with the same ``--out`` resumes (``--no-resume`` starts over): completed stages are skipped and an interrupted
  stage continues from its step, epoch, batch offset, optimizer and EMA state.
* Metrics: ``<out>/metrics.jsonl`` (``kind`` train / eval / stage_start / stage_end / data / export) with
  ``loss_*``, ``collapse_std``, ``retrieval@1`` / ``retrieval@8`` on held-out synthetic programs.
* Distributed: launched by ``torchrun`` (WORLD_SIZE > 1) each process drives one GPU and gradients are
  all-reduced every step; otherwise a single plain process runs on ``training.device``.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from arcjepa.core.types import Task
from arcjepa.data.tensorize import resolve_parser
from arcjepa.training.common import (STAGES, Checkpointer, DistInfo, MetricsLogger, SynthStore, TimeBudget,
                                     TrainContext, barrier, broadcast_module, build_model, build_target, cfg_get,
                                     cleanup_distributed, init_distributed, load_config, prepare_synthetic,
                                     remove_file_logging, set_seed, setup_logging)
from arcjepa.training.pretrain_jepa import run_stage_a
from arcjepa.training.train_program_encoder import run_stage_c, run_stage_d
from arcjepa.training.train_real import run_stage_b

log = logging.getLogger(__name__)

STAGE_FNS: Dict[str, Callable[[TrainContext, float], Dict[str, Any]]] = {
    "A": run_stage_a, "B": run_stage_b, "C": run_stage_c, "D": run_stage_d}
DEFAULT_FRACTIONS: Dict[str, float] = {"A": 0.45, "B": 0.25, "C": 0.15, "D": 0.15}


def stage_budget(stage: str, remaining_stages: Sequence[str], seconds_left: float,
                 fractions: Mapping[str, float]) -> float:
    """Seconds for ``stage``: its share of ``seconds_left`` among the stages still to run."""
    tot = sum(float(fractions.get(s, 0.0)) for s in remaining_stages)
    if tot <= 0:
        return max(0.0, seconds_left) / max(1, len(remaining_stages))
    return max(0.0, seconds_left) * float(fractions.get(stage, 0.0)) / tot


def memory_sources(cfg: Mapping[str, Any], store: Optional[SynthStore], train_idx: Sequence[int]
                   ) -> Tuple[List[Any], Dict[str, Task], Dict[str, str], List[str]]:
    """``(synthetic tasks, real training tasks, their families, blocked ids)`` for the exported memory.

    Synthetic: the first ``memory.max_synthetic`` synthetic *train*-split tasks (generated in-session; never
    sdg_hard). Real: the 700 re-split train tasks (capped at ``memory.max_real``) when ``memory.include_real``.
    ``blocked ids`` = eval_public + the 150 val + the 150 holdout ids; the export refuses any of them, and the
    real tasks are checked here to be a subset of the 700 train ids (so no sdg / non-ARC id can enter).
    """
    mcfg = cfg.get("memory") or {}
    n_syn = int(mcfg.get("max_synthetic", 1000))
    synth = [store.task(i) for i in list(train_idx)[:n_syn]] if store is not None else []
    real: Dict[str, Task] = {}
    fams: Dict[str, str] = {}
    blocked: List[str] = []
    if mcfg.get("include_real", True):
        try:
            from arcjepa.data.hf_loader import EVAL_PUBLIC, load_resplit, load_training_tasks, resolve_root
            from arcjepa.training.train_real import EvalLeakError, load_split_families, official_task_splits

            root = resolve_root(cfg_get(cfg, "data.root"))
            splits = load_resplit(root)
            train_ids = set(splits["train"])
            ids = sorted(train_ids)
            cap = mcfg.get("max_real")
            ids = ids[: int(cap)] if cap is not None else ids
            tasks = load_training_tasks(root)
            real = {t: tasks[t] for t in ids if t in tasks}
            fams = load_split_families(root, ids)
            blocked = sorted({t for t, s in official_task_splits(root).items() if s == EVAL_PUBLIC}
                             | set(splits["val"]) | set(splits["holdout"]))
            outside = sorted(set(real) - train_ids)
            if outside or set(real) & set(blocked):
                raise EvalLeakError(f"memory real tasks outside the 700 train split: {outside[:5]}")
        except (FileNotFoundError, OSError) as exc:
            log.warning("memory: real training tasks unavailable (%s)", exc)
    return synth, real, fams, blocked


def train(cfg: Dict[str, Any], *, hours: float, out_dir: str, stages: str = "ABCD", resume: bool = True,
          export: bool = True, export_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run the requested stages within ``hours`` and export; returns the run summary (main rank)."""
    t_start = time.time()
    out = Path(out_dir)
    tr = cfg.setdefault("training", {})
    dist = init_distributed(str(tr.get("device", "auto")))
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out, dist.is_main)
    try:
        total_s = max(0.0, float(hours) * 3600.0)
        budget = TimeBudget(total_s)
        reserve = float(tr.get("reserve_frac", 0.05)) * total_s if export else 0.0
        seed = int(cfg.get("seed", 0))
        set_seed(seed)
        if dist.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if tr.get("threads"):
            torch.set_num_threads(int(tr["threads"]))

        store, synth_train, synth_held = prepare_synthetic(cfg, out, dist)
        model = build_model(cfg).to(dist.device)
        target = build_target(model).to(dist.device)
        metrics = MetricsLogger(out / "metrics.jsonl", dist.is_main)
        ckpt = Checkpointer(out / "checkpoints", dist.is_main)
        ctx = TrainContext(cfg=cfg, model=model, target=target, device=dist.device, dist=dist, out_dir=out,
                           metrics=metrics, ckpt=ckpt, parser=resolve_parser(None), store=store,
                           synth_train=synth_train, synth_heldout=synth_held, budget=budget)
        state = Checkpointer(out / "checkpoints", False).load("cpu") if resume else None
        if state is not None:
            model.load_state_dict(state["model"])
            target.load_state_dict(state["target"])
            ctx.completed = list(state.get("completed", []))
            ctx.global_step = int(state.get("global_step", 0))
            ctx.resume = state if state.get("stage") else None
            log.info("resumed from %s: completed %s, in-progress %s", ckpt.last_path, ctx.completed, state.get("stage"))
            metrics.log({"kind": "resume", "completed": ",".join(ctx.completed), "stage": state.get("stage") or "",
                         "global_step": ctx.global_step})
        broadcast_module(model, dist)
        broadcast_module(target, dist)
        n_params = model.num_parameters()
        metrics.log({"kind": "start", "hours": hours, "n_params": n_params, "world_size": dist.world_size,
                     "device": str(dist.device), "stages": stages, "synthetic_train": len(synth_train),
                     "synthetic_heldout": len(synth_held)})
        log.info("model %s: %.2fM parameters on %s (world %d)", model.cfg.name, n_params / 1e6, dist.device,
                 dist.world_size)

        fractions = dict(DEFAULT_FRACTIONS)
        fractions.update({k: float(v) for k, v in (tr.get("time_fractions") or {}).items()})
        todo = [s for s in STAGES if s in stages.upper() and s not in ctx.completed
                and bool(cfg_get(cfg, f"stages.{s}.enabled", True))]
        for i, s in enumerate(todo):
            left = budget.remaining() - reserve
            b = stage_budget(s, todo[i:], left, fractions)
            metrics.log({"kind": "stage_start", "stage": s, "budget_s": round(b, 2), "global_step": ctx.global_step})
            STAGE_FNS[s](ctx, b)

        summary: Dict[str, Any] = {"out": str(out), "stages": ctx.summaries, "completed": ctx.completed,
                                   "global_step": ctx.global_step, "n_params": n_params}
        if export:
            pkg_dir = Path(export_dir) if export_dir else out / str(cfg_get(cfg, "export.dir", "package"))
            # Under torchrun the export (tens of thousands of rule latents + pseudo-label searches) runs on rank 0
            # only. Every rank meets once more, then ALL ranks destroy the process group: ranks 1..N-1 return and
            # exit cleanly and rank 0 exports with no collective pending, so no NCCL timeout can abort it.
            barrier(dist)
            cleanup_distributed(dist)
            if dist.is_main:
                from arcjepa.training.export import export_package

                synth, real, fams, eval_ids = memory_sources(cfg, store, synth_train)
                res = export_package(model, cfg, pkg_dir, synth_tasks=synth, real_tasks=real, families=fams,
                                     parser=ctx.parser, device=dist.device, eval_task_ids=eval_ids,
                                     extra_meta={"training_summary": {"completed": ctx.completed,
                                                                      "global_step": ctx.global_step}})
                metrics.log({"kind": "export", "dir": res["dir"], "memory_size": res["memory_size"],
                             "seconds": res["seconds"]})
                summary["package"] = res
        summary["seconds"] = round(time.time() - t_start, 2)
        if dist.is_main:
            (out / "train_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
        log.info("training finished in %.1fs", summary["seconds"])
        return summary
    finally:
        cleanup_distributed(dist)
        remove_file_logging(out)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="ARC-JEPA training: stages A->B->C->D time-boxed, then export.")
    ap.add_argument("--config", required=True, help="YAML config (configs/{base,debug,l4x4,kaggle}.yaml)")
    ap.add_argument("--hours", type=float, default=None, help="wall-clock budget (default training.hours or 9.5)")
    ap.add_argument("--out", required=True, help="run directory (checkpoints, metrics.jsonl, package/)")
    ap.add_argument("--stages", default="ABCD", help="subset of stages to run, e.g. A or CD")
    ap.add_argument("--no-resume", action="store_true", help="ignore an existing checkpoint in --out")
    ap.add_argument("--no-export", action="store_true", help="skip the package export")
    ap.add_argument("--export-dir", default=None, help="package directory (default <out>/package)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """CLI entry point."""
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config, args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    hours = args.hours if args.hours is not None else float(cfg_get(cfg, "training.hours", 9.5))
    return train(cfg, hours=hours, out_dir=args.out, stages=args.stages, resume=not args.no_resume,
                 export=not args.no_export, export_dir=args.export_dir)


if __name__ == "__main__":  # pragma: no cover
    main()
