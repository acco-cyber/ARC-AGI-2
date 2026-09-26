"""Offline package export (spec §Kaggle inference notebook): one folder with

* ``model.safetensors`` (``safetensors`` when importable, else ``model.pt`` = ``torch.save(state_dict)``),
* ``config.json``      -- ``{"format", "model": ModelConfig dict, "weights", "vocab", "memory", "search", ...}``,
* ``vocab.json``       -- the program tokenizer vocabulary (``ProgramTokenizer.save``),
* ``program_memory.npz`` (key ``latents`` Float32[N, rule_dim], unit rows) + ``programs.json``
  (``{"dim", "records": [{program, complexity, family, source, task_id}, ...]}``).

The transformation memory is built from synthetic ``train``-split tasks and the 700 re-split *training* tasks
only (never an evaluation set). A real task contributes a program label only when a short DSL search finds a
program that fits ALL its train pairs exactly (``memory.pseudo_label_seconds`` > 0); otherwise its record keeps
the rule latent + family with ``program == ""`` (transformation features only; the search's memory prior
skips such records).

``ARCJEPA.load_package(path)`` reads the folder back. CLI::

    python -m arcjepa.training.export --checkpoint runs/x/checkpoints/last.pt --out runs/x/package
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from arcjepa.core.types import Task
from arcjepa.model.arcjepa import (ARCJEPA, CONFIG_FILE, MEMORY_NPZ_FILE, PACKAGE_FORMAT, PROGRAMS_FILE, VOCAB_FILE,
                                   WEIGHTS_PT_FILE, WEIGHTS_SAFETENSORS_FILE, load_package_memory)
from arcjepa.model.memory import TransformationMemory

log = logging.getLogger(__name__)

PathLike = Union[str, Path]


# ============================================================================================ weights

def save_weights(model: ARCJEPA, out_dir: PathLike, prefer_safetensors: bool = True) -> str:
    """Write the model weights; returns the file name used (``model.safetensors`` or ``model.pt``)."""
    out = Path(out_dir)
    if prefer_safetensors:
        try:
            from safetensors.torch import save_model

            save_model(model, str(out / WEIGHTS_SAFETENSORS_FILE), metadata={"format": PACKAGE_FORMAT})
            return WEIGHTS_SAFETENSORS_FILE
        except Exception as exc:  # noqa: BLE001 - missing package or unsupported tensors: fall back
            log.warning("safetensors export failed (%s); writing %s", exc, WEIGHTS_PT_FILE)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state, out / WEIGHTS_PT_FILE)
    return WEIGHTS_PT_FILE


# ============================================================================================ memory

def pseudo_label(task: Task, seconds: float) -> Optional[str]:
    """A DSL program fitting ALL train pairs of ``task`` exactly (short uniform-prior beam search), else None."""
    if seconds <= 0 or not task.train:
        return None
    try:
        from arcjepa.search.beam import beam_search
        from arcjepa.search.verifier import is_exact
    except ImportError:  # search module unavailable
        return None
    try:
        cands = beam_search(task.train, prior=None, time_budget_s=float(seconds))
    except Exception as exc:  # noqa: BLE001 - a failing search just means "no label"
        log.debug("pseudo-label search failed on %s: %s", task.task_id, exc)
        return None
    for c in cands:
        if c.demo_err == 0 and is_exact(c.program, task.train):
            return c.program.to_str()
    return None


def _complexity(program: str) -> int:
    if not program:
        return 0
    try:
        from arcjepa.dsl.ast import Node

        return int(Node.from_str(program).size())
    except Exception:  # noqa: BLE001
        return len(program.split())


def build_program_memory(model: ARCJEPA, *, synth_tasks: Sequence[Any] = (),
                         real_tasks: Optional[Mapping[str, Task]] = None,
                         families: Optional[Mapping[str, str]] = None, parser: Any = None,
                         device: Optional[torch.device] = None, pseudo_label_seconds: float = 0.0,
                         max_ctx_synth: int = 5, max_ctx_real: int = 10, batch_size: int = 16,
                         precision: str = "fp32", eval_task_ids: Collection[str] = ()) -> TransformationMemory:
    """Rule-latent KNN memory over synthetic programs and (pseudo-labelled) training tasks."""
    from arcjepa.training.common import rule_latents, synth_episode

    device = device or model.device
    real_tasks = dict(real_tasks or {})
    leak = set(real_tasks) & set(eval_task_ids)
    if leak:
        raise AssertionError(f"evaluation tasks must never enter the memory: {sorted(leak)[:5]}")
    mem = TransformationMemory(model.cfg.rule_dim)
    synth = list(synth_tasks)
    if synth:
        r = rule_latents(model, [synth_episode(t, -1) for t in synth], parser, device, max_ctx=max_ctx_synth,
                         batch_size=batch_size, precision=precision).numpy()
        for t, v in zip(synth, r):
            mem.add(v, t.program, _complexity(t.program), str(getattr(t, "category", "synthetic")),
                    source="synthetic", task_id=t.task_id)
    if real_tasks:
        from arcjepa.data.hf_loader import episodes_from_task

        ids = sorted(real_tasks)
        eps = []
        for tid in ids:
            t = real_tasks[tid]
            ep = episodes_from_task(t)[0] if t.test else episodes_from_task(Task(tid, t.train, t.train[:1]))[0]
            eps.append(ep)
        r = rule_latents(model, eps, parser, device, max_ctx=max_ctx_real, batch_size=batch_size,
                         precision=precision).numpy()
        n_label = 0
        for tid, v in zip(ids, r):
            prog = pseudo_label(real_tasks[tid], pseudo_label_seconds) or ""
            n_label += bool(prog)
            mem.add(v, prog, _complexity(prog), (families or {}).get(tid, "unknown"), source="arc", task_id=tid)
        log.info("memory: %d real training tasks, %d with exact pseudo-programs", len(ids), n_label)
    return mem


def save_program_memory(mem: TransformationMemory, out_dir: PathLike) -> Dict[str, str]:
    """Write ``program_memory.npz`` (latents) + ``programs.json`` (dim + records)."""
    out = Path(out_dir)
    np.savez(out / MEMORY_NPZ_FILE, latents=mem.matrix().astype(np.float32))
    (out / PROGRAMS_FILE).write_text(json.dumps({"dim": mem.dim, "records": mem.records}), encoding="utf-8")
    return {"npz": MEMORY_NPZ_FILE, "programs": PROGRAMS_FILE}


def load_program_memory(pkg_dir: PathLike) -> Optional[TransformationMemory]:
    """Read the memory of an exported package (``None`` when absent)."""
    return load_package_memory(pkg_dir)


# ============================================================================================ package

def export_package(model: ARCJEPA, cfg: Mapping[str, Any], out_dir: PathLike, *, synth_tasks: Sequence[Any] = (),
                   real_tasks: Optional[Mapping[str, Task]] = None, families: Optional[Mapping[str, str]] = None,
                   parser: Any = None, device: Optional[torch.device] = None,
                   eval_task_ids: Collection[str] = (), extra_meta: Optional[Mapping[str, Any]] = None
                   ) -> Dict[str, Any]:
    """Write the offline package for ``model`` into ``out_dir``; returns ``{"dir", "files", "memory_size"}``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mcfg = dict(cfg.get("memory") or {})
    t0 = time.time()
    was = model.training
    model.eval()
    try:
        mem = build_program_memory(
            model, synth_tasks=synth_tasks, real_tasks=real_tasks, families=families, parser=parser, device=device,
            pseudo_label_seconds=float(mcfg.get("pseudo_label_seconds", 0.0) or 0.0),
            max_ctx_synth=int((cfg.get("synthetic") or {}).get("max_ctx", 5)),
            max_ctx_real=int(((cfg.get("stages") or {}).get("B") or {}).get("max_ctx", 10)),
            batch_size=int((cfg.get("eval") or {}).get("batch_size", 16)),
            precision=str((cfg.get("training") or {}).get("precision", "fp32")), eval_task_ids=eval_task_ids)
    finally:
        model.train(was)
    weights = save_weights(model, out, bool((cfg.get("export") or {}).get("safetensors", True)))
    model.tokenizer.save(out / VOCAB_FILE)
    mem_files = save_program_memory(mem, out)
    meta = {
        "format": PACKAGE_FORMAT,
        "model": model.cfg.to_dict(),
        "weights": weights,
        "vocab": VOCAB_FILE,
        "memory": {**mem_files, "size": len(mem), "dim": mem.dim, "top_k": int(mcfg.get("top_k", 16)),
                   "sources": "synthetic train split + 700 re-split training tasks (never evaluation)"},
        "search": cfg.get("search"), "tta": cfg.get("tta"), "repair": cfg.get("repair"),
        "outputs": cfg.get("outputs"),
        "training": {"config_path": cfg.get("config_path"), "seed": cfg.get("seed")},
        "created_unix": int(time.time()),
        "n_parameters": int(model.num_parameters()),
    }
    if extra_meta:
        meta.update(dict(extra_meta))
    (out / CONFIG_FILE).write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    files = [weights, CONFIG_FILE, VOCAB_FILE, MEMORY_NPZ_FILE, PROGRAMS_FILE]
    log.info("exported package to %s (%d memory records, %.1fs)", out, len(mem), time.time() - t0)
    return {"dir": str(out), "files": files, "memory_size": len(mem), "seconds": round(time.time() - t0, 2)}


def export_from_checkpoint(checkpoint: PathLike, out_dir: PathLike, config: Optional[PathLike] = None,
                           overrides: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Rebuild the model from a training checkpoint and export it (memory from the config's data sources)."""
    from arcjepa.training.common import (DistInfo, load_config, model_config_from, prepare_synthetic)
    from arcjepa.training.train_all import memory_sources

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = load_config(config, overrides) if config else dict(state.get("config") or {})
    from arcjepa.model.config import ModelConfig

    mcfg = ModelConfig.from_dict(state["model_config"]) if state.get("model_config") else model_config_from(cfg)
    model = ARCJEPA(mcfg)
    model.load_state_dict(state["model"])
    out = Path(out_dir)
    store, train_idx, _ = prepare_synthetic(cfg, out.parent, DistInfo())
    synth, real, fams, eval_ids = memory_sources(cfg, store, train_idx)
    return export_package(model, cfg, out, synth_tasks=synth, real_tasks=real, families=fams,
                          eval_task_ids=eval_ids)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """CLI entry point (see module docstring)."""
    ap = argparse.ArgumentParser(description="Export an ARC-JEPA offline package from a checkpoint.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], help="config override key.path=value")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    return export_from_checkpoint(args.checkpoint, args.out, args.config, args.set)


if __name__ == "__main__":  # pragma: no cover
    main()
