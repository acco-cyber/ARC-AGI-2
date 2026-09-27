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
import os
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
                         precision: str = "fp32", eval_task_ids: Collection[str] = (), num_workers: int = 0,
                         pseudo_label_total_seconds: Optional[float] = None) -> TransformationMemory:
    """Rule-latent KNN memory over synthetic programs and (pseudo-labelled) training tasks.

    ``eval_task_ids`` are the blocked ids (eval_public, and val + holdout when the caller passes them, as
    ``train_all.memory_sources`` does); any of them among ``real_tasks`` raises. ``num_workers`` > 0 tensorises
    episodes in DataLoader worker processes (the forward pass stays on ``device``). The pseudo-label search
    spends at most ``pseudo_label_seconds`` per task and ``pseudo_label_total_seconds`` overall (None = no cap);
    tasks left when the total is spent keep ``program == ""``.
    """
    from arcjepa.training.common import rule_latents, synth_episode

    device = device or model.device
    real_tasks = dict(real_tasks or {})
    leak = set(real_tasks) & set(eval_task_ids)
    if leak:
        raise AssertionError(f"evaluation / held-out tasks must never enter the memory: {sorted(leak)[:5]}")
    mem = TransformationMemory(model.cfg.rule_dim)
    synth = list(synth_tasks)
    if synth:
        r = rule_latents(model, [synth_episode(t, -1) for t in synth], parser, device, max_ctx=max_ctx_synth,
                         batch_size=batch_size, precision=precision, num_workers=num_workers).numpy()
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
                         precision=precision, num_workers=num_workers).numpy()
        n_label = 0
        t_label = time.time()
        total = None if pseudo_label_total_seconds is None else max(0.0, float(pseudo_label_total_seconds))
        for i, (tid, v) in enumerate(zip(ids, r)):
            secs = float(pseudo_label_seconds)
            if total is not None:  # share what is left of the total evenly over the tasks still to label
                secs = min(secs, (total - (time.time() - t_label)) / max(1, len(ids) - i))
            prog = (pseudo_label(real_tasks[tid], secs) or "") if secs >= 0.05 else ""
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
    """Write the offline package for ``model`` into ``out_dir``; returns ``{"dir", "files", "memory_size"}``.

    Order: weights + vocab + a provisional ``config.json`` (``"export_complete": false``, empty memory) are
    written FIRST, so an export killed while the memory is being built still leaves a loadable package; the
    memory files and the final ``config.json`` (``"export_complete": true``) follow. The memory is built on
    ``device`` (default: the model's device), tensorising with ``memory.num_workers`` DataLoader workers
    (default ``min(8, cpu_count - 1)`` on CUDA, 0 on CPU).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mcfg = dict(cfg.get("memory") or {})
    t0 = time.time()
    device = device or model.device
    weights = save_weights(model, out, bool((cfg.get("export") or {}).get("safetensors", True)))
    model.tokenizer.save(out / VOCAB_FILE)
    for stale in (MEMORY_NPZ_FILE, PROGRAMS_FILE):  # never pair an older memory with these weights
        if (out / stale).is_file():
            (out / stale).unlink()
    meta: Dict[str, Any] = {
        "format": PACKAGE_FORMAT,
        "model": model.cfg.to_dict(),
        "weights": weights,
        "vocab": VOCAB_FILE,
        "memory": {"size": 0, "dim": int(model.cfg.rule_dim), "top_k": int(mcfg.get("top_k", 16))},
        "search": cfg.get("search"), "tta": cfg.get("tta"), "repair": cfg.get("repair"),
        "outputs": cfg.get("outputs"),
        "training": {"config_path": cfg.get("config_path"), "seed": cfg.get("seed")},
        "created_unix": int(time.time()),
        "n_parameters": int(model.num_parameters()),
        "export_complete": False,
    }
    if extra_meta:
        meta.update(dict(extra_meta))
    _write_json_atomic(meta, out / CONFIG_FILE)
    nw = mcfg.get("num_workers")
    if nw is None:
        nw = min(8, max(0, (os.cpu_count() or 1) - 1)) if device.type == "cuda" else 0
    pl_total = mcfg.get("pseudo_label_total_seconds")
    was = model.training
    model.eval()
    try:
        mem = build_program_memory(
            model, synth_tasks=synth_tasks, real_tasks=real_tasks, families=families, parser=parser, device=device,
            pseudo_label_seconds=float(mcfg.get("pseudo_label_seconds", 0.0) or 0.0),
            pseudo_label_total_seconds=None if pl_total is None else float(pl_total),
            max_ctx_synth=int((cfg.get("synthetic") or {}).get("max_ctx", 5)),
            max_ctx_real=int(((cfg.get("stages") or {}).get("B") or {}).get("max_ctx", 10)),
            batch_size=int((cfg.get("eval") or {}).get("batch_size", 16)),
            precision=str((cfg.get("training") or {}).get("precision", "fp32")), eval_task_ids=eval_task_ids,
            num_workers=int(nw))
    finally:
        model.train(was)
    t_mem = time.time() - t0
    mem_files = save_program_memory(mem, out)
    meta["memory"] = {**mem_files, "size": len(mem), "dim": mem.dim, "top_k": int(mcfg.get("top_k", 16)),
                      "sources": "synthetic train split + 700 re-split training tasks "
                                 "(never eval_public / val / holdout / sdg_hard)"}
    meta["export_complete"] = True
    meta["export_seconds"] = round(time.time() - t0, 2)
    _write_json_atomic(meta, out / CONFIG_FILE)
    files = [weights, CONFIG_FILE, VOCAB_FILE, MEMORY_NPZ_FILE, PROGRAMS_FILE]
    log.info("exported package to %s (%d memory records on %s, %d workers, %.1fs, memory %.1fs)", out, len(mem),
             device, int(nw), time.time() - t0, t_mem)
    return {"dir": str(out), "files": files, "memory_size": len(mem), "seconds": round(time.time() - t0, 2)}


def _write_json_atomic(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def resolve_export_device(pref: str = "auto") -> torch.device:
    """``auto`` -> ``cuda:0`` when CUDA is available, else CPU; ``cpu`` / ``cuda`` / ``cuda:N`` force."""
    if pref == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if pref.startswith("cuda") and not torch.cuda.is_available():
        log.warning("export device %r requested but CUDA is unavailable; using CPU", pref)
        return torch.device("cpu")
    return torch.device(pref)


def export_from_checkpoint(checkpoint: PathLike, out_dir: PathLike, config: Optional[PathLike] = None,
                           overrides: Optional[Sequence[str]] = None, device: str = "auto") -> Dict[str, Any]:
    """Rebuild the model from a training checkpoint and export it (memory from the config's data sources).

    The model and the memory build run on ``device`` (``auto`` = CUDA when available): a CPU memory build of
    the v1 model takes ~9 s per synthetic episode, so the CPU path needs a tiny ``memory.max_synthetic``.
    """
    from arcjepa.training.common import (DistInfo, load_config, model_config_from, prepare_synthetic)
    from arcjepa.training.train_all import memory_sources

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = load_config(config, overrides) if config else dict(state.get("config") or {})
    from arcjepa.model.config import ModelConfig

    mcfg = ModelConfig.from_dict(state["model_config"]) if state.get("model_config") else model_config_from(cfg)
    model = ARCJEPA(mcfg)
    model.load_state_dict(state["model"])
    del state
    dev = resolve_export_device(device)
    model.to(dev)
    out = Path(out_dir)
    store, train_idx, _ = prepare_synthetic(cfg, out.parent, DistInfo())
    synth, real, fams, blocked = memory_sources(cfg, store, train_idx)
    log.info("export from %s on %s: %d synthetic + %d real memory tasks", checkpoint, dev, len(synth), len(real))
    return export_package(model, cfg, out, synth_tasks=synth, real_tasks=real, families=fams, device=dev,
                          eval_task_ids=blocked)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """CLI entry point (see module docstring)."""
    ap = argparse.ArgumentParser(description="Export an ARC-JEPA offline package from a checkpoint.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="auto", help="auto (CUDA when available) | cpu | cuda | cuda:N")
    ap.add_argument("--set", action="append", default=[], help="config override key.path=value")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    return export_from_checkpoint(args.checkpoint, args.out, args.config, args.set, device=args.device)


if __name__ == "__main__":  # pragma: no cover
    main()
