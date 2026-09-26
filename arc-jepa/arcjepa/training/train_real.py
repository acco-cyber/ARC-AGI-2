"""Stage B: real ARC adaptation on the 700-task train split (spec §Training, run-order step 3).

Data: the ``episodes``, ``episodes_aug``, ``arcgen_fresh`` and ``sdg_hard`` configs of the local HF mirror
(``arcjepa.data``), restricted to

* episodes of the 700 re-split *train* tasks, plus
* episodes whose task id is not an official ARC task at all (``sdg_hard`` synthetic-verified tasks),

and never ``eval_public`` (the file is never opened; rows labelled ``eval_public`` or carrying an eval / val /
holdout task id raise or are dropped). Rare families are oversampled by weighted sampling *without
replacement* inside each epoch (weight = family_count^-alpha), so a rare-family episode is never duplicated
within an epoch; the augmented / regenerated variants supply the extra rare-family volume. Loss = the JEPA
terms (no program terms), lr 3e-5.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from arcjepa.core.types import Episode, Pair, Task, episode_from_json
from arcjepa.training.common import (RealEpisodeDataset, StageSpec, TrainContext, cfg_get, eval_loss, run_stage)

log = logging.getLogger(__name__)

STAGE = "B"
DEFAULT_CONFIGS: Tuple[str, ...] = ("episodes", "episodes_aug", "arcgen_fresh", "sdg_hard")
TRAIN_HF_SPLITS: Tuple[str, ...] = ("train", "val", "test")
_TID_RE = re.compile(r'"task_id"\s*:\s*"([^"]+)"')


class EvalLeakError(AssertionError):
    """Raised when an evaluation / held-out task would enter real-data training."""


def official_task_splits(root: Optional[str] = None) -> Dict[str, str]:
    """task id -> HF split label for the 1,120 official tasks (ids only; used to exclude, never to train)."""
    from arcjepa.data.hf_loader import load_task_rows

    return {tid: str(row.get("split", "")) for tid, row in load_task_rows(root).items()}


def load_split_families(root: Optional[str], ids: Sequence[str]) -> Dict[str, str]:
    """Families of official training tasks: the re-split document's bookkeeping, else ``family_of``."""
    from arcjepa.data.hf_loader import PACKAGE_DATA_DIR, SPLITS_FILENAME

    fams: Dict[str, str] = {}
    path = PACKAGE_DATA_DIR / SPLITS_FILENAME
    if path.is_file():
        try:
            fams = dict(json.loads(path.read_text(encoding="utf-8")).get("families", {}))
        except (OSError, ValueError) as exc:
            log.warning("could not read families from %s: %s", path, exc)
    missing = [t for t in ids if t not in fams]
    if missing:
        from arcjepa.data import family_of, load_training_tasks

        tasks = load_training_tasks(root)
        for t in missing:
            if t in tasks:
                fams[t] = family_of(tasks[t])
    return fams


def stream_episodes(root: str, config: str, split: str, keep: Callable[[str], bool],
                    max_rows: Optional[int] = None) -> List[Episode]:
    """Episodes of ``<root>/<config>/<split>.jsonl`` whose task id passes ``keep`` (early stop at ``max_rows``).

    The task id is read with a regex before JSON decoding so filtered-out rows cost almost nothing.
    """
    from arcjepa.data.hf_loader import EVAL_PUBLIC, jsonl_path

    if split == EVAL_PUBLIC:
        raise EvalLeakError("stage B never reads eval_public")
    path = jsonl_path(root, config, split)
    out: List[Episode] = []
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            m = _TID_RE.search(line[:600])
            if m is not None and not keep(m.group(1)):
                continue
            if not line.strip():
                continue
            row = json.loads(line)
            tid = str(row.get("task_id", ""))
            if not keep(tid):
                continue
            if row.get("split") == EVAL_PUBLIC:
                raise EvalLeakError(f"row {row.get('episode_id')} of {config}/{split} is labelled eval_public")
            ep = episode_from_json(row)
            if not ep.context or not ep.target_output:
                continue
            out.append(ep)
            if max_rows is not None and len(out) >= max_rows:
                break
    return out


def family_weights(families: Sequence[str], alpha: float = 0.5) -> List[float]:
    """Per-item sampling weights ``count(family)^-alpha`` (alpha 0 = uniform, 1 = family-balanced)."""
    counts = Counter(families)
    return [float(counts[f]) ** (-float(alpha)) for f in families]


def load_real_data(cfg: Mapping[str, Any]) -> Tuple[List[Episode], List[str], List[Episode]]:
    """``(train_episodes, train_families, val_episodes)`` for stage B.

    Train: every configured episode config over the HF train/val/test files, restricted to the 700 re-split
    train tasks plus non-official (sdg) task ids. Val: canonical ``episodes`` of the 150 re-split val tasks
    (model selection only).
    """
    from arcjepa.data.hf_loader import EVAL_PUBLIC, load_resplit, resolve_root

    bcfg = cfg_get(cfg, "stages.B", {}) or {}
    root = resolve_root(cfg_get(cfg, "data.root"))
    splits = load_resplit(root)
    train_ids: Set[str] = set(splits["train"])
    val_ids: Set[str] = set(splits["val"])
    official = official_task_splits(root)
    eval_ids = {t for t, s in official.items() if s == EVAL_PUBLIC}
    if train_ids & eval_ids:
        raise EvalLeakError("the 700 train split contains eval_public ids")
    blocked = eval_ids | set(splits["val"]) | set(splits["holdout"])

    def keep(tid: str) -> bool:
        return tid in train_ids or (tid not in official and tid not in blocked)

    configs = list(bcfg.get("configs") or DEFAULT_CONFIGS)
    hf_splits = [s for s in (bcfg.get("hf_splits") or TRAIN_HF_SPLITS) if s != EVAL_PUBLIC]
    cap = bcfg.get("max_per_config")
    episodes: List[Episode] = []
    for conf in configs:
        got: List[Episode] = []
        for sp in hf_splits:
            left = None if cap is None else int(cap) - len(got)
            if left is not None and left <= 0:
                break
            got += stream_episodes(root, conf, sp, keep, left)
        log.info("stage B: %d episodes from %s", len(got), conf)
        episodes += got
    for ep in episodes:
        if ep.task_id in blocked:
            raise EvalLeakError(f"episode {ep.episode_id} of a blocked task entered stage B")

    fams_official = load_split_families(root, sorted(train_ids))
    cache: Dict[str, str] = {}
    families: List[str] = []
    for ep in episodes:
        f = fams_official.get(ep.task_id) or cache.get(ep.task_id)
        if f is None:
            from arcjepa.data import family_of

            f = family_of(Task(ep.task_id, list(ep.context), [Pair(ep.test_input, ep.target_output or [])]))
            cache[ep.task_id] = f
        families.append(f)

    n_val = int(bcfg.get("val_episodes", 64) or 0)
    val_eps: List[Episode] = []
    if n_val > 0:
        for sp in hf_splits:
            val_eps += stream_episodes(root, "episodes", sp, lambda t: t in val_ids, n_val - len(val_eps))
            if len(val_eps) >= n_val:
                break
    return episodes, families, val_eps


def build_stage_b(ctx: TrainContext) -> Optional[StageSpec]:
    """Stage B spec (``None`` when the data mirror is unavailable or yields no episodes)."""
    from arcjepa.data.hf_loader import DataRootNotFound

    try:
        episodes, families, val_eps = load_real_data(ctx.cfg)
    except (DataRootNotFound, FileNotFoundError) as exc:
        log.warning("stage B: real data unavailable (%s); skipping", exc)
        return None
    if not episodes:
        log.warning("stage B: no episodes after filtering; skipping")
        return None
    scfg: Dict[str, Any] = dict(cfg_get(ctx.cfg, "stages.B", {}) or {})
    loss = {"program_loss": 0.0, "ranking_loss": 0.0}
    loss.update(scfg.get("loss") or {})
    scfg["loss"] = loss
    max_ctx = int(scfg.get("max_ctx", 10))
    out_obj = bool(cfg_get(ctx.cfg, "data.out_objects", True))
    ds = RealEpisodeDataset(episodes, ctx.parser, max_ctx=max_ctx, out_objects=out_obj)
    weights = family_weights(families, float(scfg.get("family_alpha", 0.5)))
    ctx.metrics.log({"kind": "data", "stage": STAGE, "episodes": len(episodes), "val_episodes": len(val_eps),
                     **{f"family_{k}": v for k, v in sorted(Counter(families).items())}})
    val_ds = RealEpisodeDataset(val_eps, ctx.parser, max_ctx=max_ctx, out_objects=out_obj)

    def val_fn(c: TrainContext) -> Dict[str, float]:
        return eval_loss(c, val_ds, STAGE, max_items=len(val_ds), batch_size=int(scfg.get("batch_size", 8)))

    return StageSpec(name=STAGE, cfg=scfg, dataset=ds, weights=weights, use_programs=False, use_negatives=False,
                     val_fn=val_fn if len(val_ds) else None)


def run_stage_b(ctx: TrainContext, budget_s: float) -> Dict[str, Any]:
    """Run stage B for at most ``budget_s`` seconds; returns the stage summary."""
    spec = build_stage_b(ctx)
    if spec is None:
        ctx.completed.append(STAGE)
        return {"stage": STAGE, "skipped": "no real data"}
    return run_stage(ctx, spec, budget_s)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """CLI: run only stage B (same flags as ``train_all``)."""
    import sys

    from arcjepa.training.train_all import main as train_all_main

    args = list(argv) if argv is not None else sys.argv[1:]
    return train_all_main(args + ["--stages", STAGE, "--no-export"])


if __name__ == "__main__":  # pragma: no cover
    main()
