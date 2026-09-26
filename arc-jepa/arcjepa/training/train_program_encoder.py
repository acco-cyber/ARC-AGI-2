"""Stages C and D: program alignment and hard-negative ranking (spec §Training, run-order steps 2 and 5).

Stage C (program alignment): synthetic tasks with their exact generating programs. The program encoder, the
scorer and the rule-latent head train at lr 1e-4; the core JEPA (grid encoder + predictor) is "mostly frozen"
(``core_lr_mult``, 0 = frozen). Loss = JEPA terms + ``program_loss`` * L_prog (1 - cos(r_task, z_p)), with the
program weight raised to 1.0 ("0.5 -> 1.0 later").

Stage D (hard negatives): as C plus ``negatives`` (8) hard negatives per positive from
``arcjepa.dsl.mutations.hard_negatives`` (wrong primitive / argument / order / colour / object / relation /
under-complete / over-complete; topped up with ``mutate`` when a program admits fewer types) and the pairwise
margin loss L_rank = max(0, 0.2 - s(r, p+) + s(r, p-)) with weight 1.0.

Only synthetic ``train``-split programs are used; the compositional ``val_comp`` programs form the held-out
retrieval@1/@8 set evaluated during and after both stages.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

from arcjepa.training.common import StageSpec, SynthEpisodeDataset, TrainContext, cfg_get, run_stage

log = logging.getLogger(__name__)

HEAD_MODULES = ("program_encoder", "scorer", "rule_latent")


def _stage_cfg(ctx: TrainContext, name: str, defaults: Dict[str, float]) -> Dict[str, Any]:
    scfg: Dict[str, Any] = dict(cfg_get(ctx.cfg, f"stages.{name}", {}) or {})
    scfg.setdefault("train_modules", list(HEAD_MODULES))
    scfg.setdefault("core_lr_mult", 0.1)
    if scfg.get("n_samples") and not scfg.get("epoch_size"):
        scfg["epoch_size"] = int(scfg["n_samples"])
    loss = dict(defaults)
    loss.update(scfg.get("loss") or {})
    scfg["loss"] = loss
    return scfg


def build_program_stage(ctx: TrainContext, name: str, negatives: int) -> Optional[StageSpec]:
    """Spec for stage ``name`` (C: ``negatives=0``; D: ``negatives=k``) over synthetic train programs."""
    if ctx.store is None or not ctx.synth_train:
        log.warning("stage %s: no synthetic programs; skipping", name)
        return None
    defaults = {"program_loss": 1.0, "ranking_loss": 1.0 if negatives else 0.0}
    scfg = _stage_cfg(ctx, name, defaults)
    ds = SynthEpisodeDataset(ctx.store, ctx.synth_train, ctx.parser,
                             max_ctx=int(cfg_get(ctx.cfg, "synthetic.max_ctx", 5)),
                             out_objects=bool(cfg_get(ctx.cfg, "data.out_objects", True)),
                             with_program=True, negatives=negatives,
                             seed=int(ctx.cfg.get("seed", 0)) + (1 if name == "C" else 2))
    return StageSpec(name=name, cfg=scfg, dataset=ds, use_programs=True, use_negatives=negatives > 0)


def run_stage_c(ctx: TrainContext, budget_s: float) -> Dict[str, Any]:
    """Stage C: program alignment (positives only)."""
    spec = build_program_stage(ctx, "C", 0)
    if spec is None:
        ctx.completed.append("C")
        return {"stage": "C", "skipped": "no synthetic programs"}
    return run_stage(ctx, spec, budget_s)


def run_stage_d(ctx: TrainContext, budget_s: float) -> Dict[str, Any]:
    """Stage D: hard-negative pairwise ranking (``stages.D.negatives`` per positive, default 8)."""
    k = int(cfg_get(ctx.cfg, "stages.D.negatives", 8))
    spec = build_program_stage(ctx, "D", max(1, k))
    if spec is None:
        ctx.completed.append("D")
        return {"stage": "D", "skipped": "no synthetic programs"}
    return run_stage(ctx, spec, budget_s)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """CLI: run stages C and D only (same flags as ``train_all``; ``--stages`` may narrow it to C or D)."""
    import sys

    from arcjepa.training.train_all import main as train_all_main

    args = list(argv) if argv is not None else sys.argv[1:]
    if "--stages" not in args:
        args = args + ["--stages", "CD"]
    return train_all_main(args + ["--no-export"])


if __name__ == "__main__":  # pragma: no cover
    main()
