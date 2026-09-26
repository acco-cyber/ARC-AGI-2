"""Stage A: synthetic JEPA pretraining (spec §Training, run-order step 1).

Loss = 1.0 L_g + 1.0 L_o + 0.5 L_r + 0.05 L_var on synthetic program-generated episodes (the program terms
L_prog / L_rank are off in this stage). AdamW lr 1.5e-4, betas (0.9, 0.95), wd 0.05, warmup 5 of 50 epochs,
cosine, bf16 autocast on CUDA, grad clip 1.0, EMA target tau 0.996 -> 0.9995.

Standalone use (normally run through ``train_all``)::

    python -m arcjepa.training.pretrain_jepa --config configs/debug.yaml --hours 0.01 --out runs/stage_a
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

from arcjepa.training.common import StageSpec, SynthEpisodeDataset, TrainContext, cfg_get, run_stage

log = logging.getLogger(__name__)

STAGE = "A"


def build_stage_a(ctx: TrainContext) -> Optional[StageSpec]:
    """Stage A spec over the synthetic ``train`` split (``None`` when there is no synthetic data)."""
    if ctx.store is None or not ctx.synth_train:
        log.warning("stage A: no synthetic training tasks; skipping")
        return None
    scfg: Dict[str, Any] = dict(cfg_get(ctx.cfg, "stages.A", {}) or {})
    loss = {"program_loss": 0.0, "ranking_loss": 0.0}
    loss.update(scfg.get("loss") or {})
    scfg["loss"] = loss
    ds = SynthEpisodeDataset(ctx.store, ctx.synth_train, ctx.parser,
                             max_ctx=int(cfg_get(ctx.cfg, "synthetic.max_ctx", 5)),
                             out_objects=bool(cfg_get(ctx.cfg, "data.out_objects", True)),
                             with_program=False, negatives=0, seed=int(ctx.cfg.get("seed", 0)))
    return StageSpec(name=STAGE, cfg=scfg, dataset=ds, use_programs=False, use_negatives=False)


def run_stage_a(ctx: TrainContext, budget_s: float) -> Dict[str, Any]:
    """Run stage A for at most ``budget_s`` seconds; returns the stage summary."""
    spec = build_stage_a(ctx)
    if spec is None:
        ctx.completed.append(STAGE)
        return {"stage": STAGE, "skipped": "no synthetic data"}
    return run_stage(ctx, spec, budget_s)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """CLI: run only stage A (same flags as ``train_all``)."""
    from arcjepa.training.train_all import main as train_all_main

    args = list(argv) if argv is not None else None
    import sys

    args = args if args is not None else sys.argv[1:]
    return train_all_main(args + ["--stages", STAGE, "--no-export"])


if __name__ == "__main__":  # pragma: no cover
    main()
