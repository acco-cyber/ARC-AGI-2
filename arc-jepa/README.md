# ARC-JEPA — Object-Centric Transformation JEPA with Latent-Guided Program Search

An ARC-AGI-2 solver built from a frozen v1 specification (`docs/FROZEN_SPEC.md`): a multi-hypothesis object
parser, a hierarchical transformation-JEPA that predicts the latent of the transformed grid and distils a task
rule latent, a 72-primitive typed DSL with a deterministic interpreter, and a neural-guided typed program
search with exact demonstration verification, local AST repair, test-time latent refinement and diverse top-2
outputs. Training data: the HF dataset `koushikz1/arc-agi-2-jepa-episodes` (official ARC-AGI-2 tasks as
masked-demonstration episodes, ARC-GEN fresh episodes, hard synthetic puzzles, counterfactual negatives) plus
program-generated synthetic tasks from the DSL.

## Layout
```
arcjepa/core       shared types              arcjepa/model     encoders, predictor, rule latent, scorer, memory
arcjepa/dsl        typed DSL + interpreter   arcjepa/search    beam / A* / evolution, verifier, repair, TTA, solver
arcjepa/parser     objects, segmentation     arcjepa/training  stages A–D, configs, export
arcjepa/synthetic  program-generated tasks   arcjepa/eval      competition metric, diagnostics
arcjepa/data       HF loader, tensorization  kaggle/           train + inference notebooks, submission validator
tests/             pytest (CPU, < 60 s per module)
```

## Honest status (2026-09-27)
Measured: symbolic search (86 ops + induction), 60 s/task — **Hard-180 24/180 = 13.3 %**, Val-150 39/150 = 26.0 %
(task-level pass@2). v1 smoke model: public eval 0.0083, Kaggle LB 0.00. No full-size model has been trained yet
(GPU quota). Full log: `docs/SESSION_LOG.md`; step tracker for the v2 brief: `docs/V2_PROGRESS.md`. The spec's
accuracy gates are targets, not forecasts; no published system of this model class has a verified hidden-set
ARC-AGI-2 score above 27 %.

## Quick start
```bash
python -m pytest tests -q
python -m arcjepa.synthetic.dataset --n 2000 --out data/synth_debug.jsonl --seed 1
python -m arcjepa.training.train_all --config configs/debug.yaml --hours 0.05 --out runs/debug
python -m arcjepa.eval.evaluate --package runs/debug/export --split val --max-tasks 20
```
