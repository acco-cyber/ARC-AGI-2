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

## Honest status
See `docs/RESULTS.md` (written after each measured run). Every number there is measured; the spec's accuracy
gates are the spec's targets, not forecasts. For context: no published system of this model class has a
verified hidden-set ARC-AGI-2 score above 27 %.

## Quick start
```bash
python -m pytest tests -q
python -m arcjepa.synthetic.dataset --n 2000 --out data/synth_debug.jsonl --seed 1
python -m arcjepa.training.train_all --config configs/debug.yaml --hours 0.05 --out runs/debug
python -m arcjepa.eval.evaluate --package runs/debug/export --split val --max-tasks 20
```
