# ARC-AGI-2 / ARC-JEPA — session log, 2026-09-25 → 2026-09-27

Everything below is measured or read from a file, the Kaggle CLI or a workflow journal. No number is projected.
Account used throughout: Kaggle `poby7722` (team BlackBox), Hugging Face `koushikz1`, GitHub `acco-cyber`.

## 1. Where things stand (2026-09-27 13:50 +06:00)

| item | state |
|---|---|
| Running Kaggle kernels | **none** (all stopped; see §6) |
| Running local jobs / workflows | **none** |
| Best team LB score | 33.89 (June, NVARC-recipe notebook, not this work) |
| ARC-JEPA on the LB | v1 = **0.00** (submission 56586239) |
| ARC-JEPA Hard-180 (task pass@2, symbolic only, 60 s/task) | **24/180 = 13.3 %** |
| ARC-JEPA Val-150 (same setting) | **39/150 = 26.0 %** |
| Target of the v2 brief | 100/180 = 55.6 % — **not reached; not claimed** |
| Code | 360 tests pass; pushed to GitHub `acco-cyber/ARC-AGI-2` under `arc-jepa/` |

## 2. Timeline

| when (+06:00) | what happened |
|---|---|
| 09-25 | Competition analysis (6 research agents, 3 designs, judges, refutations): `arc2/wf50/PLAN_50_90.md`. Leaders (Tufa 83, rabbithole 77, nvbanana 74, Yi-Chia Chen 55) all run private LLM-class weights; no JEPA/latent model is behind any score above 40; P(≥50 by 11-02) estimated 5–10 %. |
| 09-25 | v6 NVARC-recipe solver pushed (`poby7722/arc2-v6-swarm-solver`) and submitted: **LB 30.28**. |
| 09-25 | HF dataset `koushikz1/arc-agi-2-jepa-episodes` built from the official tasks (tasks, masked-demo episodes, ×8 augmentations, ARC-GEN fresh episodes, hard SDG puzzles, counterfactual negatives, rule programs). |
| 09-25 | Probe of the public Qwen3-8B Nemotron grid-direct model: 100 % truncated at 1,500 tokens, 61 tok/s on HF transformers → parked. |
| 09-26 | ARC-JEPA built from the user's frozen spec by parallel agents: typed 72-op DSL + interpreter, 10-hypothesis object parser, hierarchical transformation-JEPA, synthetic generator, latent-guided search (beam / repair / TTA / A* / evolution / diverse top-2), training stages A–D, Kaggle notebooks. 299 tests. |
| 09-26 | Kaggle: 2-batch-GPU-session limit hit; v6 notebook deleted on request ("off other notebook"). Train smoke run (debug config) completed on 4×L4. |
| 09-26 | Review found: sdg_hard leak of public-eval parents into Stage B, torchrun export vs NCCL timeout, CPU export fallback, identity-grid attempts, pool-crash handling → fixed. |
| 09-26 | Infer v1 (smoke model) on the 120 public-eval tasks: **0.0083** (1/172 outputs, 0 exact-fit tasks, 48 near misses ≥ 90 % cells). Submitted → **LB 0.00**. |
| 09-27 01:00 | v1 published: HF `koushikz1/arc-jepa` (tag v1), GitHub release `arc-jepa-v1`, Kaggle model `poby7722/arc-jepa`. |
| 09-27 | Solve-rate audit on Val-150: 0.12 at 20 s = same at 90 s → the DSL, not the budget, is the limit (110 of 132 failures not expressible). DSL coverage added (86 ops: scaling, panel boolean, line connection, bbox fill, …; property→colour induction; relational ArgPool) → **Val-150 0.26 (39/150), 21 gained, 0 lost**. |
| 09-27 10:38 | Full training kernel pushed; stayed QUEUED. Kaggle quota API: `poby7722` has **6 GPU-h/week** (L4×4 burns 2×) → the 7.5 h run could never finish; replaced by a 1.5 h `arc-jepa-train-v2`, also never started. |
| 09-27 12:00 | User's v2 brief: locked Hard-180 harness, target 100/180. Split frozen: Train-670 / Val-150 / Hard-180 (sha256 `109f247cbc0ccf729e1a66d0747bcb2fde5e68c3715a90f76a271e94d447f9db`). |
| 09-27 13:28 | Step 1 measured: symbolic-only Hard-180 **24/180**. |
| 09-27 13:48 | User: "turn off everything". Queued `arc-jepa-train-v2` deleted (source saved in `kaggle_runs/train_v2_source/`). |

## 3. Kaggle submissions (this period)

| ref | date (UTC) | what | public LB |
|---|---|---|---|
| 56555583 | 09-25 16:38 | v6 NVARC-recipe swarm solver, seed 0 | 30.28 |
| 56586239 | 09-26 18:25 | ARC-JEPA v1 (smoke model + DSL search) | 0.00 |

## 4. ARC-JEPA measured results

| setting | split | task pass@2 | outputs | exact-fit | near-90 % | s/task |
|---|---|---|---|---|---|---|
| v1 smoke model + 72-op DSL (Kaggle) | public eval 120 | 1 task-output (0.0083) | 1/172 | 0 | 48 | ~6 (16 workers) |
| symbolic, 72+4 ops, 20 s | Val-150 | 18 (0.12) | 18/163 | | | 14.1 |
| symbolic, 86 ops + induce + ArgPool, 20 s | Val-150 | 39 (0.26) | | | | 14.1 |
| same, 60 s | Val-150 | **39 (0.260)** | 40/163 | 40 | 68 | 40.1 |
| same, 60 s | **Hard-180** | **24 (0.133)** | 27/199 | 26 | 54 | 47.0 |
| ↳ clean 150 (old holdout) | Hard-180 | 22 (0.147) | 25/160 | 24 | 47 | |
| ↳ 30 moved from old train | Hard-180 | 2 (0.067) | 2/39 | 2 | 7 | |

Hard-180 failure categories: solved 24 · wrong_shape 30 · exact_fit_wrong_on_test 3 · near_miss_90 30 ·
no_candidate_close 93. Per-task records: `reports/symbolic_v2dsl_h180_per_task.jsonl`.

## 5. v2 brief — status by step

| step | status |
|---|---|
| H harness (`python -m arcjepa.eval.hard180 --model … --budget 60`) | done |
| 1 symbolic-only Hard-180 | done: 24/180 |
| 2 full-v1 model prior on Hard-180 | **blocked**: no full model was ever trained (quota) |
| 3 ceilings | pending (needs 2) |
| 4 Search V2 (TypedExprBank, sketches, constraints, macros) | not started |
| 5 RuleBank adapter (4×256 slots, 72-op head, 256→64→256 adapter) | not started |
| 6 hard-negative mining | not started |
| 7 task-local TTA (64-d context, 4 steps) | not started |
| 8 ablations, publish v2 | not started |

Honest read: symbolic search reaches 13.3 % on Hard-180 and the failures are dominated by transformations the DSL
cannot express (93 tasks with no close candidate). The brief's 55.6 % would need roughly 4× the current solve count.

## 6. What was turned off / deleted

| resource | action | why |
|---|---|---|
| `poby7722/arc2-v6-swarm-solver` | deleted 09-26 | held both GPU batch slots; source in `arc2/deploy/poby_v6/` |
| `poby7722/arc-jepa-train` | deleted 09-27 (by the sibling session) | 7.5 h run could not fit the 6 h/week quota |
| `poby7722/arc-jepa-train-v2` | deleted 09-27 13:48 (queued, never ran) | user request; source in `kaggle_runs/train_v2_source/` |
| background watchers / workflows | none left running | |

Kept: Kaggle datasets `poby7722/arc-jepa-code`, `poby7722/arc-agi-2-jepa-episodes`; kernel `poby7722/arc-jepa-infer`
(complete); model `poby7722/arc-jepa`; HF `koushikz1/arc-jepa` + `koushikz1/arc-agi-2-jepa-episodes`.

## 7. How to continue on a new account

1. **Kaggle token** for the new account in `~/.kaggle/access_token` (or `KAGGLE_API_TOKEN`). Check the GPU quota
   first (the old account had 6 h/week, which cannot train the full model).
2. **Re-create the two datasets** under the new owner: `arc2/dataset/hf_arc2_episodes` (episodes; `dataset-metadata.json`
   id → `<owner>/arc-agi-2-jepa-episodes`) and the package (`launch_full_training.ps1` stages it as
   `<owner>/arc-jepa-code`).
3. **Change the owner** in `arc-jepa/kaggle/build_train_nb.py` / `build_infer_nb.py` (`OWNER = "poby7722"`), rebuild
   with `python kaggle/build_train_nb.py --full` and `python kaggle/build_infer_nb.py`, then push `kaggle/train_full/`.
4. After training: `python -m arcjepa.eval.hard180 --model <arc_jepa_pkg> --budget 60` (step 2), then steps 3–8 of
   `docs/V2_PROGRESS.md`.
5. Publishing: `publish_version.py` (targets HF `koushikz1/arc-jepa`, GitHub `acco-cyber/ARC-AGI-2`, Kaggle model
   `poby7722/arc-jepa` — change `KG_OWNER` for the new Kaggle account).

## 8. Where things are

* Code: `arc-jepa/` (package `arcjepa`, tests, configs, Kaggle notebooks, docs). Start with `docs/V2_PROGRESS.md`.
* Reports: `arc-jepa/reports/` (Hard-180 and Val-150 summaries + per-task JSONL).
* Audit: `arc-jepa/docs/SOLVE_RATE_AUDIT.md`. Reviews: `docs/REVIEW_2026-09-26.json`, `docs/INTEGRATION_ROUND_FINAL.md`.
* Kaggle run artefacts: `kaggle_runs/` (smoke training report, v1 inference log and metrics, train-v2 source).
* Tooling: `publish_version.py`, `push_github.py`, `launch_full_training.ps1`, `watch_kernel.ps1`.
