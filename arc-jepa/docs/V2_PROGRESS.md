# ARC-JEPA v2 — progress tracker (source of truth across sessions)

Spec: the user's "ARC-JEPA v2" brief of 2026-09-27 (target 100/180 = 55.6 % task-level pass@2 on a locked
Hard-180 harness; never claim unless measured). Order: H harness → 1 symbolic-only H180 → 2 full-v1 model prior
H180 → 3 ceilings → 4 Search V2 → 5 RuleBank adapter → 6 hard-negative mining → 7 task-local TTA → 8 ablations →
publish v2 (HF `koushikz1/arc-jepa`, GitHub release `arc-jepa-v2`, Kaggle model `poby7722/arc-jepa`).

## Rules in force
* The 120 public-evaluation tasks are never used for training, adapter fitting, memory, tuning or search design.
  The existing locked-evaluation guard stays.
* Split of the 1,000 official training tasks: **670 train / 150 val / 180 Hard-180**, task-level, frozen
  (`data/splits_670_150_180.json` + sha256 pinned by a test).
  * Hard-180 = the 150 tasks of the old `holdout` split (never trained on, never inspected by any design step)
    + the 30 hardest tasks of the old 700 `train` split by an objective, pre-registered hardness score.
  * Val-150 = the old `val` split (already the development set: the solve-rate audit inspected all of it).
  * Train-670 = old train minus the 30 moved tasks.
  * Caveat for the queued full-v1 Kaggle run (pushed 09-27 10:38, old 700 split): 30 Hard-180 tasks were in its
    Stage-B data. Its Hard-180 result is reported on all 180 AND on the clean 150 subset.
* Development/tuning happens on Val-150. Hard-180 is an acceptance gate: a new configuration is rejected if
  Hard-180 falls by more than 2 tasks.
* sdg_hard: only puzzles whose recorded parent tasks are all in Train-670 may be used (143/228 derive from
  NVARC mixes with public-eval parents).

## Compute constraint (measured 09-27 13:15 via the Kaggle quota API)
Account poby7722 GPU quota = **21,600 s = 6 h per week** (not 30 h), refresh 2026-10-03 00:00 UTC; used 3,499 s;
L4x4 burns at 2x → ≈2.5 h of L4x4 wall left this week. The 7.5 h full run could never finish, so the queued
`poby7722/arc-jepa-train` was deleted and replaced by `poby7722/arc-jepa-train-v2` (full v1 config, 86-op DSL,
Train-670, ARCJEPA_TRAIN_HOURS 1.5, synthetic 15 min, hard cap `kaggle kernels push -t 7200`), pushed 13:20.
~1 quota-h stays for a short inference dev run needed to submit. More GPU time needs another team account's quota
or the 10-03 refresh.

## Status
| step | state | result (measured) |
|---|---|---|
| H harness `python -m arcjepa.eval.hard180` | DONE 09-27 13:20 (360 tests pass; split sha256 109f247c…f9db) | |
| 1 symbolic-only H180 (budget 60 s) | DONE 09-27 13:28 | **24/180 = 13.3 %** task pass@2 (clean150 22/150, old-train30 2/30); outputs 27/199; exact-fit 26/180; near-90 % 54/180; mean 47.0 s/task; Val-150 @60 s: 39/150 = 26.0 % |
| 2 full-v1 prior H180 | BLOCKED: no trained full model exists. poby7722 quota 6 h/week; both full-training kernels were deleted while still queued (09-27 13:20 and 13:48, user: "turn off everything"). Continue on a new account. | |
| 3 ceilings | pending | |
| 4 Search V2 | pending | |
| 5 RuleBank adapter | pending | |
| 6 hard-negative mining | pending | |
| 7 task-local TTA | pending | |
| 8 ablations | pending | |
| publish v2 | pending | |

## Reference numbers (measured before v2)
* v1 smoke model: public eval 0.0083 (1/172 outputs), Kaggle LB 0.00.
* Symbolic search, val150 @20 s: 0.12 (72+4 ops) → 0.26 (86 ops + induce + ArgPool relational, 39/150).
