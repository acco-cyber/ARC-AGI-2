# ARC-JEPA status log — 2026-09-27 00:00 (UTC+6)

## 1. Full GPU training: NOT STARTED YET
Waiting on the pre-training workflow (fixes → DSL coverage → integration → go/no-go).
Launch target: right after the go check, then ~7.5 h on Kaggle 4×L4 (`poby7722/arc-jepa-train`, full mode).

Workflow `arc-jepa-pretrain-finish` progress:
| step | state | result |
|---|---|---|
| training-path fixes | DONE | 316 tests pass; eval-leak (sdg_hard) removed; torchrun export timeout fixed; full-mode notebook ready |
| solve-rate audit | DONE | val split 18/150 solved (score 0.120) at 20 s/task; prototype DSL additions reach 36/150 (0.240) |
| inference fixes | RUNNING | diversity (no identity attempts), worker-crash re-queue, fail-loud dev mode |
| DSL coverage additions | queued | the audit's 5 additions, re-measured on the same 150 val tasks |
| integration + go/no-go | queued | full tests, CPU end-to-end, notebook rebuild |

## 2. Smoke training on the Kaggle GPU (the only training run so far, 09-26 14:48 UTC)
Debug config, 0.56 M-parameter model, 200 synthetic tasks, 32 s total — proves the pipeline, not a real model.
```
synthetic: 200 tasks generated at 318 tasks/s (depths 1-6: 46/53/51/31/15/4; 12 held-out compositions)
stage A (JEPA pretrain)    20 steps  loss_total 2.284 -> 1.787   retrieval@1 0.06  @8 0.50
stage B (real ARC adapt)   20 steps  val_loss_total 2.590
stage C (program align)    20 steps  loss_total 2.609 -> 2.432
stage D (hard negatives)   20 steps  loss_total 2.659 -> 2.085
export: /kaggle/working/arc_jepa_pkg (28 memory records, 2.4 s)
```

## 3. Inference on the 120 public evaluation tasks (smoke package, old code), Kaggle 4×L4
```
mode: dev (public evaluation) | package found | GPUs 4 | CPUs 48 | workers 16
COMPETITION METRIC (public eval, 120 tasks): 0.0083
outputs solved: 1 / 172 | solver errors 0 | exact programs found on 0 tasks
submission.json: 240 tasks, valid = True, 12.1 min elapsed
```
Not submitted (per plan: submit on 09-27 after the full training).

## 4. What the numbers say
* On the easier training-distribution validation split the search solves 12 % (24 % with the proposed DSL
  additions). On the real evaluation distribution it solves 0.8 %. Evaluation grids are 3.4× larger by median
  area and ARC-GEN covers none of them.
* The audit found that more search time does not help (20 s, 30 s and 90 s solve the same tasks): 110 of 132
  failures are transformations the DSL cannot express. Coverage, not training time, is the binding limit.
* Honest expectation for tomorrow's submission: low single digits on the leaderboard. 50 % is not reachable
  by this architecture on the evidence measured so far.

Files: `kaggle_runs/train_v1/` (smoke training output), `kaggle_runs/infer_v1/kernel_log.txt`,
`arc-jepa/docs/SOLVE_RATE_AUDIT.md`, `arc-jepa/docs/INTEGRATION_ROUND_FINAL.md`.
