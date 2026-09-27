# Solve-rate audit: symbolic solver, model=None, val split (2026-09-26 22:39)

Measurement only. No package file was modified. Scratch lives in `runs/audit/`.

## Setup

* **Code:** a frozen copy of `arcjepa/`, `configs/`, `data/` and `pyproject.toml` taken at 22:39 into
  `runs/audit/snapshot/`. Every run imports from there (checked: `arcjepa.__file__` resolves into the snapshot).
  An older copy from 19:27 was kept as `runs/audit/snapshot_0926_1927/`.
* **Tasks:** all 150 ids of `data/splits_700_150_150.json["val"]`, loaded through
  `arcjepa.data.hf_loader.load_tasks`. There are 163 test outputs.
* **Solver:** `arcjepa.search.solver.solve_task(task, None, SolveConfig(per_task_seconds=20))` with a
  `multiprocessing.Pool(6)`. Scoring uses `arcjepa.eval.evaluate` (the competition metric: per task, the fraction
  of test outputs where attempt 1 or attempt 2 is exact, averaged over tasks).
* **Reproduce** (from `runs/audit`, `$env:PYTHONPATH` set to `runs\audit\snapshot`):
  `python audit_snap.py val150_s20 20 6 150`

## Headline

| run | score | solved tasks | correct test outputs | mean wall s/task | max | total wall (6 procs) |
|---|---|---|---|---|---|---|
| **baseline (snapshot, 20 s)** | **0.120** | 18 / 150 | 18 / 163 | 14.05 | 19.34 | 487 s (3.25 s/task throughput) |
| same code at 30 s (earlier run, `val150_s30`) | 0.120 | the same 18 | 18 | 20.7 | 28.8 | 617 s |
| 132 failures re-run at 90 s (`fail132_s90`) | +1 task (`a8610ef7`) | | | | | |
| **+ proposed additions 1–5 (prototype, 20 s)** | **0.240** | 36 / 150 | | 12.97 | 20.01 | |
| + additions 1–5 + ArgPool extension (6) | 0.240 | 36 / 150 | | 12.89 | 19.32 | |

**More search time does not help:** 20 s, 30 s and 90 s solve the same tasks. The limit is what the DSL can
express, not the budget. No test finished with a solver error, and no test got fewer than two valid attempts.

Per family (baseline): pattern 5/11, object 6/57, context 3/44, relation 1/12, composition 1/15, counting 1/5,
geometry 1/1, symmetry 0/5. By difficulty bucket 0–3: 0.14 / 0.13 / 0.09 / 0.12.

Solved at baseline: 23b5c85d 2dee498d 358ba94e 3af2c5a8 46442a0e 67e8384a 6d75e8bb 7b6016b9 84db8fc4
9565186b aabf363d ae4f1146 c8f0f002 c909285e cd3c21df ea32f347 ed36ccf7 f0df5ff0.

## Failure classification (132 unsolved tasks)

The failing set at 20 s is identical to the one at 30 s. I checked that `runs/audit/classes.py` (a hand
classification after viewing every task) covers exactly these 132 ids. I also re-ran every "not reached" program
against the snapshot DSL with `check_snap.py`: each one fits every train pair and the test.

| class | n | meaning |
|---|---|---|
| not_expressible | 110 | No program in the current 72+4-op DSL within depth 6 produces the outputs. |
| not_reached | 6 | A DSL program exists (listed below) but the beam or pool never builds it. |
| wrong_on_test | 4 | An exact-fit program was found but it is wrong on the test input. |
| wrong_shape | 12 | No exact fit, and both attempts have the wrong output dimensions. |

**not_reached** (programs verified on the snapshot):

| task | program | why it is missed |
|---|---|---|
| 67385a82 | `(RENDER (APPLY_TO_EACH (FILTER (GET_COMPONENTS4 INPUT) LARGER (SELECT_SMALLEST (GET_COMPONENTS4 INPUT))) (RECOLOR OBJ 8)) INPUT)` | the FILTER set is not in ArgPool |
| 60b61512 | `(PATTERN_FILL (FILL (FILL INPUT (GET_BBOX (SELECT_LARGEST (GET_COMPONENTS8 INPUT))) 7) (GET_BBOX (FARTHEST (GET_COMPONENTS8 INPUT) (SELECT_LARGEST (GET_COMPONENTS8 INPUT)))) 7) (SELECT_NONZERO INPUT) INPUT)` | FARTHEST object is not in ArgPool |
| 6df30ad6 | `(MAP_COLOR (RENDER_BLANK (SELECT_COLOR (GET_COMPONENTS4 INPUT) 5) INPUT) 5 (ARGMAX_SIZE (DUPLICATE (NEAREST (GET_COMPONENTS4 INPUT) (SELECT_LARGEST (GET_COMPONENTS4 INPUT))) (0 0))))` | "colour of the NEAREST object" is not in ArgPool |
| a8610ef7 | `(FILL (MAP_COLOR INPUT 8 5) (SELECT_NONZERO (PATTERN_FILL INPUT (SELECT_NONZERO INPUT) (REFLECT_V INPUT))) 2)` | AND-mask with the mirror image is not in ArgPool (found at 90 s) |
| ea959feb | `(SWAP_COLORS (PERIODIC_REPEAT (SWAP_COLORS INPUT 0 1)) 0 1)` | depth-3 spine pruned by the beam (background colour is 1, not 0) |
| f5b8619d | `(TILE (PATTERN_FILL (MAP_COLOR (EXTEND (EXTEND INPUT (MERGE (GET_COMPONENTS4 INPUT)) (1 0)) (MERGE (GET_COMPONENTS4 INPUT)) (-1 0)) (MOST_COMMON_COLOR INPUT) 8) (SELECT_NONZERO INPUT) INPUT) 2 2)` | depth-5 spine pruned by the beam |

**wrong_on_test** (the exact fit is spurious):

* 52df9849: the fit `(FILL INPUT (GET_BBOX (SELECT_SMALLEST (SELECT_ALL INPUT))) (LEAST_COMMON_COLOR INPUT))` is
  wrong. The real rule is overlap layering.
* 7fe24cdd: 2 fits, both wrong. The real output is a 2x2 concatenation of the four rotations.
* 9ddd00f0: 9 fits (22 with the extended pool), all wrong. The real rule is D4 symmetry completion.
* d9fac9be: the fit `(CROP INPUT (SELECT_SMALLEST ...))` is wrong. The correct program
  `(CROP INPUT (SELECT_LARGEST (FILTER (GET_COMPONENTS4 INPUT) INSIDE (SELECT_LARGEST (GET_COMPONENTS4 INPUT)))))`
  fits train and test, and the ArgPool extension finds it.

**wrong_shape:** 22425bda 412b6263 47c1f68c 57edb29d 68bc2e87 90c28cc7 91413438 a3325580 ba1aa698 c920a713
d4c90558 d749d46f. Most need panel extraction or a summary output (colour list, bar chart).

## The 10 most frequent missing transformations

Counted over the 110 not_expressible tasks. Tasks marked * are fixed by the measured prototype additions
(see the next section).

| # | missing transformation | n | task ids |
|---|---|---|---|
| 1 | object motion: move to contact, border or partner; stacking; pivot rotation; colour-conditioned shift. Plain cell or object gravity fits none of them (checked in all 8 D4 conjugations) | 16 | 03560426 18286ef8 1b8318e3 1efba499 230f2e48 67c52801 7d7772cc 84551f4c 8dab14c2 90347967 9f669b64 b25e450b c9680e90 d687bc17 f28a3cbb f45f5ca7 |
| 2 | line drawing: segments between same-colour points, rays, diagonals, bridges, paths with turns | 16 | 0d87d2a6 264363fd 2bee17df* 3bd67248 55059096 69889d6e 7e2bad24 85fa5666 97239e3d b527c5c6 b7f8a4d8 d4a91cb9 dbc1a6ce* e5790162 ea786f4a f3b10344 |
| 3 | object-to-object correspondence: legend or key lookup, recolour by the adjacent marker, template stamping at markers | 16 | 17b866bd 1da012fc 2bcee788 33b52de3 3f23242b 447fd412 5b37cb25 6c434453 72322fa7 88207623 93b581b8 b7256dcd c3fa4749 d4469b4b f5c89df1 fc10701f |
| 4 | pattern or symmetry completion and extrapolation. A mask-colour D4 completion fits none of them, because the mask colour also occurs as a real colour | 10 | 045e512c 456873bc 5792cb4d 58c02a16 7447852a a096bf4d bae5c565 d22278a0 e3fe1151 e40b9e2f |
| 5 | panel logic: split at separator lines, then boolean AND/OR/XOR/NOR, overlay with priority, select, sort | 9 | 17cae0c1 1be83260 31d5ba1a* 75b8110e 94f9d214* dc2aa30b e84fef15 e99362f0 (fits at depth 1, not selected) fea12743 |
| 6 | count-conditioned output: output size or colour set by a count | 9 | 5289ad53 878187ab b0c4d837 b91ae062* c8b7cc0f dce56571 df8cc377 e048c9ed ff2825db |
| 7 | integer scaling: upscale, block downscale, self-Kronecker, fit-to-frame | 8 | 007bbfb7* 5614dbcf* 5783df64 60c09cac* 68b67ca3* 6b9890af 762cd429 e57337a4 |
| 8 | per-object property → colour: recolour or delete by size, hole size, orientation, number of marker cells | 6 | 1d61978c* 320afe60 52364a65 84f2aca1* a934301b* e8593010* |
| 9 | local neighbourhood rules: contour corners, rings around dots | 5 | 15663ba9 2a28add5 9772c176 a04b2602 db93a21d |
| 10 | assembly: jigsaw, fit pieces into holes | 4 | 846bdb03 97a05b5b a61ba2ce a8c38be5 |

Rarer: per-region fill (1c0d0a4b 8fbca751* a57f2f04), canvas growth or concatenation (15696249 cad67732 db118e2a),
coordinate-conditioned edits (ba26e723 d23f8c26 e7dd8335) and a 1x1 or summary output (19bb5feb 1a2e2828).
d23f8c26 is fixed by a one-line KEEP_CENTER_COL and ba26e723 by a MAP_COLOR restricted to every k-th column; both
fit exactly but are too narrow to recommend.

## The cheapest additions (measured end to end)

**How they were measured.** Each change was prototyped as a monkeypatch inside the worker processes, against the
snapshot (`runs/audit/levers_snap.py`, with the implementations in `brute_prims.py` and `brute_induce.py`). The
full 150-task val run used the same 20 s budget and a pool of 6. Result: 0.12 → **0.24**, 18 new tasks, **no
baseline task lost**, and mean wall went down (12.97 s against 14.05 s) because exact fits stop early.

| # | change (file · function) | tasks it fixes (measured) | confidence |
|---|---|---|---|
| 1 | `arcjepa/dsl/primitives.py` · register `UPSCALE(GRID, INTEGER{2..5})`, `DOWNSCALE(GRID, INTEGER)` (block majority), `DOWNSCALE_ANY(GRID, INTEGER)` (any non-zero in the block), `KRON_SELF(GRID)`, and upscale by #colours (`UPSCALE_NC(GRID)`, or a `COUNT_COLORS` INTEGER item in `arcjepa/search/beam.py` `ArgPool._build`) | 007bbfb7, 5614dbcf, 60c09cac, 68b67ca3, b91ae062 (+5). All are depth-1 fits found in under 2 s | high |
| 2 | new `arcjepa/search/induce.py` · `induce_recolor(pairs)`, called from `arcjepa/search/solver.py` `solve_task` before the beam. It segments objects (cc4, cc8, multicolour cc8/cc4, enclosed holes, 0-components), induces a property → colour/keep table from the demos (property is one of colour, size, shape, bbox, #enclosed, #minority cells, largest/smallest, size rank …), requires the table to be conflict-free and to cover the test objects, and emits its output as a Candidate / attempt 1 | 1d61978c, 67385a82, 84f2aca1, a934301b, e8593010 (+5). It fired on 10 tasks and all 10 were correct (5 of them were already solved): 0 false positives. Induction costs at most 2.8 s | high on val, medium for hidden (a covering table can be over-fitted) |
| 3 | `arcjepa/dsl/primitives.py` · `PANEL_BOOL(GRID, INTEGER{AND,OR,XOR,NOR,A-not-B,B-not-A}, COLOR)` and `PANEL_OVERLAY(GRID, INTEGER perm)`. The grid is split at uniform separator rows/columns, or into halves | 31d5ba1a, 94f9d214, 47c1f68c via `(MIRROR_TILE (PANEL_BOOL …))` (+3). e99362f0 fits `PANEL_OVERLAY 18` at depth 1 but was not selected | high |
| 4 | `arcjepa/dsl/primitives.py` · `CONNECT_SAME(GRID, COLOR)` (fill background between two same-coloured cells on a row or column; colour 0 = their own colour) and `FILL_EMPTY_LINES(GRID, COLOR)` | dbc1a6ce, 2bee17df (+2) | high |
| 5 | `arcjepa/dsl/primitives.py` · `BBOX_FILL(GRID, COLOR)` (fill the background inside every 8-connected object's bbox) | 60b61512, 8fbca751 (+2). 1c0d0a4b was also solved, through a long program that looks coincidental | medium-high |
| 6 (optional) | `arcjepa/search/beam.py` · `ArgPool._build`: add FILTER(rel, largest/smallest) sets, NEAREST/FARTHEST objects, the colour of an object, their bboxes, and AND-masks of INPUT with its D4 images | on top of 1–5: +d9fac9be, +6df30ad6, −1c0d0a4b (the coincidental solve is displaced). Net +1 | medium |

Brute force over all 150 val tasks, with each primitive at depth 1 and wrapped by any D4 transform, pre, post or
conjugated, found that gravity (cells or rigid objects), fill-enclosed, crop to a colour's bbox, D4 symmetry
completion, rays, majority colour per object, count → row, and noise removal fix **no** failing val task. They
are still generic ARC operations, but this split gives no evidence for them.

## Constraints for whoever implements this

* `primitives.py:1036` asserts `len(SPEC_PRIMITIVES) == 72`, and `tests/test_dsl.py::test_registry_has_exactly_the_72_spec_primitives`
  and `tests/test_model.py:264` pin 72. Register the new ops with `structural=True`, as CROP and RENDER are, or
  update the spec, the assert and both tests together.
* The program-encoder vocabulary is rebuilt from `REGISTRY`, and `ProgramTokenizer.load` restores the saved order.
  **Land new primitives before `poby7722/arc-jepa-train` runs.** Otherwise the packaged `vocab.json` and the
  program memory will not contain them. The symbolic search still runs, but the neural prior sees `<unk>` for the
  new ops, and the synthetic generator never samples them.
* Additions 1, 3, 4 and 5 are pure grid → grid functions and must raise `ExecError` on bad shapes (the prototypes
  do). Addition 2 is a solver stage and belongs inside the deadline accounting of `solve_task`.

## Files

`runs/audit/audit_snap.py`, `val150_s20.jsonl`, `val150_s20_eval.json`, `val150_s20.log` (baseline) ·
`classes.py` (per-task class and type), `check_snap.py` (verifies the not_reached programs) ·
`brute_prims.py`, `brute_prims.log`, `brute_induce.py`, `brute_induce.json` (depth-1 evidence) ·
`levers_snap.py`, `lever_dsl_induce_s20.jsonl/.log` (additions 1–5), `lever_all_s20.jsonl/.log` (1–6).
