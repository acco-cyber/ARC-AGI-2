# ARC-JEPA module contracts (binding for every implementation agent)

Package root: `E:\Claude code\arc2\ARC-AGI-2\arc-jepa\` (this folder is pushed to GitHub `acco-cyber/ARC-AGI-2`
under `arc-jepa/`). Python package: `arcjepa/`. Python 3.11, PyTorch ≥ 2.6 (CPU locally; CUDA on Kaggle),
numpy, pyyaml, pytest. No other hard dependencies (no faiss, no HF `datasets` at inference time; the HF loader
may use `datasets` only when `ARCJEPA_HF=1`, otherwise it reads the local JSONL mirror).

Read `docs/FROZEN_SPEC.md` first. Where this file and the spec disagree on a concrete API, THIS file wins;
where this file is silent, follow the spec. Every module ships `tests/test_<module>.py` that passes on CPU in
under 60 s with `python -m pytest tests/test_<module>.py -q` from the package root. Code style: type hints,
docstrings on public functions, no global mutable state except the primitive REGISTRY, deterministic given a
seed, no prints in library code (use `logging`).

## 0. Core types — `arcjepa/core/types.py` (ALREADY WRITTEN; do not change signatures)
```python
Grid = List[List[int]]                      # ints 0..9, 1 <= H, W <= 30, rectangular
PAD_ID = 10                                 # tensor padding colour
MAX_SIDE = 30
@dataclass(frozen=True) class Pair: input: Grid; output: Grid
@dataclass class Task: task_id: str; train: List[Pair]; test: List[Pair]   # test[i].output may be [] when unknown
@dataclass class Episode: episode_id: str; task_id: str; split: str; context: List[Pair];
                          test_input: Grid; target_output: Optional[Grid]; source: str = "arc"
def grid_shape(g) -> Tuple[int,int]; def validate_grid(g) -> bool; def grids_equal(a,b) -> bool
def task_from_json(task_id, d) -> Task; def episode_from_json(d) -> Episode
```

## 1. DSL — `arcjepa/dsl/` (agent A1)
* `types.py`: `class T(str, Enum)` with exactly GRID, OBJECT_SET, OBJECT, MASK, COLOR, POSITION, INTEGER,
  BOOLEAN, RELATION, PROGRAM. Literal domains: COLOR ∈ 0..9; POSITION ∈ {(dr,dc): dr,dc ∈ −3..3}∪{named
  anchors "center","top","bottom","left","right"}; INTEGER ∈ 0..9 (+ "n_objects" style computed values);
  BOOLEAN literals True/False.
* `ast.py`: `@dataclass(frozen=True) class Node: op: str; args: Tuple["Node | int | str | Tuple[int,int]", ...] = ()`.
  Leaves: `Node("INPUT")` (the task input grid, type GRID); literals are raw Python values in `args`.
  Methods: `depth()`, `size()`, `primitives() -> Set[str]`, `to_str()` (S-expression, e.g.
  `(RECOLOR (SELECT_LARGEST (GET_COMPONENTS4 INPUT)) 3)`), `Node.from_str(s)` (exact inverse),
  `children()`, `replace(path: Tuple[int,...], new: Node) -> Node`, `paths() -> List[Tuple[int,...]]`.
  A **Program** is a Node of output type GRID (every value-typed program must end in a GRID-producing op;
  object-typed results are rendered with `RENDER: OBJECT_SET|OBJECT × GRID → GRID` = paint objects onto the
  input-sized canvas, and `PAINT_ON_BLANK` variants where the spec needs them — count these as part of the
  72 only if you keep the total at 72 by folding RENDER into COMPOSE; otherwise document the extra helpers as
  "structural ops" outside the 72).
* `primitives.py`: `@dataclass class Primitive: name: str; arg_types: Tuple[T,...]; out_type: T;
  fn: Callable; category: str; literal_args: Dict[int, Sequence] = {}` (index → allowed literal values);
  `REGISTRY: Dict[str, Primitive]` containing the 72 spec names (plus structural helpers, flagged
  `structural=True`). `by_out_type(t: T) -> List[Primitive]`. Every fn is pure and total over its typed
  domain or raises `ExecError`.
* `interpreter.py`: `class ExecError(Exception)`; `execute(prog: Node, grid: Grid, *, max_steps: int = 10_000,
  max_cells: int = 900, timeout_s: float = 0.5) -> Grid` — evaluates the AST with type checking, returns a
  valid Grid or raises ExecError (never returns None, never hangs; enforce H,W ≤ 30). `typecheck(prog) -> T`
  raises `TypeError` on mismatch. Values: GRID = Grid; OBJECT = `arcjepa.parser.objects.Object`;
  OBJECT_SET = `List[Object]`; MASK = `List[List[bool]]`; COLOR/INTEGER = int; POSITION = (dr,dc);
  BOOLEAN = bool.
* `grammar.py`: `expansions(target: T, depth_left: int) -> List[Primitive]`; `random_program(rng: random.Random,
  depth: int, category: Optional[str] = None) -> Node` (typed, executable on a random grid with prob ≥ 0.5);
  `enumerate_programs(max_depth: int, max_count: int) -> Iterator[Node]` (breadth-first, type-constrained,
  canonical-deduplicated).
* `canonicalize.py`: `canonicalize(node: Node) -> Node` (rotation/reflection composition table, identity
  removal MOVE(x,(0,0)), idempotents, argument ordering for commutative ops); `structural_signature(node) ->
  str` (op skeleton without literals) used by search/diversity.
* `mutations.py`: `mutate(rng, node) -> Node` (type-preserving single edit); `hard_negatives(rng, node, k=8) ->
  List[Node]` covering the 8 spec types; `crossover(rng, a, b) -> Node`.
* Tests: every primitive typechecks and executes on ≥3 random grids; interpreter determinism; from_str∘to_str
  identity on 500 random programs; canonicalize(ROTATE90∘ROTATE90) == ROTATE180; enumerate_programs(2, 2000)
  yields ≥ 500 distinct canonical programs; 10 hand-written ARC-like programs reproduce expected outputs.

## 2. Parser — `arcjepa/parser/` (agent A2)
* `objects.py`: `@dataclass(frozen=True) class Object: cells: FrozenSet[Tuple[int,int]]; color_hist:
  Tuple[Tuple[int,int],...]; primary_color: int; bbox: Tuple[int,int,int,int]  # r0,c0,r1,c1 inclusive`;
  properties `area, h, w, centroid, aspect, density, perimeter, holes, sym_h, sym_v, sym_d1, sym_d2,
  orientation, touches (top,bottom,left,right)`; `features(grid_h, grid_w) -> np.ndarray[float32, 32]` (24
  deterministic features, normalised to [0,1] where natural, zero-padded to 32); `crop(pad_to: int = 30) ->
  np.ndarray[int8, pad_to, pad_to]` (colours, PAD_ID outside); `paint(canvas: Grid, color: Optional[int] = None)
  -> Grid`; `translate(dr, dc) -> Object`; `recolor(c) -> Object`; `mask(h,w) -> List[List[bool]]`.
* `segmentation.py`: `HYPOTHESES: Tuple[str,...] = ("cc4","cc8","per_color_cc4","color_agnostic_cc8","rows",
  "cols","rect_regions","frames","repeat_blocks","symmetry")`; `segment(grid: Grid, hypothesis: str,
  background: int = 0) -> List[Object]` (deterministic order: by (r0,c0,−area)); `all_hypotheses(grid) ->
  Dict[str, List[Object]]`; every hypothesis must return within 20 ms on a 30×30 grid.
* `relations.py`: `RELATIONS: Tuple[str,...]` = the 18 spec names in spec order; `relation_features(a: Object,
  b: Object, grid_h, grid_w) -> np.ndarray[float32, 24]` (18 binary + Δr, Δc, dist, IoU, size_ratio, pad);
  `relation_matrix(objs, h, w) -> np.ndarray[float32, N, N, 24]`.
* `hypotheses.py`: `default_hypothesis(grid) -> str` (cc4 unless >64 objects, then per_color_cc4, then cc8,
  ... first with ≤64); `parse(grid, hypothesis=None, max_objects=64) -> Tuple[List[Object], np.ndarray[N,32],
  np.ndarray[N,N,24]]`.
* Tests: known grids → expected object counts/bboxes for each hypothesis; features in range; relation symmetry
  (left_of(a,b) ⇔ right_of(b,a)); ≤64 cap; timing.

## 3. Synthetic — `arcjepa/synthetic/` (agent A3, after A1+A2)
* `generators.py`: `random_input_grid(rng, *, h=None, w=None, n_objects=None, palette=None, background=0,
  style: str = "objects") -> Grid` styles: objects, lines, tiles, noise_sparse, frames, symmetric.
* `program_sampler.py`: `sample_program(rng, category: Optional[str]=None) -> Node` with the spec depth mix and
  category mix; `DEPTH_MIX`, `CATEGORY_MIX` constants.
* `dataset.py`: `@dataclass class SynthTask: task_id: str; program: str; pairs: List[Pair]; depth: int;
  primitives: List[str]; category: str; difficulty: float; adversarial: bool`; `make_task(rng, *, n_pairs=(3,6))
  -> Optional[SynthTask]` returns None if degenerate (any pair output == input for all pairs, constant outputs,
  ExecError, non-determinism check by re-execution, >30 side); `generate(n: int, out_path: str, seed: int,
  workers: int = 1, split_rule: str = "compositional") -> Dict[str,int]` writes JSONL rows
  `{task_id, program, pairs:[{input,output}], depth, primitives, category, difficulty, adversarial, split}`
  where `split` ∈ {train, val_comp} by the compositional rule (held-out primitive PAIRS listed in
  `HELDOUT_COMPOSITIONS`), throughput target ≥ 300 tasks/s/worker on CPU.
* `perturbations.py`: `add_distractors(rng, task) -> SynthTask`, `ambiguous_segmentation(rng, task)`.
* Tests: 500 generated tasks all re-execute exactly; depth histogram within ±5 pp of the mix; split leakage
  check (no val_comp composition appears in train).

## 4. Data — `arcjepa/data/` (agent A4)
* `hf_loader.py`: `load_tasks(root: str) -> Dict[str, Task]` and `load_episodes(root, config: str, split: str)
  -> List[Episode]` reading the local mirror layout `<root>/<config>/<split>.jsonl` (configs as in the HF card;
  `root` defaults to env `ARCJEPA_DATA` or `E:\Claude code\arc2\dataset\hf_arc2_episodes`; on Kaggle
  `/kaggle/input/datasets/poby7722/arc-agi-2-jepa-episodes` — probe both). `resplit_700_150_150(task_ids:
  Sequence[str], families: Dict[str,str], seed: int = 20260926) -> Dict[str, List[str]]` family-balanced,
  deterministic; write/read `splits_700_150_150.json`. `eval_public` is never returned by any training
  loader (assert).
* `families.py`: `family_of(task: Task) -> str` heuristic over the 8 spec families using shape relation,
  colour change, symmetry, object count change, tiling detection; returns one of the 8 names.
* `tensorize.py`: `grid_to_tensor(g) -> torch.LongTensor[30,30]` (PAD_ID fill) + `grid_mask(g) ->
  BoolTensor[30,30]`; `encode_episode(ep, parser) -> Dict[str, Tensor]` with keys
  `ctx_in [K,30,30], ctx_out [K,30,30], ctx_mask [K], test_in [30,30], target [30,30] (PAD if None),
  obj_feats [K+1, 64, 32], obj_crops [K+1, 64, 30, 30], obj_mask [K+1, 64], rel_feats [K+1, 64, 64, 24]`
  (K ≤ 10 demos, padded); `collate(batch) -> Dict[str, Tensor]` stacking with padding; a
  `torch.utils.data.Dataset` wrapper `EpisodeDataset(episodes, parser, max_ctx=10)`.
* Tests: loader counts equal the HF card (tasks 1120; episodes 6077); resplit sizes 700/150/150 and disjoint;
  tensor shapes; round-trip grid ↔ tensor.

## 5. Model — `arcjepa/model/` (agent A5)
All dims from the spec; config dataclass `ModelConfig` in `arcjepa/model/config.py` with a `tiny()` preset
(d 64, 2 layers everywhere) for CPU tests. Modules and forward signatures:
* `cell_encoder.py`: `CellEncoder(cfg)`: `forward(grid: Long[B,30,30], mask: Bool[B,30,30]) ->
  (tokens Float[B,900,256], pooled Float[B,256])`.
* `object_encoder.py`: `ObjectEncoder(cfg)`: `forward(crops Int[B,64,30,30], feats Float[B,64,32], mask
  Bool[B,64]) -> (obj_tokens Float[B,64,256], pooled Float[B,256])`.
* `relation_encoder.py`: `RelationEncoder(cfg)`: `forward(obj_tokens, rel_feats Float[B,64,64,24], mask) ->
  (rel_tokens Float[B,64,64,256], pooled Float[B,256])` (implement pair mixing efficiently; may subsample pairs
  to ≤ 512 per grid).
* `jepa_encoder.py`: `GridJEPAEncoder(cfg)`: `forward(batch_grid_dict) -> Dict{z: Float[B,512], z_global
  Float[B,384], obj_tokens, rel_pooled, cell_pooled}`.
* `target_encoder.py`: `EMATargetEncoder(online: GridJEPAEncoder, tau_start=0.996, tau_end=0.9995)`:
  `update(step, total_steps)`, `forward(...)` under `torch.no_grad()`.
* `predictor.py`: `TransformationPredictor(cfg)`: `forward(z_x Float[B,512], r_task Float[B,256], obj_tokens_x
  Float[B,64,256], obj_mask) -> Dict{z_hat Float[B,512], obj_hat Float[B,64,256], rel_hat Float[B,256]}`.
* `rule_latent.py`: `RuleLatent(cfg)`: `forward(z_x Float[B,K,512], z_y Float[B,K,512], k_mask Bool[B,K]) ->
  Float[B,256]` (MLP 1536→1024→512→256 per demo, attention pool).
* `program_encoder.py`: `ProgramTokenizer` (pre-order tokens: op ids + type ids + depth ids + literal ids;
  vocab from REGISTRY; `encode(node) -> List[int]`, `pad_batch`) and `ProgramEncoder(cfg)`:
  `forward(tokens Long[B,L], mask) -> Float[B,256]`.
* `scorer.py`: `Scorer(cfg)`: `forward(r Float[B,256], z_p Float[B,256]) -> Float[B]`.
* `memory.py`: `TransformationMemory`: `add(rule_latent np[256], program: str, complexity: int, family: str)`,
  `query(r, k=16) -> List[dict]` (cosine, numpy), `save(path.npz + .json)`, `load(path)`.
* `arcjepa.py`: `class ARCJEPA(nn.Module)` composing all of the above; `encode_grid(batch) -> z`,
  `rule_from_episode(batch) -> r_task` (uses ctx pairs), `predict(z_x, r) -> ẑ`, `score_programs(r, token_batch)
  -> scores`; `losses.py`: `jepa_losses(model, target_encoder, batch, programs_pos=None, programs_neg=None,
  weights=LossWeights()) -> Dict[str, Tensor]` implementing L_g, L_o, L_r, L_prog, L_rank (margin 0.2), L_var
  (γ 1.0) and `total`.
* Tests (tiny config, CPU): forward shapes; losses finite and decrease over 30 steps on a synthetic batch;
  EMA update changes target weights; program tokenizer round trip; parameter count of the full config printed
  and within 10–40 M.

## 6. Search — `arcjepa/search/` (agent A6, after A1; model API as above)
* `verifier.py`: `demo_error(prog: Node, pairs: Sequence[Pair]) -> Tuple[int, int]` = (#pairs mismatched,
  #cells mismatched over shape-matching pairs; shape mismatch counts as all cells); `is_exact(prog, pairs)`;
  `execute_safe(prog, grid) -> Optional[Grid]`.
* `candidate.py`: `@dataclass class Candidate: program: Node; score: float; demo_err: int; cell_err: int;
  neural: float; complexity: int; source: str`.
* `beam.py`: `beam_search(task_pairs, *, prior: Callable[[List[Node]], List[float]] | None, width=32,
  max_depth=6, top_primitives=8, alpha=1.0, beta=10.0, gamma=0.15, time_budget_s=5.0, seeds:
  Sequence[Node] = ()) -> List[Candidate]` type-constrained expansion from `Node("INPUT")`, partial programs
  completed greedily for scoring, returns candidates sorted by Score; must return within budget.
* `repair.py`: `repair(cands, pairs, rounds=1, rng=None, prior=None) -> List[Candidate]` local AST edits
  guided by differing-cell analysis.
* `tta.py`: `refine_rule_latent(model, r0, pairs, candidates, steps=8, lr=0.05, anchor=0.1) -> Tensor` where
  L_demo is made differentiable as a softmax-weighted expected demo error over the candidate set
  (weights = softmax of scorer outputs) — document this as the v1 realisation of the spec's L_TTA.
* `astar.py`: `astar_search(pairs, prior, max_nodes=50_000, time_budget_s=10.0) -> List[Candidate]`.
* `evolution.py`: `evolve(pairs, seeds, prior, pop=32, gens=20, p_mut=0.4, p_cross=0.2, p_neural=0.4,
  time_budget_s=10.0) -> List[Candidate]`.
* `diversity.py`: `select_two(cands, test_input) -> Tuple[Grid, Grid, dict]`: cluster exact-fit candidates by
  `structural_signature` and by output on the test input; attempt_1 = best cluster's best program output;
  attempt_2 = best other cluster (else best non-exact candidate output with the right shape, else identity /
  most common demo output shape fallback). Outputs are always valid grids.
* `memory_prior.py`: `MemoryPrior(memory, model)`: `seeds_for(r_task, k=16) -> List[Node]`.
* `solver.py`: `@dataclass class SolveConfig` (all spec knobs + `per_task_seconds`); `difficulty(task,
  parsed) -> int` (D = 0.25 H(P) + 0.2 N_obj + 0.2 N_seg + 0.2 composition + 0.15 ambiguity → bucket 0–3);
  `solve_task(task: Task, model: Optional[ARCJEPA], cfg: SolveConfig) -> Tuple[List[Tuple[Grid,Grid]], dict]`
  (one (attempt_1, attempt_2) per test input, plus the spec's diagnostics dict). Works with `model=None`
  (pure symbolic prior = uniform) so the pipeline runs before any training.
* Tests: beam finds exact programs for ≥ 8 of 10 depth-≤2 synthetic tasks within 5 s each on CPU; repair fixes
  a planted single-node error; select_two never returns invalid grids; solver returns exactly two attempts per
  test input for 20 real training tasks within the budget (correctness not required).

## 7. Training — `arcjepa/training/` + `configs/` (agent A7, after A3/A4/A5)
* `configs/{base,debug,l4x4,kaggle}.yaml` per the spec's v1 config; `debug` = tiny model, 200 synthetic
  tasks, 20 steps.
* `pretrain_jepa.py` (Stage A), `train_program_encoder.py` (Stage C + D hard negatives), `train_real.py`
  (Stage B), `train_all.py` (runs A→B→C→D time-boxed by `--hours`, checkpoints every N minutes to
  `--out`, resumable); `torchrun`-compatible DDP with a single-process fallback; bf16 autocast on CUDA;
  logs JSONL metrics (`loss_*`, `retrieval@1/@8` on a held-out synthetic program set, `collapse_std`).
* `export.py`: writes `model.safetensors` (or `.pt`), `config.json`, `program_memory.npz + programs.json`,
  `vocab.json` into one folder.
* Tests: `python -m arcjepa.training.train_all --config configs/debug.yaml --hours 0.02 --out /tmp/x` completes
  on CPU and exports a loadable package.

## 8. Eval + Kaggle — `arcjepa/eval/`, `kaggle/` (agent A8, after A6/A7)
* `eval/evaluate.py`: `evaluate(tasks: Dict[str,Task], solver_fn, *, max_tasks=None) -> dict` computing the
  **competition metric** (mean over tasks of the fraction of test outputs where attempt_1 or attempt_2 is an
  exact match) plus per-family breakdown, search stats, and the per-task diagnostics JSON of the spec;
  `diagnostics.py`, `search_stats.py`, `error_analysis.py` (near-miss cell fractions).
* `kaggle/build_train_nb.py` → `kaggle/arc-jepa-train.ipynb` + `kernel-metadata.json` (id
  `poby7722/arc-jepa-train`, NvidiaL4, no internet, dataset_sources = code dataset `poby7722/arc-jepa-code`
  and episodes dataset `poby7722/arc-agi-2-jepa-episodes`, pinned image
  `gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868`):
  copies the code, generates synthetic phase-1 tasks in-session (CPU workers, ≤ 30 min), runs `train_all`
  time-boxed (`ARCJEPA_TRAIN_HOURS`, default 9.5), exports the package to `/kaggle/working/arc_jepa_pkg`.
* `kaggle/build_infer_nb.py` → `kaggle/arc-jepa-infer.ipynb` + metadata (id `poby7722/arc-jepa-infer`, sources:
  code dataset, competition, kernel_sources = `poby7722/arc-jepa-train`): dev mode solves `eval_public` (scores
  with `evaluate`), rerun mode (`KAGGLE_IS_COMPETITION_RERUN`) solves `arc-agi_test_challenges.json` within
  a global 11 h budget (fair-share per task, shortest-first, incremental `submission.json` rewrite every 60 s,
  fallback attempts for every test input from the start), validates the file, and works with `model=None`
  if the package is missing (pure symbolic mode) so it never fails to submit.
* `kaggle/validate_submission.py`: shape/format validator (2 attempts per test input, ints 0–9, 1–30 sides,
  all task ids present).
* Tests: notebooks validate with nbformat; every code cell py_compiles; the validator rejects malformed files.

## Return format for every agent
Return JSON: `{files: [...], public_api: [...], tests: {cmd, passed, failed}, timings: {...},
open_issues: [...], deviations_from_spec: [...]}`. Deviations must be justified in one line each.
