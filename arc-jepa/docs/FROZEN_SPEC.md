# ARC-JEPA v1 frozen specification (condensed from the user's design of 2026-09-26)

**Project:** ARC-JEPA — Object-Centric Transformation JEPA with Latent-Guided Program Search.
**Competition target:** Kaggle `arc-prize-2026-arc-agi-2`: 240 hidden tasks, two outputs per test input, exact match,
`submission.json`, L4×4 (4 × 22 GiB), 12 h, no internet. The 120 public evaluation tasks are never used for
development, training, memory, or tuning.

## Pipeline
```
ARC task → multi-hypothesis object parser → cell + object + relation representation → hierarchical
Transformation-JEPA → task transformation latent r_task → {program latent, transformation memory} →
neural-guided typed program search → exact symbolic verification → local AST repair / test-time latent
refinement → top-2 diverse outputs
```
The neural system answers "what kind of transformation is this?"; the symbolic system answers "which exact
program realizes it?". It is deliberately NOT a direct grid→grid predictor.

## Data
* Real tasks: 1,000 public training tasks → **700 train / 150 val (model selection) / 150 locked holdout**,
  task-level, family-balanced where possible (families: geometry, object, relation, counting, composition,
  context, symmetry, pattern). 120 public evaluation = final benchmark only. Source of record: the HF dataset
  `koushikz1/arc-agi-2-jepa-episodes` (configs tasks / episodes / episodes_aug / arcgen_fresh / sdg_hard /
  counterfactual / rule_programs), local mirror `E:\Claude code\arc2\dataset\hf_arc2_episodes`.
* Per grid: raw int8 H×W (colours 0–9, 0 = background, no per-task colour remap in the base representation),
  colour statistics (H, W, HW, #colours, #nonzero, density, symmetry), object segmentation hypotheses.
* Synthetic program-generated tasks provide volume: phase 1 100k, phase 2 500k, phase 3 1M. Each sample:
  `program, input_grids, output_grids, objects, relations, program_depth, primitive_set, difficulty`.
  Depth mix 25/25/20/15/10/5 % for depths 1–6. Category mix 40 % object-centric, 20 % geometry, 15 % relational,
  10 % counting, 10 % contextual, 5 % adversarial (distractor objects/colours, look-alike shapes, ambiguous
  segmentation, two local rules vs one global rule, order-sensitive compositions).
  **Compositional split:** primitives seen, compositions unseen in validation (e.g. train ROTATE+RECOLOR,
  MOVE+COPY, FILTER+MOVE; validate ROTATE+COPY+RECOLOR). Never random-split synthetic programs.
* Pseudo-programs for real tasks: run DSL search; keep a program as a label only if it fits ALL train pairs
  exactly; otherwise store transformation features only. Never fabricate symbolic labels.

## Object parser
Segmentation hypotheses S1..S10: cc4, cc8, per-colour components, colour-agnostic components, row segments,
column segments, repeated-pattern blocks, frame/container detection, largest rectangular regions,
symmetry-derived components. The model learns P(S_i | X). Max 64 objects per grid (pad + mask).
Object features (24 deterministic + learned shape embedding): id, cells, colour histogram, primary colour, area,
bbox, bbox h/w, centroid x/y, aspect ratio, density, perimeter, holes, symmetry h/v/d1/d2, orientation,
touches top/bottom/left/right. Shape encoder: crop ≤30×30 → 3×3 conv → 3×3 conv → global pool → 128; concat
32 deterministic → 160 → project 256. Relations per object pair: left_of, right_of, above, below, overlap,
touching, contains, inside, same_color, same_shape, aligned_x, aligned_y, nearest, farther, same_size, larger,
smaller, symmetric_to + continuous Δx, Δy, distance, IoU, size ratio → e_ij ∈ R^64.

## Typed DSL — 72 primitives
Types: GRID, OBJECT_SET, OBJECT, MASK, COLOR, POSITION, INTEGER, BOOLEAN, RELATION, PROGRAM.
* Selection (8): SELECT_ALL SELECT_COLOR SELECT_NONZERO SELECT_LARGEST SELECT_SMALLEST SELECT_UNIQUE SELECT_BORDER SELECT_CENTER
* Shape/object analysis (8): GET_COMPONENTS4 GET_COMPONENTS8 GET_BBOX GET_CENTROID GET_AREA GET_PERIMETER GET_HOLES GET_SYMMETRY
* Spatial relations (10): LEFT_OF RIGHT_OF ABOVE BELOW TOUCHING OVERLAPPING INSIDE CONTAINS NEAREST FARTHEST
* Geometric (10): ROTATE90 ROTATE180 ROTATE270 REFLECT_H REFLECT_V REFLECT_D1 REFLECT_D2 TRANSPOSE SHIFT ALIGN
* Object manipulation (12): COPY MOVE DELETE DUPLICATE MERGE SPLIT EXTEND SHRINK GROW FILL OUTLINE FRAME
* Colour (8): RECOLOR SWAP_COLORS MAP_COLOR MOST_COMMON_COLOR LEAST_COMMON_COLOR REPLACE_BACKGROUND COLOR_OBJECT COLOR_BY_POSITION
* Pattern (8): TILE REPEAT_X REPEAT_Y REPEAT_N MIRROR_TILE PATTERN_FILL ALTERNATE PERIODIC_REPEAT
* Counting (4): COUNT_OBJECTS COUNT_CELLS ARGMAX_SIZE ARGMIN_SIZE
* Conditional/composition (4): IF APPLY_TO_EACH FILTER COMPOSE
Typed signatures gate every expansion (e.g. SELECT_LARGEST: OBJECT_SET→OBJECT; GET_AREA: OBJECT→INTEGER;
RECOLOR: OBJECT×COLOR→OBJECT; MOVE: OBJECT×POSITION→OBJECT). Canonical AST: normalise equivalents
(ROTATE90∘ROTATE90 → ROTATE180, MOVE(...,0) removed, idempotent ops collapsed). Escape hatch `NEURAL_OP`
(a learned unknown operation) is a fallback only, never the primary solver.

## Model (~15–30 M params)
* Cell embedding: colour 32 + row 32 + column 32 + neighbour stats 32 = 128 → 256. Grid encoder: 12 pre-norm
  transformer blocks, d 256, 8 heads, FFN 1024, dropout 0.
* Object encoder: 4 blocks d 256 (object token + position + shape emb + attributes), ≤64 objects.
* Relation encoder: 4 blocks d 256 over (o_i, o_j, e_ij) → r_ij ∈ R^256.
* Hierarchical latent: z_cell 256, z_object 256, z_global 384 → concat 1024 → 512 = **z_X ∈ R^512**.
* Target encoder: EMA, τ 0.996 → 0.9995 (cosine), no gradient.
* Predictor: 6 blocks d 512, 8 heads, FFN 2048; input z_X + task context + target mask info → ẑ_Y ∈ R^512.
* Rule latent: r_i = MLP([z_X; z_Y; z_Y − z_X]) 1536→1024→512→256; r_task = AttentionPool(r_1..r_n) ∈ R^256.
* Program encoder: primitive emb 128 + type emb, 6 tree/transformer blocks d 256 → z_p ∈ R^256.
* Scorer: s_neural = MLP([r_task; z_p; r_task ⊙ z_p]) → R.
* Memory: `rules.faiss`-style KNN over (rule_latent 256, program, complexity, family), K = 16, built only
  from synthetic + training tasks (never from any evaluation set).

## Losses
L_g = ||ẑ_Y^global − sg(z_Y^global)||² (1.0) · L_o = mean_i ||ẑ_Y,i^obj − sg(z_Y,i^obj)||² (1.0) ·
L_r = mean_ij ||r̂_ij − sg(r_ij)||² (0.5) · L_prog = 1 − cos(r_task, z_p) (0.5 → 1.0 later) ·
L_rank = max(0, 0.2 − s(r,p⁺) + s(r,p⁻)) (0.5 → 1.0 later) · L_var = (1/d) Σ_j max(0, 1 − std(z_j)) (0.05).
L = 1.0 L_g + 1.0 L_o + 0.5 L_r + 0.5 L_prog + 0.5 L_rank + 0.05 L_var.

## Training
* Stage A synthetic JEPA pretraining: 1 M tasks, 50 epochs, global batch 512 (128/GPU × 4, accumulation 1),
  AdamW lr 1.5e-4, betas (0.9, 0.95), eps 1e-8, wd 0.05, warmup 5 epochs, cosine, bf16, grad clip 1.0.
* Stage B real ARC adaptation: 700 tasks, 32 tasks/GPU, lr 3e-5, 100 epochs, oversample rare families
  (never plain replication).
* Stage C program alignment: 250–500k exact-program samples, 30 epochs, lr 1e-4; train program encoder,
  scoring head, rule-latent projection; core JEPA mostly frozen.
* Stage D hard negatives: 8 per positive (wrong primitive / argument / order / colour / object / relation /
  under-complete / over-complete), pairwise ranking.
* L4×4: DDP over 4 GPUs, micro-batch 64 (or 32 × accumulation 2), global 256.
* Run order: 1 synthetic JEPA only (loss ↓, no collapse) → 2 + rule-latent alignment (Program Retrieval @1/@8)
  → 3 real adaptation (train exact-fit) → 4 search integration (holdout) → 5 hard negatives + TTA →
  6 full solver (accuracy/compute Pareto).

## Search
Score(p) = α s_neural − β L_demo − γ C(p), α 1.0, β 10.0, γ 0.15.
* Stage 1 neural beam: width 32, max depth 6, top-8 primitives per expansion by neural prior, type-constrained.
* Exact verification: E(p) = Σ_i 1[p(X_i) ≠ Y_i]; E = 0 is exact; stop early on exact + minimal/confident.
* Repair: for E > 0, map differing cells to AST nodes, mutate locally (e.g. MOVE_LEFT → MOVE_RIGHT/UP/DOWN/ALIGN).
* Test-time latent refinement: freeze weights, optimise r_task from r_0: 8 steps, lr 0.05,
  L_TTA = L_demo + 0.1 ||r − r_0||².
* Budget policy by difficulty D = 0.25 H(P) + 0.2 N_objects + 0.2 N_segments + 0.2 composition + 0.15 ambiguity:
  D0 32 beam / 1 repair · D1 64 / 2 · D2 128 / 4 + TTA · D3 128 + A* fallback + TTA + evolutionary repair.
* A* fallback: state (AST, remaining slots), g = complexity, h = −neural compatibility + estimated demo error,
  max 50,000 nodes. Evolutionary fallback: population 32, 20 generations, mutation 40 %, crossover 20 %,
  neural-guided mutation 40 %, fitness F = −10 E_demo + s_neural − 0.15 C(p).
* Two outputs: cluster exact-fit programs by structural equivalence; p1 = argmax Score(cluster 1),
  p2 = argmax Score(other cluster). Always exactly two predictions per test input; validate cells/dimensions.
* Runtime targets (median): parser <50 ms, JEPA <100 ms, beam <1 s, repair <3 s, A* <10 s per task.

## Kaggle inference notebook (no training)
imports → load packaged model → load DSL → load memory → utilities → load test tasks → parser → objects →
rule latent → neural candidates → symbolic search → repair → diversity → output-1/2 → write submission.json →
validation. Package offline: `model.safetensors, program_memory.npz, programs.json, config.json, vocab.json`.
No pip from internet, no downloads.

## Evaluation / research tables
Per-task diagnostics JSON (correct, candidate_rank, program_depth, objects, hypotheses, beam_expansions,
repair_rounds, tta_steps, inference_ms, rule_retrieval_r8). Tables: overall (baseline, object-symbolic, JEPA,
+program latent, +search, ARC-JEPA); cognitive categories; search efficiency (accuracy vs nodes/latency);
representation ablation (cell / +object / +relation / full); JEPA ablation. Key plots: accuracy vs log(search
nodes); Retrieval@K for K ∈ {1,4,8,16,32,64}; compositional generalisation (unseen 3-/4-way compositions).
Gates as written in the spec: synthetic held-out compositions >90 %; locked 150 >70 %; public 120 >75 %;
>80 % strong; 85 %+ competition-grade; 90 % stretch. (Evidence context: no system of this class has a
measured hidden-set ARC-AGI-2 score above 27 %; the gates are the spec's, not a forecast.)

## v1 config (yaml)
```yaml
model: {cell_dim: 256, object_dim: 256, relation_dim: 256, jepa_dim: 512, rule_dim: 256, program_dim: 256,
        cell_layers: 12, object_layers: 4, relation_layers: 4, predictor_layers: 6, program_layers: 6,
        heads: 8, ffn_dim: 1024}
jepa: {ema_start: 0.996, ema_end: 0.9995, global_loss: 1.0, object_loss: 1.0, relation_loss: 0.5,
       program_loss: 0.5, ranking_loss: 0.5, variance_loss: 0.05}
training: {optimizer: adamw, lr: 1.5e-4, weight_decay: 0.05, betas: [0.9, 0.95], warmup_epochs: 5,
           precision: bf16, grad_clip: 1.0}
search: {beam_width: 64, max_depth: 6, top_primitives: 8, astar_max_nodes: 50000}
tta: {enabled: true, steps: 8, lr: 0.05, anchor_weight: 0.1}
repair: {enabled: true, max_rounds: 4}
memory: {top_k: 16}
outputs: {num_candidates: 2}
```

## Implementation order
1 ARC loader · 2 object parser · 3 72-op DSL · 4 deterministic interpreter · 5 synthetic generator ·
6 program canonicaliser · 7 baseline symbolic search · 8 object encoder · 9 JEPA encoder · 10 transformation
latent · 11 program encoder · 12 neural scorer · 13 beam search · 14 hard negatives · 15 TTA · 16 repair loop ·
17 transformation memory · 18 adaptive compute · 19 Kaggle packaging.
First milestone: DSL + interpreter + exact verification. Then JEPA + latent retrieval. Then latent-guided search.
