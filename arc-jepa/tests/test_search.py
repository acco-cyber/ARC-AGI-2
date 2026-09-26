"""Tests for the search module (INTERFACES.md §6).  CPU only, well under 60 s in total."""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from arcjepa.core.types import Pair, Task, task_from_json, validate_grid
from arcjepa.dsl import ExecError, Node, canonicalize, execute, random_program
from arcjepa.search import (Candidate, MemoryPrior, SolveConfig, astar_search, beam_search, candidate_from_outputs,
                            complexity, demo_error, difficulty, difficulty_score, evolve, execute_safe,
                            fallback_grids, is_exact, localise, make_candidate, parse_task, predict_output_shape,
                            refine_rule_latent, repair, score_value, select_two, solve_task, sort_candidates)

TRAIN_JSONL = Path(os.environ.get("ARCJEPA_DATA", r"E:\Claude code\arc2\dataset\hf_arc2_episodes")) / "tasks" / "train.jsonl"


# ============================================================================================ helpers

def rand_grid(rng: random.Random, colours: Tuple[int, ...] = tuple(range(1, 10)), lo: int = 4, hi: int = 10):
    h, w = rng.randint(lo, hi), rng.randint(lo, hi)
    g = [[0] * w for _ in range(h)]
    for _ in range(rng.randint(2, 4)):
        col = rng.choice(colours)
        r0, c0 = rng.randrange(h), rng.randrange(w)
        r1, c1 = min(h - 1, r0 + rng.randint(0, 3)), min(w - 1, c0 + rng.randint(0, 3))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                g[r][c] = col
    return g


def task_from_program(prog: Node, rng: random.Random, n: int = 4, **kw) -> Optional[List[Pair]]:
    pairs = []
    for _ in range(n):
        g = rand_grid(rng, **kw)
        try:
            pairs.append(Pair(g, execute(prog, g)))
        except ExecError:
            return None
    return pairs


def _nondegenerate(pairs: List[Pair]) -> bool:
    return (not all(p.input == p.output for p in pairs)) and len({str(p.output) for p in pairs}) > 1


def synthetic_tasks(n: int = 10, seed: int = 123) -> List[Tuple[str, List[Pair]]]:
    """``n`` depth<=2 synthetic tasks: arcjepa.synthetic when importable, else dsl.grammar.random_program."""
    rng = random.Random(seed)
    out: List[Tuple[str, List[Pair]]] = []
    try:  # pragma: no cover - depends on whether the synthetic module has landed
        from arcjepa.synthetic.dataset import make_task  # type: ignore
        for _ in range(3000):
            if len(out) >= n:
                break
            st = make_task(rng)
            if st is None or st.depth > 2 or len(st.pairs) < 3 or not _nondegenerate(list(st.pairs)):
                continue
            out.append((st.program, list(st.pairs)))
    except ImportError:
        pass
    while len(out) < n:
        prog = random_program(rng, 1 + len(out) % 2)
        if canonicalize(prog).to_str() == "INPUT":
            continue
        pairs = task_from_program(prog, rng)
        if pairs is None or not _nondegenerate(pairs):
            continue
        out.append((prog.to_str(), pairs))
    return out


def load_real_tasks(n: int) -> List[Task]:
    if not TRAIN_JSONL.exists():
        pytest.skip(f"real task file missing: {TRAIN_JSONL}")
    tasks = []
    with open(TRAIN_JSONL, encoding="utf-8") as f:
        for line in f:
            if len(tasks) >= n:
                break
            d = json.loads(line)
            tasks.append(task_from_json(d["task_id"], d))
    return tasks


P = Node.from_str


# ============================================================================================ verifier / candidate

def test_verifier_counts_pairs_and_cells():
    g1 = [[1, 0], [0, 2]]
    g2 = [[3, 3, 3], [0, 1, 0]]
    rot = P("(ROTATE90 INPUT)")
    pairs = [Pair(g1, execute(rot, g1)), Pair(g2, execute(rot, g2))]
    assert demo_error(rot, pairs) == (0, 0)
    assert is_exact(rot, pairs)
    # identity: pair 1 has 2x2 shape (cell mismatches), pair 2 has the wrong shape (all 6 target cells)
    wrong, cells = demo_error(P("INPUT"), pairs)
    assert wrong == 2
    exp1 = sum(a != b for ra, rb in zip(g1, pairs[0].output) for a, b in zip(ra, rb))
    assert cells == exp1 + 6
    assert not is_exact(P("INPUT"), pairs)
    # failing program: every cell of every pair counts
    bad = P("(CROP INPUT (SELECT_LARGEST (SELECT_COLOR (GET_COMPONENTS4 INPUT) 9)))")
    assert execute_safe(bad, g1) is None
    assert demo_error(bad, pairs) == (2, 4 + 6)


def test_candidate_score_and_ordering():
    g = [[1, 2], [3, 4]]
    pairs = [Pair(g, execute(P("(REFLECT_H INPUT)"), g))]
    exact = make_candidate(P("(REFLECT_H INPUT)"), pairs)
    near = make_candidate(P("(MAP_COLOR (REFLECT_H INPUT) 1 5)"), pairs)
    far = make_candidate(P("INPUT"), pairs)
    assert exact.demo_err == 0 and exact.loss == 0.0
    assert exact.score == pytest.approx(score_value(0.0, 0.0, complexity(exact.program)))
    assert complexity(P("(MAP_COLOR INPUT 1 5)")) == 3 and complexity(P("INPUT")) == 0
    ranked = sort_candidates([far, near, exact])
    assert ranked[0] is exact  # beta * E dominates complexity
    assert near.score > far.score or near.cell_err <= far.cell_err
    assert exact.with_neural(2.0).score == pytest.approx(exact.score + 2.0)


# ============================================================================================ beam

def test_beam_solves_depth_le2_synthetic_tasks():
    tasks = synthetic_tasks(10, seed=123)
    solved = 0
    for prog, pairs in tasks:
        demos = pairs[:-1] if len(pairs) >= 3 else pairs
        t0 = time.perf_counter()
        cands = beam_search(demos, prior=None, width=32, max_depth=6, time_budget_s=5.0)
        dt = time.perf_counter() - t0
        assert dt < 5.0 * 1.1, (prog, dt)
        assert cands == sort_candidates(cands)
        exact = [c for c in cands if c.demo_err == 0]
        if exact:
            solved += 1
            assert is_exact(exact[0].program, demos)  # verified by the interpreter
    assert solved >= 8, f"beam solved only {solved}/10"


def test_beam_respects_budget_on_unsolvable_task():
    rng = random.Random(5)
    pairs = []
    for _ in range(3):  # unrelated random outputs: no program fits
        a = [[rng.randrange(10) for _ in range(12)] for _ in range(12)]
        b = [[rng.randrange(10) for _ in range(12)] for _ in range(12)]
        pairs.append(Pair(a, b))
    st = {}
    t0 = time.perf_counter()
    cands = beam_search(pairs, time_budget_s=0.6, stats=st)
    dt = time.perf_counter() - t0
    assert dt < 0.6 * 1.1
    assert st["beam_expansions"] > 0
    assert all(c.demo_err > 0 for c in cands) and len(cands) > 0


def test_beam_with_prior_top_primitives_and_seeds():
    rng = random.Random(1)
    target = P("(MAP_COLOR (REFLECT_V INPUT) 2 7)")
    pairs = None
    while pairs is None or not _nondegenerate(pairs):
        pairs = task_from_program(target, rng, colours=(1, 2, 3))

    def prior(progs):
        return [1.0 if ("MAP_COLOR" in p.to_str() or "REFLECT_V" in p.to_str()) else 0.0 for p in progs]

    cands = beam_search(pairs, prior=prior, width=16, top_primitives=4, time_budget_s=5.0)
    exact = [c for c in cands if c.demo_err == 0]
    assert exact and exact[0].neural == pytest.approx(1.0)
    # an exact seed is recorded immediately
    cands2 = beam_search(pairs, seeds=[target], time_budget_s=2.0)
    assert any(c.demo_err == 0 and c.source == "seed" for c in cands2)


# ============================================================================================ repair

@pytest.mark.parametrize("truth,planted", [
    ("(MAP_COLOR (ROTATE90 INPUT) 3 5)", "(MAP_COLOR (ROTATE90 INPUT) 3 6)"),            # wrong literal
    ("(MAP_COLOR (ROTATE90 INPUT) 3 5)", "(MAP_COLOR (ROTATE270 INPUT) 3 5)"),           # wrong primitive
    ("(RENDER_BLANK (APPLY_TO_EACH (GET_COMPONENTS4 INPUT) (MOVE OBJ (0 1))) INPUT)",
     "(RENDER_BLANK (APPLY_TO_EACH (GET_COMPONENTS4 INPUT) (MOVE OBJ (1 0))) INPUT)"),   # wrong offset in a lambda
    ("(FILL INPUT (SELECT_NONZERO INPUT) 4)", "(FILL INPUT (SELECT_NONZERO INPUT) 2)"),  # wrong colour
])
def test_repair_fixes_planted_single_node_error(truth, planted):
    rng = random.Random(7)
    prog = P(truth)
    pairs = None
    while pairs is None or not _nondegenerate(pairs):
        pairs = task_from_program(prog, rng, colours=(1, 2, 3))
    bad = make_candidate(P(planted), pairs, source="planted")
    assert bad.demo_err > 0
    out = repair([bad], pairs, rounds=2, rng=random.Random(0), time_budget_s=5.0)
    exact = [c for c in out if c.demo_err == 0]
    assert exact, f"repair failed to fix {planted}"
    assert exact[0].source == "repair" and is_exact(exact[0].program, pairs)
    assert any(c.source == "planted" for c in out)  # inputs are kept


def test_localise_points_at_the_responsible_node():
    rng = random.Random(3)
    truth = P("(COLOR_OBJECT (REFLECT_H INPUT) (SELECT_LARGEST (GET_COMPONENTS4 INPUT)) 5)")
    pairs = None
    while pairs is None or not _nondegenerate(pairs):
        pairs = task_from_program(truth, rng, colours=(1, 2, 3))
    planted = P("(COLOR_OBJECT (REFLECT_H INPUT) (SELECT_LARGEST (GET_COMPONENTS4 INPUT)) 6)")
    ranked = localise(planted, pairs)
    assert ranked and ranked[0][0] == ()  # the root COLOR_OBJECT owns every wrong cell
    assert ranked[0][1] > 0.5


# ============================================================================================ select_two

def test_select_two_always_valid():
    rng = random.Random(11)
    g = rand_grid(rng)
    pairs = [Pair(g, execute(P("(ROTATE180 INPUT)"), g))]
    test_in = rand_grid(rng)
    # no candidates at all
    a1, a2, info = select_two([], test_in, pairs=pairs)
    assert validate_grid(a1) and validate_grid(a2)
    # candidates that fail on the test input + an invalid test input
    failing = candidate_from_outputs(P("(CROP INPUT (SELECT_UNIQUE (SELECT_COLOR (GET_COMPONENTS4 INPUT) 9)))"),
                                     [pairs[0].output], pairs)
    for ti in (test_in, [], [[1, 2], [3]]):
        a1, a2, _ = select_two([failing], ti, pairs=pairs)
        assert validate_grid(a1) and validate_grid(a2)
    # two exact programs with different test outputs -> two different attempts from different clusters
    sym = [[1, 2, 1], [2, 5, 2], [1, 2, 1]]
    pairs2 = [Pair(sym, sym)]
    c1 = make_candidate(P("INPUT"), pairs2)
    c2 = make_candidate(P("(REFLECT_H INPUT)"), pairs2)
    assert c1.demo_err == 0 and c2.demo_err == 0
    t2 = [[1, 2, 3], [4, 5, 6]]
    a1, a2, info = select_two([c1, c2], t2, pairs=pairs2)
    assert a1 != a2 and {str(a1), str(a2)} == {str(t2), str(execute(P("(REFLECT_H INPUT)"), t2))}
    assert info["n_clusters"] == 2 and info["attempt_2_source"] == "exact"
    # fuzz: random candidate sets
    for k in range(25):
        progs = [random_program(rng, rng.randint(1, 3)) for _ in range(rng.randint(0, 6))]
        cands = [make_candidate(p, pairs) for p in progs]
        a1, a2, _ = select_two(cands, rand_grid(rng), pairs=pairs if k % 2 else None)
        assert validate_grid(a1) and validate_grid(a2)


def test_predict_output_shape_and_fallbacks():
    a = [[1, 2], [3, 4]]
    pairs = [Pair(a, [r + r for r in a])]
    assert predict_output_shape(pairs, [[1, 2, 3]]) == (1, 6)
    same = [Pair(a, a)]
    assert predict_output_shape(same, [[1, 2, 3]]) == (1, 3)
    fbs = fallback_grids([[1, 2, 3]], pairs)
    assert fbs and all(validate_grid(g) for g in fbs)
    assert any(len(g) == 1 and len(g[0]) == 6 for g in fbs)


# ============================================================================================ A* / evolution

def test_astar_finds_simple_program_within_budget():
    rng = random.Random(2)
    truth = P("(REPLACE_BACKGROUND (REFLECT_V INPUT) 4)")
    pairs = task_from_program(truth, rng, n=3)
    st = {}
    t0 = time.perf_counter()
    cands = astar_search(pairs, None, max_nodes=20_000, time_budget_s=4.0, stats=st)
    assert time.perf_counter() - t0 < 4.0 * 1.1
    assert any(c.demo_err == 0 for c in cands)
    assert 0 < st["astar_nodes"] <= 20_000 + 500
    # node cap is honoured
    st2 = {}
    astar_search(pairs, None, max_nodes=50, time_budget_s=4.0, stats=st2, early_stop=False)
    assert st2["astar_nodes"] < 50 + 300


def test_evolve_repairs_a_near_miss_and_is_deterministic():
    rng = random.Random(4)
    truth = P("(MAP_COLOR (REFLECT_H INPUT) 1 8)")
    pairs = None
    while pairs is None or not _nondegenerate(pairs):
        pairs = task_from_program(truth, rng, colours=(1, 2))
    seed = P("(MAP_COLOR (REFLECT_H INPUT) 1 6)")
    a = evolve(pairs, [seed], None, pop=16, gens=10, time_budget_s=5.0, rng=random.Random(0))
    b = evolve(pairs, [seed], None, pop=16, gens=10, time_budget_s=5.0, rng=random.Random(0))
    assert [c.program.to_str() for c in a[:5]] == [c.program.to_str() for c in b[:5]]
    assert a == sort_candidates(a)
    assert any(c.demo_err == 0 for c in a)


# ============================================================================================ model-facing parts

def _tiny_model():
    torch = pytest.importorskip("torch")
    model_mod = pytest.importorskip("arcjepa.model")
    torch.manual_seed(0)
    return model_mod.ARCJEPA(model_mod.ModelConfig.tiny())


def test_tta_refines_rule_latent_without_touching_weights():
    torch = pytest.importorskip("torch")
    model = _tiny_model()
    rng = random.Random(0)
    truth = P("(ROTATE90 INPUT)")
    pairs = task_from_program(truth, rng, n=3)
    cands = [make_candidate(p, pairs) for p in
             (truth, P("INPUT"), P("(REFLECT_H INPUT)"), P("(MAP_COLOR INPUT 1 2)"), P("(ROTATE180 INPUT)"))]
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    r0 = torch.randn(model.cfg.rule_dim)
    hist: List[float] = []
    r = refine_rule_latent(model, r0, pairs, cands, steps=8, lr=0.05, anchor=0.1, history=hist)
    assert r.shape == r0.shape and torch.isfinite(r).all()
    assert len(hist) == 9 and hist[-1] <= hist[0] + 1e-6
    assert not torch.equal(r, r0)
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert all(p.grad is None for p in model.parameters())
    # degenerate inputs return r0 unchanged
    assert torch.equal(refine_rule_latent(model, r0, pairs, [], steps=8), r0)
    assert torch.equal(refine_rule_latent(None, r0, pairs, cands), r0)


def test_memory_prior_returns_typed_seeds():
    np = pytest.importorskip("numpy")
    from arcjepa.model.memory import TransformationMemory
    mem = TransformationMemory(dim=8)
    rs = np.random.RandomState(0)
    good = ["(ROTATE90 INPUT)", "(MAP_COLOR INPUT 1 2)"]
    for prog in good + ["(GET_COMPONENTS4 INPUT)", "not a program ((("]:
        mem.add(rs.randn(8).astype("float32"), prog, 1, "geometry")
    seeds = MemoryPrior(mem, None).seeds_for(rs.randn(8).astype("float32"), k=16)
    assert sorted(s.to_str() for s in seeds) == sorted(good)  # non-GRID / unparsable records skipped
    assert MemoryPrior(None).seeds_for(rs.randn(8), k=4) == []
    assert MemoryPrior(TransformationMemory(dim=8)).seeds_for(rs.randn(8), k=4) == []


# ============================================================================================ solver

def test_solve_config_round_trip_and_spec_yaml():
    cfg = SolveConfig(per_task_seconds=12.5, memory=object())
    d = cfg.to_dict()
    assert "memory" not in d and d["per_task_seconds"] == 12.5
    json.dumps(d)  # JSON-friendly
    back = SolveConfig.from_dict(d)
    assert back.to_dict() == d and back.difficulty_thresholds == cfg.difficulty_thresholds
    spec = {"search": {"beam_width": 64, "max_depth": 6, "top_primitives": 8, "astar_max_nodes": 50000},
            "tta": {"enabled": True, "steps": 8, "lr": 0.05, "anchor_weight": 0.1},
            "repair": {"enabled": True, "max_rounds": 4}, "memory": {"top_k": 16}, "outputs": {"num_candidates": 2}}
    y = SolveConfig.from_spec_yaml(spec, per_task_seconds=3.0)
    assert (y.beam_width, y.max_depth, y.top_primitives, y.astar_max_nodes) == (64, 6, 8, 50000)
    assert (y.tta_steps, y.tta_lr, y.tta_anchor, y.repair_max_rounds, y.memory_top_k) == (8, 0.05, 0.1, 4, 16)
    assert (y.alpha, y.beta, y.gamma, y.per_task_seconds) == (1.0, 10.0, 0.15, 3.0)


def test_difficulty_buckets():
    tasks = load_real_tasks(6)
    for t in tasks:
        parsed = parse_task(t)
        d, terms = difficulty_score(t, parsed)
        assert 0.0 <= d <= 1.0 and set(terms) == {"H", "N_obj", "N_seg", "composition", "ambiguity"}
        assert difficulty(t, parsed) in (0, 1, 2, 3)
    g = [[1, 0], [0, 0]]
    easy = Task("easy", [Pair(g, g)] * 4, [Pair(g, g)])
    assert difficulty(easy) == 0


def test_solver_two_attempts_per_test_input_on_real_tasks_within_budget():
    tasks = load_real_tasks(20)
    cfg = SolveConfig(per_task_seconds=0.75)
    worst = 0.0
    for t in tasks:
        t0 = time.perf_counter()
        attempts, diag = solve_task(t, None, cfg)
        dt = time.perf_counter() - t0
        worst = max(worst, dt)
        assert dt <= cfg.per_task_seconds * 1.1, (t.task_id, dt)
        assert len(attempts) == len(t.test)
        for a1, a2 in attempts:
            assert validate_grid(a1) and validate_grid(a2)
        assert "error" not in diag, diag.get("error")
        for key in ("correct", "candidate_rank", "program_depth", "objects", "hypotheses", "beam_expansions",
                    "repair_rounds", "tta_steps", "inference_ms", "rule_retrieval_r8"):
            assert key in diag
        assert diag["bucket"] in (0, 1, 2, 3)


def test_solver_with_tiny_model_full_pipeline():
    model = _tiny_model()
    np = pytest.importorskip("numpy")
    from arcjepa.model.memory import TransformationMemory
    rng = random.Random(9)
    truth = P("(REFLECT_V (MAP_COLOR INPUT 1 4))")
    pairs = None
    while pairs is None or not _nondegenerate(pairs):
        pairs = task_from_program(truth, rng, n=4, colours=(1, 2, 3))
    task = Task("synthetic", pairs[:3], [Pair(pairs[3].input, pairs[3].output)])
    mem = TransformationMemory(dim=model.cfg.rule_dim)
    mem.add(np.zeros(model.cfg.rule_dim, dtype="float32") + 0.1, truth.to_str(), 2, "geometry")
    # thresholds (0,0,0) force bucket 3: beam + repair + TTA + A* + evolution all run with the neural prior
    cfg = SolveConfig(per_task_seconds=3.0, difficulty_thresholds=(0.0, 0.0, 0.0), memory=mem)
    t0 = time.perf_counter()
    attempts, diag = solve_task(task, model, cfg)
    assert time.perf_counter() - t0 <= 3.0 * 1.1
    assert len(attempts) == 1 and all(validate_grid(g) for g in attempts[0])
    assert "error" not in diag and "model_error" not in diag
    assert diag["bucket"] == 3
    assert diag["rule_retrieval_r8"] == 1.0  # the memory seed is exact on the demos
    assert diag["correct"] == [True]
    assert diag["tta_steps"] == cfg.tta_steps
    # an unsolvable task runs every stage (beam, repair, TTA + re-guided beam, A*, evolution) under the prior
    noise = [Pair([[rng.randrange(10) for _ in range(6)] for _ in range(6)],
                  [[rng.randrange(10) for _ in range(6)] for _ in range(6)]) for _ in range(4)]
    hard = Task("noise", noise[:3], [Pair(noise[3].input, [])])
    cfg2 = SolveConfig(per_task_seconds=2.0, difficulty_thresholds=(0.0, 0.0, 0.0))
    t0 = time.perf_counter()
    attempts, diag = solve_task(hard, model, cfg2)
    assert time.perf_counter() - t0 <= 2.0 * 1.1
    assert len(attempts) == 1 and all(validate_grid(g) for g in attempts[0])
    assert "error" not in diag and diag["correct"] is None
    assert {"beam", "repair", "tta", "astar", "evolution"} <= set(diag["stages"])
