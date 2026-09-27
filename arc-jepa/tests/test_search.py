"""Tests for the search module (INTERFACES.md §6).  CPU only, well under 60 s in total."""
from __future__ import annotations

import gc
import json
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from arcjepa.core.types import Pair, Task, task_from_json, validate_grid
from arcjepa.dsl import ExecError, Node, canonicalize, execute, random_program
from arcjepa.search import (MIN_TIMEOUT_S, Candidate, MemoryPrior, SolveConfig, astar_search, beam_search,
                            candidate_from_outputs, clipped_timeout, complexity, demo_error, difficulty,
                            difficulty_score, evolve, execute_safe, fallback_grids, identity_plausible,
                            induce_recolor, induced_candidate, is_exact, localise, make_candidate, parse_task,
                            past_search_deadline, predict_output_shape, refine_rule_latent, repair, score_value,
                            search_deadline, select_two, solve_task, sort_candidates)

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


def _map_colour_task():
    """Demos recolour 1 -> 5; the test input has no colour 1, so the exact program returns it unchanged."""
    pairs = [Pair([[1, 0], [0, 1]], [[5, 0], [0, 5]]), Pair([[1, 1], [0, 0]], [[5, 5], [0, 0]])]
    return pairs, [[2, 0], [0, 2]]


def test_select_two_never_spends_an_attempt_on_the_identity():
    pairs, test_in = _map_colour_task()
    exact = make_candidate(P("(MAP_COLOR INPUT 1 5)"), pairs)
    assert exact.demo_err == 0 and execute(exact.program, test_in) == test_in  # exact, but identity on the test
    ident = make_candidate(P("INPUT"), pairs)
    near = make_candidate(P("(MAP_COLOR INPUT 2 5)"), pairs)  # near miss that changes the test input
    assert ident.demo_err > 0 and near.demo_err > 0
    for cands in ([exact, ident, near], [ident, near], [exact], [ident], []):
        a1, a2, info = select_two(cands, test_in, pairs=pairs)
        assert validate_grid(a1) and validate_grid(a2) and a1 != a2
        assert test_in not in (a1, a2), (cands, info)
    a1, a2, info = select_two([exact, ident, near], test_in, pairs=pairs)
    assert a1 == [[5, 0], [0, 5]] and info["attempt_1_source"] == "near_miss"
    assert info["attempt_2_source"] == "fallback" and info["identity_demoted"]
    assert (len(a2), len(a2[0])) == (2, 2)  # the shape-inferred fill (predicted shape = input shape)
    # the identity is only a last resort: without demos and candidates nothing better exists
    a1, a2, _ = select_two([], test_in)
    assert a1 == test_in and a2 == [[0]]
    # identity tasks (every demo output equals its input) keep the identity
    same = [Pair([[1, 2], [3, 4]], [[1, 2], [3, 4]]), Pair([[7]], [[7]])]
    assert identity_plausible(same) and not identity_plausible(pairs)
    a1, _, _ = select_two([make_candidate(P("INPUT"), same)], test_in, pairs=same)
    assert a1 == test_in


def test_predict_output_shape_and_fallbacks():
    a = [[1, 2], [3, 4]]
    pairs = [Pair(a, [r + r for r in a])]
    assert predict_output_shape(pairs, [[1, 2, 3]]) == (1, 6)
    same = [Pair(a, a)]
    assert predict_output_shape(same, [[1, 2, 3]]) == (1, 3)
    fbs = fallback_grids([[1, 2, 3]], pairs)
    assert fbs and all(validate_grid(g) for g in fbs)
    assert any(len(g) == 1 and len(g[0]) == 6 for g in fbs)
    # the identity comes after every shape-inferred fill (last resort) unless the task is an identity task
    mc_pairs, test_in = _map_colour_task()
    fbs = fallback_grids(test_in, mc_pairs)
    assert fbs[-2:] == [test_in, [[0]]] and test_in not in fbs[:-2] and len(fbs[:-2]) == 2
    assert fallback_grids(test_in, mc_pairs, last_resort=False) == fbs[:-2]
    two_identity_demos = [Pair(a, a), Pair([[7]], [[7]])]
    assert fallback_grids(test_in, two_identity_demos)[0] == test_in


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


@pytest.fixture
def pinned_torch_threads():
    """One intra-op torch thread for the duration of a test, as the Kaggle pool workers pin theirs
    (``threads_per_worker``).  With every core busy (external load), torch's default pool of one thread per core
    stalls a tiny CPU forward from ~0.1 s to several seconds, which no per-task budget can absorb."""
    torch = pytest.importorskip("torch")
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(n)


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


def test_search_deadline_clips_interpreter_timeouts():
    assert clipped_timeout(5.0) == 5.0 and not past_search_deadline()  # no deadline block: unchanged
    now = time.perf_counter()
    with search_deadline(now + 2.0):
        assert clipped_timeout(5.0) <= 2.0 and clipped_timeout(0.02) <= 0.02
        with search_deadline(now + 10.0):  # nested blocks keep the earlier deadline
            assert clipped_timeout(5.0) <= 2.0
        with search_deadline(now - 1.0):
            assert clipped_timeout(5.0) == MIN_TIMEOUT_S and past_search_deadline()
        assert not past_search_deadline()
    assert clipped_timeout(5.0) == 5.0
    g = [[1, 2], [3, 4]]
    with search_deadline(time.perf_counter() - 1.0):  # a legitimate program still runs within the floor
        assert execute_safe(P("(ROTATE90 INPUT)"), g) == execute(P("(ROTATE90 INPUT)"), g)


def test_parse_task_respects_its_deadline():
    t = load_real_tasks(1)[0]
    full = parse_task(t)
    cut = parse_task(t, deadline=time.perf_counter() - 1.0)
    assert "truncated" not in full and cut["truncated"] is True
    assert len(cut["n_objects"]) == 1 and cut["n_objects"][0] == full["n_objects"][0]  # default hypothesis kept


def _scheduling_jitter(seconds: float = 0.25) -> float:
    """Largest gap between consecutive clock reads in a short busy loop: how long this process is descheduled at
    the moment (about 1 ms on an idle machine, tens of ms under heavy external CPU load)."""
    end = time.perf_counter() + seconds
    last = time.perf_counter()
    gap = 0.0
    while True:
        now = time.perf_counter()
        gap = max(gap, now - last)
        last = now
        if now >= end:
            return gap


def test_solver_two_attempts_per_test_input_on_real_tasks_within_budget():
    """Correctness is strict for every run (two valid attempts per test input, no solver error, the spec's
    diagnostics keys); the wall-clock check is calibrated to the machine.

    * The solver runs on a frozen heap (``gc.freeze``), as in the Kaggle pool workers, which freeze theirs after
      loading the package: a gen-2 collection of pytest's large heap alone takes ~0.1 s, more than the 10 %
      tolerance of this 0.75 s budget.
    * Tolerance = 10 % of the budget + 2 x the scheduling jitter measured now (external CPU load deschedules the
      process; the solver cannot run while it is not scheduled).
    * A task that still overshoots is re-timed once with a fresh jitter measurement: external load is transient,
      a budget bug repeats. At most 3 of the 20 tasks may need that.
    """
    tasks = load_real_tasks(20)
    cfg = SolveConfig(per_task_seconds=0.75)
    retimed: List[Tuple[str, float, float]] = []
    gc.collect()
    gc.freeze()
    try:
        jitter = _scheduling_jitter()
        for t in tasks:
            for run in range(2):
                t0 = time.perf_counter()
                attempts, diag = solve_task(t, None, cfg)
                dt = time.perf_counter() - t0
                assert len(attempts) == len(t.test)
                for a1, a2 in attempts:
                    assert validate_grid(a1) and validate_grid(a2)
                assert "error" not in diag, diag.get("error")
                for key in ("correct", "candidate_rank", "program_depth", "objects", "hypotheses",
                            "beam_expansions", "repair_rounds", "tta_steps", "inference_ms", "rule_retrieval_r8"):
                    assert key in diag
                assert diag["bucket"] in (0, 1, 2, 3)
                limit = cfg.per_task_seconds * 1.1 + 2.0 * jitter
                if dt <= limit:
                    break
                assert run == 0, (t.task_id, dt, limit, diag["stages"])
                retimed.append((t.task_id, round(dt, 3), round(limit, 3)))
                jitter = max(jitter, _scheduling_jitter())
    finally:
        gc.unfreeze()
    assert len(retimed) <= 3, retimed


def _model_stage_seconds(model, task: Task) -> float:
    """Wall time of one rule-latent forward for ``task`` right now: the fixed cost of the solver's model stage on
    this machine under its current load (warm-up call first)."""
    from arcjepa.search import rule_latent_for_task
    rule_latent_for_task(model, task)
    t0 = time.perf_counter()
    rule_latent_for_task(model, task)
    return time.perf_counter() - t0


def test_solver_with_tiny_model_full_pipeline(pinned_torch_threads):
    """Every stage runs under the neural prior; assertions on stages, TTA, retrieval and correctness are strict.

    Calibrated budgets: the model stage (and TTA's refinement) cannot be interrupted, so each budget is
    ``max(spec value, k x the model-stage cost measured now)`` (the spec value on an idle machine; more under heavy
    external load, where a tiny CPU forward takes ~0.8 s); the wall-clock bound stays 1.1 x the budget plus
    the measured scheduling jitter.
    """
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
    jitter = _scheduling_jitter()
    budget = max(3.0, 6.0 * _model_stage_seconds(model, task))
    # thresholds (0,0,0) force bucket 3: beam + repair + TTA + A* + evolution all run with the neural prior
    cfg = SolveConfig(per_task_seconds=budget, difficulty_thresholds=(0.0, 0.0, 0.0), memory=mem)
    t0 = time.perf_counter()
    attempts, diag = solve_task(task, model, cfg)
    assert time.perf_counter() - t0 <= budget * 1.1 + 2.0 * jitter, (budget, diag["stages"])
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
    # Parsing and TTA's refinement are un-interruptible too; under a burst of external load they can eat the time
    # the last fallback needed (measured: parse 478 ms + model 457 ms of a 2.4 s budget, evolution never started).
    # Such a run is re-run once with a re-measured, doubled budget: load is transient, a stage-plan bug repeats.
    # Every run must meet the wall-clock bound and return valid attempts; the stage coverage must hold on the last.
    for run in range(2):
        budget2 = (1 + run) * max(2.0, 6.0 * _model_stage_seconds(model, hard))
        jitter = max(jitter, _scheduling_jitter()) if run else jitter
        cfg2 = SolveConfig(per_task_seconds=budget2, difficulty_thresholds=(0.0, 0.0, 0.0))
        t0 = time.perf_counter()
        attempts, diag = solve_task(hard, model, cfg2)
        assert time.perf_counter() - t0 <= budget2 * 1.1 + 2.0 * jitter, (budget2, diag["stages"])
        assert len(attempts) == 1 and all(validate_grid(g) for g in attempts[0])
        assert "error" not in diag and diag["correct"] is None
        if {"beam", "repair", "tta", "astar", "evolution"} <= set(diag["stages"]):
            break
        assert run == 0, (budget2, diag["stages"])


# ============================================================================================ ArgPool relational fillers

def test_argpool_relational_fillers_values_and_beam_use():
    """docs/SOLVE_RATE_AUDIT.md addition 6: FILTER sets vs. the largest / smallest object, NEAREST / FARTHEST, object
    colours, and AND-masks of INPUT with its D4 images are pool fillers (rich pools only), with correct values."""
    from arcjepa.dsl.types import T
    from arcjepa.search.beam import ArgPool, _freeze

    # cc4 objects: A = the 2x2 block of 1 (largest), B = the 2 at (0, 4), C = the 3 at (3, 0) (nearer to A than B)
    g = [[1, 1, 0, 0, 2], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [3, 0, 0, 0, 0]]
    pool = ArgPool([g], [0, 1, 2, 3])
    lean = ArgPool([g], [0, 1, 2, 3], rich=False)
    largest = "(SELECT_LARGEST (GET_COMPONENTS4 INPUT))"
    smaller = f"(FILTER (GET_COMPONENTS4 INPUT) SMALLER {largest})"
    nearest = f"(NEAREST (GET_COMPONENTS4 INPUT) {largest})"
    by_str = {it.expr.to_str(): it for t in T for it in pool.items[t]}
    assert sorted(o.primary_color for o in by_str[smaller].values[0]) == [2, 3]
    assert [o.primary_color for o in by_str[f"(APPLY_TO_EACH {smaller} (RECOLOR OBJ 1))"].values[0]] == [1, 1]
    assert by_str[nearest].values[0].primary_color == 3
    assert by_str[f"(ARGMAX_SIZE (DUPLICATE {nearest} (0 0)))"].values == (3,)  # the colour of that object
    # the AND-mask of INPUT with its 180-degree rotation: cells set in both g and ROTATE180(g)
    rot = "(SELECT_NONZERO (PATTERN_FILL INPUT (SELECT_NONZERO INPUT) (ROTATE180 INPUT)))"
    assert by_str[rot].values[0] == [[bool(g[r][c] and g[3 - r][4 - c]) for c in range(5)] for r in range(4)]
    # every relational filler is in the pool or duplicates the value of an earlier item of its type
    fars = f"(FARTHEST (GET_COMPONENTS4 INPUT) {largest})"
    for t, src in ((T.OBJECT_SET, smaller), (T.OBJECT, nearest), (T.OBJECT, fars), (T.MASK, rot)):
        assert _freeze([pool_value(src, g)]) in pool._seen[t], src
    assert not any(k in s for s in (it.expr.to_str() for t in T for it in lean.items[t])
                   for k in ("FILTER", "NEAREST", "FARTHEST", "PATTERN_FILL", "DUPLICATE"))
    assert pool.summary()["OBJECT_SET"] > lean.summary()["OBJECT_SET"]

    # 67385a82-like: every object bigger than the smallest one turns 8 -- a depth-1 RENDER over a FILTER filler
    prog = P(f"(RENDER (APPLY_TO_EACH (FILTER (GET_COMPONENTS4 INPUT) LARGER (SELECT_SMALLEST (GET_COMPONENTS4 "
             f"INPUT))) (RECOLOR OBJ 8)) INPUT)")
    grids = [[[3, 3, 0, 0, 3], [3, 0, 0, 0, 0], [0, 0, 3, 3, 0], [3, 0, 3, 3, 0]],
             [[3, 0, 0, 3, 3, 3], [0, 0, 0, 0, 0, 0], [3, 3, 0, 0, 3, 0], [0, 0, 0, 0, 0, 0]],
             [[0, 3, 0, 0], [0, 0, 0, 3], [3, 3, 0, 3], [3, 3, 0, 0]],
             [[3, 3, 3, 0, 3], [0, 0, 0, 0, 0], [0, 3, 0, 3, 3], [0, 0, 0, 3, 0]]]
    pairs = [Pair(x, execute(prog, x)) for x in grids]
    assert pairs[0].output == [[8, 8, 0, 0, 3], [8, 0, 0, 0, 0], [0, 0, 8, 8, 0], [3, 0, 8, 8, 0]]  # by hand
    cands = beam_search(pairs[:3], time_budget_s=5.0)
    exact = [c for c in cands if c.demo_err == 0]
    assert exact and execute(exact[0].program, pairs[3].input) == pairs[3].output, [c.program.to_str() for c in cands[:3]]


def pool_value(src: str, g):
    """Value of a pool expression on ``g`` (the interpreter's untyped evaluator, as ArgPool.add uses it)."""
    from arcjepa.dsl.interpreter import evaluate
    return evaluate(P(src), g)


# ============================================================================================ induced recolour tables

INDUCE_TASKS = Path(__file__).parent / "data" / "induce_tasks.json"
#: Audit tasks (docs/SOLVE_RATE_AUDIT.md addition 2) where the induced table is correct, with the rule that fires.
INDUCE_EXPECTED = {"67385a82": ("cc4", "size"), "84f2aca1": ("holes", "size"), "c8f0f002": ("cc4", "color"),
                   "e8593010": ("zero4", "size"), "9565186b": ("cc4", "is_largest"), "ea32f347": ("cc4", "shape")}


def _induce_tasks() -> dict:
    raw = json.loads(INDUCE_TASKS.read_text(encoding="utf-8"))
    return {k: task_from_json(k, v) for k, v in raw.items() if not k.startswith("_")}


def test_induce_recolor_fires_correctly_on_audit_tasks():
    tasks = _induce_tasks()
    assert set(INDUCE_EXPECTED) < set(tasks)
    for tid, (seg, prop) in INDUCE_EXPECTED.items():
        t = tasks[tid]
        tests_in = [p.input for p in t.test]
        t0 = time.perf_counter()
        rule = induce_recolor(t.train, tests_in)
        assert time.perf_counter() - t0 < 2.0
        assert rule is not None and (rule.segmentation, rule.prop) == (seg, prop), (tid, rule)
        assert all(rule.apply(p.input) == p.output for p in t.train), tid  # conflict-free on the demos
        assert all(rule.apply(p.input) == p.output for p in t.test), tid   # and right on the hidden answers
        c = induced_candidate(rule, t.train, tests_in)
        assert c is not None and c.demo_err == 0 and c.source == "induce"
        assert c.meta["test_outputs"] == [p.output for p in t.test]
        assert c.program.to_str() == f"(INDUCE_RECOLOR {seg} {prop})"
    # coverage: 9565186b's demos also admit a colour+size table, but a test object has an unseen value
    t = tasks["9565186b"]
    assert (induce_recolor(t.train).segmentation, induce_recolor(t.train).prop) == ("cc4", "color_size")
    # 6df30ad6: a size table fits the demos, but a test object has a size no demo shows -> nothing
    t = tasks["6df30ad6"]
    assert induce_recolor(t.train) is not None
    assert induce_recolor(t.train, [p.input for p in t.test]) is None


def test_induce_recolor_returns_nothing_when_the_table_conflicts():
    g = [[0, 0, 0, 0, 0], [0, 5, 0, 5, 0], [0, 0, 0, 0, 0]]
    same = [[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0]]
    # two interior single cells, equal in every property, repainted differently within one demo
    split = [[0, 0, 0, 0, 0], [0, 1, 0, 2, 0], [0, 0, 0, 0, 0]]
    assert induce_recolor([Pair(g, split)], [g]) is None
    # ... or differently in two demos
    g2 = [[0, 0, 0, 0, 0], [0, 5, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 5, 0], [0, 0, 0, 0, 0]]
    g2_out = [[0, 0, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 0, 0]]
    assert induce_recolor([Pair(g, same), Pair(g2, g2_out)], [g]) is None
    # control: the consistent demo alone induces colour 5 -> 1
    rule = induce_recolor([Pair(g, same)], [g2])
    assert rule is not None and (rule.segmentation, rule.prop) == ("cc4", "color") and rule.apply(g2) == [
        [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]]
    # no rule for identity tasks, shape-changing tasks, or when a changed cell lies outside every segment
    assert induce_recolor([Pair(g, g)], [g]) is None
    assert induce_recolor([Pair(g, [[1, 2]])], [g]) is None
    assert induce_recolor([Pair([[0, 0], [0, 3]], [[4, 0], [0, 3]])], [[[0, 0], [0, 3]]]) is None
    assert induce_recolor([Pair(g, same)], [g], time_budget_s=0.0) is None  # out of time: nothing


def test_place_induced_keeps_an_exact_search_fit_first():
    from arcjepa.search.solver import _place_induced
    a1, a2, g = [[1]], [[2]], [[3]]
    info = {"attempt_1_source": "exact", "attempt_2_source": "near_miss"}
    assert _place_induced(g, a1, a2, info) == (a1, g) and info["attempt_2_source"] == "induce"
    info = {"attempt_1_source": "near_miss", "attempt_2_source": "fallback"}
    assert _place_induced(g, a1, a2, info) == (g, a1) and info["attempt_1_source"] == "induce"
    info = {"attempt_1_source": "exact", "attempt_2_source": "exact"}
    assert _place_induced(a1, a1, a2, info) == (a1, a2)  # the exact fit already predicts it
    info = {"attempt_1_source": "near_miss", "attempt_2_source": "fallback"}
    assert _place_induced(a1, a1, a2, info) == (a1, a2)  # no duplicate attempt
    assert _place_induced(None, a1, a2, {}) == (a1, a2)


def test_solver_places_the_induced_prediction_inside_its_budget():
    t = _induce_tasks()["84f2aca1"]  # not expressible in the DSL: only the induced table solves it
    cfg = SolveConfig(per_task_seconds=2.0)
    jitter = _scheduling_jitter()
    t0 = time.perf_counter()
    attempts, diag = solve_task(t, None, cfg)
    assert time.perf_counter() - t0 <= cfg.per_task_seconds * 1.1 + 2.0 * jitter + 0.2, diag["stages"]
    assert diag["correct"] == [True] and "error" not in diag
    assert diag["induced"]["program"] == "(INDUCE_RECOLOR holes size)" and "induce" in diag["stages"]
    # the stage is capped at 10 % of the budget (it takes a few ms here)
    assert diag["stages"]["induce"] <= 1000.0 * (0.1 * cfg.per_task_seconds + 2.0 * jitter + 0.05)
    assert diag["selection"][0]["attempt_1_source"] == "induce"
    _, diag_off = solve_task(t, None, SolveConfig(per_task_seconds=2.0, induce=False))
    assert "induced" not in diag_off and diag_off["correct"] == [False]


# ============================================================================================ package compatibility

def test_package_exported_before_the_extensions_loads_and_solves(tmp_path, pinned_torch_threads):
    """A package trained before the spec extensions (its vocab.json has the 76 ops of the 72 spec primitives + 4
    structural helpers) must load strictly under the extended DSL: the saved vocabulary order is kept, extension
    ops encode to <unk>, the prior scores programs that contain them, and solve_task still finds them."""
    torch = pytest.importorskip("torch")
    from arcjepa.dsl.primitives import EXTENSION_PRIMITIVES, REGISTRY
    from arcjepa.model import ARCJEPA, ModelConfig
    from arcjepa.model.program_encoder import ProgramTokenizer
    from arcjepa.search import NeuralPrior, rule_latent_for_task
    from arcjepa.training.export import export_package

    old_registry = {k: v for k, v in REGISTRY.items() if not v.extension}
    assert len(old_registry) == 76 and len(REGISTRY) == 76 + len(EXTENSION_PRIMITIVES)
    cfg = ModelConfig.tiny()
    torch.manual_seed(0)
    old = ARCJEPA(cfg, ProgramTokenizer(registry=old_registry, max_depth_tokens=cfg.program_depth_tokens))
    export_package(old, {"memory": {"top_k": 4}}, tmp_path / "pkg")
    saved = json.loads((tmp_path / "pkg" / "vocab.json").read_text(encoding="utf-8"))
    assert not set(EXTENSION_PRIMITIVES) & set(saved["itos"])
    assert ProgramTokenizer().vocab_size == len(saved["itos"]) + len(EXTENSION_PRIMITIVES)  # the new DSL's vocab

    model = ARCJEPA.load_package(tmp_path / "pkg")  # strict weight load: any embedding size mismatch raises
    tok = model.tokenizer
    assert tok.itos == saved["itos"]
    assert model.program_encoder.sym_emb.num_embeddings == len(saved["itos"])
    for n in EXTENSION_PRIMITIVES:
        assert tok.encode(Node(n, (P("INPUT"),) + (2,) * (REGISTRY[n].arity - 1)))[1] == tok.UNK
    assert tok.UNK not in tok.encode(P("(ROTATE90 INPUT)"))

    def kron(g):
        return REGISTRY["KRON_SELF"].fn(g)

    grids = [[[1, 0, 1], [0, 1, 0], [1, 1, 0]], [[2, 2, 0], [0, 2, 0], [0, 0, 2]], [[0, 3, 0], [3, 3, 3], [0, 3, 0]],
             [[4, 0, 0], [0, 4, 4], [4, 0, 4]]]
    pairs = [Pair(g, kron(g)) for g in grids]
    task = Task("old-vocab", pairs[:3], [pairs[3]])
    r_task = rule_latent_for_task(model, task)
    progs = [P("(KRON_SELF INPUT)"), P("(UPSCALE INPUT 3)"), P("(PANEL_BOOL INPUT 0 2)"), P("(ROTATE90 INPUT)")]
    with torch.no_grad():  # directly, not through NeuralPrior (which maps a scoring failure to a neutral 0)
        raw = model.score_programs(r_task, progs)
    assert raw.shape == (4,) and bool(torch.isfinite(raw).all())
    scores = NeuralPrior(model, r_task)(progs)
    assert scores == pytest.approx([float(v) for v in raw.clamp(-4.0, 4.0)])
    budget = max(3.0, 6.0 * _model_stage_seconds(model, task))
    attempts, diag = solve_task(task, model, SolveConfig(per_task_seconds=budget))
    assert "error" not in diag and "model_error" not in diag
    assert diag["correct"] == [True] and diag["best_program"] == "(KRON_SELF INPUT)"
