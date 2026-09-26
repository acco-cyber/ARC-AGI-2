"""Tests for the synthetic task module (INTERFACES.md §3): grid generators, program sampler, make_task degeneracy
filter, JSONL generation (exact re-execution, depth / category mix, compositional split without leakage, worker
invariance, CLI), perturbations and the difficulty score.

Run from the package root:  python -m pytest tests/test_synthetic.py -q
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pytest

from arcjepa.core.types import MAX_SIDE, Pair, Task, validate_grid
from arcjepa.dsl import Node, T, canonicalize, execute, typecheck
from arcjepa.parser import segment
from arcjepa.synthetic import (CATEGORIES, CATEGORY_MIX, DEPTH_MIX, HELDOUT_COMPOSITIONS, STYLES, SynthTask,
                               add_distractors, ambiguous_segmentation, degeneracy_reason, difficulty_score,
                               generate, iter_rows, load_jsonl, make_task, make_task_with_reason, random_input_grid,
                               random_palette, row_to_task, sample_category, sample_depth, sample_program, split_of,
                               task_to_row, to_arc_task)
from arcjepa.synthetic.dataset import count_cc4, main as cli_main
from arcjepa.synthetic.perturbations import task_background

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)

N_MAIN = 2000  # tasks in the shared generated file (depth/category histograms need a few thousand samples)


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    """One JSONL file of N_MAIN tasks shared by the dataset-level tests."""
    out = tmp_path_factory.mktemp("synth") / "tasks.jsonl"
    t0 = time.perf_counter()
    stats = generate(N_MAIN, str(out), seed=20260926, workers=1)
    dt = time.perf_counter() - t0
    rows = list(iter_rows(str(out)))
    return {"path": out, "stats": stats, "rows": rows, "seconds": dt}


# ======================================================================================= generators

@pytest.mark.parametrize("style", STYLES)
def test_generator_styles_valid_and_deterministic(style: str) -> None:
    for seed in range(40):
        g1 = random_input_grid(random.Random(seed), style=style)
        g2 = random_input_grid(random.Random(seed), style=style)
        assert g1 == g2
        assert validate_grid(g1)
        assert any(v != 0 for row in g1 for v in row), "grid must contain foreground"


def test_generator_respects_shape_palette_background() -> None:
    rng = random.Random(3)
    for style in STYLES:
        for _ in range(10):
            g = random_input_grid(rng, h=12, w=9, palette=[2, 5], background=1, style=style)
            assert len(g) == 12 and len(g[0]) == 9
            assert {v for row in g for v in row} <= {1, 2, 5}
    g = random_input_grid(rng, h=40, w=0, style="objects")  # clipped to 1..30
    assert len(g) == MAX_SIDE and len(g[0]) == 1
    with pytest.raises(ValueError):
        random_input_grid(rng, style="nope")


def test_objects_style_object_count() -> None:
    rng = random.Random(11)
    counts = [count_cc4(random_input_grid(rng, h=20, w=20, n_objects=4, palette=[3], style="objects"))
              for _ in range(50)]
    assert max(counts) <= 4  # every shape is 4-connected, so single-colour scenes never exceed the shape count
    assert sum(1 for c in counts if c >= 3) >= 35  # clearance-1 placement keeps objects separate


def test_random_palette() -> None:
    rng = random.Random(0)
    for _ in range(100):
        pal = random_palette(rng, background=0, include=[0, 7, 3])
        assert 7 in pal and 3 in pal and 0 not in pal
        assert len(set(pal)) == len(pal) and 1 <= len(pal) <= 9


# ======================================================================================= program sampler

def test_mix_constants() -> None:
    assert sorted(DEPTH_MIX) == [1, 2, 3, 4, 5, 6]
    assert [DEPTH_MIX[d] for d in range(1, 7)] == [0.25, 0.25, 0.20, 0.15, 0.10, 0.05]
    assert CATEGORY_MIX == {"object": 0.40, "geometry": 0.20, "relational": 0.15, "counting": 0.10,
                            "contextual": 0.10, "adversarial": 0.05}
    assert abs(sum(DEPTH_MIX.values()) - 1) < 1e-9 and abs(sum(CATEGORY_MIX.values()) - 1) < 1e-9


def test_depth_and_category_sampling_follow_mix() -> None:
    rng = random.Random(1)
    n = 20000
    dc = Counter(sample_depth(rng) for _ in range(n))
    cc = Counter(sample_category(rng) for _ in range(n))
    for d, p in DEPTH_MIX.items():
        assert abs(dc[d] / n - p) < 0.015
    for c, p in CATEGORY_MIX.items():
        assert abs(cc[c] / n - p) < 0.015


def test_sample_program_typed_canonical_deterministic() -> None:
    hits = 0
    for seed in range(150):
        rng = random.Random(seed)
        cat = CATEGORIES[seed % len(CATEGORIES)]
        depth = 1 + seed % 6
        p = sample_program(rng, cat, depth=depth)
        assert p == sample_program(random.Random(seed), cat, depth=depth)
        assert typecheck(p) is T.GRID
        assert canonicalize(p) == p
        assert Node.from_str(p.to_str()) == p
        hits += p.depth() == depth
    assert hits >= 140  # canonical depth equals the target depth almost always
    # the probe-based path (grammar.random_program) works too
    q = sample_program(random.Random(5), "geometry", depth=2, probe=True)
    assert typecheck(q) is T.GRID


# ======================================================================================= degeneracy filter

def _grid(*rows: str) -> List[List[int]]:
    return [[int(ch) for ch in r] for r in rows]


def test_degeneracy_reasons() -> None:
    prog = Node.from_str("(ROTATE180 INPUT)")
    a, b, c = _grid("120", "000"), _grid("003", "300"), _grid("40", "04")
    good = [Pair(g, execute(prog, g)) for g in (a, b, c)]
    assert degeneracy_reason(prog, good, recheck=True) is None
    assert degeneracy_reason(prog, good[:1]) == "too_few_pairs"
    assert degeneracy_reason(prog, [good[0], good[0], good[1]]) == "duplicate_inputs"
    assert degeneracy_reason(prog, [Pair(g, g) for g in (a, b)]) == "identity"
    assert degeneracy_reason(prog, [Pair(a, _grid("55")), Pair(b, _grid("5", "5"))]) == "constant_output"
    assert degeneracy_reason(prog, [Pair(a, _grid("12")), Pair(b, _grid("12"))]) == "same_output"
    wrong = [good[0], good[1], Pair(c, _grid("11", "22"))]
    assert degeneracy_reason(prog, wrong, recheck=True) == "nondeterministic"
    big = [[1] * 31]
    assert degeneracy_reason(prog, [Pair(a, big), good[1]]) == "oversize"


def test_make_task_rejects_degenerate_programs() -> None:
    rng = random.Random(0)
    # canonicalises to the identity (depth 0)
    assert make_task_with_reason(rng, program="(ROTATE90 (ROTATE270 INPUT))")[1] == "trivial_program"
    # a program that always fails is an ExecError reject
    t, why = make_task_with_reason(rng, program="(CROP INPUT (SELECT_COLOR (GET_COMPONENTS4 INPUT) 0))")
    assert t is None
    # explicit depth that cannot be met by the canonical program is rejected, never mislabelled
    for seed in range(60):
        task, reason = make_task_with_reason(random.Random(seed), depth=3)
        assert (task is None) or task.depth == 3


def test_make_task_valid_and_deterministic() -> None:
    made = 0
    for seed in range(120):
        t1 = make_task(random.Random(seed))
        t2 = make_task(random.Random(seed))
        assert (t1 is None) == (t2 is None)
        if t1 is None:
            continue
        assert t1 == t2
        made += 1
        assert isinstance(t1, SynthTask)
        assert 3 <= len(t1.pairs) <= 6
        prog = Node.from_str(t1.program)
        assert canonicalize(prog) == prog and prog.depth() == t1.depth
        assert t1.primitives == sorted(prog.primitives())
        assert t1.category in CATEGORIES
        assert 0.0 <= t1.difficulty <= 1.0
        assert degeneracy_reason(prog, t1.pairs, recheck=True) is None
    assert made >= 50


def test_make_task_fixed_program_and_n_pairs() -> None:
    rng = random.Random(4)
    tasks = [make_task(rng, program="(REFLECT_H INPUT)", n_pairs=(4, 4), category="geometry") for _ in range(20)]
    tasks = [t for t in tasks if t is not None]
    assert len(tasks) >= 15
    for t in tasks:
        assert len(t.pairs) == 4 and t.program == "(REFLECT_H INPUT)" and t.depth == 1
        for p in t.pairs:
            assert p.output == [row[::-1] for row in p.input] or p.output == p.input[::-1]


# ======================================================================================= generated dataset

def test_generated_rows_schema(generated: Dict[str, Any]) -> None:
    rows = generated["rows"]
    assert len(rows) == N_MAIN == generated["stats"]["n"]
    keys = ["task_id", "program", "pairs", "depth", "primitives", "category", "difficulty", "adversarial", "split"]
    ids = set()
    for r in rows:
        assert list(r) == keys
        assert r["split"] in ("train", "val_comp")
        assert 3 <= len(r["pairs"]) <= 6
        assert isinstance(r["adversarial"], bool) and 0.0 <= r["difficulty"] <= 1.0
        ids.add(r["task_id"])
    assert len(ids) == N_MAIN


def test_500_tasks_reexecute_exactly(generated: Dict[str, Any]) -> None:
    for r in generated["rows"][:500]:
        prog = Node.from_str(r["program"])
        assert prog.to_str() == r["program"] and prog.depth() == r["depth"]
        assert sorted(prog.primitives()) == r["primitives"]
        pairs = [Pair(p["input"], p["output"]) for p in r["pairs"]]
        for p in pairs:
            assert validate_grid(p.input) and validate_grid(p.output)
            assert execute(prog, p.input) == p.output
        assert degeneracy_reason(prog, pairs, recheck=True) is None


def test_depth_histogram_matches_mix(generated: Dict[str, Any]) -> None:
    rows = generated["rows"]
    hist = Counter(r["depth"] for r in rows)
    assert set(hist) <= set(DEPTH_MIX)
    for d, p in DEPTH_MIX.items():
        assert abs(hist[d] / len(rows) - p) <= 0.05, (d, hist[d] / len(rows), p)


def test_category_histogram_matches_mix(generated: Dict[str, Any]) -> None:
    rows = generated["rows"]
    hist = Counter(r["category"] for r in rows)
    for c, p in CATEGORY_MIX.items():
        assert abs(hist[c] / len(rows) - p) <= 0.05, (c, hist[c] / len(rows), p)
    adv = [r for r in rows if r["category"] == "adversarial"]
    assert adv and all(r["adversarial"] for r in adv)
    assert not any(r["adversarial"] for r in rows if r["category"] != "adversarial")


def test_compositional_split_no_leakage(generated: Dict[str, Any]) -> None:
    rows = generated["rows"]
    held = [frozenset(p) for p in HELDOUT_COMPOSITIONS]
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val_comp"]
    assert val, "val_comp must not be empty"
    assert 0.02 <= len(val) / len(rows) <= 0.25
    for r in train:
        ps = set(r["primitives"])
        assert not any(h <= ps for h in held), r["program"]
    for r in val:
        assert any(h <= set(r["primitives"]) for h in held)
    # primitives seen: every primitive of a held-out pair (and of any val_comp task) also occurs in train
    train_prims = set().union(*(set(r["primitives"]) for r in train))
    for a, b in HELDOUT_COMPOSITIONS:
        assert a in train_prims and b in train_prims
    # (a primitive the DSL sampler draws very rarely, e.g. FILTER at ~0.1 %, may by chance occur only in val)
    seen_all = sum(1 for r in val if set(r["primitives"]) <= train_prims)
    assert seen_all / len(val) >= 0.97
    # the pairs themselves never co-occur in train
    train_pairs = set()
    for r in train:
        train_pairs |= {frozenset(x) for x in itertools.combinations(sorted(set(r["primitives"])), 2)}
    assert not (set(held) & train_pairs)
    assert generated["stats"]["train"] == len(train) and generated["stats"]["val_comp"] == len(val)


def test_split_rule_function() -> None:
    a, b = HELDOUT_COMPOSITIONS[0]
    assert split_of([a, b, "RECOLOR"]) == "val_comp"
    assert split_of([a]) == "train" and split_of([b]) == "train"
    assert split_of([a, b], "none") == "train"
    with pytest.raises(ValueError):
        split_of([a], "random")


def test_generated_difficulty_spread(generated: Dict[str, Any]) -> None:
    rows = generated["rows"]
    by_depth: Dict[int, List[float]] = {}
    for r in rows:
        by_depth.setdefault(r["depth"], []).append(r["difficulty"])
    means = [sum(v) / len(v) for _, v in sorted(by_depth.items())]
    assert means == sorted(means), "mean difficulty must grow with depth"
    assert means[-1] - means[0] > 0.3


def test_load_jsonl_and_row_roundtrip(generated: Dict[str, Any]) -> None:
    tasks = load_jsonl(str(generated["path"]))
    assert len(tasks) == N_MAIN
    val = load_jsonl(str(generated["path"]), split="val_comp")
    assert len(val) == generated["stats"]["val_comp"]
    t = tasks[0]
    row = task_to_row(t, "train")
    assert row_to_task(row) == t
    arc = to_arc_task(t)
    assert isinstance(arc, Task) and len(arc.test) == 1 and len(arc.train) == len(t.pairs) - 1


def test_throughput(generated: Dict[str, Any]) -> None:
    per_worker = generated["stats"]["tasks_per_s_per_worker"]
    log.warning("synthetic throughput: %d tasks/s/worker (%d tasks in %.2fs)", per_worker, N_MAIN,
                generated["seconds"])
    assert per_worker >= 100  # target is >= 300 on an idle CPU; loose bound keeps the test stable under load


# ======================================================================================= determinism / workers / CLI

def test_generate_deterministic_and_worker_invariant(tmp_path: Path) -> None:
    p1, p2, p3 = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    s1 = generate(240, str(p1), seed=7, workers=1, chunk_size=80)
    s2 = generate(240, str(p2), seed=7, workers=2, chunk_size=80)
    generate(240, str(p3), seed=8, workers=1, chunk_size=80)
    assert p1.read_bytes() == p2.read_bytes()
    assert p1.read_bytes() != p3.read_bytes()
    assert s1["n"] == s2["n"] == 240 and s1["train"] + s1["val_comp"] == 240
    assert s1["attempts"] == s2["attempts"]


def test_generate_split_rule_none(tmp_path: Path) -> None:
    p = tmp_path / "n.jsonl"
    st = generate(50, str(p), seed=1, split_rule="none")
    assert st["train"] == 50 and st["val_comp"] == 0
    with pytest.raises(ValueError):
        generate(5, str(p), seed=1, split_rule="random")


def test_cli(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "cli.jsonl"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    res = subprocess.run([sys.executable, "-m", "arcjepa.synthetic.dataset", "--n", "40", "--out", str(out),
                          "--seed", "3", "--workers", "2", "--chunk-size", "20"],
                         cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    stats = json.loads(res.stdout.strip().splitlines()[-1])
    assert stats["n"] == 40
    rows = list(iter_rows(str(out)))
    assert len(rows) == 40
    # in-process main() with the same arguments reproduces the file byte for byte
    out2 = tmp_path / "cli2.jsonl"
    cli_main(["--n", "40", "--out", str(out2), "--seed", "3", "--workers", "1", "--chunk-size", "20"])
    assert out.read_bytes() == out2.read_bytes()


# ======================================================================================= perturbations / difficulty

def _object_task(seed: int) -> SynthTask:
    rng = random.Random(seed)
    while True:
        t = make_task(rng, program="(REFLECT_H INPUT)", category="adversarial")
        if t is not None:
            return t


def _check_consistent(t: SynthTask) -> None:
    prog = Node.from_str(t.program)
    for p in t.pairs:
        assert execute(prog, p.input) == p.output
    assert degeneracy_reason(prog, t.pairs, recheck=True) is None


def test_add_distractors() -> None:
    changed = 0
    for seed in range(20):
        base = _object_task(seed)
        bg = task_background(base)
        pert = add_distractors(random.Random(seed), base)
        _check_consistent(pert)
        if pert.pairs != base.pairs:
            changed += 1
            assert pert.adversarial
            for a, b in zip(base.pairs, pert.pairs):
                diff = [(r, c) for r in range(len(a.input)) for c in range(len(a.input[0]))
                        if a.input[r][c] != b.input[r][c]]
                assert all(a.input[r][c] == bg for r, c in diff)  # distractors only cover background
    assert changed >= 15


def test_ambiguous_segmentation() -> None:
    changed = 0
    for seed in range(20):
        base = _object_task(seed)
        bg = task_background(base)
        pert = ambiguous_segmentation(random.Random(seed), base)
        _check_consistent(pert)
        if pert.pairs == base.pairs:
            continue
        changed += 1
        assert any(len(segment(p.input, "cc4", background=bg)) > len(segment(p.input, "cc8", background=bg))
                   for p in pert.pairs)
    assert changed >= 15


def test_perturbation_falls_back_on_degenerate() -> None:
    # a program whose perturbed inputs would all fail (CROP of a missing colour) keeps the original task
    t = _object_task(0)
    bad = SynthTask(t.task_id, "(CROP INPUT (SELECT_COLOR (GET_COMPONENTS4 INPUT) 0))", t.pairs, 3, [], "object",
                    0.1, False)
    assert add_distractors(random.Random(0), bad) is bad


def test_difficulty_score_and_cc4() -> None:
    assert difficulty_score(1, 0, 1, False) == 0.0
    assert difficulty_score(6, 50, 12, True) == 1.0
    assert difficulty_score(3, 2, 3, False) < difficulty_score(5, 2, 3, False)
    assert difficulty_score(3, 2, 3, False) < difficulty_score(3, 2, 3, True)
    rng = random.Random(9)
    for i in range(60):
        g = random_input_grid(rng, style=STYLES[i % len(STYLES)])
        assert count_cc4(g) == len(segment(g, "cc4"))
