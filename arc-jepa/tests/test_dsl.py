"""Tests for the typed ARC-JEPA DSL (INTERFACES.md §1): registry, AST, interpreter, grammar, canonicaliser,
mutations, plus a suite of hand-written ARC-like programs with expected outputs.

Run from the package root:  python -m pytest tests/test_dsl.py -q
"""
from __future__ import annotations

import copy
import dataclasses
import random
import statistics
import time
from collections import Counter
from typing import Dict, List

import pytest

from arcjepa.core.types import Grid, validate_grid
from arcjepa.dsl import (EXTENSION_PRIMITIVES, HARD_NEGATIVE_TYPES, INPUT, OBJ, REGISTRY, SPEC_PRIMITIVES,
                         STRUCTURAL_PRIMITIVES, DSLTypeError, ExecError, Node, Object, T, by_out_type, canonicalize,
                         crossover, enumerate_programs, evaluate, execute, expansions, hard_negatives, mutate,
                         random_program, structural_signature, typecheck)
from arcjepa.dsl.canonicalize import D4_TABLE, GEOMETRIC_OPS
from arcjepa.dsl.grammar import DEPTH_MIX, can_host, sample_depth
from arcjepa.dsl.interpreter import value_type_ok
from arcjepa.dsl.mutations import hard_negatives_typed

# ======================================================================================= helpers

SPEC_NAMES: Dict[str, List[str]] = {
    "selection": ["SELECT_ALL", "SELECT_COLOR", "SELECT_NONZERO", "SELECT_LARGEST", "SELECT_SMALLEST",
                  "SELECT_UNIQUE", "SELECT_BORDER", "SELECT_CENTER"],
    "analysis": ["GET_COMPONENTS4", "GET_COMPONENTS8", "GET_BBOX", "GET_CENTROID", "GET_AREA", "GET_PERIMETER",
                 "GET_HOLES", "GET_SYMMETRY"],
    "relation": ["LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW", "TOUCHING", "OVERLAPPING", "INSIDE", "CONTAINS",
                 "NEAREST", "FARTHEST"],
    "geometric": ["ROTATE90", "ROTATE180", "ROTATE270", "REFLECT_H", "REFLECT_V", "REFLECT_D1", "REFLECT_D2",
                  "TRANSPOSE", "SHIFT", "ALIGN"],
    "manipulation": ["COPY", "MOVE", "DELETE", "DUPLICATE", "MERGE", "SPLIT", "EXTEND", "SHRINK", "GROW", "FILL",
                     "OUTLINE", "FRAME"],
    "color": ["RECOLOR", "SWAP_COLORS", "MAP_COLOR", "MOST_COMMON_COLOR", "LEAST_COMMON_COLOR",
              "REPLACE_BACKGROUND", "COLOR_OBJECT", "COLOR_BY_POSITION"],
    "pattern": ["TILE", "REPEAT_X", "REPEAT_Y", "REPEAT_N", "MIRROR_TILE", "PATTERN_FILL", "ALTERNATE",
                "PERIODIC_REPEAT"],
    "counting": ["COUNT_OBJECTS", "COUNT_CELLS", "ARGMAX_SIZE", "ARGMIN_SIZE"],
    "conditional": ["IF", "APPLY_TO_EACH", "FILTER", "COMPOSE"],
}


def rect_grid(rng: random.Random, h: int = 0, w: int = 0, n_objects: int = 3) -> Grid:
    """Random grid of separated filled rectangles (>= 3x3); colours [a, a, b, ...] so one colour is unique."""
    h = h or rng.randint(10, 14)
    w = w or rng.randint(10, 14)
    a, b, c = rng.sample(range(1, 10), 3)
    colours = [a, a, b] + [c] * max(0, n_objects - 3)
    for _attempt in range(100):
        g = [[0] * w for _ in range(h)]
        placed = 0
        for col in colours:
            for _ in range(200):
                rh, rw = rng.randint(3, 4), rng.randint(3, 4)
                r0, c0 = rng.randint(0, h - rh), rng.randint(0, w - rw)
                # keep a one-cell background gap around every rectangle
                if any(g[r][cc] for r in range(max(0, r0 - 1), min(h, r0 + rh + 1))
                       for cc in range(max(0, c0 - 1), min(w, c0 + rw + 1))):
                    continue
                for r in range(r0, r0 + rh):
                    for cc in range(c0, c0 + rw):
                        g[r][cc] = col
                placed += 1
                break
        if placed >= 3:
            return g
    raise AssertionError("could not place 3 rectangles")


def noisy_grid(rng: random.Random, h: int = 0, w: int = 0) -> Grid:
    """Random grid of small (possibly overlapping / touching) coloured blobs."""
    h = h or rng.randint(4, 15)
    w = w or rng.randint(4, 15)
    g = [[0] * w for _ in range(h)]
    for _ in range(rng.randint(1, 7)):
        col = rng.randint(1, 9)
        r0, c0 = rng.randrange(h), rng.randrange(w)
        r1, c1 = min(h - 1, r0 + rng.randint(0, 3)), min(w - 1, c0 + rng.randint(0, 3))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if rng.random() < 0.85:
                    g[r][c] = col
    return g


def run(src: str, grid: Grid) -> Grid:
    return execute(Node.from_str(src), grid)


# ======================================================================================= registry / types

def test_type_enum_exact() -> None:
    assert [t.value for t in T] == ["GRID", "OBJECT_SET", "OBJECT", "MASK", "COLOR", "POSITION", "INTEGER",
                                    "BOOLEAN", "RELATION", "PROGRAM"]


def test_registry_has_exactly_the_72_spec_primitives() -> None:
    expected = [n for names in SPEC_NAMES.values() for n in names]
    assert len(expected) == 72 and len(set(expected)) == 72
    assert set(SPEC_PRIMITIVES) == set(expected)
    assert len(SPEC_PRIMITIVES) == 72
    for cat, names in SPEC_NAMES.items():
        for n in names:
            assert REGISTRY[n].category == cat, n
            assert not REGISTRY[n].structural
    for n in STRUCTURAL_PRIMITIVES:
        assert REGISTRY[n].structural and n not in expected
    # spec extensions (INTERFACES.md §1): outside the 72, never structural, and nothing else is registered
    assert set(EXTENSION_PRIMITIVES) == {"UPSCALE", "DOWNSCALE", "DOWNSCALE_ANY", "KRON_SELF", "UPSCALE_NC",
                                         "PANEL_BOOL", "PANEL_OVERLAY", "CONNECT_SAME", "FILL_EMPTY_LINES",
                                         "BBOX_FILL"}
    for n in EXTENSION_PRIMITIVES:
        assert REGISTRY[n].extension and not REGISTRY[n].structural and n not in expected
    assert set(REGISTRY) == set(expected) | set(STRUCTURAL_PRIMITIVES) | set(EXTENSION_PRIMITIVES)


def test_spec_signature_examples() -> None:
    assert REGISTRY["SELECT_LARGEST"].signature() == ((T.OBJECT_SET,), T.OBJECT)
    assert REGISTRY["GET_AREA"].signature() == ((T.OBJECT,), T.INTEGER)
    assert REGISTRY["RECOLOR"].signature() == ((T.OBJECT, T.COLOR), T.OBJECT)
    assert REGISTRY["MOVE"].signature() == ((T.OBJECT, T.POSITION), T.OBJECT)
    for t in T:
        assert all(p.out_type is t for p in by_out_type(t))
    assert {p.name for p in by_out_type(T.GRID, include_structural=False)}.isdisjoint(STRUCTURAL_PRIMITIVES)


def test_object_type_is_parser_compatible() -> None:
    names = [f.name for f in dataclasses.fields(Object)]
    assert names[:4] == ["cells", "color_hist", "primary_color", "bbox"]
    from arcjepa.dsl._objects_fallback import Object as Fallback
    assert [f.name for f in dataclasses.fields(Fallback)][:4] == names[:4]


# ======================================================================================= every primitive

_OS = "(GET_COMPONENTS4 INPUT)"
_L = f"(SELECT_LARGEST {_OS})"
_S = f"(SELECT_SMALLEST {_OS})"

#: One expression per registered op whose ROOT is that op (72 spec primitives + structural helpers + spec
#: extensions).
PRIMITIVE_EXAMPLES: Dict[str, str] = {
    "SELECT_ALL": "(SELECT_ALL INPUT)",
    "SELECT_COLOR": f"(SELECT_COLOR {_OS} (ARGMAX_SIZE {_OS}))",
    "SELECT_NONZERO": "(SELECT_NONZERO INPUT)",
    "SELECT_LARGEST": _L,
    "SELECT_SMALLEST": _S,
    "SELECT_UNIQUE": f"(SELECT_UNIQUE {_OS})",
    "SELECT_BORDER": f"(SELECT_BORDER {_OS})",
    "SELECT_CENTER": f"(SELECT_CENTER {_OS})",
    "GET_COMPONENTS4": _OS,
    "GET_COMPONENTS8": "(GET_COMPONENTS8 INPUT)",
    "GET_BBOX": f"(GET_BBOX {_L})",
    "GET_CENTROID": f"(GET_CENTROID {_L})",
    "GET_AREA": f"(GET_AREA {_L})",
    "GET_PERIMETER": f"(GET_PERIMETER {_L})",
    "GET_HOLES": f"(GET_HOLES {_L})",
    "GET_SYMMETRY": f"(GET_SYMMETRY {_L})",
    "LEFT_OF": f"(LEFT_OF {_S} {_L})",
    "RIGHT_OF": f"(RIGHT_OF {_S} {_L})",
    "ABOVE": f"(ABOVE {_S} {_L})",
    "BELOW": f"(BELOW {_S} {_L})",
    "TOUCHING": f"(TOUCHING {_S} {_L})",
    "OVERLAPPING": f"(OVERLAPPING {_S} (GROW {_L}))",
    "INSIDE": f"(INSIDE {_S} {_L})",
    "CONTAINS": f"(CONTAINS {_L} {_S})",
    "NEAREST": f"(NEAREST {_OS} {_L})",
    "FARTHEST": f"(FARTHEST {_OS} {_L})",
    "ROTATE90": "(ROTATE90 INPUT)",
    "ROTATE180": "(ROTATE180 INPUT)",
    "ROTATE270": "(ROTATE270 INPUT)",
    "REFLECT_H": "(REFLECT_H INPUT)",
    "REFLECT_V": "(REFLECT_V INPUT)",
    "REFLECT_D1": "(REFLECT_D1 INPUT)",
    "REFLECT_D2": "(REFLECT_D2 INPUT)",
    "TRANSPOSE": "(TRANSPOSE INPUT)",
    "SHIFT": "(SHIFT INPUT (1 -1))",
    "ALIGN": f"(ALIGN {_L} top)",
    "COPY": f"(COPY INPUT {_L} (2 2))",
    "MOVE": f"(MOVE {_L} (0 1))",
    "DELETE": f"(DELETE INPUT {_L})",
    "DUPLICATE": f"(DUPLICATE {_L} (3 0))",
    "MERGE": f"(MERGE {_OS})",
    "SPLIT": f"(SPLIT (MERGE {_OS}))",
    "EXTEND": f"(EXTEND INPUT {_S} (0 1))",
    "SHRINK": f"(SHRINK {_L})",
    "GROW": f"(GROW {_S})",
    "FILL": f"(FILL INPUT (GET_BBOX {_S}) 7)",
    "OUTLINE": f"(OUTLINE INPUT {_L} 8)",
    "FRAME": "(FRAME INPUT 6)",
    "RECOLOR": f"(RECOLOR {_L} 4)",
    "SWAP_COLORS": "(SWAP_COLORS INPUT (MOST_COMMON_COLOR INPUT) (LEAST_COMMON_COLOR INPUT))",
    "MAP_COLOR": f"(MAP_COLOR INPUT (ARGMIN_SIZE {_OS}) 9)",
    "MOST_COMMON_COLOR": "(MOST_COMMON_COLOR INPUT)",
    "LEAST_COMMON_COLOR": "(LEAST_COMMON_COLOR INPUT)",
    "REPLACE_BACKGROUND": "(REPLACE_BACKGROUND INPUT 5)",
    "COLOR_OBJECT": f"(COLOR_OBJECT INPUT {_S} 3)",
    "COLOR_BY_POSITION": "(COLOR_BY_POSITION INPUT center)",
    "TILE": "(TILE INPUT 2 2)",
    "REPEAT_X": "(REPEAT_X INPUT 2)",
    "REPEAT_Y": "(REPEAT_Y INPUT 2)",
    "REPEAT_N": "(REPEAT_N INPUT 2)",
    "MIRROR_TILE": "(MIRROR_TILE INPUT)",
    "PATTERN_FILL": f"(PATTERN_FILL INPUT (GET_BBOX {_L}) (ROTATE90 INPUT))",
    "ALTERNATE": f"(ALTERNATE {_OS} 1 2)",
    "PERIODIC_REPEAT": "(PERIODIC_REPEAT INPUT)",
    "COUNT_OBJECTS": f"(COUNT_OBJECTS {_OS})",
    "COUNT_CELLS": "(COUNT_CELLS (SELECT_NONZERO INPUT))",
    "ARGMAX_SIZE": f"(ARGMAX_SIZE {_OS})",
    "ARGMIN_SIZE": f"(ARGMIN_SIZE {_OS})",
    "IF": f"(IF (GET_SYMMETRY {_L}) (ROTATE90 INPUT) (REFLECT_H INPUT))",
    "APPLY_TO_EACH": f"(APPLY_TO_EACH {_OS} (RECOLOR OBJ 5))",
    "FILTER": f"(FILTER {_OS} SAME_COLOR {_L})",
    "COMPOSE": "(COMPOSE (MOVE OBJ (0 1)) (RECOLOR OBJ 3))",
    "RENDER": f"(RENDER (APPLY_TO_EACH {_OS} (GROW OBJ)) INPUT)",
    "RENDER_OBJ": f"(RENDER_OBJ (RECOLOR {_L} 2) INPUT)",
    "RENDER_BLANK": f"(RENDER_BLANK {_OS} INPUT)",
    "CROP": f"(CROP INPUT {_L})",
    # spec extensions
    "UPSCALE": "(UPSCALE INPUT 2)",
    "DOWNSCALE": "(DOWNSCALE INPUT 2)",
    "DOWNSCALE_ANY": "(DOWNSCALE_ANY INPUT 2)",
    "KRON_SELF": "(KRON_SELF INPUT)",
    "UPSCALE_NC": "(UPSCALE_NC INPUT)",
    "PANEL_BOOL": "(PANEL_BOOL INPUT 2 (MOST_COMMON_COLOR INPUT))",
    "PANEL_OVERLAY": "(PANEL_OVERLAY INPUT 1)",
    "CONNECT_SAME": "(CONNECT_SAME INPUT 0)",
    "FILL_EMPTY_LINES": "(FILL_EMPTY_LINES INPUT 3)",
    "BBOX_FILL": f"(BBOX_FILL INPUT (ARGMIN_SIZE {_OS}))",
}


def small_grid(rng: random.Random, lo: int = 2, hi: int = 5) -> Grid:
    """Random grid with sides ``lo..hi`` and at least one non-zero cell (KRON_SELF squares the sides)."""
    h, w = rng.randint(lo, hi), rng.randint(lo, hi)
    g = [[rng.choice((0, 0, rng.randint(1, 9))) for _ in range(w)] for _ in range(h)]
    g[rng.randrange(h)][rng.randrange(w)] = rng.randint(1, 9)
    return g


def panel_grid(rng: random.Random, n: int = 0) -> Grid:
    """``n`` (2 or 3) equally shaped random panels side by side, joined by one-column separators of a colour that
    does not occur inside the panels."""
    n = n or rng.choice((2, 3))
    ph, pw = rng.randint(3, 6), rng.randint(3, 6)
    sep = rng.randint(1, 9)
    inner = [c for c in range(1, 10) if c != sep]
    panels = [[[rng.choice(inner) if rng.random() < 0.5 else 0 for _ in range(pw)] for _ in range(ph)]
              for _ in range(n)]
    rows: Grid = []
    for r in range(ph):
        row: List[int] = []
        for i, p in enumerate(panels):
            if i:
                row.append(sep)
            row.extend(p[r])
        rows.append(row)
    return rows


#: Valid-input generators for ops whose domain is narrower than rect_grid's (they raise ExecError otherwise, by
#: design): block downscaling needs sides divisible by the factor, KRON_SELF needs sides <= 5 (30 = 5 x 6 cap on
#: the squared side), the panel ops need a panel structure.
EXAMPLE_GRIDS = {
    "DOWNSCALE": lambda rng: rect_grid(rng, rng.choice((10, 12, 14)), rng.choice((10, 12, 14))),
    "DOWNSCALE_ANY": lambda rng: rect_grid(rng, rng.choice((10, 12, 14)), rng.choice((10, 12, 14))),
    "KRON_SELF": small_grid,
    "PANEL_BOOL": panel_grid,
    "PANEL_OVERLAY": panel_grid,
}


def test_examples_cover_every_registered_op() -> None:
    assert set(PRIMITIVE_EXAMPLES) == set(REGISTRY)
    for name, src in PRIMITIVE_EXAMPLES.items():
        assert Node.from_str(src).op == name
    assert set(EXAMPLE_GRIDS) <= set(REGISTRY)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_primitive_typechecks_and_executes(name: str) -> None:
    node = Node.from_str(PRIMITIVE_EXAMPLES[name])
    prim = REGISTRY[name]
    assert typecheck(node) is prim.out_type
    rng = random.Random(1000 + sorted(REGISTRY).index(name))
    make_grid = EXAMPLE_GRIDS.get(name, rect_grid)
    ok = 0
    for _ in range(4):
        g = make_grid(rng)
        before = copy.deepcopy(g)
        val = evaluate(node, g)
        assert g == before, "primitives must not mutate their inputs"
        assert value_type_ok(val, prim.out_type), (name, val)
        ok += 1
        # GRID-typed wrapper where the output is a grid: must also pass the strict execute() path
        if prim.out_type is T.GRID:
            assert validate_grid(execute(node, g))
    assert ok >= 3


def test_primitive_semantics_spot_checks() -> None:
    g = [[1, 2, 0], [0, 0, 3]]
    assert run("(ROTATE180 INPUT)", g) == [[3, 0, 0], [0, 2, 1]]
    assert run("(REFLECT_H INPUT)", g) == [[0, 2, 1], [3, 0, 0]]
    assert run("(REFLECT_V INPUT)", g) == [[0, 0, 3], [1, 2, 0]]
    assert run("(TRANSPOSE INPUT)", g) == [[1, 0], [2, 0], [0, 3]]
    assert run("(REFLECT_D2 INPUT)", [[1, 2], [3, 4]]) == [[4, 2], [3, 1]]
    assert run("(SHIFT INPUT (0 1))", g) == [[0, 1, 2], [0, 0, 0]]
    assert run("(SWAP_COLORS INPUT 1 3)", g) == [[3, 2, 0], [0, 0, 1]]
    assert run("(REPLACE_BACKGROUND INPUT 7)", g) == [[1, 2, 7], [7, 7, 3]]
    assert run("(FRAME INPUT 5)", [[0, 0, 0], [0, 1, 0], [0, 0, 0]]) == [[5, 5, 5], [5, 1, 5], [5, 5, 5]]
    assert evaluate(Node.from_str("(COUNT_OBJECTS (GET_COMPONENTS4 INPUT))"), g) == 3
    assert evaluate(Node.from_str("(MOST_COMMON_COLOR INPUT)"), [[1, 2, 2], [0, 0, 0]]) == 2
    assert run("(PERIODIC_REPEAT INPUT)", [[1, 2, 1, 0, 0, 0]]) == [[1, 2, 1, 2, 1, 2]]
    # 8-connectivity joins diagonal cells, 4-connectivity does not
    diag = [[4, 0], [0, 4]]
    assert evaluate(Node.from_str("(COUNT_OBJECTS (GET_COMPONENTS4 INPUT))"), diag) == 2
    assert evaluate(Node.from_str("(COUNT_OBJECTS (GET_COMPONENTS8 INPUT))"), diag) == 1


def test_multicolour_objects_keep_their_colours() -> None:
    g = [[1, 2, 0], [0, 0, 0]]
    prog = "(RENDER_BLANK (SPLIT (MOVE (MERGE (GET_COMPONENTS4 INPUT)) (1 0))) INPUT)"
    assert run(prog, g) == [[0, 0, 0], [1, 2, 0]]
    merged = evaluate(Node.from_str("(MERGE (GET_COMPONENTS4 INPUT))"), g)
    assert merged.area == 2 and {c for c, _ in merged.color_hist} == {1, 2}


# ======================================================================================= AST

def test_ast_example_round_trip_and_methods() -> None:
    s = "(RECOLOR (SELECT_LARGEST (GET_COMPONENTS4 INPUT)) 3)"
    n = Node.from_str(s)
    assert n.to_str() == s
    assert n == Node("RECOLOR", (Node("SELECT_LARGEST", (Node("GET_COMPONENTS4", (INPUT,)),)), 3))
    assert n.depth() == 3 and INPUT.depth() == 0
    assert n.size() == 4
    assert n.primitives() == {"RECOLOR", "SELECT_LARGEST", "GET_COMPONENTS4"}
    assert n.children() == [n.args[0]]
    assert n.paths() == [(), (0,), (0, 0), (0, 0, 0)]
    m = n.replace((0, 0), Node("GET_COMPONENTS8", (INPUT,)))
    assert m.to_str() == "(RECOLOR (SELECT_LARGEST (GET_COMPONENTS8 INPUT)) 3)"
    assert n.to_str() == s  # immutable
    lit = Node.from_str("(SHIFT (ALIGN_DUMMY INPUT center True) (-2 3))")
    assert lit.args[1] == (-2, 3) and lit.args[0].args[1] == "center" and lit.args[0].args[2] is True
    with pytest.raises(ValueError):
        Node.from_str("(ROTATE90 INPUT")
    with pytest.raises(ValueError):
        Node.from_str("(ROTATE90 INPUT))")


def test_from_str_to_str_identity_on_500_random_programs() -> None:
    rng = random.Random(7)
    for _ in range(500):
        p = random_program(rng)
        s = p.to_str()
        q = Node.from_str(s)
        assert q == p
        assert q.to_str() == s


# ======================================================================================= interpreter

def test_interpreter_is_deterministic() -> None:
    rng = random.Random(11)
    progs = [random_program(rng, d) for d in (1, 2, 3, 4, 5, 6) for _ in range(15)]
    grids = [noisy_grid(rng) for _ in range(4)]
    for p in progs:
        for g in grids:
            outs = []
            for _ in range(2):
                try:
                    outs.append(execute(p, copy.deepcopy(g)))
                except ExecError as e:
                    outs.append(("err", str(e).split(":")[0]))
            assert outs[0] == outs[1], p.to_str()


def test_interpreter_errors_are_exec_errors() -> None:
    g = [[1, 0], [0, 2]]
    with pytest.raises(ExecError):
        execute(Node("NO_SUCH_OP", (INPUT,)), g)
    with pytest.raises(ExecError):
        execute(Node("ROTATE90", (INPUT, INPUT)), g)  # arity
    with pytest.raises(ExecError):
        execute(Node.from_str("(SELECT_LARGEST (GET_COMPONENTS4 INPUT))"), g)  # not GRID-typed
    with pytest.raises(ExecError):
        execute(Node.from_str("(FRAME INPUT 12)"), g)  # colour literal out of domain
    with pytest.raises(ExecError):
        execute(Node.from_str("(TILE INPUT 3 3)"), [[1] * 12 for _ in range(12)])  # 36 > 30
    with pytest.raises(ExecError):
        execute(Node.from_str("(RENDER_OBJ (SELECT_LARGEST (GET_COMPONENTS4 INPUT)) INPUT)"), [[0, 0], [0, 0]])
    with pytest.raises(ExecError):
        execute(Node("ROTATE90", (INPUT,)), [[1, 2], [3]])  # ragged input
    with pytest.raises(ExecError):
        execute(Node.from_str("(ROTATE90 (ROTATE90 (ROTATE90 INPUT)))"), g, max_steps=2)
    with pytest.raises(ExecError):
        execute(Node("ROTATE90", (INPUT,)), g, timeout_s=0.0)
    with pytest.raises(ExecError):
        execute(Node.from_str("(REPEAT_X INPUT 4)"), [[1] * 5 for _ in range(5)], max_cells=50)
    assert issubclass(DSLTypeError, TypeError)
    with pytest.raises(TypeError):
        typecheck(Node("GET_AREA", (INPUT,)))
    with pytest.raises(TypeError):
        typecheck(Node("RECOLOR", (OBJ, 3)))  # OBJ outside a PROGRAM body


def test_execute_never_returns_invalid_or_raises_other_exceptions() -> None:
    rng = random.Random(3)
    progs = [random_program(rng) for _ in range(250)]
    progs += [mutate(rng, p) for p in progs[:150]]
    n_ok = 0
    for p in progs:
        for g in (noisy_grid(rng), rect_grid(rng), [[rng.randint(0, 9)]]):
            try:
                out = execute(p, g)
            except ExecError:
                continue
            assert out is not None and validate_grid(out), p.to_str()
            n_ok += 1
    assert n_ok > 0


def test_execute_depth4_on_30x30_is_fast() -> None:
    rng = random.Random(5)
    g = rect_grid(rng, 30, 30, n_objects=12)
    progs = [random_program(random.Random(s), 4) for s in range(60)]
    for p in progs[:5]:  # warm-up
        try:
            execute(p, g)
        except ExecError:
            pass
    times = []
    for p in progs:
        t0 = time.perf_counter()
        try:
            execute(p, g)
        except ExecError:
            pass
        times.append(time.perf_counter() - t0)
    assert statistics.median(times) < 0.020, statistics.median(times)
    assert max(times) < 0.5 + 0.1  # hard time budget (timeout_s) is respected


# ======================================================================================= grammar

def test_expansions_are_type_constrained() -> None:
    d1 = {p.name for p in expansions(T.GRID, 1)}
    assert "ROTATE90" in d1 and "FRAME" in d1
    assert "RENDER_OBJ" not in d1 and "IF" not in d1  # need an OBJECT / BOOLEAN-driven subtree
    assert all(p.out_type is T.GRID for p in expansions(T.GRID, 3))
    assert {p.name for p in expansions(T.OBJECT, 2)} >= {"SELECT_LARGEST", "MERGE"}
    assert expansions(T.GRID, 0) == []
    reach = set()
    for t in T:
        for d in range(1, 7):
            reach |= {p.name for p in expansions(t, d)}
            reach |= {p.name for p in expansions(t, d, in_lambda=True)}
    assert reach == set(REGISTRY)


def test_random_program_typed_exact_depth_and_executable() -> None:
    rng = random.Random(0)
    grid_rng = random.Random(99)
    n_ok = n_total = 0
    for depth in range(1, 7):
        for _ in range(30):
            p = random_program(rng, depth)
            assert typecheck(p) is T.GRID
            assert p.depth() == depth, (depth, p.to_str())
            g = noisy_grid(grid_rng)
            n_total += 1
            try:
                execute(p, g)
                n_ok += 1
            except ExecError:
                pass
    assert n_ok / n_total >= 0.5, n_ok / n_total


def test_random_program_deterministic_and_depth_mix() -> None:
    a = [random_program(random.Random(42)).to_str() for _ in range(3)]
    assert len(set(a)) == 1
    s1 = [random_program(r).to_str() for r in [random.Random(5)] for _ in range(20)]
    s2 = [random_program(r).to_str() for r in [random.Random(5)] for _ in range(20)]
    assert s1 == s2
    rng = random.Random(1)
    hist = Counter(sample_depth(rng) for _ in range(4000))
    for d, frac in DEPTH_MIX.items():
        assert abs(hist[d] / 4000 - frac) < 0.03
    assert abs(sum(DEPTH_MIX.values()) - 1.0) < 1e-9


@pytest.mark.parametrize("category,prim_cats", [
    ("object", {"selection", "manipulation", "analysis", "structural"}),
    ("geometry", {"geometric", "pattern"}),
    ("relational", {"relation", "conditional"}),
    ("counting", {"counting"}),
    ("contextual", {"color", "pattern", "conditional"}),
])
def test_random_program_category_bias(category: str, prim_cats: set) -> None:
    rng = random.Random(17)
    hits = 0
    for _ in range(40):
        depth = rng.randint(3, 5)
        assert can_host(T.GRID, depth, False, frozenset(prim_cats))
        p = random_program(rng, depth, category)
        assert typecheck(p) is T.GRID and p.depth() == depth
        if any(REGISTRY[o].category in prim_cats for o in p.primitives()):
            hits += 1
    assert hits >= 36, hits
    # infeasible depth: relation/counting primitives need >= 3 levels, sampling still returns a typed program
    assert not can_host(T.GRID, 1, False, frozenset({"relation", "counting"}))
    assert typecheck(random_program(rng, 1, "relational")) is T.GRID


def test_sampler_covers_most_primitives() -> None:
    rng = random.Random(2)
    used = set()
    for _ in range(1500):
        used |= random_program(rng).primitives()
    assert len(used & set(SPEC_PRIMITIVES)) >= 60, sorted(set(SPEC_PRIMITIVES) - used)
    assert set(EXTENSION_PRIMITIVES) <= used, sorted(set(EXTENSION_PRIMITIVES) - used)


def test_enumerate_programs_breadth_first_canonical_dedup() -> None:
    t0 = time.perf_counter()
    progs = list(enumerate_programs(2, 2000))
    assert time.perf_counter() - t0 < 20
    keys = [p.to_str() for p in progs]
    assert len(set(keys)) == len(keys) >= 500
    assert len(progs) <= 2000
    last = 0
    for p in progs:
        assert typecheck(p) is T.GRID
        assert canonicalize(p) == p
        d = p.depth()
        assert d <= 2
        if d >= 1:  # breadth-first: depth never decreases (the canonical identity INPUT has depth 0)
            assert d >= last
            last = d
    assert any(p.depth() == 1 for p in progs) and any(p.depth() == 2 for p in progs)
    assert list(enumerate_programs(1, 10)) == list(enumerate_programs(1, 10))  # deterministic
    assert list(enumerate_programs(2, 0)) == []


# ======================================================================================= canonicalisation

def test_canonicalize_rotation_composition() -> None:
    r = Node.from_str("(ROTATE90 (ROTATE90 INPUT))")
    assert canonicalize(r) == Node.from_str("(ROTATE180 INPUT)")
    assert canonicalize(Node.from_str("(ROTATE90 (ROTATE270 INPUT))")) == INPUT
    assert canonicalize(Node.from_str("(REFLECT_H (REFLECT_H (ROTATE90 INPUT)))")) == Node.from_str(
        "(ROTATE90 INPUT)")
    assert canonicalize(Node.from_str("(TRANSPOSE INPUT)")) == Node.from_str("(REFLECT_D1 INPUT)")


def test_d4_table_matches_execution() -> None:
    g = [[1, 2, 3], [4, 5, 6]]  # non-square and asymmetric
    for (outer, inner), res in D4_TABLE.items():
        got = run(f"({outer} ({inner} INPUT))", g)
        want = g if res is None else run(f"({res} INPUT)", g)
        assert got == want, (outer, inner, res)
    assert len(D4_TABLE) == len(GEOMETRIC_OPS) ** 2


def test_canonicalize_identities_idempotents_ordering() -> None:
    c = canonicalize
    L = _L
    assert c(Node.from_str(f"(RENDER_OBJ (MOVE {L} (0 0)) INPUT)")) == Node.from_str(f"(RENDER_OBJ {L} INPUT)")
    assert c(Node.from_str("(SHIFT INPUT (0 0))")) == INPUT
    assert c(Node.from_str("(MAP_COLOR INPUT 3 3)")) == INPUT
    assert c(Node.from_str("(TILE INPUT 1 1)")) == INPUT
    assert c(Node.from_str(f"(RENDER (APPLY_TO_EACH {_OS} OBJ) INPUT)")) == Node.from_str(
        f"(RENDER {_OS} INPUT)")
    assert c(Node.from_str(f"(RENDER_OBJ (RECOLOR (RECOLOR {L} 2) 5) INPUT)")) == Node.from_str(
        f"(RENDER_OBJ (RECOLOR {L} 5) INPUT)")
    assert c(Node.from_str("(FRAME (FRAME INPUT 1) 2)")) == Node.from_str("(FRAME INPUT 2)")
    assert c(Node.from_str(f"(RENDER_OBJ (MOVE (MOVE {L} (1 0)) (1 2)) INPUT)")) == Node.from_str(
        f"(RENDER_OBJ (MOVE {L} (2 2)) INPUT)")
    assert c(Node.from_str("(SWAP_COLORS INPUT 5 2)")) == Node.from_str("(SWAP_COLORS INPUT 2 5)")
    a = Node.from_str(f"(IF (TOUCHING {L} {_S}) (ROTATE90 INPUT) INPUT)")
    b = Node.from_str(f"(IF (TOUCHING {_S} {L}) (ROTATE90 INPUT) INPUT)")
    assert c(a) == c(b)
    assert c(Node.from_str(f"(IF (RIGHT_OF {L} {_S}) INPUT (ROTATE90 INPUT))")) == Node.from_str(
        f"(IF (LEFT_OF {_S} {L}) INPUT (ROTATE90 INPUT))")
    assert c(Node.from_str("(IF True (ROTATE90 INPUT) INPUT)")) == Node.from_str("(ROTATE90 INPUT)")
    assert c(Node.from_str(f"(IF (GET_SYMMETRY {L}) INPUT INPUT)")) == INPUT
    # computed background colour may be 0: must NOT collapse
    keep = Node.from_str("(REPLACE_BACKGROUND (REPLACE_BACKGROUND INPUT (MOST_COMMON_COLOR INPUT)) 4)")
    assert c(keep) == keep


def test_compose_inlining_respects_scope() -> None:
    body = Node.from_str("(MERGE (APPLY_TO_EACH (SPLIT OBJ) (RECOLOR OBJ 3)))")
    comp = Node("COMPOSE", (Node.from_str("(MOVE OBJ (1 0))"), body))
    prog = Node("RENDER", (Node("APPLY_TO_EACH", (Node.from_str(_OS), comp)), INPUT))
    canon = canonicalize(prog)
    assert "COMPOSE" not in canon.primitives()
    # the inner lambda's OBJ must stay bound to the inner APPLY_TO_EACH
    assert "(RECOLOR OBJ 3)" in canon.to_str()
    g = [[1, 1, 0, 0], [0, 0, 0, 2], [0, 0, 0, 0]]
    assert execute(prog, g) == execute(canon, g)


def test_canonicalize_preserves_semantics_and_is_idempotent() -> None:
    rng = random.Random(23)
    grids = [noisy_grid(rng) for _ in range(3)] + [rect_grid(rng)]
    for _ in range(300):
        p = random_program(rng)
        cp = canonicalize(p)
        assert canonicalize(cp) == cp
        assert typecheck(cp) is T.GRID
        for g in grids:
            try:
                want = execute(p, g)
            except ExecError:
                continue
            assert execute(cp, g) == want, (p.to_str(), cp.to_str())


def test_structural_signature_drops_literals() -> None:
    a = Node.from_str("(FRAME (SHIFT INPUT (1 0)) 3)")
    b = Node.from_str("(FRAME (SHIFT INPUT (0 -2)) 7)")
    assert structural_signature(a) == structural_signature(b) == "(FRAME (SHIFT INPUT _) _)"
    assert structural_signature(a) != structural_signature(Node.from_str("(FRAME (ROTATE90 INPUT) 3)"))


# ======================================================================================= mutations

_RICH = [
    f"(RENDER (APPLY_TO_EACH (FILTER {_OS} LEFT_OF {_L}) (RECOLOR OBJ 8)) (MAP_COLOR INPUT 1 2))",
    f"(IF (TOUCHING {_S} {_L}) (ROTATE90 (FRAME INPUT 3)) (COLOR_OBJECT INPUT {_L} 4))",
    f"(RENDER_OBJ (MOVE (RECOLOR {_L} 5) (1 1)) (REFLECT_H (SHIFT INPUT (0 1))))",
    f"(OUTLINE (FILL INPUT (GET_HOLES {_L}) 6) (NEAREST {_OS} {_S}) 2)",
]


def test_mutate_is_type_preserving() -> None:
    rng = random.Random(8)
    changed = 0
    for _ in range(200):
        p = random_program(rng)
        m = mutate(rng, p)
        assert typecheck(m) is T.GRID
        changed += m != p
    assert changed >= 180


def test_hard_negatives_cover_the_eight_types() -> None:
    rng = random.Random(4)
    covered = set()
    for src in _RICH:
        pos = Node.from_str(src)
        typed = hard_negatives_typed(rng, pos, k=8)
        assert 1 <= len(typed) <= 8
        keys = {canonicalize(pos).to_str()}
        for kind, neg in typed:
            assert typecheck(neg) is T.GRID
            key = canonicalize(neg).to_str()
            assert key not in keys, (kind, neg.to_str())
            keys.add(key)
            covered.add(kind)
    assert set(HARD_NEGATIVE_TYPES) <= covered, set(HARD_NEGATIVE_TYPES) - covered
    assert len(HARD_NEGATIVE_TYPES) == 8


def test_hard_negatives_on_random_programs() -> None:
    rng = random.Random(9)
    for _ in range(60):
        p = random_program(rng, rng.randint(2, 5))
        negs = hard_negatives(rng, p, k=8)
        assert len(negs) <= 8
        pk = canonicalize(p).to_str()
        assert all(typecheck(n) is T.GRID and canonicalize(n).to_str() != pk for n in negs)
    a = hard_negatives(random.Random(1), Node.from_str(_RICH[0]))
    b = hard_negatives(random.Random(1), Node.from_str(_RICH[0]))
    assert a == b


def test_crossover_is_typed() -> None:
    rng = random.Random(6)
    for _ in range(100):
        a, b = random_program(rng), random_program(rng)
        c = crossover(rng, a, b)
        assert typecheck(c) is T.GRID
        assert c.depth() >= 1 or c == a


# ======================================================================================= ARC-like programs

ARC_CASES = [
    ("rotate", "(ROTATE90 INPUT)",
     [([[1, 2, 3], [4, 5, 6]], [[4, 1], [5, 2], [6, 3]])]),
    ("recolor_largest", "(RENDER_OBJ (RECOLOR (SELECT_LARGEST (GET_COMPONENTS4 INPUT)) 3) INPUT)",
     [([[1, 1, 0, 0], [1, 1, 0, 2], [0, 0, 0, 2], [3, 0, 0, 0]],
       [[3, 3, 0, 0], [3, 3, 0, 2], [0, 0, 0, 2], [3, 0, 0, 0]])]),
    ("move_objects", "(RENDER_BLANK (APPLY_TO_EACH (GET_COMPONENTS4 INPUT) (MOVE OBJ (1 0))) INPUT)",
     [([[1, 0, 2], [0, 0, 2], [0, 0, 0]], [[0, 0, 0], [1, 0, 2], [0, 0, 2]])]),
    ("tile", "(TILE INPUT 2 3)",
     [([[1, 2], [3, 4]], [[1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4], [1, 2, 1, 2, 1, 2], [3, 4, 3, 4, 3, 4]])]),
    ("mirror", "(MIRROR_TILE INPUT)",
     [([[1, 2], [3, 0]], [[1, 2, 2, 1], [3, 0, 0, 3], [3, 0, 0, 3], [1, 2, 2, 1]])]),
    ("crop_bbox", "(CROP INPUT (SELECT_LARGEST (GET_COMPONENTS8 INPUT)))",
     [([[0, 0, 0, 0, 0], [0, 5, 5, 0, 0], [0, 5, 0, 5, 0], [0, 0, 5, 5, 0], [0, 0, 0, 0, 1]],
       [[5, 5, 0], [5, 0, 5], [0, 5, 5]])]),
    ("fill_holes", "(FILL INPUT (GET_HOLES (SELECT_LARGEST (GET_COMPONENTS4 INPUT))) 4)",
     [([[2, 2, 2, 2, 0], [2, 0, 0, 2, 0], [2, 2, 2, 2, 0], [0, 0, 0, 0, 1]],
       [[2, 2, 2, 2, 0], [2, 4, 4, 2, 0], [2, 2, 2, 2, 0], [0, 0, 0, 0, 1]])]),
    ("count_based", "(REPEAT_X INPUT (COUNT_OBJECTS (GET_COMPONENTS4 INPUT)))",
     [([[1, 0, 2], [0, 0, 0]], [[1, 0, 2, 1, 0, 2], [0, 0, 0, 0, 0, 0]]),
      ([[1, 0, 2], [0, 3, 0]], [[1, 0, 2, 1, 0, 2, 1, 0, 2], [0, 3, 0, 0, 3, 0, 0, 3, 0]])]),
    ("conditional_recolor",
     "(IF (GET_SYMMETRY (SELECT_LARGEST (GET_COMPONENTS4 INPUT))) (MAP_COLOR INPUT 1 2) (MAP_COLOR INPUT 1 3))",
     [([[1, 1, 1], [0, 1, 0], [0, 0, 0]], [[2, 2, 2], [0, 2, 0], [0, 0, 0]]),
      ([[1, 1, 0], [0, 1, 1], [0, 0, 0]], [[3, 3, 0], [0, 3, 3], [0, 0, 0]])]),
    ("compose",
     "(RENDER_BLANK (APPLY_TO_EACH (GET_COMPONENTS4 INPUT) (COMPOSE (MOVE OBJ (0 1)) (RECOLOR OBJ 5))) INPUT)",
     [([[1, 0, 0], [0, 2, 0]], [[0, 5, 0], [0, 0, 5]])]),
    ("relational_recolor",
     "(RENDER (APPLY_TO_EACH (FILTER (GET_COMPONENTS4 INPUT) LEFT_OF (SELECT_LARGEST (GET_COMPONENTS4 INPUT)))"
     " (RECOLOR OBJ 8)) INPUT)",
     [([[1, 0, 3, 3, 0, 2], [0, 0, 3, 3, 0, 0], [4, 0, 0, 0, 0, 0]],
       [[8, 0, 3, 3, 0, 2], [0, 0, 3, 3, 0, 0], [8, 0, 0, 0, 0, 0]])]),
]


@pytest.mark.parametrize("name,src,pairs", ARC_CASES, ids=[c[0] for c in ARC_CASES])
def test_hand_written_arc_programs(name: str, src: str, pairs) -> None:
    prog = Node.from_str(src)
    assert prog.to_str() == src
    assert typecheck(prog) is T.GRID
    canon = canonicalize(prog)
    for inp, want in pairs:
        assert execute(prog, inp) == want, name
        assert execute(canon, inp) == want, name  # canonical form is semantically identical


# ======================================================================================= spec extensions

def test_extension_scaling_semantics() -> None:
    g = [[1, 2], [0, 3]]
    assert run("(UPSCALE INPUT 2)", g) == [[1, 1, 2, 2], [1, 1, 2, 2], [0, 0, 3, 3], [0, 0, 3, 3]]
    assert run("(UPSCALE INPUT 3)", [[4]]) == [[4, 4, 4]] * 3
    # blocks [[1,1],[1,0]] / [[0,0],[0,3]] / [[1,2],[2,1]]: majority (ties -> larger colour) vs. any non-zero
    blocks = [[1, 1, 0, 0, 1, 2], [1, 0, 0, 3, 2, 1]]
    assert run("(DOWNSCALE INPUT 2)", blocks) == [[1, 0, 2]]
    assert run("(DOWNSCALE_ANY INPUT 2)", blocks) == [[1, 3, 2]]
    assert run("(DOWNSCALE INPUT 3)", [[5, 5, 0], [0, 5, 0], [0, 0, 0]]) == [[0]]
    assert run("(DOWNSCALE_ANY INPUT 3)", [[5, 5, 0], [0, 5, 0], [0, 0, 0]]) == [[5]]
    assert run("(KRON_SELF INPUT)", [[1, 0], [0, 2]]) == [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]]
    assert run("(KRON_SELF INPUT)", [[3, 3, 0]]) == [[3, 3, 0, 3, 3, 0, 0, 0, 0]]
    assert run("(UPSCALE_NC INPUT)", [[1, 0], [0, 2]]) == [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]]
    assert run("(UPSCALE_NC INPUT)", [[5, 5]]) == [[5, 5]]  # one colour: factor 1
    big = [[1] * 16 for _ in range(16)]
    for bad, grid in [("(UPSCALE INPUT 2)", big),                    # 32 > 30
                      ("(UPSCALE INPUT 1)", g),                      # literal outside the factor domain 2..5
                      ("(UPSCALE INPUT (COUNT_OBJECTS (GET_COMPONENTS4 INPUT)))", [[0, 0], [0, 0]]),  # factor 0
                      ("(DOWNSCALE INPUT 2)", [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]),  # 3 rows, k 2
                      ("(DOWNSCALE_ANY INPUT 3)", [[1, 2], [3, 4]]),
                      ("(KRON_SELF INPUT)", [[1] * 6 for _ in range(6)]),  # 36 > 30
                      ("(UPSCALE_NC INPUT)", [[0, 0]])]:                  # no colour
        with pytest.raises(ExecError):
            run(bad, grid)


#: Two 3x2 panels separated by a column of 5: A = [[1,0],[0,1],[1,0]], B = [[1,1],[0,0],[0,1]].
_PANELS = [[1, 0, 5, 1, 1], [0, 1, 5, 0, 0], [1, 0, 5, 0, 1]]


@pytest.mark.parametrize("op,want", [
    (0, [[1, 0], [0, 0], [0, 0]]),   # AND
    (1, [[1, 1], [0, 1], [1, 1]]),   # OR
    (2, [[0, 1], [0, 1], [1, 1]]),   # XOR
    (3, [[0, 0], [1, 0], [0, 0]]),   # NOR
    (4, [[0, 0], [0, 1], [1, 0]]),   # FIRST_ONLY: A and not B
    (5, [[0, 1], [0, 0], [0, 1]]),   # LAST_ONLY: B and not A
])
def test_extension_panel_bool_semantics(op: int, want: Grid) -> None:
    assert run(f"(PANEL_BOOL INPUT {op} 7)", _PANELS) == [[7 * v for v in row] for row in want]


def test_extension_panel_splitting_and_overlay() -> None:
    from arcjepa.dsl.primitives import grid_panels, panel_priority
    # no separator lines: left / right halves when W >= 2H - 1, else top / bottom halves
    assert grid_panels([[1, 0, 0, 2], [1, 1, 2, 0]]) == [[[1, 0], [1, 1]], [[0, 2], [2, 0]]]
    assert run("(PANEL_BOOL INPUT 0 4)", [[1, 0, 0, 2], [1, 1, 2, 0]]) == [[0, 0], [4, 0]]
    assert run("(PANEL_BOOL INPUT 0 9)", [[1, 2], [0, 3], [4, 0], [0, 5]]) == [[9, 0], [0, 9]]
    with pytest.raises(ExecError):
        run("(PANEL_BOOL INPUT 0 1)", [[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # odd sides, no separators
    with pytest.raises(ExecError):
        REGISTRY["PANEL_BOOL"].fn(_PANELS, 6, 1)  # computed op outside 0..5
    # overlay: the highest-priority panel's non-zero colour wins
    p = [[1, 0, 5, 2, 2], [0, 1, 5, 0, 0], [1, 0, 5, 0, 2]]
    first_wins = [[1, 2], [0, 1], [1, 2]]
    second_wins = [[2, 2], [0, 1], [1, 2]]
    assert run("(PANEL_OVERLAY INPUT 0)", p) == first_wins
    assert run("(PANEL_OVERLAY INPUT 1)", p) == second_wins
    assert run("(PANEL_OVERLAY INPUT 2)", p) == second_wins  # reverse of the rotation starting at panel 0
    assert run("(PANEL_OVERLAY INPUT 3)", p) == first_wins
    with pytest.raises(ExecError):
        run("(PANEL_OVERLAY INPUT 4)", p)  # 2 panels have 4 orders
    # 2n dihedral orders: all 6 permutations of 3 panels
    assert [panel_priority(3, k) for k in range(6)] == [[0, 1, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0], [0, 2, 1],
                                                        [1, 0, 2]]


def test_extension_line_and_region_fill_semantics() -> None:
    g = [[3, 0, 0, 3, 0], [0, 0, 0, 0, 0], [3, 0, 4, 0, 0]]
    assert run("(CONNECT_SAME INPUT 0)", g) == [[3, 3, 3, 3, 0], [3, 0, 0, 0, 0], [3, 0, 4, 0, 0]]
    assert run("(CONNECT_SAME INPUT 8)", g) == [[3, 8, 8, 3, 0], [8, 0, 0, 0, 0], [3, 0, 4, 0, 0]]
    assert run("(CONNECT_SAME INPUT 8)", [[2, 2, 0], [0, 0, 0]]) == [[2, 2, 0], [0, 0, 0]]  # no gap
    assert run("(CONNECT_SAME INPUT 8)", [[2, 1, 2], [0, 0, 0]]) == [[2, 1, 2], [0, 0, 0]]  # blocked
    f = [[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 2, 1], [1, 1, 1, 1]]
    assert run("(FILL_EMPTY_LINES INPUT 3)", f) == [[1, 1, 1, 1], [1, 3, 3, 1], [1, 3, 2, 1], [1, 1, 1, 1]]
    assert run("(FILL_EMPTY_LINES INPUT 3)", [[0, 0, 0], [0, 1, 0]]) == [[0, 0, 0], [0, 1, 0]]  # thinner than 3
    b = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 2], [0, 0, 2, 2]]
    assert run("(BBOX_FILL INPUT 5)", b) == [[1, 5, 0, 0], [5, 1, 0, 0], [0, 0, 5, 2], [0, 0, 2, 2]]
    assert run("(BBOX_FILL INPUT 5)", [[1, 2], [0, 3]]) == [[1, 2], [5, 3]]  # colour-agnostic components
    for bad in ("(FILL_EMPTY_LINES INPUT 0)", "(BBOX_FILL INPUT 0)"):  # colour 0 would be a no-op
        with pytest.raises(ExecError):
            run(bad, f)


_D4 = ("ROTATE90", "ROTATE180", "ROTATE270", "REFLECT_H", "REFLECT_V", "REFLECT_D1", "REFLECT_D2")


def test_extension_canonical_rules_are_sound() -> None:
    c = canonicalize
    P = Node.from_str
    assert c(P("(DOWNSCALE (UPSCALE INPUT 3) 3)")) == INPUT
    assert c(P("(DOWNSCALE_ANY (UPSCALE INPUT 2) 2)")) == INPUT
    assert c(P("(UPSCALE (UPSCALE INPUT 2) 2)")) == P("(UPSCALE INPUT 4)")
    assert c(P("(UPSCALE (UPSCALE INPUT 2) 3)")) == P("(UPSCALE (UPSCALE INPUT 2) 3)")  # 6 is not a literal
    assert c(P("(UPSCALE INPUT (COUNT_OBJECTS (SELECT_ALL INPUT)))")) == P("(UPSCALE_NC INPUT)")
    assert c(P("(FILL_EMPTY_LINES (FILL_EMPTY_LINES INPUT 3) 4)")) == P("(FILL_EMPTY_LINES INPUT 3)")
    assert c(P("(BBOX_FILL (ROTATE90 INPUT) 2)")) == P("(ROTATE90 (BBOX_FILL INPUT 2))")
    assert c(P("(ROTATE270 (UPSCALE (ROTATE90 INPUT) 2))")) == P("(UPSCALE INPUT 2)")
    assert c(P("(KRON_SELF (REFLECT_H INPUT))")) == P("(REFLECT_H (KRON_SELF INPUT))")
    # left alone: order-dependent ops, and computed arguments (moving the D4 op out would add a level)
    for src in ("(CONNECT_SAME (ROTATE90 INPUT) 3)", "(PANEL_BOOL (ROTATE90 INPUT) 0 2)",
                "(PANEL_OVERLAY (REFLECT_H INPUT) 1)", "(BBOX_FILL (ROTATE90 INPUT) (MOST_COMMON_COLOR INPUT))"):
        assert c(P(src)) == P(src), src
    rng = random.Random(31)
    grids = [small_grid(rng, 2, 6) for _ in range(6)] + [[[1, 1, 0, 0], [1, 0, 0, 3], [2, 2, 0, 3], [2, 0, 0, 0]]]
    progs = ["(DOWNSCALE (UPSCALE INPUT 3) 3)", "(DOWNSCALE_ANY (UPSCALE INPUT 2) 2)", "(UPSCALE (UPSCALE INPUT 2) 2)",
             "(UPSCALE INPUT (COUNT_OBJECTS (SELECT_ALL INPUT)))", "(FILL_EMPTY_LINES (FILL_EMPTY_LINES INPUT 3) 4)",
             "(BBOX_FILL (ROTATE90 INPUT) 2)", "(ROTATE270 (UPSCALE (ROTATE90 INPUT) 2))",
             "(KRON_SELF (REFLECT_H INPUT))"]
    checked = 0
    for src in progs:
        prog = P(src)
        for g in grids:
            try:
                want = execute(prog, g)
            except ExecError:
                continue
            assert execute(c(prog), g) == want, (src, g)
            checked += 1
    assert checked >= 30
    # the D4-equivariance the move-outside rule relies on: op(D(g)) == D(op(g)) for every D4 op
    for op_src in ("(UPSCALE {} 2)", "(DOWNSCALE {} 2)", "(DOWNSCALE_ANY {} 2)", "(UPSCALE_NC {})", "(KRON_SELF {})",
                   "(BBOX_FILL {} 4)", "(FILL_EMPTY_LINES {} 4)"):
        for d in _D4:
            for g in grids + [[[1, 2, 0, 0], [3, 3, 0, 1]]]:
                try:
                    want = execute(P(f"({d} {op_src.format('INPUT')})"), g)
                except ExecError:
                    continue
                assert execute(P(op_src.format(f"({d} INPUT)")), g) == want, (op_src, d, g)


def test_extensions_are_seen_by_grammar_enumerator_and_mutations() -> None:
    ext = set(EXTENSION_PRIMITIVES)
    assert ext <= {p.name for p in expansions(T.GRID, 1)}
    roots = {p.op for p in enumerate_programs(1, 3000)}
    assert ext <= roots, ext - roots
    for n in ext:  # typed GRID -> GRID search primitives: the GRID argument comes first
        assert REGISTRY[n].out_type is T.GRID and REGISTRY[n].arg_types[0] is T.GRID
    # an op swap moves literals into the new op's domain: REPEAT_X(g, 1) -> UPSCALE(g, 2) (factors 2..5)
    rng = random.Random(12)
    src = Node.from_str("(REPEAT_X INPUT 1)")
    swapped = {mutate(rng, src, "swap_op").to_str() for _ in range(300)}
    assert "(UPSCALE INPUT 2)" in swapped and "(DOWNSCALE INPUT 2)" in swapped, swapped
    assert all(typecheck(Node.from_str(s)) is T.GRID for s in swapped)
    # hard negatives of an extension program stay typed and distinct
    pos = Node.from_str("(MIRROR_TILE (PANEL_BOOL INPUT 1 (MOST_COMMON_COLOR INPUT)))")
    typed = hard_negatives_typed(random.Random(3), pos, k=8)
    assert len(typed) >= 6
    keys = {canonicalize(pos).to_str()}
    for _, neg in typed:
        assert typecheck(neg) is T.GRID and canonicalize(neg).to_str() not in keys
        keys.add(canonicalize(neg).to_str())
