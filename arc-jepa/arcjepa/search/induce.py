"""Per-object property -> colour / keep table induction (solver stage; docs/SOLVE_RATE_AUDIT.md addition 2).

Many ARC tasks recolour (or delete, or fill) every object by one of its properties: its size, its shape, the
number of cells it encloses, whether it is the largest, ...  Such rules are awkward for the typed DSL (they need
one branch per property value) but trivial to induce directly from the demos:

1. segment every demo input with one of :data:`SEGMENTATIONS` (single-colour 4- / 8-connected components,
   colour-agnostic 8- / 4-connected components, enclosed background regions, all background regions);
2. every segment must end up either unchanged (``"keep"``) or painted in one uniform colour in the demo output, and
   every cell outside the segments must be unchanged;
3. the map ``property value -> action`` over all segments of all demos (property one of :data:`PROPERTIES`) must be
   a function (conflict-free) and must not be all ``"keep"``;
4. every segment of every test input must have a property value seen in the demos (the table covers the test).

The first (segmentation, property) pair in declaration order that passes all four checks is the rule.  By
construction it reproduces every demo output exactly.  :func:`induce_recolor` returns it as a
:class:`RecolorRule`; :func:`induced_candidate` wraps it as a search :class:`~arcjepa.search.candidate.Candidate`
(program ``(INDUCE_RECOLOR <segmentation> <property>)``, a diagnostic label, not a DSL program; the predicted test
outputs are in ``meta["test_outputs"]``).  :func:`arcjepa.search.solver.solve_task` runs the induction inside its
deadline accounting and places the prediction as attempt 1 when the search has no exact fit and as attempt 2
otherwise.

Measured on the 150 val tasks (audit prototype ``runs/audit/brute_induce.py``): the rule fired on 10 tasks and was
correct on all 10.  A covering table can still be over-fitted on hidden tasks, which is why an exact DSL program,
when one exists, keeps attempt 1.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from arcjepa.core.types import Grid, Pair, validate_grid
from arcjepa.dsl.ast import Node

from .candidate import Candidate, score_value

__all__ = ["SEGMENTATIONS", "PROPERTIES", "KEEP", "INDUCE_OP", "RecolorRule", "segments", "object_properties",
           "induce_recolor", "induced_candidate"]

log = logging.getLogger(__name__)

#: Segmentations, in the order they are tried.
SEGMENTATIONS: Tuple[str, ...] = ("cc4", "cc8", "multi8", "multi4", "holes", "zero4")
#: Object properties, in the order they are tried (per segmentation).
PROPERTIES: Tuple[str, ...] = ("color", "size", "color_size", "shape", "bbox", "touch", "n_enclosed", "colcounts",
                               "n_minor", "is_largest", "is_smallest", "size_rank_desc", "is_rect")
#: Action of a segment whose cells are unchanged.
KEEP = "keep"
#: Op name of the diagnostic program of an induced candidate (not a registered DSL primitive).
INDUCE_OP = "INDUCE_RECOLOR"
#: Segmentations with more segments than this (per grid) are skipped (noise grids; the table would over-fit).
MAX_SEGMENTS = 200

Cell = Tuple[int, int]
_N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_N8 = _N4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))


# ============================================================================================ segmentation

def _components(grid: Grid, member: Any, diag: bool) -> List[List[Cell]]:
    """Connected components (8-connected when ``diag``, else 4-connected) of the cells whose value satisfies
    ``member(value)``, in reading order of their first cell."""
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    nbrs = _N8 if diag else _N4
    out: List[List[Cell]] = []
    for r in range(h):
        row = grid[r]
        for c in range(w):
            if seen[r][c] or not member(row[c]):
                continue
            seen[r][c] = True
            comp = [(r, c)]
            dq = deque(comp)
            while dq:
                cr, cc = dq.popleft()
                for dr, dc in nbrs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and member(grid[nr][nc]):
                        seen[nr][nc] = True
                        comp.append((nr, nc))
                        dq.append((nr, nc))
            out.append(comp)
    return out


def _same_colour_components(grid: Grid, diag: bool) -> List[List[Cell]]:
    out: List[List[Cell]] = []
    for col in sorted({v for row in grid for v in row} - {0}):
        out.extend(_components(grid, lambda v, col=col: v == col, diag))
    return out


def segments(grid: Grid, kind: str) -> List[List[Cell]]:
    """Segments of ``grid`` under segmentation ``kind`` (one of :data:`SEGMENTATIONS`), as cell lists.

    ``cc4`` / ``cc8``: single-colour 4- / 8-connected components of the non-zero cells; ``multi8`` / ``multi4``:
    colour-agnostic 8- / 4-connected components of the non-zero cells; ``zero4``: 4-connected components of the
    zero cells; ``holes``: the ``zero4`` components that do not touch the grid border.
    """
    h, w = len(grid), len(grid[0])
    if kind == "cc4":
        return _same_colour_components(grid, False)
    if kind == "cc8":
        return _same_colour_components(grid, True)
    if kind == "multi8":
        return _components(grid, lambda v: v != 0, True)
    if kind == "multi4":
        return _components(grid, lambda v: v != 0, False)
    if kind == "zero4":
        return _components(grid, lambda v: v == 0, False)
    if kind == "holes":
        return [o for o in _components(grid, lambda v: v == 0, False)
                if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in o)]
    raise KeyError(f"unknown segmentation {kind!r}")


# ============================================================================================ properties

def _n_enclosed(cells: Sequence[Cell]) -> int:
    """Number of non-object cells inside the object's bbox that are 4-enclosed (cannot reach the bbox ring)."""
    s = set(cells)
    r0 = min(r for r, _ in cells)
    r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    c1 = max(c for _, c in cells)
    seen = set()
    dq: deque = deque()
    for r in range(r0 - 1, r1 + 2):
        for c in (c0 - 1, c1 + 1):
            seen.add((r, c))
            dq.append((r, c))
    for c in range(c0, c1 + 1):
        for r in (r0 - 1, r1 + 1):
            seen.add((r, c))
            dq.append((r, c))
    while dq:
        a, b = dq.popleft()
        for dr, dc in _N4:
            x, y = a + dr, b + dc
            if r0 - 1 <= x <= r1 + 1 and c0 - 1 <= y <= c1 + 1 and (x, y) not in s and (x, y) not in seen:
                seen.add((x, y))
                dq.append((x, y))
    return (r1 - r0 + 3) * (c1 - c0 + 3) - len(seen) - len(s)


def object_properties(grid: Grid, cells: Sequence[Cell], sizes: Sequence[int]) -> Dict[str, Any]:
    """Every property of :data:`PROPERTIES` for one segment (``sizes`` = the sizes of all segments of the grid)."""
    h, w = len(grid), len(grid[0])
    counts: Dict[int, int] = {}
    for r, c in cells:
        v = grid[r][c]
        counts[v] = counts.get(v, 0) + 1
    r0 = min(r for r, _ in cells)
    r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    c1 = max(c for _, c in cells)
    n = len(cells)
    colours = tuple(sorted(counts))
    distinct_desc = sorted(set(sizes), reverse=True)
    return {
        "color": colours,
        "size": n,
        "color_size": (colours, n),
        "shape": tuple(sorted((r - r0, c - c0) for r, c in cells)),
        "bbox": (r1 - r0 + 1, c1 - c0 + 1),
        "touch": any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells),
        "n_enclosed": _n_enclosed(cells),
        "colcounts": tuple(sorted(counts.items())),
        "n_minor": n - max(counts.values()),
        "is_largest": n == max(sizes),
        "is_smallest": n == min(sizes),
        "size_rank_desc": distinct_desc.index(n),
        "is_rect": n == (r1 - r0 + 1) * (c1 - c0 + 1),
    }


def _action(gi: Grid, go: Grid, cells: Sequence[Cell]) -> Any:
    """``KEEP`` when the segment is unchanged, its uniform output colour when repainted, else ``None``."""
    if all(gi[r][c] == go[r][c] for r, c in cells):
        return KEEP
    outs = {go[r][c] for r, c in cells}
    return next(iter(outs)) if len(outs) == 1 else None


# ============================================================================================ rules

@dataclass(frozen=True)
class RecolorRule:
    """An induced rule: segment with ``segmentation``, look each segment's ``prop`` value up in ``table``."""

    segmentation: str
    prop: str
    table: Tuple[Tuple[Any, Any], ...]  # (property value, action) with action = colour 0..9 or KEEP
    _lookup: Dict[Any, Any] = field(default_factory=dict, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        self._lookup.update(dict(self.table))

    @property
    def program(self) -> Node:
        """Diagnostic label ``(INDUCE_RECOLOR <segmentation> <property>)`` (not an executable DSL program)."""
        return Node(INDUCE_OP, (self.segmentation, self.prop))

    def apply(self, grid: Grid) -> Optional[Grid]:
        """The rule's output on ``grid``; ``None`` when a segment's property value is not in the table."""
        segs = segments(grid, self.segmentation)
        if not segs or len(segs) > MAX_SEGMENTS:
            return None
        sizes = [len(o) for o in segs]
        out = [list(row) for row in grid]
        for o in segs:
            key = object_properties(grid, o, sizes)[self.prop]
            if key not in self._lookup:
                return None
            act = self._lookup[key]
            if act != KEEP:
                for r, c in o:
                    out[r][c] = act
        return out


def _grid_segments(grid: Grid, kind: str) -> Optional[List[Tuple[List[Cell], Dict[str, Any]]]]:
    segs = segments(grid, kind)
    if not segs or len(segs) > MAX_SEGMENTS:
        return None
    sizes = [len(o) for o in segs]
    return [(o, object_properties(grid, o, sizes)) for o in segs]


def _late(deadline: Optional[float]) -> bool:
    return deadline is not None and time.perf_counter() > deadline


def induce_recolor(pairs: Sequence[Pair], test_inputs: Iterable[Grid] = (), *,
                   time_budget_s: Optional[float] = None, deadline: Optional[float] = None) -> Optional[RecolorRule]:
    """Induce a property -> colour / keep table from the demo ``pairs`` (see the module docstring).

    Returns the first rule (in :data:`SEGMENTATIONS` x :data:`PROPERTIES` order) that is conflict-free on the demos,
    changes something, and covers every segment of every ``test_inputs`` grid; ``None`` when there is none, when
    a demo changes the grid shape, or when the time runs out (``time_budget_s`` and / or an absolute
    ``time.perf_counter()`` ``deadline``; the earlier one wins).
    """
    if time_budget_s is not None:
        d = time.perf_counter() + max(0.0, float(time_budget_s))
        deadline = d if deadline is None else min(deadline, d)
    demos = [p for p in pairs if validate_grid(p.input) and validate_grid(p.output)]
    tests = [g for g in test_inputs if validate_grid(g)]
    if not demos:
        return None
    for p in demos:
        if (len(p.input), len(p.input[0])) != (len(p.output), len(p.output[0])):
            return None
    if all(p.input == p.output for p in demos):
        return None  # identity task: nothing to recolour
    for kind in SEGMENTATIONS:
        if _late(deadline):
            return None
        per_demo: List[List[Tuple[List[Cell], Dict[str, Any], Any]]] = []
        ok = True
        for p in demos:
            segs = _grid_segments(p.input, kind)
            if segs is None:
                ok = False
                break
            covered = [[False] * len(p.input[0]) for _ in p.input]
            rows: List[Tuple[List[Cell], Dict[str, Any], Any]] = []
            for cells, props in segs:
                act = _action(p.input, p.output, cells)
                if act is None:
                    ok = False
                    break
                for r, c in cells:
                    covered[r][c] = True
                rows.append((cells, props, act))
            if not ok:
                break
            gi, go = p.input, p.output
            if any(gi[r][c] != go[r][c] for r in range(len(gi)) for c in range(len(gi[0])) if not covered[r][c]):
                ok = False
                break
            per_demo.append(rows)
            if _late(deadline):
                return None
        if not ok:
            continue
        test_props: List[List[Dict[str, Any]]] = []
        for g in tests:
            segs = _grid_segments(g, kind)
            if segs is None:
                ok = False
                break
            test_props.append([props for _, props in segs])
        if not ok:
            continue
        for prop in PROPERTIES:
            table: Dict[Any, Any] = {}
            conflict = False
            for rows in per_demo:
                for _, props, act in rows:
                    key = props[prop]
                    if table.setdefault(key, act) != act:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict or all(a == KEEP for a in table.values()):
                continue
            if any(props[prop] not in table for tp in test_props for props in tp):
                continue
            items = tuple(sorted(table.items(), key=lambda kv: repr(kv[0])))
            log.debug("induced %s/%s with %d entries", kind, prop, len(items))
            return RecolorRule(kind, prop, items)
    return None


def induced_candidate(rule: RecolorRule, pairs: Sequence[Pair], test_inputs: Sequence[Grid]) -> Optional[Candidate]:
    """Wrap ``rule`` as a Candidate (``source="induce"``): exact on the demos, predicted test outputs in
    ``meta["test_outputs"]`` (``None`` when the rule does not reproduce every demo or misses a test input)."""
    outs = [rule.apply(p.input) for p in pairs]
    if any(o is None or o != p.output for o, p in zip(outs, pairs)):
        return None
    test_outs = [rule.apply(g) for g in test_inputs]
    if any(o is None for o in test_outs):
        return None
    compl = 3 + len(rule.table)
    return Candidate(program=rule.program, score=score_value(0.0, 0.0, compl), demo_err=0, cell_err=0, neural=0.0,
                     complexity=compl, source="induce", loss=0.0, outputs=list(outs),
                     meta={"rule": rule, "test_outputs": test_outs, "segmentation": rule.segmentation,
                           "property": rule.prop, "table_size": len(rule.table)})
