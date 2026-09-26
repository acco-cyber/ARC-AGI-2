"""The ten object-segmentation hypotheses S1..S10 of the ARC-JEPA parser.

All hypotheses are implemented with plain Python list scans / stack-based flood
fills (no scipy, no numpy in the hot loop) and finish well under 20 ms on a
30 x 30 grid. Every hypothesis returns objects in the deterministic order
``(r0, c0, -area, primary_color, pixels)``.

Hypothesis definitions (v1 realisation of the spec's S1..S10):

``cc4``               same-colour 4-connected components of non-background cells.
``cc8``               same-colour 8-connected components.
``per_color_cc4``     one object per non-background colour = the union of that colour's
                      cc4 components (the "colour layer" hypothesis; always <= 9 objects).
``color_agnostic_cc8`` 8-connected components of *all* non-background cells (multicolour objects).
``rows``              maximal horizontal runs of equal non-background colour.
``cols``              maximal vertical runs of equal non-background colour.
``rect_regions``      greedy decomposition into maximal solid single-colour rectangles
                      (scan order anchor, largest area rectangle at each anchor).
``frames``            hollow single-colour rectangular rings (bbox >= 3 x 3, ring == cc4
                      component) plus, per frame, one colour-agnostic object with the
                      non-background content strictly inside the ring.
``repeat_blocks``     blocks delimited by full uniform non-background separator rows/cols;
                      otherwise tiles of the minimal exact 2-D period; ``[]`` if no structure.
``symmetry``          cc4 components merged with their exact mirror images under the grid's
                      global left-right / top-bottom / 180-degree (and diagonal, if square)
                      symmetries (union-find over components).
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple

from arcjepa.core.types import Grid

from .objects import Object

__all__ = ["HYPOTHESES", "segment", "all_hypotheses", "order_objects"]

logger = logging.getLogger(__name__)

HYPOTHESES: Tuple[str, ...] = (
    "cc4",
    "cc8",
    "per_color_cc4",
    "color_agnostic_cc8",
    "rows",
    "cols",
    "rect_regions",
    "frames",
    "repeat_blocks",
    "symmetry",
)

_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_N8 = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))

CellList = List[Tuple[int, int, int]]


def _dims(grid: Grid) -> Tuple[int, int]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def order_objects(objs: Iterable[Object]) -> List[Object]:
    """Sort objects by ``(r0, c0, -area, primary_color, pixels)`` (deterministic)."""
    return sorted(objs, key=lambda o: (o.bbox[0], o.bbox[1], -o.area, o.primary_color, o.iter_pixels()))


def _finalize(groups: Iterable[CellList]) -> List[Object]:
    return order_objects(Object.from_cells(g) for g in groups if g)


# ---------------------------------------------------------------------------- components
def _components(grid: Grid, background: int, nbrs: Sequence[Tuple[int, int]], color_agnostic: bool) -> List[CellList]:
    """Stack-based flood fill returning components as lists of ``(r, c, colour)``."""
    h, w = _dims(grid)
    seen = [[False] * w for _ in range(h)]
    comps: List[CellList] = []
    for r in range(h):
        row = grid[r]
        seen_r = seen[r]
        for c in range(w):
            if seen_r[c] or row[c] == background:
                continue
            color = row[c]
            seen_r[c] = True
            stack = [(r, c)]
            cells: CellList = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc, grid[cr][cc]))
                for dr, dc in nbrs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc]:
                        v = grid[nr][nc]
                        if v != background and (color_agnostic or v == color):
                            seen[nr][nc] = True
                            stack.append((nr, nc))
            comps.append(cells)
    return comps


def _seg_cc4(grid: Grid, background: int) -> List[Object]:
    return _finalize(_components(grid, background, _N4, False))


def _seg_cc8(grid: Grid, background: int) -> List[Object]:
    return _finalize(_components(grid, background, _N8, False))


def _seg_color_agnostic_cc8(grid: Grid, background: int) -> List[Object]:
    return _finalize(_components(grid, background, _N8, True))


def _seg_per_color(grid: Grid, background: int) -> List[Object]:
    layers: Dict[int, CellList] = {}
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v != background:
                layers.setdefault(v, []).append((r, c, v))
    return _finalize(layers[k] for k in sorted(layers))


# ---------------------------------------------------------------------------- runs
def _seg_rows(grid: Grid, background: int) -> List[Object]:
    h, w = _dims(grid)
    groups: List[CellList] = []
    for r in range(h):
        row = grid[r]
        c = 0
        while c < w:
            v = row[c]
            if v == background:
                c += 1
                continue
            start = c
            while c < w and row[c] == v:
                c += 1
            groups.append([(r, cc, v) for cc in range(start, c)])
    return _finalize(groups)


def _seg_cols(grid: Grid, background: int) -> List[Object]:
    h, w = _dims(grid)
    groups: List[CellList] = []
    for c in range(w):
        r = 0
        while r < h:
            v = grid[r][c]
            if v == background:
                r += 1
                continue
            start = r
            while r < h and grid[r][c] == v:
                r += 1
            groups.append([(rr, c, v) for rr in range(start, r)])
    return _finalize(groups)


# ---------------------------------------------------------------------------- rectangles
def _seg_rect_regions(grid: Grid, background: int) -> List[Object]:
    h, w = _dims(grid)
    taken = [[False] * w for _ in range(h)]
    groups: List[CellList] = []
    for r in range(h):
        row = grid[r]
        for c in range(w):
            if taken[r][c] or row[c] == background:
                continue
            color = row[c]
            # widest run at the anchor row
            wmax = 0
            while c + wmax < w and row[c + wmax] == color and not taken[r][c + wmax]:
                wmax += 1
            best_w, best_h, best_area = wmax, 1, wmax
            cur_w = wmax
            for rr in range(r + 1, h):
                row2 = grid[rr]
                taken2 = taken[rr]
                ww = 0
                while ww < cur_w and row2[c + ww] == color and not taken2[c + ww]:
                    ww += 1
                if ww == 0:
                    break
                if ww < cur_w:
                    cur_w = ww
                area = cur_w * (rr - r + 1)
                if area > best_area:
                    best_w, best_h, best_area = cur_w, rr - r + 1, area
            cells: CellList = []
            for rr in range(r, r + best_h):
                taken_rr = taken[rr]
                for cc in range(c, c + best_w):
                    taken_rr[cc] = True
                    cells.append((rr, cc, color))
            groups.append(cells)
    return _finalize(groups)


# ---------------------------------------------------------------------------- frames
def _ring_cells(r0: int, c0: int, r1: int, c1: int) -> Set[Tuple[int, int]]:
    ring: Set[Tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        ring.add((r0, c))
        ring.add((r1, c))
    for r in range(r0, r1 + 1):
        ring.add((r, c0))
        ring.add((r, c1))
    return ring


def _seg_frames(grid: Grid, background: int) -> List[Object]:
    groups: List[CellList] = []
    for comp in _components(grid, background, _N4, False):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        ring = _ring_cells(r0, c0, r1, c1)
        if len(comp) != len(ring) or any((p[0], p[1]) not in ring for p in comp):
            continue
        groups.append(comp)
        interior: CellList = []
        for r in range(r0 + 1, r1):
            row = grid[r]
            for c in range(c0 + 1, c1):
                v = row[c]
                if v != background:
                    interior.append((r, c, v))
        if interior:
            groups.append(interior)
    return _finalize(groups)


# ---------------------------------------------------------------------------- repeat blocks
def _uniform_lines(grid: Grid, background: int) -> Tuple[List[int], List[int]]:
    h, w = _dims(grid)
    sep_rows = [r for r in range(h) if grid[r][0] != background and all(v == grid[r][0] for v in grid[r])]
    sep_cols = [c for c in range(w) if grid[0][c] != background and all(grid[r][c] == grid[0][c] for r in range(h))]
    return sep_rows, sep_cols


def _intervals(n: int, seps: List[int]) -> List[Tuple[int, int]]:
    """Half-open intervals of ``range(n)`` between separator indices."""
    out: List[Tuple[int, int]] = []
    start = 0
    for s in seps:
        if s > start:
            out.append((start, s))
        start = s + 1
    if start < n:
        out.append((start, n))
    return out


def _min_period_rows(grid: Grid) -> int:
    """Smallest row period ``p`` with at least two full repetitions (``2p <= H``), else ``H``."""
    h = len(grid)
    for p in range(1, h // 2 + 1):
        if all(grid[r] == grid[r - p] for r in range(p, h)):
            return p
    return h


def _min_period_cols(grid: Grid) -> int:
    """Smallest column period ``p`` with at least two full repetitions (``2p <= W``), else ``W``."""
    h, w = _dims(grid)
    for p in range(1, w // 2 + 1):
        ok = True
        for row in grid:
            for c in range(p, w):
                if row[c] != row[c - p]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return p
    return w


def _blocks_from_intervals(grid: Grid, background: int, rows: List[Tuple[int, int]], cols: List[Tuple[int, int]]) -> List[CellList]:
    groups: List[CellList] = []
    for ra, rb in rows:
        for ca, cb in cols:
            cells: CellList = []
            for r in range(ra, rb):
                row = grid[r]
                for c in range(ca, cb):
                    v = row[c]
                    if v != background:
                        cells.append((r, c, v))
            if cells:
                groups.append(cells)
    return groups


def _seg_repeat_blocks(grid: Grid, background: int) -> List[Object]:
    h, w = _dims(grid)
    sep_rows, sep_cols = _uniform_lines(grid, background)
    if sep_rows or sep_cols:
        rows = _intervals(h, sep_rows)
        cols = _intervals(w, sep_cols)
        if len(rows) * len(cols) >= 2:
            return _finalize(_blocks_from_intervals(grid, background, rows, cols))
    ph, pw = _min_period_rows(grid), _min_period_cols(grid)
    if ph == h and pw == w:
        return []
    rows = [(i, min(i + ph, h)) for i in range(0, h, ph)]
    cols = [(j, min(j + pw, w)) for j in range(0, w, pw)]
    return _finalize(_blocks_from_intervals(grid, background, rows, cols))


# ---------------------------------------------------------------------------- symmetry
def _seg_symmetry(grid: Grid, background: int) -> List[Object]:
    h, w = _dims(grid)
    comps = _components(grid, background, _N4, False)
    n = len(comps)
    if n <= 1:
        return _finalize(comps)
    # label grid: cell -> component index (components are disjoint)
    label = [[-1] * w for _ in range(h)]
    for i, comp in enumerate(comps):
        for r, c, _ in comp:
            label[r][c] = i
    H1, W1 = h - 1, w - 1
    transforms: List[Callable[[int, int], Tuple[int, int]]] = [
        lambda r, c: (r, W1 - c),  # left-right mirror
        lambda r, c: (H1 - r, c),  # top-bottom mirror
        lambda r, c: (H1 - r, W1 - c),  # 180-degree rotation
    ]
    if h == w:
        transforms.append(lambda r, c: (c, r))  # transpose
        transforms.append(lambda r, c: (W1 - c, H1 - r))  # anti-transpose
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, comp in enumerate(comps):
        size = len(comp)
        r, c, col = comp[0]
        for f in transforms:
            rr, cc = f(r, c)
            j = label[rr][cc]
            # cheap pre-checks on the first cell: partner exists, is different, same size and colour
            if j < 0 or j == i or len(comps[j]) != size or grid[rr][cc] != col:
                continue
            if size > 1:
                ok = True
                for r2, c2, _ in comp:
                    rr2, cc2 = f(r2, c2)
                    if label[rr2][cc2] != j:
                        ok = False
                        break
                if not ok:
                    continue
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)
    merged: Dict[int, CellList] = {}
    for i, comp in enumerate(comps):
        merged.setdefault(find(i), []).extend(comp)
    return _finalize(merged[k] for k in sorted(merged))


# ---------------------------------------------------------------------------- public API
_SEGMENTERS: Dict[str, Callable[[Grid, int], List[Object]]] = {
    "cc4": _seg_cc4,
    "cc8": _seg_cc8,
    "per_color_cc4": _seg_per_color,
    "color_agnostic_cc8": _seg_color_agnostic_cc8,
    "rows": _seg_rows,
    "cols": _seg_cols,
    "rect_regions": _seg_rect_regions,
    "frames": _seg_frames,
    "repeat_blocks": _seg_repeat_blocks,
    "symmetry": _seg_symmetry,
}


def segment(grid: Grid, hypothesis: str, background: int = 0) -> List[Object]:
    """Segment ``grid`` under ``hypothesis`` (one of :data:`HYPOTHESES`).

    Returns objects sorted by ``(r0, c0, -area)``; an empty list for an all-background
    grid or when a structural hypothesis (frames, repeat_blocks) finds nothing.
    Raises ``ValueError`` for an unknown hypothesis.
    """
    fn = _SEGMENTERS.get(hypothesis)
    if fn is None:
        raise ValueError(f"unknown segmentation hypothesis {hypothesis!r}; expected one of {HYPOTHESES}")
    if not grid or not grid[0]:
        return []
    return fn(grid, background)


def all_hypotheses(grid: Grid, background: int = 0) -> Dict[str, List[Object]]:
    """Run every hypothesis; keys follow :data:`HYPOTHESES` order."""
    return {name: segment(grid, name, background) for name in HYPOTHESES}
