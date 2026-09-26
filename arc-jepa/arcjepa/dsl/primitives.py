"""The 72 typed DSL primitives (FROZEN_SPEC.md "Typed DSL") plus four structural helpers.

Every primitive function is pure and total over its typed domain or raises :class:`ExecError`.  Value
representations (INTERFACES.md §1): GRID = ``List[List[int]]``; OBJECT = ``Object``; OBJECT_SET =
``List[Object]``; MASK = ``List[List[bool]]``; COLOR / INTEGER = ``int``; POSITION = ``(dr, dc)`` tuple or a
named anchor string; BOOLEAN = ``bool``; RELATION = relation name string; PROGRAM = an AST ``Node`` (lambda body
over the ``OBJ`` leaf, OBJECT -> OBJECT).

Primitives flagged ``needs_ctx`` receive an :class:`ExecContext` (the task input grid and its shape) as their
first argument because MASK values and canvas anchors need a canvas size.  ``higher_order`` primitives receive
an evaluator callable as their last argument.  ``lazy`` primitives (IF) have their arguments evaluated by the
interpreter on demand.

Structural helpers (outside the 72, flagged ``structural=True``): RENDER, RENDER_OBJ, RENDER_BLANK, CROP.
"""
from __future__ import annotations

import dataclasses
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from arcjepa.core.types import Grid, MAX_SIDE
from arcjepa.dsl.ast import Node
from arcjepa.dsl.types import (BOOLEANS, COLORS, LEAF_OBJ, Object, POSITION_ANCHORS, POSITION_OFFSETS, RELATIONS,
                               T, UNIT_DIRECTIONS)

__all__ = [
    "ExecError", "ExecContext", "Primitive", "REGISTRY", "SPEC_PRIMITIVES", "STRUCTURAL_PRIMITIVES", "CATEGORIES",
    "by_out_type", "by_category", "make_object", "components", "paint_object", "sort_objects",
    "MAX_OBJECTS",
]

log = logging.getLogger(__name__)

Mask = List[List[bool]]
MAX_OBJECTS = 256


class ExecError(Exception):
    """Raised by primitives / the interpreter for any runtime failure (never returns an invalid value)."""


@dataclass(frozen=True)
class ExecContext:
    """Execution context: the task input grid and its shape (the canvas for masks and anchors)."""

    input: Grid
    h: int
    w: int

    @staticmethod
    def of(grid: Grid) -> "ExecContext":
        return ExecContext(grid, len(grid), len(grid[0]))


@dataclass
class Primitive:
    """Registry entry for one DSL operator."""

    name: str
    arg_types: Tuple[T, ...]
    out_type: T
    fn: Callable[..., Any]
    category: str
    literal_args: Dict[int, Sequence] = field(default_factory=dict)
    structural: bool = False
    needs_ctx: bool = False
    higher_order: bool = False
    lazy: bool = False

    @property
    def arity(self) -> int:
        return len(self.arg_types)

    def signature(self) -> Tuple[Tuple[T, ...], T]:
        return (self.arg_types, self.out_type)


# ======================================================================================= helpers

def _check_dims(h: int, w: int) -> None:
    if h < 1 or w < 1 or h > MAX_SIDE or w > MAX_SIDE:
        raise ExecError(f"grid shape {h}x{w} outside 1..{MAX_SIDE}")


def _copy(g: Grid) -> Grid:
    return [list(r) for r in g]


def _blank(h: int, w: int, fill: int = 0) -> Grid:
    return [[fill] * w for _ in range(h)]


def _shape(g: Grid) -> Tuple[int, int]:
    return len(g), len(g[0])


#: True when the Object class in use stores per-cell colours (the parser Object's optional ``pixels`` field).
_HAS_PIXELS: bool = "pixels" in {f.name for f in dataclasses.fields(Object)}


def make_object(cells: Iterable[Tuple[int, int]], color: int) -> Object:
    """Construct a single-colour ``Object`` from cells (keyword construction, compatible with the parser Object)."""
    fs = frozenset(cells)
    if not fs:
        raise ExecError("empty object")
    r0 = min(r for r, _ in fs)
    r1 = max(r for r, _ in fs)
    c0 = min(c for _, c in fs)
    c1 = max(c for _, c in fs)
    return Object(cells=fs, color_hist=((int(color), len(fs)),), primary_color=int(color), bbox=(r0, c0, r1, c1))


def pixel_map(o: Object) -> Dict[Tuple[int, int], int]:
    """``{(r, c): colour}`` of an object (per-cell colours when the Object carries them, else primary colour)."""
    px = getattr(o, "pixels", ())
    if px:
        return {(r, c): col for r, c, col in px}
    col = o.primary_color
    return {rc: col for rc in o.cells}


def make_colored_object(colored: Dict[Tuple[int, int], int]) -> Object:
    """Construct an ``Object`` from a ``{(r, c): colour}`` map (multi-colour aware when the Object supports it).

    The colour histogram is sorted by colour and the primary colour is the most frequent one (ties: smallest
    colour), matching the parser's convention.
    """
    if not colored:
        raise ExecError("empty object")
    counts: Dict[int, int] = {}
    for col in colored.values():
        counts[col] = counts.get(col, 0) + 1
    if len(counts) == 1:
        return make_object(colored.keys(), next(iter(counts)))
    primary = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    cells = frozenset(colored)
    r0 = min(r for r, _ in cells)
    r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    c1 = max(c for _, c in cells)
    kw: Dict[str, Any] = dict(cells=cells, color_hist=tuple(sorted(counts.items())), primary_color=int(primary),
                              bbox=(r0, c0, r1, c1))
    if _HAS_PIXELS:
        kw["pixels"] = tuple(sorted((r, c, int(col)) for (r, c), col in colored.items()))
    return Object(**kw)


def _bbox_center(o: Object) -> Tuple[float, float]:
    r0, c0, r1, c1 = o.bbox
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def _centroid(o: Object) -> Tuple[float, float]:
    n = len(o.cells)
    return (sum(r for r, _ in o.cells) / n, sum(c for _, c in o.cells) / n)


def _local_cells(o: Object) -> FrozenSet[Tuple[int, int]]:
    r0, c0, _, _ = o.bbox
    return frozenset((r - r0, c - c0) for r, c in o.cells)


def sort_objects(objs: Iterable[Object]) -> List[Object]:
    """Deterministic reading order: by (r0, c0, -area, colour)."""
    return sorted(objs, key=lambda o: (o.bbox[0], o.bbox[1], -len(o.cells), o.primary_color))


def _check_objs(objs: List[Object]) -> List[Object]:
    if len(objs) > MAX_OBJECTS:
        raise ExecError(f"too many objects ({len(objs)} > {MAX_OBJECTS})")
    return objs


def components(grid: Grid, conn: int = 4) -> List[Object]:
    """Per-colour connected components of the non-background cells (4- or 8-connectivity), reading order."""
    h, w = _shape(grid)
    seen = [[False] * w for _ in range(h)]
    if conn == 4:
        nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    out: List[Object] = []
    for r in range(h):
        row = grid[r]
        for c in range(w):
            col = row[c]
            if col == 0 or seen[r][c]:
                continue
            seen[r][c] = True
            comp = [(r, c)]
            dq = deque(comp)
            while dq:
                cr, cc = dq.popleft()
                for dr, dc in nbrs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == col:
                        seen[nr][nc] = True
                        comp.append((nr, nc))
                        dq.append((nr, nc))
            out.append(make_object(comp, col))
            if len(out) > MAX_OBJECTS:
                raise ExecError("too many components")
    return sort_objects(out)


def _paint_into(out: Grid, obj: Object, color: Optional[int] = None) -> None:
    """Paint ``obj`` into ``out`` in place (per-cell colours unless ``color`` is given; clipped to the canvas)."""
    h, w = _shape(out)
    if color is None and getattr(obj, "pixels", ()):
        for r, c, col in obj.pixels:  # type: ignore[attr-defined]
            if 0 <= r < h and 0 <= c < w:
                out[r][c] = col
        return
    col = obj.primary_color if color is None else int(color)
    for r, c in obj.cells:
        if 0 <= r < h and 0 <= c < w:
            out[r][c] = col


def paint_object(grid: Grid, obj: Object, color: Optional[int] = None) -> Grid:
    """Copy of ``grid`` with the object's cells painted (clipped to the canvas)."""
    out = _copy(grid)
    _paint_into(out, obj, color)
    return out


def _translate(o: Object, dr: int, dc: int) -> Object:
    r0, c0, r1, c1 = o.bbox
    kw: Dict[str, Any] = dict(cells=frozenset((r + dr, c + dc) for r, c in o.cells), color_hist=o.color_hist,
                              primary_color=o.primary_color, bbox=(r0 + dr, c0 + dc, r1 + dr, c1 + dc))
    px = getattr(o, "pixels", ())
    if px:
        kw["pixels"] = tuple((r + dr, c + dc, col) for r, c, col in px)
    return Object(**kw)


def _recolor(o: Object, color: int) -> Object:
    return Object(cells=o.cells, color_hist=((int(color), len(o.cells)),), primary_color=int(color), bbox=o.bbox)


def _as_offset(pos: Any) -> Tuple[int, int]:
    if isinstance(pos, tuple) and len(pos) == 2:
        return int(pos[0]), int(pos[1])
    raise ExecError(f"position offset expected, got {pos!r}")


def _mask(ctx: ExecContext, cells: Iterable[Tuple[int, int]]) -> Mask:
    m = [[False] * ctx.w for _ in range(ctx.h)]
    for r, c in cells:
        if 0 <= r < ctx.h and 0 <= c < ctx.w:
            m[r][c] = True
    return m


def _hole_cells(o: Object) -> Set[Tuple[int, int]]:
    """Background cells inside the bbox that are 4-enclosed by the object (do not reach the bbox border)."""
    r0, c0, r1, c1 = o.bbox
    cells = o.cells
    seen: Set[Tuple[int, int]] = set()
    holes: Set[Tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if (r, c) in cells or (r, c) in seen:
                continue
            comp = [(r, c)]
            seen.add((r, c))
            touches = False
            i = 0
            while i < len(comp):
                cr, cc = comp[i]
                i += 1
                if cr in (r0, r1) or cc in (c0, c1):
                    touches = True
                for nr, nc in ((cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)):
                    if r0 <= nr <= r1 and c0 <= nc <= c1 and (nr, nc) not in cells and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        comp.append((nr, nc))
            if not touches:
                holes.update(comp)
    return holes


def _bbox_gap(a: Object, b: Object) -> int:
    ar0, ac0, ar1, ac1 = a.bbox
    br0, bc0, br1, bc1 = b.bbox
    dr = max(0, br0 - ar1, ar0 - br1)
    dc = max(0, bc0 - ac1, ac0 - bc1)
    return dr + dc


def _dist2(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


def _disjoint(a: Object, b: Object) -> bool:
    small, big = (a.cells, b.cells) if len(a.cells) <= len(b.cells) else (b.cells, a.cells)
    return not any(c in big for c in small)


def _rank_color(objs: List[Object], largest: bool) -> int:
    if not objs:
        raise ExecError("empty object set")
    key = (lambda o: -len(o.cells)) if largest else (lambda o: len(o.cells))
    return min(sort_objects(objs), key=key).primary_color


def _color_counts(g: Grid) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for row in g:
        for v in row:
            if v != 0:
                counts[v] = counts.get(v, 0) + 1
    return counts


def _tile(g: Grid, n: int, m: int) -> Grid:
    if n < 1 or m < 1:
        raise ExecError("tile factor must be >= 1")
    h, w = _shape(g)
    _check_dims(h * n, w * m)
    rows = [list(row) * m for row in g]
    return [list(r) for _ in range(n) for r in rows]


# ======================================================================================= grid geometry

def _rot90(g: Grid) -> Grid:
    return [list(r) for r in zip(*g[::-1])]


def _rot180(g: Grid) -> Grid:
    return [r[::-1] for r in g[::-1]]


def _rot270(g: Grid) -> Grid:
    return [list(r) for r in zip(*g)][::-1]


def _reflect_h(g: Grid) -> Grid:
    return [r[::-1] for r in g]


def _reflect_v(g: Grid) -> Grid:
    return [list(r) for r in g[::-1]]


def _transpose(g: Grid) -> Grid:
    return [list(r) for r in zip(*g)]


def _reflect_d2(g: Grid) -> Grid:
    return _rot180(_transpose(g))


def _shift(g: Grid, pos: Any) -> Grid:
    h, w = _shape(g)
    if isinstance(pos, str):
        cells = [(r, c) for r in range(h) for c in range(w) if g[r][c] != 0]
        if not cells:
            return _copy(g)
        r0 = min(r for r, _ in cells)
        r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        c1 = max(c for _, c in cells)
        dr, dc = _anchor_delta(pos, h, w, r0, c0, r1, c1)
    else:
        dr, dc = _as_offset(pos)
    out = _blank(h, w)
    for r in range(h):
        nr = r + dr
        if not 0 <= nr < h:
            continue
        row = g[r]
        orow = out[nr]
        for c in range(w):
            nc = c + dc
            if 0 <= nc < w:
                orow[nc] = row[c]
    return out


def _anchor_delta(anchor: str, h: int, w: int, r0: int, c0: int, r1: int, c1: int) -> Tuple[int, int]:
    if anchor == "top":
        return (-r0, 0)
    if anchor == "bottom":
        return (h - 1 - r1, 0)
    if anchor == "left":
        return (0, -c0)
    if anchor == "right":
        return (0, w - 1 - c1)
    if anchor == "center":
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        return ((h - bh) // 2 - r0, (w - bw) // 2 - c0)
    raise ExecError(f"unknown anchor {anchor!r}")


def _align(ctx: ExecContext, o: Object, pos: Any) -> Object:
    r0, c0, r1, c1 = o.bbox
    if isinstance(pos, str):
        dr, dc = _anchor_delta(pos, ctx.h, ctx.w, r0, c0, r1, c1)
    else:
        tr, tc = _as_offset(pos)
        dr, dc = tr - r0, tc - c0
    return _translate(o, dr, dc)


# ======================================================================================= selection

def _select_all(g: Grid) -> List[Object]:
    by_color: Dict[int, List[Tuple[int, int]]] = {}
    for r, row in enumerate(g):
        for c, v in enumerate(row):
            if v != 0:
                by_color.setdefault(v, []).append((r, c))
    return sort_objects(make_object(cells, col) for col, cells in by_color.items())


def _select_color(objs: List[Object], color: int) -> List[Object]:
    return [o for o in objs if o.primary_color == color]


def _select_nonzero(g: Grid) -> Mask:
    return [[v != 0 for v in row] for row in g]


def _select_largest(objs: List[Object]) -> Object:
    if not objs:
        raise ExecError("SELECT_LARGEST on empty set")
    return min(sort_objects(objs), key=lambda o: -len(o.cells))


def _select_smallest(objs: List[Object]) -> Object:
    if not objs:
        raise ExecError("SELECT_SMALLEST on empty set")
    return min(sort_objects(objs), key=lambda o: len(o.cells))


def _select_unique(objs: List[Object]) -> Object:
    if not objs:
        raise ExecError("SELECT_UNIQUE on empty set")
    objs = sort_objects(objs)
    colors: Dict[int, int] = {}
    for o in objs:
        colors[o.primary_color] = colors.get(o.primary_color, 0) + 1
    cand = [o for o in objs if colors[o.primary_color] == 1]
    if len(cand) == 1:
        return cand[0]
    shapes: Dict[FrozenSet[Tuple[int, int]], int] = {}
    locs = [_local_cells(o) for o in objs]
    for s in locs:
        shapes[s] = shapes.get(s, 0) + 1
    cand = [o for o, s in zip(objs, locs) if shapes[s] == 1]
    if len(cand) == 1:
        return cand[0]
    raise ExecError("no unique object")


def _select_border(ctx: ExecContext, objs: List[Object]) -> List[Object]:
    out = []
    for o in objs:
        r0, c0, r1, c1 = o.bbox
        if r0 <= 0 or c0 <= 0 or r1 >= ctx.h - 1 or c1 >= ctx.w - 1:
            out.append(o)
    return out


def _select_center(ctx: ExecContext, objs: List[Object]) -> Object:
    if not objs:
        raise ExecError("SELECT_CENTER on empty set")
    centre = ((ctx.h - 1) / 2.0, (ctx.w - 1) / 2.0)
    return min(sort_objects(objs), key=lambda o: _dist2(_bbox_center(o), centre))


# ======================================================================================= analysis

def _get_bbox(ctx: ExecContext, o: Object) -> Mask:
    r0, c0, r1, c1 = o.bbox
    return _mask(ctx, ((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)))


def _get_centroid(o: Object) -> Tuple[int, int]:
    cr, cc = _centroid(o)
    return (int(round(cr)), int(round(cc)))


def _get_area(o: Object) -> int:
    return len(o.cells)


def _get_perimeter(o: Object) -> int:
    cells = o.cells
    return sum(1 for (r, c) in cells for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (r + dr, c + dc) not in cells)


def _get_holes(ctx: ExecContext, o: Object) -> Mask:
    return _mask(ctx, _hole_cells(o))


def _get_symmetry(o: Object) -> bool:
    loc = _local_cells(o)
    h = o.bbox[2] - o.bbox[0] + 1
    w = o.bbox[3] - o.bbox[1] + 1
    sym_h = all((r, w - 1 - c) in loc for r, c in loc)
    sym_v = all((h - 1 - r, c) in loc for r, c in loc)
    return sym_h or sym_v


# ======================================================================================= relations

def _left_of(a: Object, b: Object) -> bool:
    return a.bbox[3] < b.bbox[1]


def _right_of(a: Object, b: Object) -> bool:
    return a.bbox[1] > b.bbox[3]


def _above(a: Object, b: Object) -> bool:
    return a.bbox[2] < b.bbox[0]


def _below(a: Object, b: Object) -> bool:
    return a.bbox[0] > b.bbox[2]


def _touching(a: Object, b: Object) -> bool:
    if _bbox_gap(a, b) > 1 or not _disjoint(a, b):
        return False
    small, big = (a.cells, b.cells) if len(a.cells) <= len(b.cells) else (b.cells, a.cells)
    for r, c in small:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and (r + dr, c + dc) in big:
                    return True
    return False


def _overlapping(a: Object, b: Object) -> bool:
    return not _disjoint(a, b)


def _inside(a: Object, b: Object) -> bool:
    ar0, ac0, ar1, ac1 = a.bbox
    br0, bc0, br1, bc1 = b.bbox
    return ar0 > br0 and ac0 > bc0 and ar1 < br1 and ac1 < bc1 and _disjoint(a, b)


def _contains(a: Object, b: Object) -> bool:
    return _inside(b, a)


def _same_shape(a: Object, b: Object) -> bool:
    return _local_cells(a) == _local_cells(b)


_RELATION_FNS: Dict[str, Callable[[Object, Object], bool]] = {
    "LEFT_OF": _left_of, "RIGHT_OF": _right_of, "ABOVE": _above, "BELOW": _below, "TOUCHING": _touching,
    "OVERLAPPING": _overlapping, "INSIDE": _inside, "CONTAINS": _contains,
    "SAME_COLOR": lambda a, b: a.primary_color == b.primary_color,
    "SAME_SHAPE": _same_shape,
    "SAME_SIZE": lambda a, b: len(a.cells) == len(b.cells),
    "LARGER": lambda a, b: len(a.cells) > len(b.cells),
    "SMALLER": lambda a, b: len(a.cells) < len(b.cells),
}


def _others(objs: List[Object], o: Object) -> List[Object]:
    return [x for x in sort_objects(objs) if x.cells != o.cells]


def _nearest(objs: List[Object], o: Object) -> Object:
    cand = _others(objs, o)
    if not cand:
        raise ExecError("NEAREST: no other object")
    co = _centroid(o)
    return min(cand, key=lambda x: (_bbox_gap(x, o), _dist2(_centroid(x), co)))


def _farthest(objs: List[Object], o: Object) -> Object:
    cand = _others(objs, o)
    if not cand:
        raise ExecError("FARTHEST: no other object")
    co = _centroid(o)
    return max(cand, key=lambda x: (_bbox_gap(x, o), _dist2(_centroid(x), co)))


# ======================================================================================= object manipulation

def _copy_obj(g: Grid, o: Object, pos: Any) -> Grid:
    dr, dc = _as_offset(pos)
    return paint_object(g, _translate(o, dr, dc))


def _move(o: Object, pos: Any) -> Object:
    dr, dc = _as_offset(pos)
    return _translate(o, dr, dc)


def _delete(g: Grid, o: Object) -> Grid:
    return paint_object(g, o, 0)


def _duplicate(o: Object, pos: Any) -> List[Object]:
    dr, dc = _as_offset(pos)
    return [o, _translate(o, dr, dc)]


def _merge(objs: List[Object]) -> Object:
    if not objs:
        raise ExecError("MERGE on empty set")
    colored: Dict[Tuple[int, int], int] = {}
    for o in sort_objects(objs):  # earlier objects (reading order) win on overlapping cells
        for cell, col in pixel_map(o).items():
            colored.setdefault(cell, col)
    return make_colored_object(colored)


def _split(o: Object) -> List[Object]:
    cells = set(o.cells)
    colors = pixel_map(o)
    out: List[Object] = []
    while cells:
        start = min(cells)
        comp = [start]
        cells.discard(start)
        i = 0
        while i < len(comp):
            r, c = comp[i]
            i += 1
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if n in cells:
                    cells.discard(n)
                    comp.append(n)
        out.append(make_colored_object({rc: colors[rc] for rc in comp}))
    return _check_objs(sort_objects(out))


def _extend(g: Grid, o: Object, pos: Any) -> Grid:
    dr, dc = _as_offset(pos)
    dr = (dr > 0) - (dr < 0)
    dc = (dc > 0) - (dc < 0)
    out = _copy(g)
    if dr == 0 and dc == 0:
        return out
    h, w = _shape(g)
    col = o.primary_color
    for r, c in o.cells:
        nr, nc = r + dr, c + dc
        while 0 <= nr < h and 0 <= nc < w:
            if out[nr][nc] == 0:
                out[nr][nc] = col
            nr += dr
            nc += dc
    return out


def _shrink(o: Object) -> Object:
    """Morphological erosion (4-neighbourhood); raises when nothing survives."""
    cells = o.cells
    kept = [(r, c) for r, c in cells
            if (r + 1, c) in cells and (r - 1, c) in cells and (r, c + 1) in cells and (r, c - 1) in cells]
    if not kept:
        raise ExecError("SHRINK emptied the object")
    colors = pixel_map(o)
    return make_colored_object({rc: colors[rc] for rc in kept})


def _grow(o: Object) -> Object:
    """Morphological dilation (4-neighbourhood); new cells take the primary colour."""
    colored = pixel_map(o)
    new: Dict[Tuple[int, int], int] = dict(colored)
    col = o.primary_color
    for r, c in colored:
        for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if n not in new:
                new[n] = col
    if len(new) > MAX_SIDE * MAX_SIDE * 2:
        raise ExecError("GROW exceeded cell budget")
    return make_colored_object(new)


def _fill(g: Grid, m: Mask, color: int) -> Grid:
    out = _copy(g)
    h, w = _shape(g)
    mh = len(m)
    for r in range(min(h, mh)):
        mrow = m[r]
        orow = out[r]
        for c in range(min(w, len(mrow))):
            if mrow[c]:
                orow[c] = color
    return out


def _outline(g: Grid, o: Object, color: int) -> Grid:
    out = _copy(g)
    h, w = _shape(g)
    cells = o.cells
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if (dr or dc) and 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    out[nr][nc] = color
    return out


def _frame(g: Grid, color: int) -> Grid:
    out = _copy(g)
    h, w = _shape(g)
    for c in range(w):
        out[0][c] = color
        out[h - 1][c] = color
    for r in range(h):
        out[r][0] = color
        out[r][w - 1] = color
    return out


# ======================================================================================= colour

def _swap_colors(g: Grid, a: int, b: int) -> Grid:
    return [[b if v == a else (a if v == b else v) for v in row] for row in g]


def _map_color(g: Grid, a: int, b: int) -> Grid:
    return [[b if v == a else v for v in row] for row in g]


def _most_common_color(g: Grid) -> int:
    counts = _color_counts(g)
    if not counts:
        return 0
    return min(counts, key=lambda k: (-counts[k], k))


def _least_common_color(g: Grid) -> int:
    counts = _color_counts(g)
    if not counts:
        return 0
    return min(counts, key=lambda k: (counts[k], k))


def _replace_background(g: Grid, color: int) -> Grid:
    return [[color if v == 0 else v for v in row] for row in g]


def _color_object(g: Grid, o: Object, color: int) -> Grid:
    return paint_object(g, o, color)


def _color_by_position(g: Grid, pos: Any) -> int:
    h, w = _shape(g)
    if isinstance(pos, str):
        anchors = {"center": (h // 2, w // 2), "top": (0, w // 2), "bottom": (h - 1, w // 2),
                   "left": (h // 2, 0), "right": (h // 2, w - 1)}
        if pos not in anchors:
            raise ExecError(f"unknown anchor {pos!r}")
        r, c = anchors[pos]
    else:
        r, c = _as_offset(pos)
        r = min(max(r, 0), h - 1)
        c = min(max(c, 0), w - 1)
    return g[r][c]


# ======================================================================================= pattern

def _repeat_x(g: Grid, n: int) -> Grid:
    return _tile(g, 1, n)


def _repeat_y(g: Grid, n: int) -> Grid:
    return _tile(g, n, 1)


def _repeat_n(g: Grid, n: int) -> Grid:
    return _tile(g, n, n)


def _mirror_tile(g: Grid) -> Grid:
    h, w = _shape(g)
    _check_dims(2 * h, 2 * w)
    top = [row + row[::-1] for row in g]
    bottom = [row + row[::-1] for row in g[::-1]]
    return [list(r) for r in top + bottom]


def _pattern_fill(g: Grid, m: Mask, pattern: Grid) -> Grid:
    out = _copy(g)
    h, w = _shape(g)
    ph, pw = _shape(pattern)
    for r in range(min(h, len(m))):
        mrow = m[r]
        for c in range(min(w, len(mrow))):
            if mrow[c]:
                out[r][c] = pattern[r % ph][c % pw]
    return out


def _alternate(objs: List[Object], a: int, b: int) -> List[Object]:
    return [_recolor(o, a if i % 2 == 0 else b) for i, o in enumerate(sort_objects(objs))]


def _period(lines: List[List[int]]) -> int:
    """Smallest period p such that every pair of non-zero cells p lines apart agrees."""
    n = len(lines)
    for p in range(1, n):
        ok = True
        for i in range(n - p):
            a, b = lines[i], lines[i + p]
            for x, y in zip(a, b):
                if x and y and x != y:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return p
    return n


def _periodic_repeat(g: Grid) -> Grid:
    h, w = _shape(g)
    pr = _period(g)
    pc = _period(_transpose(g))
    table: Dict[Tuple[int, int], int] = {}
    for r in range(h):
        for c in range(w):
            v = g[r][c]
            if v:
                table.setdefault((r % pr, c % pc), v)
    out = _copy(g)
    for r in range(h):
        for c in range(w):
            if out[r][c] == 0:
                out[r][c] = table.get((r % pr, c % pc), 0)
    return out


# ======================================================================================= counting

def _count_objects(objs: List[Object]) -> int:
    return len(objs)


def _count_cells(m: Mask) -> int:
    return sum(1 for row in m for v in row if v)


# ======================================================================================= conditional / composition

def _if(cond: bool, a: Grid, b: Grid) -> Grid:
    return a if cond else b


def _apply_to_each(objs: List[Object], body: Node, ev: Callable[[Node, Object], Any]) -> List[Object]:
    out = []
    for o in objs:
        res = ev(body, o)
        if not isinstance(res, Object):
            raise ExecError("APPLY_TO_EACH body must return an OBJECT")
        out.append(res)
    return _check_objs(out)


def _filter(objs: List[Object], rel: str, anchor: Object) -> List[Object]:
    fn = _RELATION_FNS.get(rel)
    if fn is None:
        raise ExecError(f"unknown relation {rel!r}")
    return [o for o in objs if fn(o, anchor)]


def substitute_obj(body: Node, replacement: Node) -> Node:
    """Replace every free ``OBJ`` leaf in ``body`` by ``replacement`` (used by COMPOSE).

    PROGRAM-typed arguments (APPLY_TO_EACH bodies, COMPOSE operands) bind their own ``OBJ`` and are left untouched,
    so the substitution respects lexical scope.
    """
    if body.op == LEAF_OBJ and not body.args:
        return replacement
    if not body.args:
        return body
    prim = REGISTRY.get(body.op)
    new_args: List[Any] = []
    for i, a in enumerate(body.args):
        binds = prim is not None and i < len(prim.arg_types) and prim.arg_types[i] is T.PROGRAM
        new_args.append(substitute_obj(a, replacement) if isinstance(a, Node) and not binds else a)
    return Node(body.op, tuple(new_args))


def _compose(p: Node, q: Node) -> Node:
    """Sequential composition of two OBJECT -> OBJECT lambda bodies: first ``p`` then ``q``."""
    if not isinstance(p, Node) or not isinstance(q, Node):
        raise ExecError("COMPOSE expects two program bodies")
    return substitute_obj(q, p)


# ======================================================================================= structural helpers

def _render(objs: List[Object], g: Grid) -> Grid:
    """Paint every object (in set order, later wins) onto a copy of ``g``."""
    out = _copy(g)
    for o in objs:
        _paint_into(out, o)
    return out


def _render_obj(o: Object, g: Grid) -> Grid:
    return paint_object(g, o)


def _render_blank(objs: List[Object], g: Grid) -> Grid:
    h, w = _shape(g)
    return _render(objs, _blank(h, w))


def _crop(g: Grid, o: Object) -> Grid:
    h, w = _shape(g)
    r0, c0, r1, c1 = o.bbox
    r0, c0 = max(r0, 0), max(c0, 0)
    r1, c1 = min(r1, h - 1), min(c1, w - 1)
    if r0 > r1 or c0 > c1:
        raise ExecError("CROP region outside the grid")
    return [row[c0:c1 + 1] for row in g[r0:r1 + 1]]


# ======================================================================================= registry

REGISTRY: Dict[str, Primitive] = {}

CATEGORIES: Tuple[str, ...] = ("selection", "analysis", "relation", "geometric", "manipulation", "color",
                               "pattern", "counting", "conditional", "structural")

_TILE_N: Tuple[int, ...] = (1, 2, 3)
_REPEAT_N: Tuple[int, ...] = (1, 2, 3, 4)


def _reg(name: str, args: Sequence[T], out: T, fn: Callable[..., Any], category: str,
         literal_args: Optional[Dict[int, Sequence]] = None, **flags: bool) -> None:
    if name in REGISTRY:  # pragma: no cover - guards accidental double registration
        raise ValueError(f"duplicate primitive {name}")
    REGISTRY[name] = Primitive(name=name, arg_types=tuple(args), out_type=out, fn=fn, category=category,
                               literal_args=dict(literal_args or {}), **flags)


G, OS, O, M, C, P, I, B, R, PR = (T.GRID, T.OBJECT_SET, T.OBJECT, T.MASK, T.COLOR, T.POSITION, T.INTEGER,
                                  T.BOOLEAN, T.RELATION, T.PROGRAM)

# selection (8)
_reg("SELECT_ALL", [G], OS, _select_all, "selection")
_reg("SELECT_COLOR", [OS, C], OS, _select_color, "selection", {1: COLORS})
_reg("SELECT_NONZERO", [G], M, _select_nonzero, "selection")
_reg("SELECT_LARGEST", [OS], O, _select_largest, "selection")
_reg("SELECT_SMALLEST", [OS], O, _select_smallest, "selection")
_reg("SELECT_UNIQUE", [OS], O, _select_unique, "selection")
_reg("SELECT_BORDER", [OS], OS, _select_border, "selection", needs_ctx=True)
_reg("SELECT_CENTER", [OS], O, _select_center, "selection", needs_ctx=True)
# shape / object analysis (8)
_reg("GET_COMPONENTS4", [G], OS, lambda g: components(g, 4), "analysis")
_reg("GET_COMPONENTS8", [G], OS, lambda g: components(g, 8), "analysis")
_reg("GET_BBOX", [O], M, _get_bbox, "analysis", needs_ctx=True)
_reg("GET_CENTROID", [O], P, _get_centroid, "analysis")
_reg("GET_AREA", [O], I, _get_area, "analysis")
_reg("GET_PERIMETER", [O], I, _get_perimeter, "analysis")
_reg("GET_HOLES", [O], M, _get_holes, "analysis", needs_ctx=True)
_reg("GET_SYMMETRY", [O], B, _get_symmetry, "analysis")
# spatial relations (10)
for _n in ("LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW", "TOUCHING", "OVERLAPPING", "INSIDE", "CONTAINS"):
    _reg(_n, [O, O], B, _RELATION_FNS[_n], "relation")
_reg("NEAREST", [OS, O], O, _nearest, "relation")
_reg("FARTHEST", [OS, O], O, _farthest, "relation")
# geometric (10)
_reg("ROTATE90", [G], G, _rot90, "geometric")
_reg("ROTATE180", [G], G, _rot180, "geometric")
_reg("ROTATE270", [G], G, _rot270, "geometric")
_reg("REFLECT_H", [G], G, _reflect_h, "geometric")
_reg("REFLECT_V", [G], G, _reflect_v, "geometric")
_reg("REFLECT_D1", [G], G, _transpose, "geometric")
_reg("REFLECT_D2", [G], G, _reflect_d2, "geometric")
_reg("TRANSPOSE", [G], G, _transpose, "geometric")
_reg("SHIFT", [G, P], G, _shift, "geometric", {1: POSITION_OFFSETS})
_reg("ALIGN", [O, P], O, _align, "geometric", {1: POSITION_ANCHORS}, needs_ctx=True)
# object manipulation (12)
_reg("COPY", [G, O, P], G, _copy_obj, "manipulation", {2: POSITION_OFFSETS})
_reg("MOVE", [O, P], O, _move, "manipulation", {1: POSITION_OFFSETS})
_reg("DELETE", [G, O], G, _delete, "manipulation")
_reg("DUPLICATE", [O, P], OS, _duplicate, "manipulation", {1: POSITION_OFFSETS})
_reg("MERGE", [OS], O, _merge, "manipulation")
_reg("SPLIT", [O], OS, _split, "manipulation")
_reg("EXTEND", [G, O, P], G, _extend, "manipulation", {2: UNIT_DIRECTIONS})
_reg("SHRINK", [O], O, _shrink, "manipulation")
_reg("GROW", [O], O, _grow, "manipulation")
_reg("FILL", [G, M, C], G, _fill, "manipulation", {2: COLORS})
_reg("OUTLINE", [G, O, C], G, _outline, "manipulation", {2: COLORS})
_reg("FRAME", [G, C], G, _frame, "manipulation", {1: COLORS})
# colour (8)
_reg("RECOLOR", [O, C], O, _recolor, "color", {1: COLORS})
_reg("SWAP_COLORS", [G, C, C], G, _swap_colors, "color", {1: COLORS, 2: COLORS})
_reg("MAP_COLOR", [G, C, C], G, _map_color, "color", {1: COLORS, 2: COLORS})
_reg("MOST_COMMON_COLOR", [G], C, _most_common_color, "color")
_reg("LEAST_COMMON_COLOR", [G], C, _least_common_color, "color")
_reg("REPLACE_BACKGROUND", [G, C], G, _replace_background, "color", {1: COLORS})
_reg("COLOR_OBJECT", [G, O, C], G, _color_object, "color", {2: COLORS})
_reg("COLOR_BY_POSITION", [G, P], C, _color_by_position, "color", {1: POSITION_ANCHORS})
# pattern (8)
_reg("TILE", [G, I, I], G, _tile, "pattern", {1: _TILE_N, 2: _TILE_N})
_reg("REPEAT_X", [G, I], G, _repeat_x, "pattern", {1: _REPEAT_N})
_reg("REPEAT_Y", [G, I], G, _repeat_y, "pattern", {1: _REPEAT_N})
_reg("REPEAT_N", [G, I], G, _repeat_n, "pattern", {1: _TILE_N})
_reg("MIRROR_TILE", [G], G, _mirror_tile, "pattern")
_reg("PATTERN_FILL", [G, M, G], G, _pattern_fill, "pattern")
_reg("ALTERNATE", [OS, C, C], OS, _alternate, "pattern", {1: COLORS, 2: COLORS})
_reg("PERIODIC_REPEAT", [G], G, _periodic_repeat, "pattern")
# counting (4)
_reg("COUNT_OBJECTS", [OS], I, _count_objects, "counting")
_reg("COUNT_CELLS", [M], I, _count_cells, "counting")
_reg("ARGMAX_SIZE", [OS], C, lambda objs: _rank_color(objs, True), "counting")
_reg("ARGMIN_SIZE", [OS], C, lambda objs: _rank_color(objs, False), "counting")
# conditional / composition (4)
_reg("IF", [B, G, G], G, _if, "conditional", {0: BOOLEANS}, lazy=True)
_reg("APPLY_TO_EACH", [OS, PR], OS, _apply_to_each, "conditional", higher_order=True)
_reg("FILTER", [OS, R, O], OS, _filter, "conditional", {1: RELATIONS})
_reg("COMPOSE", [PR, PR], PR, _compose, "conditional")
# structural helpers (outside the 72)
_reg("RENDER", [OS, G], G, _render, "structural", structural=True)
_reg("RENDER_OBJ", [O, G], G, _render_obj, "structural", structural=True)
_reg("RENDER_BLANK", [OS, G], G, _render_blank, "structural", structural=True)
_reg("CROP", [G, O], G, _crop, "structural", structural=True)

SPEC_PRIMITIVES: Tuple[str, ...] = tuple(n for n, p in REGISTRY.items() if not p.structural)
STRUCTURAL_PRIMITIVES: Tuple[str, ...] = tuple(n for n, p in REGISTRY.items() if p.structural)
assert len(SPEC_PRIMITIVES) == 72, len(SPEC_PRIMITIVES)


def by_out_type(t: T, include_structural: bool = True) -> List[Primitive]:
    """Primitives producing type ``t`` (registry order)."""
    return [p for p in REGISTRY.values() if p.out_type == t and (include_structural or not p.structural)]


def by_category(category: str) -> List[Primitive]:
    """Primitives of one category (registry order)."""
    return [p for p in REGISTRY.values() if p.category == category]


def same_signature(p: Primitive) -> List[Primitive]:
    """Other primitives with identical arg / out types (used by mutations and hard negatives)."""
    return [q for q in REGISTRY.values() if q.name != p.name and q.signature() == p.signature()]
