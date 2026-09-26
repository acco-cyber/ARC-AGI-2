"""Heuristic assignment of an ARC task to one of the eight spec families.

Families (FROZEN_SPEC "Data"): ``geometry, object, relation, counting, composition, context, symmetry,
pattern``. The assignment is a deterministic decision cascade over cheap grid statistics of the task's
training pairs (shape relation, dihedral equality, tiling / periodicity, output symmetry, sub-grid cropping,
separator lines, object-count change, added/removed cells). It is used only to *balance* the 700/150/150
re-split and to break down evaluation results; it is not a label the model trains on.

Decision order (first hit wins)::

    geometry     every output is the same non-identity dihedral transform of its input
    pattern      output tiles a dihedral copy of the input (or vice versa), or every output is periodic
    symmetry     every output is mirror/rotation symmetric while the inputs are not (symmetry completion)
    context      output is an exact (>= 4 cells, >= 2 colours) crop of the input
    counting     small outputs whose shape or cell count varies with the number of input objects
    context      inputs are divided by full separator lines, or the output is a small constant-shape summary
    relation     same shape, >= 2 objects and only straight-line cells are added between objects,
                 or object-dependent recolouring (no global colour map)
    object       same shape with a global colour map, object removal, movement, count change or cells added
                 inside object bounding boxes
    composition  everything else (grows without tiling, mixed shape relations, several signals at once)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from arcjepa.core.types import Grid, Pair, Task

FAMILIES: Tuple[str, ...] = ("geometry", "object", "relation", "counting", "composition", "context", "symmetry", "pattern")

_Cells = Set[Tuple[int, int]]


# --------------------------------------------------------------------------------------------------------------
# Grid helpers
# --------------------------------------------------------------------------------------------------------------


def _arr(g: Grid) -> np.ndarray:
    return np.asarray(g, dtype=np.int16)


def _eq(a: np.ndarray, b: np.ndarray) -> bool:
    """Cheap exact equality (shape + values) without ``np.array_equal``'s Python overhead."""
    return a.shape == b.shape and bool((a == b).all())


def dihedral_transforms(a: np.ndarray) -> Dict[str, np.ndarray]:
    """The eight dihedral images of ``a`` keyed by the names used in the dataset (``identity`` first)."""
    t = a.T  # slicing views are much cheaper than np.rot90 for thousands of small grids
    return {
        "identity": a,
        "rot90": t[::-1, :],
        "rot180": a[::-1, ::-1],
        "rot270": t[:, ::-1],
        "flip_h": a[:, ::-1],
        "flip_v": a[::-1, :],
        "transpose": t,
        "anti_transpose": t[::-1, ::-1],
    }


def _same_shape(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape


def dihedral_match(inp: np.ndarray, out: np.ndarray) -> Optional[str]:
    """Name of a non-identity dihedral transform ``t`` with ``t(inp) == out`` (None if there is none)."""
    for name, img in dihedral_transforms(inp).items():
        if name == "identity":
            continue
        if _same_shape(img, out) and np.array_equal(img, out):
            return name
    return None


def components(a: np.ndarray, background: int = 0, diagonal: bool = True) -> List[_Cells]:
    """Colour-agnostic connected components of the non-background cells (8-connectivity by default).

    Implemented as vectorised min-label propagation (each cell takes the smallest label among itself and its
    neighbours until convergence), which is much faster than a Python BFS on 30 x 30 grids. Components are
    returned in order of their smallest raster index.
    """
    h, w = a.shape
    nz = a != background
    if not nz.any():
        return []
    try:  # fast path: scipy is available locally and on Kaggle; the propagation below is the fallback
        from scipy import ndimage  # type: ignore

        structure = np.ones((3, 3), dtype=bool) if diagonal else None
        lab, n = ndimage.label(nz, structure=structure)
        out: List[_Cells] = [set() for _ in range(int(n))]
        rs, cs = np.nonzero(nz)
        for r, c, k in zip(rs.tolist(), cs.tolist(), lab[rs, cs].tolist()):
            out[k - 1].add((r, c))
        return out  # scipy numbers components in raster order of first occurrence
    except ImportError:
        pass
    labels = np.where(nz, np.arange(h * w, dtype=np.int32).reshape(h, w), h * w).astype(np.int32)
    if diagonal:
        shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    padded = np.full((h + 2, w + 2), h * w, dtype=np.int32)
    while True:
        padded[1:-1, 1:-1] = labels
        new = labels.copy()
        for dr, dc in shifts:
            np.minimum(new, padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w], out=new)
        new = np.where(nz, new, h * w)
        if np.array_equal(new, labels):
            break
        labels = new
    comps: Dict[int, _Cells] = {}
    rs, cs = np.nonzero(nz)
    for r, c, lab in zip(rs.tolist(), cs.tolist(), labels[rs, cs].tolist()):
        comps.setdefault(lab, set()).add((r, c))
    return [comps[k] for k in sorted(comps)]


def _shape_key(cells: _Cells) -> Tuple[Tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _is_periodic(a: np.ndarray) -> bool:
    """True when ``a`` repeats with a period strictly smaller than its size along at least one axis."""
    h, w = a.shape
    if h * w < 4:
        return False
    for p in range(1, h // 2 + 1):
        if (a[p:, :] == a[:-p, :]).all():
            return True
    for p in range(1, w // 2 + 1):
        if (a[:, p:] == a[:, :-p]).all():
            return True
    return False


def _tiling_of(unit: np.ndarray, big: np.ndarray) -> bool:
    """True when ``big`` is an integer tiling of dihedral images of ``unit`` (at least two tiles)."""
    uh, uw = unit.shape
    bh, bw = big.shape
    if bh % uh or bw % uw:
        return False
    nr, nc = bh // uh, bw // uw
    if nr * nc < 2:
        return False
    imgs = [img for img in dihedral_transforms(unit).values() if img.shape == unit.shape]
    for i in range(nr):
        for j in range(nc):
            tile = big[i * uh:(i + 1) * uh, j * uw:(j + 1) * uw]
            if not any(_eq(tile, img) for img in imgs):
                return False
    return True


def _symmetries(a: np.ndarray) -> Set[str]:
    out: Set[str] = set()
    if _eq(a, a[:, ::-1]):
        out.add("h")
    if _eq(a, a[::-1, :]):
        out.add("v")
    if _eq(a, a[::-1, ::-1]):
        out.add("rot180")
    if a.shape[0] == a.shape[1]:
        if _eq(a, a.T):
            out.add("d1")
        if _eq(a, a.T[::-1, ::-1]):
            out.add("d2")
    return out


def _is_subgrid(small: np.ndarray, big: np.ndarray) -> bool:
    """True when ``small`` occurs verbatim somewhere inside ``big`` (vectorised sliding window)."""
    sh, sw = small.shape
    bh, bw = big.shape
    if sh > bh or sw > bw:
        return False
    windows = np.lib.stride_tricks.sliding_window_view(big, (sh, sw))
    return bool((windows == small).all(axis=(-1, -2)).any())


def _separator_lines(a: np.ndarray) -> int:
    """Number of full rows plus full columns painted in one non-background colour (panel separators)."""
    h, w = a.shape
    n = 0
    for r in range(h):
        row = a[r]
        if row[0] != 0 and np.all(row == row[0]):
            n += 1
    for c in range(w):
        col = a[:, c]
        if col[0] != 0 and np.all(col == col[0]):
            n += 1
    if n >= h or n >= w:  # the whole grid is one colour: not a separator structure
        return 0
    return n


# --------------------------------------------------------------------------------------------------------------
# Task statistics
# --------------------------------------------------------------------------------------------------------------


@dataclass
class TaskFeatures:
    """Cheap deterministic statistics of a task's demonstration pairs used by :func:`family_of`."""

    n_pairs: int
    shape_relation: str  # same | constant | shrinks | grows | mixed
    dihedral: Optional[str]  # common non-identity dihedral name or None
    tiling: bool  # output tiles (a dihedral copy of) the input, or input tiles the output
    periodic_output: bool
    symmetry_completion: bool
    crop: bool  # every output is an exact sub-grid of its input
    separators: bool  # every input has full separator lines
    output_max_side: int
    output_shape_varies: bool
    output_cells_track_objects: bool  # #non-bg output cells == #input objects for every pair
    n_objects_in: List[int]
    n_objects_out: List[int]
    same_positions: bool  # non-background masks identical for every pair (colour-only change)
    global_color_map: bool  # a single colour -> colour map explains every pair
    added_line_only: bool  # every added component is a straight segment
    added_inside_bboxes: bool  # every added cell falls in some input object's bbox
    removed_only: bool  # cells removed, none added
    moved_objects: bool  # multiset of object shapes preserved, positions changed


def _shape_relation(pairs: Sequence[Pair]) -> str:
    rels = []
    for p in pairs:
        hi, wi = len(p.input), len(p.input[0])
        ho, wo = len(p.output), len(p.output[0])
        if (hi, wi) == (ho, wo):
            rels.append("same")
        elif ho * wo < hi * wi:
            rels.append("shrinks")
        elif ho * wo > hi * wi:
            rels.append("grows")
        else:
            rels.append("mixed")
    if all(r == "same" for r in rels):
        return "same"
    out_shapes = {(len(p.output), len(p.output[0])) for p in pairs}
    if len(out_shapes) == 1:
        return "constant"
    if all(r == "shrinks" for r in rels):
        return "shrinks"
    if all(r == "grows" for r in rels):
        return "grows"
    return "mixed"


def task_features(pairs: Sequence[Pair]) -> TaskFeatures:
    """Compute :class:`TaskFeatures` for a sequence of demonstration pairs (at least one)."""
    ins = [_arr(p.input) for p in pairs]
    outs = [_arr(p.output) for p in pairs]
    n = len(pairs)

    dih_names = [dihedral_match(a, b) for a, b in zip(ins, outs)]
    dihedral = dih_names[0] if dih_names and all(d == dih_names[0] and d is not None for d in dih_names) else None

    tiling = all(
        (a.shape != b.shape) and (_tiling_of(a, b) if b.size > a.size else _tiling_of(b, a)) for a, b in zip(ins, outs)
    )
    periodic_output = all(_is_periodic(b) and not np.array_equal(a, b) if a.shape == b.shape else _is_periodic(b)
                          for a, b in zip(ins, outs))

    out_syms = [_symmetries(b) for b in outs]
    in_syms = [_symmetries(a) for a in ins]
    common = set.intersection(*out_syms) if out_syms else set()
    symmetry_completion = bool(common) and not all(common <= s for s in in_syms) and not all(
        np.array_equal(a, b) for a, b in zip(ins, outs)
    )

    # a crop must be informative: at least 4 cells and 2 colours, otherwise tiny outputs (e.g. [[1]]) would
    # trivially "occur" inside the input and swallow the counting family
    crop = all(
        a.shape != b.shape and b.size < a.size and b.size >= 4 and len(np.unique(b)) >= 2 and _is_subgrid(b, a)
        for a, b in zip(ins, outs)
    )
    separators = all(_separator_lines(a) >= 1 for a in ins)

    comps_in = [components(a) for a in ins]
    comps_out = [components(b) for b in outs]
    n_in = [len(c) for c in comps_in]
    n_out = [len(c) for c in comps_out]

    out_shapes = {b.shape for b in outs}
    output_max_side = max(max(b.shape) for b in outs)
    output_cells = [int((b != 0).sum()) for b in outs]
    output_cells_track_objects = n >= 2 and all(oc == ni for oc, ni in zip(output_cells, n_in)) and len(set(n_in)) > 1

    same_shape = all(a.shape == b.shape for a, b in zip(ins, outs))
    same_positions = same_shape and all(np.array_equal(a != 0, b != 0) for a, b in zip(ins, outs))

    global_color_map = False
    if same_shape:
        cmap: Dict[int, int] = {}
        ok = True
        for a, b in zip(ins, outs):
            for ca, cb in zip(a.ravel().tolist(), b.ravel().tolist()):
                if cmap.setdefault(ca, cb) != cb:
                    ok = False
                    break
            if not ok:
                break
        global_color_map = ok and any(k != v for k, v in cmap.items())

    added_line_only = False
    added_inside_bboxes = False
    removed_only = False
    moved_objects = False
    if same_shape:
        any_added = False
        any_removed = False
        line_only = True
        inside = True
        for a, b, comps in zip(ins, outs, comps_in):
            added = (a == 0) & (b != 0)
            removed = (a != 0) & (b == 0)
            if removed.any():
                any_removed = True
            if added.any():
                any_added = True
                for comp in components(added.astype(np.int16)):
                    rs = [r for r, _ in comp]
                    cs = [c for _, c in comp]
                    if not (min(rs) == max(rs) or min(cs) == max(cs)) or len(comp) < 2:
                        line_only = False
                bboxes = []
                for comp in comps:
                    rs = [r for r, _ in comp]
                    cs = [c for _, c in comp]
                    bboxes.append((min(rs), min(cs), max(rs), max(cs)))
                for r, c in zip(*np.nonzero(added)):
                    if not any(r0 <= r <= r1 and c0 <= c <= c1 for r0, c0, r1, c1 in bboxes):
                        inside = False
                        break
        added_line_only = any_added and line_only
        added_inside_bboxes = any_added and inside
        removed_only = any_removed and not any_added
        moved_objects = all(
            len(ci) == len(co) and sorted(map(_shape_key, ci)) == sorted(map(_shape_key, co)) and not np.array_equal(a, b)
            for a, b, ci, co in zip(ins, outs, comps_in, comps_out)
        ) and all(n_ >= 1 for n_ in n_in)

    return TaskFeatures(
        n_pairs=n,
        shape_relation=_shape_relation(pairs),
        dihedral=dihedral,
        tiling=tiling,
        periodic_output=periodic_output,
        symmetry_completion=symmetry_completion,
        crop=crop,
        separators=separators,
        output_max_side=output_max_side,
        output_shape_varies=len(out_shapes) > 1,
        output_cells_track_objects=output_cells_track_objects,
        n_objects_in=n_in,
        n_objects_out=n_out,
        same_positions=same_positions,
        global_color_map=global_color_map,
        added_line_only=added_line_only,
        added_inside_bboxes=added_inside_bboxes,
        removed_only=removed_only,
        moved_objects=moved_objects,
    )


# --------------------------------------------------------------------------------------------------------------
# Family decision
# --------------------------------------------------------------------------------------------------------------


def family_from_features(f: TaskFeatures) -> str:
    """Map :class:`TaskFeatures` to one of :data:`FAMILIES` with the documented decision order."""
    if f.dihedral is not None:
        return "geometry"
    if f.tiling or f.periodic_output:
        return "pattern"
    if f.symmetry_completion and f.shape_relation in ("same", "constant", "grows"):
        return "symmetry"
    if f.crop:
        return "context"
    small = f.output_max_side <= 5
    if f.shape_relation in ("shrinks", "constant", "mixed") and small and (
        f.output_cells_track_objects or (f.output_shape_varies and f.shape_relation != "constant")
    ):
        return "counting"
    if f.separators and f.shape_relation != "grows":
        return "context"
    if f.shape_relation == "constant" and small:
        return "counting" if f.output_cells_track_objects else "context"
    if f.shape_relation == "same":
        multi = min(f.n_objects_in) >= 2 if f.n_objects_in else False
        if f.same_positions:
            if f.global_color_map:
                return "object"
            return "relation" if multi else "object"
        if f.global_color_map:
            return "object"
        if f.added_line_only and multi:
            return "relation"
        if f.removed_only or f.moved_objects or f.added_inside_bboxes:
            return "object"
        if any(a != b for a, b in zip(f.n_objects_in, f.n_objects_out)):
            return "object"
        return "relation" if multi else "composition"
    if f.shape_relation == "shrinks":
        return "context"
    return "composition"


def family_of(task: Task) -> str:
    """Return the spec family (one of :data:`FAMILIES`) of ``task`` from its training pairs.

    Deterministic and pure; tasks without demonstration pairs fall back to ``"composition"``.
    """
    pairs = task.train if task.train else task.all_pairs
    if not pairs:
        return "composition"
    fam = family_from_features(task_features(pairs))
    assert fam in FAMILIES
    return fam


def family_histogram(tasks: Dict[str, Task]) -> Dict[str, int]:
    """Count of tasks per family (all eight keys present, sorted by spec order)."""
    hist = {f: 0 for f in FAMILIES}
    for t in tasks.values():
        hist[family_of(t)] += 1
    return hist
