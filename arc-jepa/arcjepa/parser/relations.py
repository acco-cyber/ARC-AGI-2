"""Pairwise object relations: 18 binary relations + 5 continuous features -> 24-dim vectors.

The relation channels follow the spec order exactly (:data:`RELATIONS`). Continuous
features are appended in :data:`CONTINUOUS` order and the vector is zero padded to 24.

Definitions (a = row object i, b = column object j):

``left_of``   a's bbox lies strictly left of b's (``c1_a < c0_b``);  ``right_of`` is the transpose.
``above``     ``r1_a < r0_b``;  ``below`` is the transpose.
``overlap``   the cell sets intersect.
``touching``  not overlapping and some cell of a is 8-adjacent to a cell of b.
``contains``  a's bbox strictly encloses b's bbox on all four sides and the cells are disjoint;
              ``inside`` is the transpose.
``same_color`` equal primary colours.  ``same_shape`` equal translation-normalised cell sets.
``aligned_x`` equal left edge, equal right edge or centroid columns within 0.5 (share a column line).
``aligned_y`` the same for rows.
``nearest``   b is (one of) a's closest other objects by centroid distance; ``farther`` = farthest.
``same_size`` equal areas; ``larger`` area_a > area_b; ``smaller`` area_a < area_b.
``symmetric_to`` b's shape is a non-identity D4 image (mirror / rotation) of a's shape.

Continuous: ``delta_r`` = (r_b - r_a)/(H-1), ``delta_c`` = (c_b - c_a)/(W-1) (signed, in [-1, 1]),
``distance`` = centroid distance / grid diagonal, ``iou`` = |a & b| / |a | b|,
``size_ratio`` = area_a / (area_a + area_b).  Slot 23 is padding.

The diagonal (i == j) of every binary channel is zero.
"""
from __future__ import annotations

import math
from typing import Dict, FrozenSet, List, Sequence, Tuple

import numpy as np

from .objects import Object

__all__ = ["RELATIONS", "CONTINUOUS", "REL_DIM", "relation_features", "relation_matrix"]

RELATIONS: Tuple[str, ...] = (
    "left_of",
    "right_of",
    "above",
    "below",
    "overlap",
    "touching",
    "contains",
    "inside",
    "same_color",
    "same_shape",
    "aligned_x",
    "aligned_y",
    "nearest",
    "farther",
    "same_size",
    "larger",
    "smaller",
    "symmetric_to",
)
CONTINUOUS: Tuple[str, ...] = ("delta_r", "delta_c", "distance", "iou", "size_ratio")
REL_DIM: int = 24
assert len(RELATIONS) == 18 and len(RELATIONS) + len(CONTINUOUS) <= REL_DIM

_REL_INDEX: Dict[str, int] = {name: i for i, name in enumerate(RELATIONS)}


def _cell_masks(objs: Sequence[Object]) -> np.ndarray:
    """Boolean ``[N, Hm, Wm]`` masks on a canvas large enough for every (possibly translated) object."""
    rmin = min(0, min(o.bbox[0] for o in objs))
    cmin = min(0, min(o.bbox[1] for o in objs))
    rmax = max(o.bbox[2] for o in objs)
    cmax = max(o.bbox[3] for o in objs)
    hm, wm = rmax - rmin + 1, cmax - cmin + 1
    masks = np.zeros((len(objs), hm, wm), dtype=bool)
    for i, o in enumerate(objs):
        if o.cells:
            idx = np.fromiter((r - rmin for r, _ in o.cells), dtype=np.int64, count=len(o.cells))
            jdx = np.fromiter((c - cmin for _, c in o.cells), dtype=np.int64, count=len(o.cells))
            masks[i, idx, jdx] = True
    return masks


def _dilate8(masks: np.ndarray) -> np.ndarray:
    n, h, w = masks.shape
    padded = np.zeros((n, h + 2, w + 2), dtype=bool)
    padded[:, 1:-1, 1:-1] = masks
    out = np.zeros_like(masks)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            out |= padded[:, 1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
    return out


def relation_matrix(objs: Sequence[Object], h: int, w: int) -> np.ndarray:
    """``float32[N, N, 24]`` relation tensor for ``objs`` living on an ``h x w`` grid."""
    n = len(objs)
    out = np.zeros((n, n, REL_DIM), dtype=np.float32)
    if n == 0:
        return out
    H = max(int(h), 1)
    W = max(int(w), 1)
    bbox = np.array([o.bbox for o in objs], dtype=np.float64).reshape(n, 4)
    r0, c0, r1, c1 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
    area = np.array([o.area for o in objs], dtype=np.float64)
    cent = np.array([o.centroid for o in objs], dtype=np.float64).reshape(n, 2)
    cr, cc = cent[:, 0], cent[:, 1]
    color = np.array([o.primary_color for o in objs], dtype=np.int64)
    eye = np.eye(n, dtype=bool)
    off = ~eye

    # --- cell-set relations via mask products -------------------------------------
    masks = _cell_masks(objs)
    mf = masks.reshape(n, -1).astype(np.float32)
    inter = mf @ mf.T  # |a & b|
    overlap = (inter > 0) & off
    df = _dilate8(masks).reshape(n, -1).astype(np.float32)
    touching = ((df @ mf.T) > 0) & ~overlap & off

    # --- bbox relations -------------------------------------------------------------
    left_of = c1[:, None] < c0[None, :]
    above = r1[:, None] < r0[None, :]
    contains = (
        (r0[:, None] < r0[None, :])
        & (c0[:, None] < c0[None, :])
        & (r1[:, None] > r1[None, :])
        & (c1[:, None] > c1[None, :])
        & ~overlap
    )
    same_color = color[:, None] == color[None, :]

    # --- shape relations ------------------------------------------------------------
    shape_ids: Dict[FrozenSet[Tuple[int, int]], int] = {}
    sid = np.zeros(n, dtype=np.int64)
    for i, o in enumerate(objs):
        sid[i] = shape_ids.setdefault(o.shape, len(shape_ids))
    same_shape = sid[:, None] == sid[None, :]
    symmetric_to = np.zeros((n, n), dtype=bool)
    d4 = [set(o.d4_shape_keys()) for o in objs]
    shapes = [o.shape for o in objs]
    for i in range(n):
        keys = d4[i]
        for j in range(n):
            if i != j and shapes[j] in keys:
                symmetric_to[i, j] = True

    # --- alignment / distance -------------------------------------------------------
    dcc = cc[None, :] - cc[:, None]  # column delta a -> b
    drr = cr[None, :] - cr[:, None]
    aligned_x = (c0[:, None] == c0[None, :]) | (c1[:, None] == c1[None, :]) | (np.abs(dcc) < 0.5)
    aligned_y = (r0[:, None] == r0[None, :]) | (r1[:, None] == r1[None, :]) | (np.abs(drr) < 0.5)
    dist = np.sqrt(drr * drr + dcc * dcc)
    if n >= 2:
        dist_off = np.where(off, dist, np.inf)
        nearest = (dist_off <= dist_off.min(axis=1, keepdims=True) + 1e-9) & off
        dist_off_max = np.where(off, dist, -np.inf)
        farther = (dist_off_max >= dist_off_max.max(axis=1, keepdims=True) - 1e-9) & off
    else:
        nearest = np.zeros((n, n), dtype=bool)
        farther = np.zeros((n, n), dtype=bool)

    same_size = area[:, None] == area[None, :]
    larger = area[:, None] > area[None, :]

    binary = {
        "left_of": left_of,
        "right_of": left_of.T,
        "above": above,
        "below": above.T,
        "overlap": overlap,
        "touching": touching,
        "contains": contains,
        "inside": contains.T,
        "same_color": same_color,
        "same_shape": same_shape,
        "aligned_x": aligned_x,
        "aligned_y": aligned_y,
        "nearest": nearest,
        "farther": farther,
        "same_size": same_size,
        "larger": larger,
        "smaller": larger.T,
        "symmetric_to": symmetric_to,
    }
    for name, mat in binary.items():
        out[:, :, _REL_INDEX[name]] = (mat & off).astype(np.float32)

    # --- continuous -----------------------------------------------------------------
    k = len(RELATIONS)
    hd = max(H - 1, 1)
    wd = max(W - 1, 1)
    diag = math.sqrt(hd * hd + wd * wd)
    union = area[:, None] + area[None, :] - inter
    out[:, :, k + 0] = np.clip(drr / hd, -1.0, 1.0)
    out[:, :, k + 1] = np.clip(dcc / wd, -1.0, 1.0)
    out[:, :, k + 2] = np.clip(dist / diag, 0.0, 1.0)
    out[:, :, k + 3] = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    out[:, :, k + 4] = area[:, None] / (area[:, None] + area[None, :])
    return out


def relation_features(a: Object, b: Object, grid_h: int, grid_w: int) -> np.ndarray:
    """``float32[24]`` relation vector for the ordered pair ``(a, b)`` (see module docstring)."""
    return relation_matrix([a, b], grid_h, grid_w)[0, 1].copy()
