"""Hypothesis selection and the top-level :func:`parse` entry point of the parser."""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from arcjepa.core.types import Grid

from .objects import FEATURE_DIM, Object
from .relations import REL_DIM, relation_matrix
from .segmentation import HYPOTHESES, order_objects, segment

__all__ = ["DEFAULT_ORDER", "MAX_OBJECTS", "default_hypothesis", "parse"]

logger = logging.getLogger(__name__)

#: Fallback order used by :func:`default_hypothesis` (spec: cc4, then per_color_cc4, then cc8, ...).
DEFAULT_ORDER: Tuple[str, ...] = (
    "cc4",
    "per_color_cc4",
    "cc8",
    "color_agnostic_cc8",
    "rect_regions",
    "rows",
    "cols",
    "frames",
    "repeat_blocks",
    "symmetry",
)
MAX_OBJECTS: int = 64


def default_hypothesis(grid: Grid, max_objects: int = MAX_OBJECTS, background: int = 0) -> str:
    """First hypothesis in :data:`DEFAULT_ORDER` that yields at most ``max_objects`` objects.

    ``cc4`` is used unless it produces more than ``max_objects`` objects; ``per_color_cc4``
    (<= 9 objects) always terminates the chain in practice. Falls back to ``cc4``.
    """
    for name in DEFAULT_ORDER:
        n = len(segment(grid, name, background))
        if n <= max_objects:
            return name
    return "cc4"


def _cap(objs: List[Object], max_objects: int) -> List[Object]:
    """Keep the ``max_objects`` largest objects (ties by scan order) and restore the canonical order."""
    if len(objs) <= max_objects:
        return objs
    ranked = sorted(range(len(objs)), key=lambda i: (-objs[i].area, i))
    keep = sorted(ranked[:max_objects])
    return order_objects(objs[i] for i in keep)


def parse(
    grid: Grid,
    hypothesis: Optional[str] = None,
    max_objects: int = MAX_OBJECTS,
    background: int = 0,
) -> Tuple[List[Object], np.ndarray, np.ndarray]:
    """Segment ``grid`` and return ``(objects, features[N, 32], relations[N, N, 24])``.

    ``hypothesis`` defaults to :func:`default_hypothesis`. When the segmentation yields more
    than ``max_objects`` objects the largest ones are kept (deterministic tie-break by scan
    order) so ``N <= max_objects`` always holds. ``N`` may be 0 for an all-background grid.
    """
    h = len(grid)
    w = len(grid[0]) if h else 0
    if hypothesis is None:
        hypothesis = default_hypothesis(grid, max_objects, background)
    elif hypothesis not in HYPOTHESES:
        raise ValueError(f"unknown hypothesis {hypothesis!r}")
    objs = segment(grid, hypothesis, background)
    if len(objs) > max_objects:
        logger.debug("parse: %s produced %d objects, capping to %d", hypothesis, len(objs), max_objects)
        objs = _cap(objs, max_objects)
    if objs:
        feats = np.stack([o.features(h, w) for o in objs]).astype(np.float32)
    else:
        feats = np.zeros((0, FEATURE_DIM), dtype=np.float32)
    rels = relation_matrix(objs, h, w)
    assert rels.shape == (len(objs), len(objs), REL_DIM)
    return objs, feats, rels
