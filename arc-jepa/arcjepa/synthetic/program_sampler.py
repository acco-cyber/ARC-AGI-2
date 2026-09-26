"""Program sampling with the spec depth and category mixes (INTERFACES.md §3, ``program_sampler.py``).

Depth mix (FROZEN_SPEC "Data"): 25/25/20/15/10/5 % for depths 1..6.  Category mix: 40 % object-centric, 20 %
geometry, 15 % relational, 10 % counting, 10 % contextual, 5 % adversarial.  Programs come from
:func:`arcjepa.dsl.grammar.random_program` and are returned in canonical form.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from arcjepa.dsl.ast import Node
from arcjepa.dsl.canonicalize import canonicalize
from arcjepa.dsl.grammar import CATEGORY_PRIMS, can_host, random_expression, random_program
from arcjepa.dsl.primitives import REGISTRY
from arcjepa.dsl.types import T

__all__ = ["DEPTH_MIX", "CATEGORY_MIX", "CATEGORIES", "sample_depth", "sample_category", "sample_program"]

#: Spec depth mix for depths 1..6 (fractions sum to 1).
DEPTH_MIX: Dict[int, float] = {1: 0.25, 2: 0.25, 3: 0.20, 4: 0.15, 5: 0.10, 6: 0.05}
#: Spec category mix (fractions sum to 1).
CATEGORY_MIX: Dict[str, float] = {"object": 0.40, "geometry": 0.20, "relational": 0.15, "counting": 0.10,
                                  "contextual": 0.10, "adversarial": 0.05}
CATEGORIES: List[str] = list(CATEGORY_MIX)

_DEPTHS = sorted(DEPTH_MIX)
_DEPTH_W = [DEPTH_MIX[d] for d in _DEPTHS]
_CAT_W = [CATEGORY_MIX[c] for c in CATEGORIES]


def sample_depth(rng: random.Random) -> int:
    """Draw a program depth (1..6) from :data:`DEPTH_MIX`."""
    return rng.choices(_DEPTHS, weights=_DEPTH_W)[0]


def sample_category(rng: random.Random) -> str:
    """Draw a sample category from :data:`CATEGORY_MIX`."""
    return rng.choices(CATEGORIES, weights=_CAT_W)[0]


def _typed_program(rng: random.Random, depth: int, category: str) -> Node:
    """Well-typed GRID program of exactly ``depth`` levels biased to ``category`` (no probe execution).

    Same construction as :func:`arcjepa.dsl.grammar.random_program` (category-hosting constraint included) but
    without executing probe grids: the task builder executes every program on its real inputs anyway, so the
    probe run only costs time.
    """
    prefs = set(CATEGORY_PRIMS.get(category, {category}))
    feasible = can_host(T.GRID, depth, False, frozenset(prefs))
    fallback: Optional[Node] = None
    for _ in range(8):
        node = random_expression(rng, T.GRID, depth, prefs=prefs, host=feasible)
        if not isinstance(node, Node):
            continue
        if not feasible or any(REGISTRY[o].category in prefs for o in node.primitives() if o in REGISTRY):
            return node
        fallback = fallback or node
    if fallback is not None:
        return fallback
    return random_program(rng, depth, category)


def sample_program(rng: random.Random, category: Optional[str] = None, *, depth: Optional[int] = None,
                   max_tries: int = 8, probe: bool = False) -> Node:
    """Sample a canonical GRID program of the requested (or mix-sampled) depth and category.

    Canonicalisation can shorten a program (e.g. ``ROTATE90∘ROTATE90 → ROTATE180``); up to ``max_tries`` draws
    are made to obtain a canonical program whose depth equals the target depth, otherwise the last canonical
    draw is returned (callers that need the exact depth check ``node.depth()``).

    Args:
        rng: source of randomness.
        category: one of :data:`CATEGORIES` (sampled from :data:`CATEGORY_MIX` when ``None``).
        depth: target depth 1..6 (sampled from :data:`DEPTH_MIX` when ``None``).
        max_tries: draws before giving up on an exact canonical depth.
        probe: use :func:`arcjepa.dsl.grammar.random_program` (prefers programs that execute on its probe
            grids; ~3x slower) instead of the probe-free typed sampler.
    """
    if category is None:
        category = sample_category(rng)
    if depth is None:
        depth = sample_depth(rng)
    node: Optional[Node] = None
    for _ in range(max(1, max_tries)):
        raw = random_program(rng, depth, category) if probe else _typed_program(rng, depth, category)
        node = canonicalize(raw)
        if node.depth() == depth:
            return node
    assert node is not None
    return node
