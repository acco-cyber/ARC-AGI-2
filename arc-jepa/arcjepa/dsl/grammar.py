"""Type-constrained grammar: expansions, random programs and breadth-first enumeration (INTERFACES.md §1).

* :func:`expansions` lists the primitives that can produce a type within a remaining depth budget.
* :func:`random_program` samples a GRID-typed program of a requested depth (optionally biased to a spec
  category) that executes on at least one probe grid whenever the sampler can find one within a few attempts.
* :func:`enumerate_programs` streams programs breadth-first by depth, canonicalised and de-duplicated.
"""
from __future__ import annotations

import functools
import itertools
import logging
import random
from typing import Any, Dict, FrozenSet, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from arcjepa.core.types import Grid
from arcjepa.dsl.ast import INPUT, OBJ, Node
from arcjepa.dsl.canonicalize import canonicalize
from arcjepa.dsl.interpreter import ExecError, execute, typecheck
from arcjepa.dsl.primitives import REGISTRY, Primitive
from arcjepa.dsl.types import BOOLEANS, COLORS, LITERAL_TYPES, RELATIONS, T, UNIT_DIRECTIONS

__all__ = ["expansions", "random_program", "random_expression", "enumerate_programs", "min_depth",
           "reachable", "can_host", "CATEGORY_PRIMS", "PROBE_GRIDS", "random_literal", "DEPTH_MIX",
           "sample_depth", "ENUM_DEFAULT"]

log = logging.getLogger(__name__)

#: Spec sample categories -> primitive categories that should dominate such programs.
CATEGORY_PRIMS: Dict[str, Set[str]] = {
    "object": {"selection", "manipulation", "analysis", "structural"},
    "geometry": {"geometric", "pattern"},
    "relational": {"relation", "conditional"},
    "counting": {"counting"},
    "contextual": {"color", "pattern", "conditional"},
    "adversarial": {"selection", "manipulation", "relation"},
}

#: Literal domains used by the enumerator (and as defaults by the sampler) when a primitive has no restriction.
ENUM_DEFAULT: Dict[T, Sequence[Any]] = {
    T.COLOR: COLORS, T.INTEGER: (1, 2, 3), T.POSITION: UNIT_DIRECTIONS, T.BOOLEAN: BOOLEANS, T.RELATION: RELATIONS,
}
#: Primitives that are degenerate below a certain depth (IF with a literal condition folds away).
_MIN_USEFUL: Dict[str, int] = {"IF": 4}
_MEMO_CAP = 4000


# ======================================================================================= depth bookkeeping

def min_depth(t: T, in_lambda: bool = False) -> int:
    """Smallest depth of an expression of type ``t`` (0 = leaf or literal)."""
    if t is T.GRID:
        return 0
    if t is T.OBJECT:
        return 0 if in_lambda else 2
    if t in (T.OBJECT_SET, T.MASK, T.PROGRAM):
        return 1
    return 0


def _arg_mode(u: T, in_lambda: bool) -> Tuple[T, bool]:
    """Type and lambda-mode used to generate an argument of declared type ``u``."""
    if u is T.PROGRAM:
        return T.OBJECT, True
    return u, in_lambda


def expansions(target: T, depth_left: int, *, in_lambda: bool = False,
               include_structural: bool = True) -> List[Primitive]:
    """Primitives producing ``target`` whose arguments can all be completed within ``depth_left - 1`` levels."""
    if depth_left < 1:
        return []
    out: List[Primitive] = []
    for p in REGISTRY.values():
        if p.out_type is not target or (p.structural and not include_structural):
            continue
        if _MIN_USEFUL.get(p.name, 1) > depth_left:
            continue
        need = max((min_depth(*_arg_mode(u, in_lambda)) for u in p.arg_types), default=0)
        if need <= depth_left - 1:
            out.append(p)
    return out


@functools.lru_cache(maxsize=None)
def reachable(t: T, depth: int, in_lambda: bool = False) -> bool:
    """True when an expression of type ``t`` with exactly ``depth`` levels exists (pure, memoised)."""
    if depth == 0:
        return t is T.GRID or (t is T.OBJECT and in_lambda) or t in LITERAL_TYPES
    for p in expansions(t, depth, in_lambda=in_lambda):
        if any(_arg_reachable(u, in_lambda, depth - 1) for u in p.arg_types):
            return True
    return False


def _arg_reachable(u: T, in_lambda: bool, depth: int) -> bool:
    """``reachable`` for an argument of declared type ``u`` (PROGRAM args are OBJECT bodies in lambda mode)."""
    gt, lam = _arg_mode(u, in_lambda)
    return reachable(gt, depth, lam)


#: Spec depth mix for synthetic programs (depths 1..6 -> 25/25/20/15/10/5 %).
DEPTH_MIX: Dict[int, float] = {1: 0.25, 2: 0.25, 3: 0.20, 4: 0.15, 5: 0.10, 6: 0.05}


def sample_depth(rng: random.Random) -> int:
    """Draw a program depth from :data:`DEPTH_MIX`."""
    depths = sorted(DEPTH_MIX)
    return rng.choices(depths, weights=[DEPTH_MIX[d] for d in depths])[0]


# ======================================================================================= random sampling

def random_literal(rng: random.Random, t: T, domain: Optional[Sequence[Any]] = None) -> Any:
    """Sample a literal of type ``t`` from ``domain`` (or the default domain)."""
    if domain is None:
        domain = ENUM_DEFAULT[t]
    if t is T.COLOR and 0 in domain and len(domain) > 1 and rng.random() < 0.85:
        return rng.choice([c for c in domain if c != 0])
    return rng.choice(list(domain))


def _choice(rng: random.Random, cands: List[Primitive], weights: Optional[Mapping[str, float]]) -> Primitive:
    """Uniform choice (``weights`` None: the exact historical RNG stream), else weighted by ``weights[name]``
    (default 1.0 for names not in the table)."""
    if not weights:
        return rng.choice(cands)
    return rng.choices(cands, weights=[max(0.0, float(weights.get(p.name, 1.0))) + 1e-12 for p in cands])[0]


def _pick_primitive(rng: random.Random, cands: List[Primitive], prefs: Optional[Set[str]],
                    weights: Optional[Mapping[str, float]] = None) -> Primitive:
    if prefs:
        preferred = [p for p in cands if p.category in prefs]
        if preferred and rng.random() < 0.75:
            return _choice(rng, preferred, weights)
    return _choice(rng, cands, weights)


def _choose_child_depth(rng: random.Random, t: T, max_d: int, in_lambda: bool) -> Optional[int]:
    opts = [d for d in range(min_depth(t, in_lambda), max_d + 1) if reachable(t, d, in_lambda)]
    if not opts:
        return None
    # literal BOOLEAN conditions fold away under canonicalisation, so conditions are computed 90 % of the time
    p_literal = 0.1 if t is T.BOOLEAN else 0.8
    if t in LITERAL_TYPES and 0 in opts and (len(opts) == 1 or rng.random() < p_literal):
        return 0
    weights = [1.0 / (1 + d) for d in opts]
    return rng.choices(opts, weights=weights)[0]


_CANDIDATES_MEMO: Dict[Tuple[T, int, bool, int], List[Primitive]] = {}


def _candidates(target: T, depth: int, in_lambda: bool) -> List[Primitive]:
    """Primitives that can head an expression of ``target`` with exactly ``depth`` levels (registry order).

    Memoised (a pure function of its arguments and the registry, like :func:`reachable`; keyed on the registry
    size too): recomputing it was ~15 % of the synthetic generator's time.  Callers must not mutate the list."""
    key = (target, depth, in_lambda, len(REGISTRY))
    got = _CANDIDATES_MEMO.get(key)
    if got is None:
        got = [p for p in expansions(target, depth, in_lambda=in_lambda)
               if any(_arg_reachable(u, in_lambda, depth - 1) for u in p.arg_types)]
        _CANDIDATES_MEMO[key] = got
    return got


@functools.lru_cache(maxsize=None)
def can_host(t: T, depth: int, in_lambda: bool, prefs: FrozenSet[str]) -> bool:
    """True when an expression of type ``t`` with exactly ``depth`` levels can contain a primitive whose category
    is in ``prefs`` (the hosting argument is always the deepest one, which keeps the depth exact)."""
    if depth == 0 or not prefs:
        return False
    for p in _candidates(t, depth, in_lambda):
        if p.category in prefs or _host_args(p, depth, in_lambda, prefs):
            return True
    return False


def _host_args(p: Primitive, depth: int, in_lambda: bool, prefs: FrozenSet[str]) -> List[int]:
    """Argument indices of ``p`` that can host a preferred primitive at exactly ``depth - 1`` levels."""
    out = []
    for i, u in enumerate(p.arg_types):
        gt, lam = _arg_mode(u, in_lambda)
        if can_host(gt, depth - 1, lam, prefs):
            out.append(i)
    return out


def random_expression(rng: random.Random, target: T, depth: int, *, in_lambda: bool = False,
                      prefs: Optional[Set[str]] = None, domain: Optional[Sequence[Any]] = None,
                      host: bool = False, weights: Optional[Mapping[str, float]] = None) -> Any:
    """Random expression of type ``target`` with exactly ``depth`` levels (Node, or a literal at depth 0).

    ``prefs`` biases primitive choice towards those categories; with ``host=True`` the expression is additionally
    forced to contain at least one preferred primitive whenever that is feasible at this depth.  ``weights``
    (primitive name -> relative weight, default 1.0) re-weights every primitive choice; ``None`` keeps the uniform
    choice.  Returns ``None`` when no expression exists.
    """
    if depth == 0:
        if target is T.GRID:
            return INPUT
        if target is T.OBJECT and in_lambda:
            return OBJ
        if target in LITERAL_TYPES:
            return random_literal(rng, target, domain)
        return None
    cands = _candidates(target, depth, in_lambda)
    if not cands:
        return None
    fprefs: FrozenSet[str] = frozenset(prefs or ())
    if host and fprefs:
        hosting = [p for p in cands if p.category in fprefs or _host_args(p, depth, in_lambda, fprefs)]
        if hosting:
            cands = hosting
        else:
            host = False
    for _ in range(4):
        p = _pick_primitive(rng, cands, prefs, weights)
        need_host = host and p.category not in fprefs
        if need_host:
            deep = _host_args(p, depth, in_lambda, fprefs)
        else:
            deep = [i for i, u in enumerate(p.arg_types) if _arg_reachable(u, in_lambda, depth - 1)]
        j = rng.choice(deep)
        args: List[Any] = []
        ok = True
        for i, u in enumerate(p.arg_types):
            gt, lam = _arg_mode(u, in_lambda)
            d_i = depth - 1 if i == j else _choose_child_depth(rng, gt, depth - 1, lam)
            if d_i is None:
                ok = False
                break
            h_i = need_host and i == j
            if u is T.PROGRAM:
                val = _random_body(rng, d_i, prefs, host=h_i, weights=weights)
            else:
                val = random_expression(rng, gt, d_i, in_lambda=lam, prefs=prefs, domain=p.literal_args.get(i),
                                        host=h_i, weights=weights)
            if val is None:
                ok = False
                break
            args.append(val)
        if ok:
            return Node(p.name, tuple(args))
    return None


def _random_body(rng: random.Random, depth: int, prefs: Optional[Set[str]], host: bool = False,
                 weights: Optional[Mapping[str, float]] = None) -> Optional[Node]:
    """Random PROGRAM argument (OBJECT -> OBJECT lambda body over ``OBJ``) with exactly ``depth`` levels.

    With probability 0.15 (depth >= 2, no hosting constraint) the body is a ``COMPOSE`` of two smaller bodies, so
    every one of the 72 primitives is reachable by the sampler.  Bodies that ignore ``OBJ`` are re-drawn a few
    times.
    """
    if depth >= 2 and not host and rng.random() < 0.15:
        first = _random_body(rng, depth - 1, prefs, weights=weights)
        second = _random_body(rng, rng.randint(1, depth - 1), prefs, weights=weights)
        if first is not None and second is not None:
            return Node("COMPOSE", (first, second))
    val: Any = None
    for _ in range(4):
        val = random_expression(rng, T.OBJECT, depth, in_lambda=True, prefs=prefs, host=host, weights=weights)
        if isinstance(val, Node) and val.uses_obj():
            return val
    return val if isinstance(val, Node) else None


def _make_probe(h: int, w: int, rects: Sequence[Tuple[int, int, int, int, int]]) -> Grid:
    g = [[0] * w for _ in range(h)]
    for r0, c0, r1, c1, col in rects:
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                g[r][c] = col
    return g


def _probe_grids() -> List[Grid]:
    a = _make_probe(10, 10, [(1, 1, 3, 4, 2), (6, 2, 8, 3, 3), (2, 7, 5, 8, 5)])
    a[2][2] = 0  # a hole inside the first rectangle
    b = _make_probe(12, 12, [(0, 0, 1, 1, 1), (3, 4, 7, 8, 4), (9, 1, 10, 5, 6), (4, 10, 4, 11, 7)])
    b[4][5] = 0
    b[5][6] = 0
    return [a, b]


PROBE_GRIDS: List[Grid] = _probe_grids()


def _executes(node: Node) -> bool:
    for g in PROBE_GRIDS:
        try:
            execute(node, g, timeout_s=0.25)
            return True
        except ExecError:
            continue
    return False


def random_program(rng: random.Random, depth: Optional[int] = None, category: Optional[str] = None, *,
                   weights: Optional[Mapping[str, float]] = None) -> Node:
    """Sample a GRID-typed program with exactly ``depth`` levels, biased towards ``category``.

    ``depth=None`` draws the depth from the spec mix :data:`DEPTH_MIX`.  ``category`` is a spec sample category
    (object, geometry, relational, counting, contextual, adversarial) or a primitive category name; when a program
    of that depth can contain a primitive of the category, the sample is forced to contain one.  ``weights``
    re-weights primitive choices (see :func:`random_expression`).  Up to 16 candidates are drawn; the first that
    type-checks and executes on a probe grid is returned, else the first well-typed candidate.
    """
    if depth is None:
        depth = sample_depth(rng)
    if depth < 1:
        raise ValueError("depth must be >= 1")
    prefs: Optional[Set[str]] = None
    feasible = False
    if category is not None:
        prefs = set(CATEGORY_PRIMS.get(category, {category}))
        feasible = can_host(T.GRID, depth, False, frozenset(prefs))
    fallback: Optional[Node] = None
    for _ in range(16):
        node = random_expression(rng, T.GRID, depth, prefs=prefs, host=feasible, weights=weights)
        if node is None or not isinstance(node, Node):
            continue
        try:
            typecheck(node)
        except TypeError:  # pragma: no cover - construction is typed
            continue
        if feasible and not any(REGISTRY[o].category in prefs for o in node.primitives() if o in REGISTRY):
            fallback = fallback or node
            continue
        if _executes(node):
            return node
        fallback = fallback or node
    if fallback is not None:
        return fallback
    node = random_expression(rng, T.GRID, depth, weights=weights)
    return node if isinstance(node, Node) else Node("ROTATE90", (INPUT,))


# ======================================================================================= enumeration

class _Enumerator:
    def __init__(self, include_structural: bool) -> None:
        self.include_structural = include_structural
        self.memo: Dict[Tuple[T, int, bool], List[Any]] = {}

    def options(self, t: T, d: int, lam: bool, domain: Optional[Sequence[Any]]) -> List[Any]:
        if d == 0 and t in LITERAL_TYPES:
            return list(domain if domain is not None else ENUM_DEFAULT[t])
        return self.exprs(t, d, lam)

    def exprs(self, t: T, d: int, lam: bool) -> List[Any]:
        key = (t, d, lam)
        if key in self.memo:
            return self.memo[key]
        out: List[Any] = []
        if d == 0:
            if t is T.GRID:
                out = [INPUT]
            elif t is T.OBJECT and lam:
                out = [OBJ]
            elif t in LITERAL_TYPES:
                out = list(ENUM_DEFAULT[t])
        else:
            for p in expansions(t, d, in_lambda=lam, include_structural=self.include_structural):
                for node in self.level(p, d, lam):
                    out.append(node)
                    if len(out) >= _MEMO_CAP:
                        break
                if len(out) >= _MEMO_CAP:
                    break
        self.memo[key] = out
        return out

    def level(self, p: Primitive, d: int, lam: bool) -> Iterator[Node]:
        """All applications of ``p`` whose deepest argument has exactly ``d - 1`` levels."""
        arity = len(p.arg_types)
        per_arg: List[Tuple[T, bool, Optional[Sequence[Any]]]] = []
        for i, u in enumerate(p.arg_types):
            gt, lam_i = _arg_mode(u, lam)
            per_arg.append((gt, lam_i, p.literal_args.get(i)))

        def opts(i: int, lo: int, hi: int) -> List[Any]:
            gt, lam_i, dom = per_arg[i]
            vals: List[Any] = []
            for k in range(lo, hi + 1):
                for v in self.options(gt, k, lam_i, dom):
                    if p.arg_types[i] is T.PROGRAM and isinstance(v, Node) and not v.uses_obj():
                        continue
                    vals.append(v)
            return vals

        for j in range(arity):
            lists = []
            for i in range(arity):
                if i < j:
                    lists.append(opts(i, 0, d - 2))
                elif i == j:
                    lists.append(opts(i, d - 1, d - 1))
                else:
                    lists.append(opts(i, 0, d - 1))
            if any(not lst for lst in lists):
                continue
            for combo in itertools.product(*lists):
                yield Node(p.name, tuple(combo))


def enumerate_programs(max_depth: int, max_count: int, *, include_structural: bool = True) -> Iterator[Node]:
    """Breadth-first (by depth) stream of canonical, de-duplicated GRID programs; at most ``max_count``."""
    if max_count <= 0:
        return
    seen: Set[str] = set()
    count = 0
    en = _Enumerator(include_structural)
    for d in range(1, max_depth + 1):
        gens = [en.level(p, d, False) for p in expansions(T.GRID, d, include_structural=include_structural)]
        active = gens
        while active:
            nxt = []
            for g in active:
                try:
                    node = next(g)
                except StopIteration:
                    continue
                nxt.append(g)
                canon = canonicalize(node)
                key = canon.to_str()
                if key in seen:
                    continue
                seen.add(key)
                yield canon
                count += 1
                if count >= max_count:
                    return
            active = nxt
