"""Neural-guided, type-constrained beam search over DSL programs (FROZEN_SPEC "Search" stage 1, INTERFACES §6).

v1 realisation
--------------
* A beam state is a complete GRID program (the *spine*) together with its cached value on every demo input.  The
  search starts from ``Node("INPUT")`` (plus any seed programs and their GRID sub-programs).
* Expanding a state with a GRID-producing primitive ``p`` (a :class:`Template`) puts the spine into one or more
  GRID slots of ``p``; the remaining slots are holes, i.e. ``p(spine, ?, ?)`` is a *partial program*.  Holes are
  type-constrained and filled from the task's :class:`ArgPool`: literal domains of the primitive (colours
  restricted to the task palette) plus small computed expressions over ``INPUT`` such as
  ``(SELECT_LARGEST (GET_COMPONENTS4 INPUT))`` or ``(MOST_COMMON_COLOR INPUT)``.  Partial programs are completed
  for scoring by exhaustive enumeration when the joint hole domain is small (<= ``max_joint``) and greedily,
  hole by hole, keeping the ``greedy_keep`` best partial fillings otherwise.
* Values are computed incrementally from the cached spine values (one primitive call per demo); programs that are
  observationally equivalent on the demos are merged (the first, i.e. simplest, survives) which also drops no-op
  wrappers.
* Ranking uses ``Score = alpha * s_neural - beta * L - gamma * C(p)`` where ``L`` is a smooth version of L_demo
  (wrong pairs + mean per-pair loss, shape mismatches graded by colour-histogram overlap).  Returned candidates
  carry the exact spec quantities (see :mod:`arcjepa.search.candidate`).
* With a neural prior, only the ``top_primitives`` primitives per expansion (ranked by the prior on their greedy
  completions) are expanded, and the best ``4 * width`` expansions of a level are re-ranked with the prior before
  the beam is cut.  The uniform prior (``prior=None``) expands every primitive.
* Exact programs are verified with the real interpreter.  With ``early_stop`` the search stops once the beam
  state that produced the first exact program has been fully expanded (minimal depth + its siblings for
  diversity) or ``max_exact`` exact programs are known.
* A quarter of the beam is reserved for the best state of each distinct output-shape class so that shape-changing
  intermediate programs (crops, tilings, transposes) are not crowded out.
"""
from __future__ import annotations

import heapq
import itertools
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from arcjepa.core.types import MAX_SIDE, Grid, Pair
from arcjepa.dsl.ast import INPUT, OBJ, Node
from arcjepa.dsl.canonicalize import structural_signature
from arcjepa.dsl.grammar import ENUM_DEFAULT
from arcjepa.dsl.interpreter import evaluate, infer_types
from arcjepa.dsl.primitives import REGISTRY, ExecContext, ExecError, Primitive, pixel_map
from arcjepa.dsl.types import LITERAL_TYPES, POSITION_ANCHORS, UNIT_DIRECTIONS, Object, T

from .candidate import (ALPHA, BETA, GAMMA, Candidate, Prior, apply_prior, candidate_from_outputs, canonical_key,
                        complexity, make_candidate, sort_candidates)
from .verifier import TargetInfo, clipped_timeout, demo_outputs, outputs_key, total_cells

__all__ = ["beam_search", "ArgPool", "PoolItem", "Template", "TEMPLATES", "SearchContext", "SearchTimeout",
           "task_palette", "grid_ok"]

log = logging.getLogger(__name__)


class SearchTimeout(Exception):
    """Internal signal: the search deadline passed."""


# ============================================================================================ templates

@dataclass(frozen=True)
class Template:
    """A GRID-producing primitive seen as a wrapper: its GRID slots take the spine / INPUT, other slots are holes."""

    prim: Primitive
    grid_slots: Tuple[int, ...]
    hole_slots: Tuple[int, ...]

    @property
    def name(self) -> str:
        return self.prim.name

    def grid_variants(self) -> List[Tuple[bool, ...]]:
        """Assignments of the GRID slots to (spine=True / GRID-pool hole=False) with at least one spine slot."""
        k = len(self.grid_slots)
        if k == 1:
            return [(True,)]
        return [flags for flags in itertools.product((True, False), repeat=k) if any(flags)]

    def holes(self, flags: Sequence[bool]) -> Tuple[int, ...]:
        """Hole slots of one variant: the non-spine GRID slots followed by the non-GRID slots."""
        return tuple(s for s, f in zip(self.grid_slots, flags) if not f) + self.hole_slots


def _build_templates() -> List[Template]:
    out: List[Template] = []
    for p in REGISTRY.values():
        if p.out_type is not T.GRID or p.lazy or p.higher_order or T.PROGRAM in p.arg_types:
            continue
        gs = tuple(i for i, t in enumerate(p.arg_types) if t is T.GRID)
        if not gs:
            continue
        hs = tuple(i for i, t in enumerate(p.arg_types) if t is not T.GRID)
        out.append(Template(p, gs, hs))
    return out


#: Every wrapper template (registry order): all GRID-output primitives except IF (lazy, degenerate below depth 4).
TEMPLATES: List[Template] = _build_templates()


def grid_ok(v: Any) -> bool:
    """Cheap structural check of a primitive's GRID output (the interpreter re-validates final programs)."""
    if not isinstance(v, list) or not v:
        return False
    r0 = v[0]
    if not isinstance(r0, list):
        return False
    h, w = len(v), len(r0)
    if h > MAX_SIDE or w < 1 or w > MAX_SIDE:
        return False
    for r in v:
        if len(r) != w:
            return False
    return True


# ============================================================================================ argument pools

@dataclass(frozen=True)
class PoolItem:
    """A hole filler: a raw literal or an expression over INPUT, with its value on every demo input."""

    expr: Any
    values: Tuple[Any, ...]
    cost: int
    depth: int


def _freeze(v: Any) -> Any:
    if isinstance(v, Object):
        return ("O", frozenset(pixel_map(v).items()))
    if isinstance(v, list):
        return tuple(_freeze(x) for x in v)
    return v


def task_palette(pairs: Sequence[Pair], extra_grids: Sequence[Grid] = ()) -> Tuple[List[int], List[int]]:
    """``(palette, output_palette)``: colours of all demo grids (+ extras) and of the demo outputs, 0 included."""
    allc: Set[int] = {0}
    outc: Set[int] = {0}
    for p in pairs:
        for row in p.input:
            allc.update(row)
        for row in p.output:
            allc.update(row)
            outc.update(row)
    for g in extra_grids:
        for row in g:
            allc.update(row)
    return sorted(allc), sorted(outc)


def _n(op: str, *args: Any) -> Node:
    return Node(op, tuple(args))


class ArgPool:
    """Type-constrained hole fillers for one task, evaluated once on every demo input.

    ``items[t]`` holds computed expressions of type ``t`` (deduplicated by their demo values; values that fail on
    any demo are dropped); :meth:`fillers` adds the primitive's literal domain (colours restricted to the task
    palette) for literal-typed slots.
    """

    OS_BASE: Tuple[Node, ...] = (_n("GET_COMPONENTS4", INPUT), _n("GET_COMPONENTS8", INPUT), _n("SELECT_ALL", INPUT))

    def __init__(self, grids: Sequence[Grid], palette: Sequence[int], *, out_palette: Optional[Sequence[int]] = None,
                 rich: bool = True, deadline: Optional[float] = None) -> None:
        self.grids: List[Grid] = list(grids)
        self.n = len(self.grids)
        self.ctxs = [ExecContext.of(g) for g in self.grids]
        self.palette: List[int] = sorted(set(palette) | {0})
        self.out_palette: List[int] = sorted(set(out_palette if out_palette is not None else palette) | {0})
        self.items: Dict[T, List[PoolItem]] = {t: [] for t in T}
        self._seen: Dict[T, Set[Any]] = {t: set() for t in T}
        self._filler_cache: Dict[Tuple[str, int], List[PoolItem]] = {}
        self._deadline = deadline
        self._build(rich)

    @classmethod
    def for_pairs(cls, pairs: Sequence[Pair], *, rich: bool = True, deadline: Optional[float] = None,
                  extra_grids: Sequence[Grid] = ()) -> "ArgPool":
        """Pool over the demo inputs of ``pairs`` with the task palette."""
        pal, out_pal = task_palette(pairs, extra_grids)
        return cls([p.input for p in pairs], pal, out_palette=out_pal, rich=rich, deadline=deadline)

    # ------------------------------------------------------------------ building
    def _late(self) -> bool:
        return self._deadline is not None and time.perf_counter() > self._deadline

    def add(self, t: T, expr: Node) -> Optional[PoolItem]:
        """Evaluate ``expr`` on every demo input and add it under type ``t`` (None when failing / duplicate)."""
        if self._late():
            return None
        vals: List[Any] = []
        for g in self.grids:
            try:
                vals.append(evaluate(expr, g, timeout_s=clipped_timeout(0.05)))
            except ExecError:
                return None
            except Exception:  # pragma: no cover - defensive
                return None
        if t is T.MASK and not any(v for m in vals for row in m for v in row):
            return None  # all-False masks only produce no-ops
        if t is T.OBJECT_SET and not any(vals):
            return None
        key = _freeze(vals)
        if key in self._seen[t]:
            return None
        self._seen[t].add(key)
        item = PoolItem(expr, tuple(vals), complexity(expr), expr.depth())
        self.items[t].append(item)
        return item

    def _build(self, rich: bool) -> None:
        cc4, cc8, sall = self.OS_BASE
        # non-spine GRID slots (e.g. the pattern of PATTERN_FILL): INPUT and its D4 images
        self.items[T.GRID].append(PoolItem(INPUT, tuple(self.grids), 0, 0))
        self._seen[T.GRID].add(_freeze(list(self.grids)))
        for op in ("ROTATE90", "ROTATE180", "ROTATE270", "REFLECT_H", "REFLECT_V", "REFLECT_D1", "REFLECT_D2"):
            self.add(T.GRID, _n(op, INPUT))
        for b in self.OS_BASE:
            self.add(T.OBJECT_SET, b)
        for sel in ("SELECT_LARGEST", "SELECT_SMALLEST", "SELECT_UNIQUE", "SELECT_CENTER"):
            for b in self.OS_BASE:
                self.add(T.OBJECT, _n(sel, b))
        self.add(T.OBJECT, _n("MERGE", cc4))
        self.add(T.MASK, _n("SELECT_NONZERO", INPUT))
        for it in list(self.items[T.OBJECT]):
            self.add(T.MASK, _n("GET_BBOX", it.expr))
            self.add(T.MASK, _n("GET_HOLES", it.expr))
        for e in (_n("MOST_COMMON_COLOR", INPUT), _n("LEAST_COMMON_COLOR", INPUT), _n("ARGMAX_SIZE", cc4),
                  _n("ARGMIN_SIZE", cc4)):
            self.add(T.COLOR, e)
        for a in POSITION_ANCHORS:
            self.add(T.COLOR, _n("COLOR_BY_POSITION", INPUT, a))
        for b in self.OS_BASE:
            self.add(T.INTEGER, _n("COUNT_OBJECTS", b))
        if not rich:
            return
        self.add(T.OBJECT_SET, _n("SELECT_BORDER", cc4))
        for c in self.palette:
            if c:
                self.add(T.OBJECT_SET, _n("SELECT_COLOR", cc4, c))
        bodies: List[Node] = [_n("RECOLOR", OBJ, c) for c in self.out_palette if c]
        bodies += [_n("MOVE", OBJ, d) for d in UNIT_DIRECTIONS]
        bodies += [_n("GROW", OBJ), _n("SHRINK", OBJ)]
        for b in (cc4, cc8):
            for body in bodies:
                self.add(T.OBJECT_SET, _n("APPLY_TO_EACH", b, body))
        self._build_relational()

    #: FILTER relations used for the relational object sets of :meth:`_build_relational`.
    FILTER_RELATIONS: Tuple[str, ...] = ("LARGER", "SMALLER", "INSIDE", "CONTAINS", "TOUCHING", "SAME_COLOR",
                                         "SAME_SHAPE")

    def _build_relational(self) -> None:
        """Relational fillers (docs/SOLVE_RATE_AUDIT.md addition 6): for the largest / smallest cc4 / cc8 object
        ``a``, the sets ``FILTER(objects, rel, a)`` with their largest / smallest member and recoloured copies, the
        NEAREST / FARTHEST object to ``a``; then the colour and bbox of every object filler; and the AND-masks of
        INPUT with each of its D4 images (``SELECT_NONZERO (PATTERN_FILL INPUT (SELECT_NONZERO INPUT) (D INPUT))``).
        """
        cc4, cc8, _ = self.OS_BASE
        cols = [c for c in self.out_palette if c]
        for b in (cc4, cc8):
            for a in (_n("SELECT_LARGEST", b), _n("SELECT_SMALLEST", b)):
                for rel in self.FILTER_RELATIONS:
                    s = _n("FILTER", b, rel, a)
                    if self.add(T.OBJECT_SET, s) is not None:
                        self.add(T.OBJECT, _n("SELECT_LARGEST", s))
                        self.add(T.OBJECT, _n("SELECT_SMALLEST", s))
                        for c in cols:
                            self.add(T.OBJECT_SET, _n("APPLY_TO_EACH", s, _n("RECOLOR", OBJ, c)))
                for op in ("NEAREST", "FARTHEST"):
                    self.add(T.OBJECT, _n(op, b, a))
        for it in list(self.items[T.OBJECT]):
            self.add(T.COLOR, _n("ARGMAX_SIZE", _n("DUPLICATE", it.expr, (0, 0))))  # the object's colour
            self.add(T.MASK, _n("GET_BBOX", it.expr))
        nz = _n("SELECT_NONZERO", INPUT)
        for op in ("ROTATE90", "ROTATE180", "ROTATE270", "REFLECT_H", "REFLECT_V", "REFLECT_D1", "REFLECT_D2"):
            self.add(T.MASK, _n("SELECT_NONZERO", _n("PATTERN_FILL", INPUT, nz, _n(op, INPUT))))

    # ------------------------------------------------------------------ fillers
    def literal_items(self, prim: Primitive, i: int) -> List[PoolItem]:
        """Literal fillers of slot ``i`` of ``prim`` (task-conditioned domain)."""
        t = prim.arg_types[i]
        dom = prim.literal_args.get(i)
        lits: Sequence[Any] = dom if dom is not None else ENUM_DEFAULT.get(t, ())
        if t is T.COLOR:
            lits = [c for c in lits if c in self.palette]
        return [PoolItem(v, (v,) * self.n, 1, 0) for v in lits]

    def fillers(self, prim: Primitive, i: int) -> List[PoolItem]:
        """All fillers of slot ``i`` of ``prim``: literal domain (if literal-typed) + computed expressions."""
        key = (prim.name, i)
        got = self._filler_cache.get(key)
        if got is not None:
            return got
        t = prim.arg_types[i]
        out: List[PoolItem] = []
        if t in LITERAL_TYPES:
            out.extend(self.literal_items(prim, i))
            if t in (T.COLOR, T.INTEGER):
                out.extend(self.items[t])
        else:
            out.extend(self.items[t])
        self._filler_cache[key] = out
        return out

    def summary(self) -> Dict[str, int]:
        """Pool sizes per type (diagnostics)."""
        return {t.value: len(v) for t, v in self.items.items() if v}


# ============================================================================================ search context

@dataclass
class _State:
    program: Node
    values: List[Grid]
    loss: float
    compl: int
    depth: int
    neural: float = 0.0
    rank: float = 0.0


class SearchContext:
    """Shared per-task search state: demos, argument pool, verified exact programs, equivalence keys, counters."""

    def __init__(self, pairs: Sequence[Pair], *, pool: Optional[ArgPool] = None, alpha: float = ALPHA,
                 beta: float = BETA, gamma: float = GAMMA, deadline: Optional[float] = None) -> None:
        self.pairs: List[Pair] = list(pairs)
        self.inputs: List[Grid] = [p.input for p in self.pairs]
        self.targets: List[Grid] = [p.output for p in self.pairs]
        self.target_info: List[TargetInfo] = [TargetInfo(t) for t in self.targets]
        self.n = len(self.pairs)
        self.n_cells = total_cells(self.pairs)
        self.pool = pool if pool is not None else ArgPool.for_pairs(self.pairs, deadline=deadline)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.exact: Dict[str, Candidate] = {}
        self.seen: Set[int] = set()
        self.evals = 0

    # ------------------------------------------------------------------ losses
    def rank_loss(self, outs: Sequence[Grid]) -> float:
        """Smooth L_demo used for ranking: wrong pairs + mean per-pair loss (0 iff exact)."""
        wrong = 0
        tot = 0.0
        for out, info in zip(outs, self.target_info):
            v = info.loss(out)
            if v > 0.0:
                wrong += 1
                tot += v
        return wrong + tot / max(1, self.n)

    def rank(self, loss: float, compl: int, neural: float = 0.0) -> float:
        return self.alpha * neural - self.beta * loss - self.gamma * compl

    # ------------------------------------------------------------------ evaluation
    def apply(self, tpl: Template, flags: Sequence[bool], spine_vals: Sequence[Grid],
              combo: Sequence[PoolItem]) -> Optional[List[Grid]]:
        """Values of ``tpl(spine, holes=combo)`` on every demo (None if any demo fails)."""
        prim = tpl.prim
        fn = prim.fn
        arity = prim.arity
        holes = tpl.holes(flags) if len(tpl.grid_slots) > 1 else tpl.hole_slots
        self.evals += 1
        outs: List[Grid] = []
        for d in range(self.n):
            args: List[Any] = [None] * arity
            for slot, is_spine in zip(tpl.grid_slots, flags):
                if is_spine:
                    args[slot] = spine_vals[d]
            for slot, item in zip(holes, combo):
                args[slot] = item.values[d]
            try:
                out = fn(self.pool.ctxs[d], *args) if prim.needs_ctx else fn(*args)
            except Exception:
                return None
            if not grid_ok(out):
                return None
            outs.append(out)
        return outs

    @staticmethod
    def build(tpl: Template, flags: Sequence[bool], spine: Node, combo: Sequence[PoolItem]) -> Node:
        """AST of ``tpl`` applied to ``spine`` with the given hole fillers."""
        args: List[Any] = [None] * tpl.prim.arity
        for slot, is_spine in zip(tpl.grid_slots, flags):
            if is_spine:
                args[slot] = spine
        for slot, item in zip(tpl.holes(flags), combo):
            args[slot] = item.expr
        return Node(tpl.prim.name, tuple(args))

    def record_exact(self, prog: Node, source: str) -> Optional[Candidate]:
        """Verify ``prog`` with the real interpreter; keep it when exact.  Returns the new Candidate or None."""
        key = canonical_key(prog)
        if key in self.exact:
            return None
        c = make_candidate(prog, self.pairs, source=source, alpha=self.alpha, beta=self.beta, gamma=self.gamma,
                           n_cells=self.n_cells)
        if c.demo_err != 0:
            log.debug("fast path said exact but the interpreter disagrees: %s", prog.to_str())
            return None
        self.exact[key] = c
        return c

    def completions(self, tpl: Template, flags: Sequence[bool], spine_vals: Sequence[Grid], *, deadline: float,
                    max_joint: int = 400, greedy_keep: int = 4, max_item_depth: int = 99
                    ) -> Iterator[Tuple[Tuple[PoolItem, ...], List[Grid], float]]:
        """Complete the partial program ``tpl(spine, ?, ...)``; yields ``(fillers, values, rank_loss)``.

        Exhaustive over the joint hole domain when it has at most ``max_joint`` combinations, otherwise greedy
        hole-by-hole completion keeping the ``greedy_keep`` best partial fillings (unfilled holes take their
        first filler).  Raises :class:`SearchTimeout` at the deadline.
        """
        lists: List[List[PoolItem]] = []
        for slot in tpl.holes(flags):
            lst = [it for it in self.pool.fillers(tpl.prim, slot) if it.depth <= max_item_depth]
            if not lst:
                return
            lists.append(lst)
        if not lists:
            if time.perf_counter() > deadline:
                raise SearchTimeout
            outs = self.apply(tpl, flags, spine_vals, ())
            if outs is not None:
                yield (), outs, self.rank_loss(outs)
            return
        total = 1
        for lst in lists:
            total *= len(lst)
        if total <= max_joint:
            for combo in itertools.product(*lists):
                if time.perf_counter() > deadline:
                    raise SearchTimeout
                outs = self.apply(tpl, flags, spine_vals, combo)
                if outs is not None:
                    yield combo, outs, self.rank_loss(outs)
            return
        defaults = tuple(lst[0] for lst in lists)
        frontier: List[Tuple[PoolItem, ...]] = [()]
        for k, lst in enumerate(lists):
            scored: List[Tuple[float, int, Tuple[PoolItem, ...]]] = []
            for partial in frontier:
                for item in lst:
                    if time.perf_counter() > deadline:
                        raise SearchTimeout
                    combo = partial + (item,) + defaults[k + 1:]
                    outs = self.apply(tpl, flags, spine_vals, combo)
                    if outs is None:
                        continue
                    loss = self.rank_loss(outs)
                    yield combo, outs, loss
                    scored.append((loss, len(scored), partial + (item,)))
            scored.sort()
            frontier = [p for _, _, p in scored[:greedy_keep]]
            if not frontier:
                return


# ============================================================================================ beam search

def _grid_subprograms(prog: Node) -> List[Node]:
    """GRID-typed sub-programs of ``prog`` outside lambda bodies (root first)."""
    try:
        types = infer_types(prog)
    except TypeError:
        return []
    out = []
    for path, node in prog.iter_nodes():
        t, lam = types.get(path, (None, True))
        if t is T.GRID and not lam and node.args:
            out.append(node)
    return out


def _shape_class(values: Sequence[Grid]) -> Tuple[Tuple[int, int], ...]:
    return tuple((len(v), len(v[0])) for v in values)


def beam_search(task_pairs: Sequence[Pair], *, prior: Optional[Prior] = None, width: int = 32, max_depth: int = 6,
                top_primitives: int = 8, alpha: float = ALPHA, beta: float = BETA, gamma: float = GAMMA,
                time_budget_s: float = 5.0, seeds: Sequence[Node] = (), max_exact: int = 16,
                early_stop: bool = True, pool: Optional[ArgPool] = None, context: Optional[SearchContext] = None,
                max_joint: int = 400, greedy_keep: int = 4, stats: Optional[Dict[str, Any]] = None
                ) -> List[Candidate]:
    """Type-constrained beam search from ``Node("INPUT")``; returns candidates sorted by Score (best first).

    Exact programs (verified with the interpreter) come first because ``L_demo`` contains the wrong-pair count;
    the best non-exact programs of the search follow (useful for repair).  The call returns within
    ``time_budget_s`` (generation stops at 95 % of the budget; the rest is reserved for finalisation).  Extra
    keyword arguments beyond the INTERFACES signature are optional: ``max_exact``, ``early_stop``, a prebuilt
    ``pool`` / ``context`` (shared with other search stages), ``max_joint`` / ``greedy_keep`` (hole completion)
    and a ``stats`` dict filled in place.
    """
    t0 = time.perf_counter()
    budget = max(0.0, float(time_budget_s))
    gen_deadline = t0 + 0.95 * budget
    pairs = list(task_pairs)
    st: Dict[str, Any] = stats if stats is not None else {}
    st.update({"beam_expansions": 0, "beam_states": 0, "beam_levels": 0, "timed_out": False})
    if not pairs:
        return []
    ctx = context if context is not None else SearchContext(pairs, pool=pool, alpha=alpha, beta=beta, gamma=gamma,
                                                            deadline=t0 + 0.3 * budget)
    ctx.alpha, ctx.beta, ctx.gamma = alpha, beta, gamma
    evals0 = ctx.evals
    width = max(1, int(width))
    heap_cap = 4 * width

    # ---------------------------------------------------------------- initial states
    init: List[_State] = []
    init_programs: List[Node] = [INPUT]
    for s in seeds:
        init_programs.extend(_grid_subprograms(s) if isinstance(s, Node) else [])
    seen_prog: Set[str] = set()
    for prog in init_programs:
        if time.perf_counter() > gen_deadline or len(init) >= width:
            break
        key = prog.to_str()
        if key in seen_prog:
            continue
        seen_prog.add(key)
        outs = demo_outputs(prog, ctx.inputs)
        if any(o is None for o in outs):
            continue
        loss = ctx.rank_loss(outs)  # type: ignore[arg-type]
        if loss == 0.0:
            ctx.record_exact(prog, "seed" if prog is not INPUT else "beam")
        h = outputs_key(outs)  # type: ignore[arg-type]
        if h in ctx.seen and prog is not INPUT:
            continue
        ctx.seen.add(h)
        c = complexity(prog)
        init.append(_State(prog, outs, loss, c, prog.depth(), 0.0, ctx.rank(loss, c)))  # type: ignore[arg-type]
    beam = sorted(init, key=lambda s: -s.rank)
    best_nonexact: List[Tuple[float, int, _State]] = []  # min-heap of the best non-exact states seen
    seq = itertools.count()

    def keep_best(state: _State) -> None:
        if state.loss == 0.0:
            return
        item = (state.rank, next(seq), state)
        if len(best_nonexact) < width:
            heapq.heappush(best_nonexact, item)
        elif item[0] > best_nonexact[0][0]:
            heapq.heapreplace(best_nonexact, item)

    for s in beam:
        keep_best(s)

    # ---------------------------------------------------------------- levels
    stop = bool(ctx.exact) and early_stop and len(ctx.exact) >= max_exact
    try:
        for _level in range(max_depth):
            if stop or not beam:
                break
            st["beam_levels"] += 1
            heap: List[Tuple[float, int, Any]] = []
            n_exact_before = len(ctx.exact)
            for s in beam:
                if s.depth >= max_depth:
                    continue
                if time.perf_counter() > gen_deadline:
                    raise SearchTimeout
                st["beam_states"] += 1
                variants = _expansion_variants(ctx, s, prior, top_primitives)
                for tpl, flags in variants:
                    item_depth_cap = max_depth - 1
                    for combo, outs, loss in ctx.completions(tpl, flags, s.values, deadline=gen_deadline,
                                                             max_joint=max_joint, greedy_keep=greedy_keep,
                                                             max_item_depth=item_depth_cap):
                        if loss == 0.0 and len(ctx.exact) < 4 * max_exact:
                            ctx.record_exact(ctx.build(tpl, flags, s.program, combo), "beam")
                        h = outputs_key(outs)
                        if h in ctx.seen:
                            continue
                        ctx.seen.add(h)
                        compl = s.compl + 1 + sum(it.cost for it in combo)
                        depth = 1 + max([s.depth] + [it.depth for it in combo])
                        r = ctx.rank(loss, compl)
                        entry = (r, next(seq), (s, tpl, flags, combo, outs, loss, compl, depth))
                        if len(heap) < heap_cap:
                            heapq.heappush(heap, entry)
                        elif r > heap[0][0]:
                            heapq.heapreplace(heap, entry)
                if early_stop and (len(ctx.exact) > n_exact_before or len(ctx.exact) >= max_exact):
                    stop = True
                    break
            beam = _select_beam(ctx, heap, prior, width)
            for s in beam:
                keep_best(s)
    except SearchTimeout:
        st["timed_out"] = True

    # ---------------------------------------------------------------- finalisation
    st["beam_expansions"] = ctx.evals - evals0
    results: List[Candidate] = list(ctx.exact.values())
    for _, _, s in sorted(best_nonexact, key=lambda x: -x[0]):
        results.append(candidate_from_outputs(s.program, s.values, pairs, source="beam", alpha=alpha, beta=beta,
                                              gamma=gamma, n_cells=ctx.n_cells))
    if prior is not None and time.perf_counter() < t0 + budget:
        results = apply_prior(results, prior, alpha, beta, gamma)
    out = sort_candidates(results)
    st["beam_exact"] = sum(1 for c in out if c.demo_err == 0)
    st["beam_seconds"] = time.perf_counter() - t0
    return out


def _expansion_variants(ctx: SearchContext, s: _State, prior: Optional[Prior],
                        top_primitives: int) -> List[Tuple[Template, Tuple[bool, ...]]]:
    """(template, grid-slot flags) pairs to expand for state ``s`` (top primitives by the prior if given).

    Primitives are ranked by the prior's best score over their representatives (each hole takes its first filler).
    Primitives tied at the cut are ordered by their best representative's demo loss (then registry order), not by
    registry order alone: when the spine dominates the prior's view (every representative scores the same), the
    first registered primitives would otherwise always win the remaining slots.  The loss is computed only for
    that tied group, so a prior without ties costs no extra evaluation.

    Primitives in ``prior.unseen_ops`` (ops the model's vocabulary lacks, e.g. the DSL spec extensions under a
    package exported before them) are always expanded and never take one of the ``top_primitives`` slots: their
    ``<unk>`` score says nothing about them.
    """
    variants = [(tpl, flags) for tpl in TEMPLATES for flags in tpl.grid_variants()]
    if prior is None or top_primitives <= 0:
        return variants
    always = frozenset(getattr(prior, "unseen_ops", None) or ())
    reps: List[Node] = []
    idx: List[int] = []
    combos: List[Tuple[PoolItem, ...]] = []
    for i, (tpl, flags) in enumerate(variants):
        if tpl.name in always:
            continue
        combo = []
        ok = True
        for slot in tpl.holes(flags):
            f = ctx.pool.fillers(tpl.prim, slot)
            if not f:
                ok = False
                break
            combo.append(f[0])
        if ok:
            reps.append(ctx.build(tpl, flags, s.program, combo))
            idx.append(i)
            combos.append(tuple(combo))
    if not reps:
        return variants
    scores = prior(reps)
    best: Dict[str, float] = {}
    for i, sc in zip(idx, scores):
        name = variants[i][0].name
        best[name] = max(best.get(name, float("-inf")), float(sc))
    ranked = sorted(best, key=lambda n: -best[n])
    if len(ranked) > top_primitives:
        cut = best[ranked[top_primitives - 1]]
        above = [n for n in ranked if best[n] > cut]
        tied = [n for n in ranked if best[n] == cut]
        if len(above) + len(tied) > top_primitives:
            loss: Dict[str, float] = {}
            tied_set = set(tied)
            for i, sc, combo in zip(idx, scores, combos):
                tpl, flags = variants[i]
                if tpl.name not in tied_set or float(sc) != cut:
                    continue
                outs = ctx.apply(tpl, flags, s.values, combo)
                if outs is not None:
                    loss[tpl.name] = min(loss.get(tpl.name, float("inf")), ctx.rank_loss(outs))
            tied.sort(key=lambda n: loss.get(n, float("inf")))  # stable: registry order breaks remaining ties
            ranked = above + tied
    keep = set(ranked[:top_primitives]) | always
    return [(tpl, flags) for tpl, flags in variants if tpl.name in keep]


def _select_beam(ctx: SearchContext, heap: List[Tuple[float, int, Any]], prior: Optional[Prior],
                 width: int) -> List[_State]:
    """Turn the level's best expansions into the next beam (prior re-ranking + shape-class diversity)."""
    states: List[_State] = []
    for r, _, (s, tpl, flags, combo, outs, loss, compl, depth) in heap:
        prog = ctx.build(tpl, flags, s.program, combo)
        states.append(_State(prog, outs, loss, compl, depth, 0.0, r))
    if prior is not None and states:
        scores = prior([st.program for st in states])
        for st, sc in zip(states, scores):
            st.neural = float(sc)
            st.rank = ctx.rank(st.loss, st.compl, st.neural)
    states.sort(key=lambda st: (-st.rank, st.compl))
    chosen: List[_State] = []
    chosen_ids: Set[int] = set()
    classes: Set[Tuple[Tuple[int, int], ...]] = set()
    per_sig: Dict[str, int] = {}
    sig_cap = max(2, width // 8)
    n_div = max(1, width // 4)

    def take(st: _State) -> None:
        chosen.append(st)
        chosen_ids.add(id(st))
        sig = structural_signature(st.program)
        per_sig[sig] = per_sig.get(sig, 0) + 1

    # (1) the best state of each distinct output-shape class (up to a quarter of the beam)
    for st in states:
        if len(chosen) >= n_div:
            break
        cls = _shape_class(st.values)
        if cls not in classes:
            classes.add(cls)
            take(st)
    # (2) best-first, at most ``sig_cap`` states per operator skeleton (literal variants of one program)
    for st in states:
        if len(chosen) >= width:
            break
        if id(st) in chosen_ids or per_sig.get(structural_signature(st.program), 0) >= sig_cap:
            continue
        take(st)
    # (3) fill any remaining slots best-first
    for st in states:
        if len(chosen) >= width:
            break
        if id(st) not in chosen_ids:
            take(st)
    chosen.sort(key=lambda st: (-st.rank, st.compl))
    return chosen
