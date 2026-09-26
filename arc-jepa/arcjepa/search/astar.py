"""Bounded A* fallback search (FROZEN_SPEC "A* fallback", INTERFACES §6).

State = (AST, remaining slots).  Two kinds of states share one priority queue:

* **complete**: a GRID program (spine) with its cached demo values;
* **partial**: a wrapper template applied to a complete spine whose first ``k`` holes are filled and whose
  remaining slots are still open.

Expanding a complete state opens every wrapper template (one partial state per template / GRID-slot variant);
expanding a partial state fills its next slot with every type-constrained filler from the task's
:class:`~arcjepa.search.beam.ArgPool` (so the branching factor is one slot's domain, not the joint domain).
Filling the last slot yields a complete child (observationally-equivalent programs are merged).

Priority ``f = g + h`` with ``g = gamma * C(p)`` (complexity of the greedy completion, open slots taking their
first filler) and ``h = beta * L(p) - alpha * s_neural(p)`` (estimated demo error of that completion minus the
neural compatibility).  The search stops at ``max_nodes`` generated nodes, at the time budget, or (with
``early_stop``) once the expansion that produced the first exact program is finished.
"""
from __future__ import annotations

import heapq
import itertools
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from arcjepa.core.types import Grid, Pair
from arcjepa.dsl.ast import INPUT, Node

from .beam import TEMPLATES, ArgPool, PoolItem, SearchContext, Template
from .candidate import (ALPHA, BETA, GAMMA, Candidate, Prior, apply_prior, candidate_from_outputs, complexity,
                        sort_candidates)
from .verifier import execute_safe, outputs_key

__all__ = ["astar_search"]

log = logging.getLogger(__name__)


@dataclass
class _Complete:
    program: Node
    values: List[Grid]
    compl: int
    depth: int
    loss: float


@dataclass
class _Partial:
    spine: _Complete
    tpl: Template
    flags: Tuple[bool, ...]
    filled: Tuple[PoolItem, ...]
    lists: Tuple[Tuple[PoolItem, ...], ...]


def astar_search(pairs: Sequence[Pair], prior: Optional[Prior], max_nodes: int = 50_000, time_budget_s: float = 10.0,
                 *, alpha: float = ALPHA, beta: float = BETA, gamma: float = GAMMA, max_depth: int = 6,
                 pool: Optional[ArgPool] = None, seeds: Sequence[Node] = (), max_results: int = 32,
                 max_exact: int = 8, early_stop: bool = True, stats: Optional[Dict[str, Any]] = None
                 ) -> List[Candidate]:
    """Best-first search with ``f = g + h`` (see module docstring); returns candidates sorted by Score.

    Extra keyword arguments beyond the INTERFACES signature are optional (scoring weights, depth cap, a shared
    ``pool``, ``seeds`` added as start states, result caps, ``early_stop`` and a ``stats`` dict).
    """
    t0 = time.perf_counter()
    budget = max(0.0, float(time_budget_s))
    deadline = t0 + 0.95 * budget
    pairs = list(pairs)
    st: Dict[str, Any] = stats if stats is not None else {}
    st.update({"astar_nodes": 0, "astar_expansions": 0, "astar_timed_out": False})
    if not pairs:
        return []
    ctx = SearchContext(pairs, pool=pool, alpha=alpha, beta=beta, gamma=gamma, deadline=t0 + 0.3 * budget)
    seq = itertools.count()
    open_: List[Tuple[float, int, Any]] = []
    best_complete: List[Tuple[float, int, _Complete]] = []  # min-heap of the best non-exact complete states

    def f_value(loss: float, compl: int, neural: float = 0.0) -> float:
        return gamma * compl + beta * loss - alpha * neural

    def keep(state: _Complete, f: float) -> None:
        if state.loss == 0.0:
            return
        item = (-f, next(seq), state)
        if len(best_complete) < max_results:
            heapq.heappush(best_complete, item)
        elif item[0] > best_complete[0][0]:
            heapq.heapreplace(best_complete, item)

    def push_complete(prog: Node, values: List[Grid], compl: int, depth: int, neural: float = 0.0) -> None:
        h = outputs_key(values)
        loss = ctx.rank_loss(values)
        if loss == 0.0:
            ctx.record_exact(prog, "astar")
        if h in ctx.seen:
            return
        ctx.seen.add(h)
        state = _Complete(prog, values, compl, depth, loss)
        f = f_value(loss, compl, neural)
        keep(state, f)
        heapq.heappush(open_, (f, next(seq), state))
        st["astar_nodes"] += 1

    # start states: INPUT and the seeds
    starts: List[Node] = [INPUT] + [s for s in seeds if isinstance(s, Node)]
    for prog in starts:
        vals = []
        ok = True
        for g in ctx.inputs:
            v = execute_safe(prog, g)
            if v is None:
                ok = False
                break
            vals.append(v)
        if ok:
            push_complete(prog, vals, complexity(prog), prog.depth())

    n_exact0 = len(ctx.exact)
    try:
        while open_ and st["astar_nodes"] < max_nodes:
            if time.perf_counter() > deadline:
                st["astar_timed_out"] = True
                break
            _, _, state = heapq.heappop(open_)
            st["astar_expansions"] += 1
            children: List[Tuple[Any, float, float, int, Node]] = []  # (state, loss, compl) + completion
            if isinstance(state, _Complete):
                if state.depth >= max_depth:
                    continue
                for tpl in TEMPLATES:
                    for flags in tpl.grid_variants():
                        lists = []
                        for slot in tpl.holes(flags):
                            lst = tuple(it for it in ctx.pool.fillers(tpl.prim, slot) if it.depth < max_depth)
                            if not lst:
                                break
                            lists.append(lst)
                        else:
                            part = _Partial(state, tpl, flags, (), tuple(lists))
                            _child(ctx, part, children, push_complete, max_depth, deadline)
            else:
                k = len(state.filled)
                for item in state.lists[k]:
                    part = _Partial(state.spine, state.tpl, state.flags, state.filled + (item,), state.lists)
                    _child(ctx, part, children, push_complete, max_depth, deadline)
            if children:
                neural = [0.0] * len(children)
                if prior is not None:
                    neural = [float(x) for x in prior([c[4] for c in children])]
                for (part, loss, _, compl, _node), s in zip(children, neural):
                    heapq.heappush(open_, (f_value(loss, compl, s), next(seq), part))
                    st["astar_nodes"] += 1
            if early_stop and (len(ctx.exact) > n_exact0 or len(ctx.exact) >= max_exact):
                break
    except _Timeout:
        st["astar_timed_out"] = True

    results: List[Candidate] = list(ctx.exact.values())
    for _, _, s in sorted(best_complete, key=lambda x: -x[0]):
        results.append(candidate_from_outputs(s.program, s.values, pairs, source="astar", alpha=alpha, beta=beta,
                                              gamma=gamma, n_cells=ctx.n_cells))
    if prior is not None and time.perf_counter() < t0 + budget:
        results = apply_prior(results, prior, alpha, beta, gamma)
    st["astar_seconds"] = time.perf_counter() - t0
    return sort_candidates(results)


class _Timeout(Exception):
    pass


def _child(ctx: SearchContext, part: _Partial, children: List[Tuple[Any, float, float, int, Node]],
           push_complete: Any, max_depth: int, deadline: float) -> None:
    """Evaluate the greedy completion of ``part``; complete children go straight to the queue."""
    if time.perf_counter() > deadline:
        raise _Timeout
    n_holes = len(part.lists)
    combo = part.filled + tuple(lst[0] for lst in part.lists[len(part.filled):])
    values = ctx.apply(part.tpl, part.flags, part.spine.values, combo)
    if values is None:
        return
    compl = part.spine.compl + 1 + sum(it.cost for it in combo)
    depth = 1 + max([part.spine.depth] + [it.depth for it in combo])
    if depth > max_depth:
        return
    node = ctx.build(part.tpl, part.flags, part.spine.program, combo)
    if len(part.filled) == n_holes:  # every slot filled: a complete program
        push_complete(node, values, compl, depth)
        return
    loss = ctx.rank_loss(values)
    if loss == 0.0:
        ctx.record_exact(node, "astar")
    children.append((part, loss, 0.0, compl, node))
