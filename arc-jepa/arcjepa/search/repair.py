"""Local AST repair guided by differing-cell analysis (FROZEN_SPEC "Search / Repair", INTERFACES §6).

For a candidate with ``E > 0``:

1. **Localise.**  Every sub-node outside lambda bodies is evaluated on the demo inputs and given a *footprint*:
   the cells a GRID node changes relative to its GRID child, the cells of an OBJECT / OBJECT_SET, the True cells
   of a MASK.  A node's localisation score is the fraction of the differing cells (over shape-matching pairs)
   its footprint covers; nodes without a footprint (colours, integers, lambda bodies) inherit their parent's
   score, and differing cells that no footprint covers are credited to the root (a missing outer step).
2. **Edit locally**, most responsible nodes first: literal changes (colours ordered by the colours the target
   wants at the differing cells, all offsets / anchors / counts), same-signature operator swaps with literal
   domains re-enumerated (e.g. MOVE -> ALIGN over all anchors), replacement of computed arguments by pool
   expressions of the same type, and for GRID nodes: removal (keep a GRID child), a MAP_COLOR wrap for the
   dominant (predicted -> expected) colour confusions, and a D4 wrap.  A few random type-preserving mutations
   (:func:`arcjepa.dsl.mutations.mutate`) close the list.
3. **Verify** every edit exactly with the interpreter; edits that lower L_demo are kept and are repaired again in
   the next round (``rounds`` rounds).

``repair`` returns the input candidates merged with every improving / exact edit, deduplicated and sorted.
"""
from __future__ import annotations

import logging
import random
import time
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

from arcjepa.core.types import Grid, Pair
from arcjepa.dsl.ast import Node
from arcjepa.dsl.grammar import ENUM_DEFAULT
from arcjepa.dsl.interpreter import evaluate, infer_types
from arcjepa.dsl.mutations import mutate
from arcjepa.dsl.primitives import REGISTRY, ExecError, Primitive, same_signature
from arcjepa.dsl.types import LITERAL_TYPES, Object, T

from .beam import ArgPool, task_palette
from .candidate import (ALPHA, BETA, GAMMA, Candidate, Prior, apply_prior, canonical_key, dedup_candidates,
                        demo_loss, make_candidate, sort_candidates)
from .verifier import execute_safe, total_cells

__all__ = ["repair", "localise", "local_edits", "diff_hints"]

log = logging.getLogger(__name__)

Path = Tuple[int, ...]
_D4 = ("ROTATE90", "ROTATE180", "ROTATE270", "REFLECT_H", "REFLECT_V", "REFLECT_D1", "REFLECT_D2")


# ============================================================================================ analysis

def _diff_cells(out: Optional[Grid], target: Grid) -> Optional[Set[Tuple[int, int]]]:
    """Differing cells when shapes match, else None."""
    if out is None or len(out) != len(target) or len(out[0]) != len(target[0]):
        return None
    return {(r, c) for r, (ro, rt) in enumerate(zip(out, target)) for c, (a, b) in enumerate(zip(ro, rt)) if a != b}


def diff_hints(prog: Node, pairs: Sequence[Pair]) -> Tuple[List[int], List[Tuple[int, int]]]:
    """``(colours wanted at differing cells, most common (predicted, expected) colour confusions)``."""
    want: Counter = Counter()
    conf: Counter = Counter()
    for p in pairs:
        out = execute_safe(prog, p.input)
        d = _diff_cells(out, p.output)
        if d is None or out is None:
            for row in p.output:
                want.update(row)
            continue
        for r, c in d:
            want[p.output[r][c]] += 1
            conf[(out[r][c], p.output[r][c])] += 1
    return [c for c, _ in want.most_common()], [m for m, _ in conf.most_common(4)]


def _footprint(v: Any, t: T) -> Optional[Set[Tuple[int, int]]]:
    if t is T.OBJECT and isinstance(v, Object):
        return set(v.cells)
    if t is T.OBJECT_SET and isinstance(v, list):
        out: Set[Tuple[int, int]] = set()
        for o in v:
            if isinstance(o, Object):
                out |= set(o.cells)
        return out
    if t is T.MASK and isinstance(v, list):
        return {(r, c) for r, row in enumerate(v) for c, b in enumerate(row) if b}
    return None


def _grid_change(v: Any, base: Any) -> Optional[Set[Tuple[int, int]]]:
    if not isinstance(v, list) or not v or not isinstance(v[0], list):
        return None
    if isinstance(base, list) and base and len(base) == len(v) and len(base[0]) == len(v[0]):
        return {(r, c) for r, (rv, rb) in enumerate(zip(v, base)) for c, (a, b) in enumerate(zip(rv, rb)) if a != b}
    return {(r, c) for r in range(len(v)) for c in range(len(v[0]))}


def localise(prog: Node, pairs: Sequence[Pair], *, timeout_s: float = 0.05) -> List[Tuple[Path, float]]:
    """Operator-node paths of ``prog`` ranked by how much of the demo error their footprint explains."""
    try:
        types = infer_types(prog)
    except TypeError:
        return []
    op_paths = [p for p, n in prog.iter_nodes() if n.op in REGISTRY]
    if not op_paths:
        return []
    diffs = [_diff_cells(execute_safe(prog, p.input), p.output) for p in pairs]
    n_diff = sum(len(d) for d in diffs if d)
    if n_diff == 0:  # exact, or every pair has the wrong shape: fall back to root-first order
        return [(p, 1.0 / (1 + len(p))) for p in op_paths]
    values: Dict[Path, List[Any]] = {}
    for path in op_paths:
        t, lam = types[path]
        if lam:
            continue
        node = prog.get(path)
        vals: List[Any] = []
        for p, d in zip(pairs, diffs):
            if d is None:
                vals.append(None)
                continue
            try:
                vals.append(evaluate(node, p.input, timeout_s=timeout_s))  # type: ignore[arg-type]
            except ExecError:
                vals.append(None)
        values[path] = vals
    score: Dict[Path, float] = {}
    covered: List[Set[Tuple[int, int]]] = [set() for _ in pairs]
    for path in op_paths:
        if path not in values:
            continue
        t, _ = types[path]
        node = prog.get(path)
        assert isinstance(node, Node)
        hit = 0
        for i, (p, d) in enumerate(zip(pairs, diffs)):
            if not d:
                continue
            v = values[path][i]
            if v is None:
                continue
            if t is T.GRID:
                child = next((path + (k,) for k, a in enumerate(node.args)
                              if isinstance(a, Node) and types.get(path + (k,), (None,))[0] is T.GRID), None)
                base = values.get(child, [None] * len(pairs))[i] if child is not None else p.input
                fp = _grid_change(v, base)
            else:
                fp = _footprint(v, t)
            if fp is None:
                continue
            inter = fp & d
            hit += len(inter)
            covered[i] |= inter
        if hit or t in (T.GRID, T.OBJECT, T.OBJECT_SET, T.MASK):
            score[path] = hit / float(n_diff)
    uncovered = sum(len(d - c) for d, c in zip(diffs, covered) if d) / float(n_diff)
    score[()] = score.get((), 0.0) + uncovered
    ranked: List[Tuple[Path, float]] = []
    for path in op_paths:
        s = score.get(path)
        if s is None:  # inherit the nearest scored ancestor
            q = path
            while q and q not in score:
                q = q[:-1]
            s = 0.95 * score.get(q, 0.0)
        ranked.append((path, s))
    ranked.sort(key=lambda x: (-x[1], len(x[0])))
    return ranked


# ============================================================================================ edits

def _domain(prim: Primitive, i: int) -> Sequence[Any]:
    t = prim.arg_types[i]
    dom = prim.literal_args.get(i)
    return dom if dom is not None else ENUM_DEFAULT.get(t, ())


def _ordered_literals(prim: Primitive, i: int, colour_hint: Sequence[int], palette: Sequence[int]) -> List[Any]:
    dom = list(_domain(prim, i))
    if prim.arg_types[i] is T.COLOR:
        first = [c for c in colour_hint if c in dom]
        rest = [c for c in palette if c in dom and c not in first]
        tail = [c for c in dom if c not in first and c not in rest]
        return first + rest + tail
    return dom


def _same(a: Any, b: Any) -> bool:
    return type(a) is type(b) and a == b


def local_edits(prog: Node, path: Path, types: Dict[Path, Tuple[T, bool]], pool: ArgPool,
                colour_hint: Sequence[int], confusions: Sequence[Tuple[int, int]]) -> Iterator[Node]:
    """Type-preserving single-node edits of the node at ``path`` (see module docstring)."""
    node = prog.get(path)
    if not isinstance(node, Node) or node.op not in REGISTRY:
        return
    prim = REGISTRY[node.op]
    t, lam = types.get(path, (prim.out_type, False))
    palette = pool.palette
    # 1. literal changes
    for i, a in enumerate(node.args):
        if isinstance(a, Node) or prim.arg_types[i] not in LITERAL_TYPES:
            continue
        for v in _ordered_literals(prim, i, colour_hint, palette):
            if not _same(v, a):
                yield prog.replace(path + (i,), v)
    # 2. same-signature operator swaps (literal domains re-enumerated when incompatible)
    for q in same_signature(prim):
        if q.lazy or q.higher_order:
            continue
        slots: List[List[Any]] = []
        for i, a in enumerate(node.args):
            if isinstance(a, Node):
                slots.append([a])
                continue
            dom = _domain(q, i)
            slots.append([a] if any(_same(a, v) for v in dom) else list(dom)[:49])
        n_combo = 1
        for s in slots:
            n_combo *= max(1, len(s))
        if n_combo > 64:
            continue
        for combo in _product(slots):
            yield prog.replace(path, Node(q.name, tuple(combo)))
    # 3. computed arguments replaced by pool expressions of the same type
    for i, a in enumerate(node.args):
        at = prim.arg_types[i]
        if not isinstance(a, Node) or at is T.PROGRAM:
            continue
        for item in pool.items.get(at, []):
            if item.expr != a and isinstance(item.expr, Node):
                yield prog.replace(path + (i,), item.expr)
        if at in LITERAL_TYPES:
            for v in _ordered_literals(prim, i, colour_hint, palette)[:10]:
                yield prog.replace(path + (i,), v)
    # 4. GRID nodes: remove / colour-fix wrap / D4 wrap
    if t is T.GRID and not lam:
        for k, a in enumerate(node.args):
            if isinstance(a, Node) and types.get(path + (k,), (None,))[0] is T.GRID:
                yield prog.replace(path, a)
        for src, dst in confusions:
            if src != dst:
                yield prog.replace(path, Node("MAP_COLOR", (node, int(src), int(dst))))
        for op in _D4:
            yield prog.replace(path, Node(op, (node,)))


def _product(slots: List[List[Any]]) -> Iterator[List[Any]]:
    if not slots:
        yield []
        return
    head, rest = slots[0], slots[1:]
    for v in head:
        for tail in _product(rest):
            yield [v] + tail


def _edits_for(prog: Node, pairs: Sequence[Pair], pool: ArgPool, rng: random.Random, *, n_random: int = 8,
               max_nodes: int = 12) -> Iterator[Node]:
    try:
        types = infer_types(prog)
    except TypeError:
        return
    colour_hint, confusions = diff_hints(prog, pairs)
    for path, _score in localise(prog, pairs)[:max_nodes]:
        yield from local_edits(prog, path, types, pool, colour_hint, confusions)
    for _ in range(n_random):
        yield mutate(rng, prog)


# ============================================================================================ main entry

def repair(cands: Sequence[Candidate], pairs: Sequence[Pair], rounds: int = 1, rng: Optional[random.Random] = None,
           prior: Optional[Prior] = None, *, alpha: float = ALPHA, beta: float = BETA, gamma: float = GAMMA,
           time_budget_s: float = 3.0, max_repair: int = 8, pool: Optional[ArgPool] = None,
           stop_on_exact: bool = True, stats: Optional[Dict[str, Any]] = None) -> List[Candidate]:
    """Repair the best non-exact candidates by localised AST edits (``rounds`` rounds, within the budget).

    Returns ``cands`` merged with every edit that lowered L_demo or became exact (deduplicated, sorted by
    Score).  ``prior`` (if given) supplies s_neural for the new candidates.  Extra keyword arguments are
    optional: ``time_budget_s``, ``max_repair`` candidates per round, a shared ``pool``, ``stop_on_exact`` (stop
    once an exact repair exists) and a ``stats`` dict filled in place.
    """
    t0 = time.perf_counter()
    deadline = t0 + max(0.0, float(time_budget_s))
    rng = rng if rng is not None else random.Random(0)
    pairs = list(pairs)
    st: Dict[str, Any] = stats if stats is not None else {}
    st.update({"repair_rounds": 0, "repair_edits": 0, "repair_exact": 0})
    if not pairs or not cands:
        return sort_candidates(cands)
    n_cells = total_cells(pairs)
    if pool is None:
        pal, out_pal = task_palette(pairs)
        pool = ArgPool([p.input for p in pairs], pal, out_palette=out_pal, deadline=t0 + 0.25 * time_budget_s)
    results: List[Candidate] = list(cands)
    seen: Set[str] = {c.key for c in cands}

    def base_loss(c: Candidate) -> float:
        return demo_loss(c.demo_err, c.cell_err, n_cells)

    work = [c for c in sort_candidates(cands) if c.demo_err > 0][:max_repair]
    found_exact = False
    for _round in range(max(0, int(rounds))):
        if not work or time.perf_counter() > deadline:
            break
        improved: List[Candidate] = []
        for c in work:
            if time.perf_counter() > deadline:
                break
            ref = base_loss(c)
            for edit in _edits_for(c.program, pairs, pool, rng):
                if time.perf_counter() > deadline:
                    break
                key = canonical_key(edit)
                if key in seen:
                    continue
                seen.add(key)
                st["repair_edits"] += 1
                nc = make_candidate(edit, pairs, source="repair", alpha=alpha, beta=beta, gamma=gamma, n_cells=n_cells)
                if nc.demo_err == 0:
                    results.append(nc)
                    st["repair_exact"] += 1
                    found_exact = True
                    break  # this candidate is fixed; move on
                if nc.loss < ref:
                    improved.append(nc)
                    results.append(nc)
            if found_exact and stop_on_exact:
                break
        st["repair_rounds"] += 1
        if found_exact and stop_on_exact:
            break
        work = sort_candidates(improved)[:max_repair]
    if prior is not None:
        results = apply_prior(results, prior, alpha, beta, gamma)
    st["repair_seconds"] = time.perf_counter() - t0
    return dedup_candidates(results)
