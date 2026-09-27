"""Type-preserving program edits: mutation, the eight spec hard-negative types, crossover (INTERFACES.md §1).

Hard negatives (FROZEN_SPEC "Stage D"): wrong primitive / argument / order / colour / object / relation /
under-complete / over-complete.  Every returned program type-checks to the same output type as the source.
"""
from __future__ import annotations

import logging
import random
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from arcjepa.dsl.ast import Node
from arcjepa.dsl.canonicalize import canonicalize
from arcjepa.dsl.grammar import ENUM_DEFAULT, random_expression, random_literal, reachable
from arcjepa.dsl.interpreter import infer_types, typecheck
from arcjepa.dsl.primitives import REGISTRY, Primitive, same_signature
from arcjepa.dsl.types import LITERAL_TYPES, T

__all__ = ["mutate", "hard_negatives", "hard_negatives_typed", "crossover", "HARD_NEGATIVE_TYPES",
           "MUTATION_KINDS"]

log = logging.getLogger(__name__)

HARD_NEGATIVE_TYPES: Tuple[str, ...] = ("wrong_primitive", "wrong_argument", "wrong_order", "wrong_color",
                                        "wrong_object", "wrong_relation", "under_complete", "over_complete")

Path = Tuple[int, ...]
TypeTable = Dict[Path, Tuple[T, bool]]


# ======================================================================================= utilities

def _types(node: Node) -> Optional[TypeTable]:
    try:
        return infer_types(node)
    except TypeError:
        return None


def _out_type(node: Node) -> Optional[T]:
    try:
        return typecheck(node)
    except TypeError:
        return None


def _same_type(new: Optional[Node], root_t: Optional[T]) -> bool:
    return new is not None and root_t is not None and _out_type(new) is root_t


def _op_paths(node: Node, types: TypeTable, pred: Optional[Callable[[Path, Node], bool]] = None) -> List[Path]:
    out = []
    for path, n in node.iter_nodes():
        if n.op not in REGISTRY:
            continue
        if pred is None or pred(path, n):
            out.append(path)
    return out


def _literal_slots(node: Node) -> List[Tuple[Path, int, T, Primitive]]:
    slots = []
    for path, n in node.iter_nodes():
        prim = REGISTRY.get(n.op)
        if prim is None:
            continue
        for i, a in enumerate(n.args):
            if not isinstance(a, Node):
                slots.append((path, i, prim.arg_types[i], prim))
    return slots


def _domain(prim: Primitive, i: int, t: T) -> Sequence:
    dom = prim.literal_args.get(i)
    return dom if dom is not None else ENUM_DEFAULT[t]


def _change_literal(rng: random.Random, node: Node, slot: Tuple[Path, int, T, Primitive]) -> Optional[Node]:
    path, i, t, prim = slot
    cur = node.get(path + (i,))
    options = [v for v in _domain(prim, i, t) if not (type(v) is type(cur) and v == cur)]
    if not options:
        return None
    return node.replace(path + (i,), rng.choice(options))


def _fit_literals(q: Primitive, args: Tuple[object, ...]) -> Optional[Tuple[object, ...]]:
    """``args`` with every literal moved into ``q``'s literal domains: an out-of-domain integer becomes the nearest
    allowed integer (e.g. ``REPEAT_X(g, 1) -> UPSCALE(g, 2)``, whose factors are 2..5); ``None`` when another
    literal does not fit (the swap would not type-check)."""
    out = list(args)
    for i, a in enumerate(args):
        if isinstance(a, Node):
            continue
        dom = q.literal_args.get(i)
        if dom is None or any(type(v) is type(a) and v == a for v in dom):
            continue
        ints = [v for v in dom if isinstance(v, int) and not isinstance(v, bool)]
        if isinstance(a, int) and not isinstance(a, bool) and ints:
            out[i] = min(ints, key=lambda v: (abs(v - a), v))
            continue
        return None
    return tuple(out)


def _swap_op(rng: random.Random, node: Node, path: Path,
             pred: Optional[Callable[[Primitive], bool]] = None) -> Optional[Node]:
    n = node.get(path)
    if not isinstance(n, Node) or n.op not in REGISTRY:
        return None
    alts = [(q, fitted) for q in same_signature(REGISTRY[n.op]) if pred is None or pred(q)
            for fitted in [_fit_literals(q, n.args)] if fitted is not None]
    if not alts:
        return None
    q, fitted = rng.choice(alts)
    return node.replace(path, Node(q.name, fitted))


def _wrapper_candidates(t: T, category: Optional[str] = None) -> List[Primitive]:
    out = []
    for p in REGISTRY.values():
        if p.out_type is not t or t not in p.arg_types:
            continue
        if category is not None and p.category != category:
            continue
        others = [u for i, u in enumerate(p.arg_types) if i != p.arg_types.index(t)]
        if all(u in LITERAL_TYPES for u in others):
            out.append(p)
    return out


def _insert_wrapper(rng: random.Random, node: Node, types: TypeTable, category: Optional[str] = None,
                    type_filter: Optional[Set[T]] = None) -> Optional[Node]:
    paths = [p for p, (t, _) in types.items() if (type_filter is None or t in type_filter)]
    rng.shuffle(paths)
    for path in paths:
        t, _ = types[path]
        cands = _wrapper_candidates(t, category)
        if not cands:
            continue
        p = rng.choice(cands)
        k = p.arg_types.index(t)
        args: List[object] = []
        for i, u in enumerate(p.arg_types):
            if i == k:
                args.append(node.get(path))
            else:
                args.append(random_literal(rng, u, p.literal_args.get(i)))
        wrapped = Node(p.name, tuple(args))
        if canonicalize(wrapped) == canonicalize(node.get(path)):  # identity wrapper, e.g. MOVE (0 0)
            continue
        return node.replace(path, wrapped)
    return None


def _remove_wrapper(rng: random.Random, node: Node, types: TypeTable) -> Optional[Node]:
    cands: List[Tuple[Path, int]] = []
    for path, n in node.iter_nodes():
        if n.op not in REGISTRY:
            continue
        t = types[path][0]
        for i, a in enumerate(n.args):
            if isinstance(a, Node) and types.get(path + (i,), (None,))[0] is t:
                cands.append((path, i))
    if not cands:
        return None
    path, i = rng.choice(cands)
    return node.replace(path, node.get(path + (i,)))


def _replace_subtree(rng: random.Random, node: Node, types: TypeTable, type_filter: Optional[Set[T]] = None,
                     depth_delta: int = 0, allow_root: bool = True) -> Optional[Node]:
    paths = [p for p, (t, _) in types.items() if (type_filter is None or t in type_filter) and (allow_root or p)]
    if not paths:
        return None
    rng.shuffle(paths)
    for path in paths[:6]:
        t, lam = types[path]
        old = node.get(path)
        assert isinstance(old, Node)
        target_depth = max(0, old.depth() + depth_delta)
        opts = [d for d in range(max(0, target_depth - 1), target_depth + 2) if reachable(t, d, lam)]
        if not opts:
            continue
        new = random_expression(rng, t, rng.choice(opts), in_lambda=lam)
        if new is None or new == old:
            continue
        if not path and not isinstance(new, Node):
            continue
        return node.replace(path, new)
    return None


def _swap_order(rng: random.Random, node: Node, types: TypeTable) -> Optional[Node]:
    cands: List[Callable[[], Node]] = []
    for path, n in node.iter_nodes():
        if n.op not in REGISTRY:
            continue
        prim = REGISTRY[n.op]
        # (a) two sibling nodes / literals of the same declared type
        for i in range(len(n.args)):
            for j in range(i + 1, len(n.args)):
                if prim.arg_types[i] is prim.arg_types[j] and n.args[i] != n.args[j]:
                    def swap(path=path, n=n, i=i, j=j) -> Node:
                        args = list(n.args)
                        args[i], args[j] = args[j], args[i]
                        return node.replace(path, Node(n.op, tuple(args)))
                    cands.append(swap)
        # (c) swap the order of two nested same-type wrappers: n(m(x, ..), ..) -> m(n(x, ..), ..)
        t = types[path][0]
        for k, m in enumerate(n.args):
            if not isinstance(m, Node) or m.op not in REGISTRY or types[path + (k,)][0] is not t:
                continue
            for k2, x in enumerate(m.args):
                if isinstance(x, Node) and types[path + (k, k2)][0] is t:
                    def rotate(path=path, n=n, m=m, k=k, k2=k2, x=x) -> Node:
                        inner = Node(n.op, n.args[:k] + (x,) + n.args[k + 1:])
                        outer = Node(m.op, m.args[:k2] + (inner,) + m.args[k2 + 1:])
                        return node.replace(path, outer)
                    cands.append(rotate)
    if not cands:
        return None
    return rng.choice(cands)()


# ======================================================================================= hard-negative types

def _hn_wrong_primitive(rng: random.Random, node: Node, types: TypeTable) -> Optional[Node]:
    preferred = _op_paths(node, types, lambda p, n: types[p][0] not in (T.OBJECT, T.OBJECT_SET))
    rest = [p for p in _op_paths(node, types) if p not in preferred]
    rng.shuffle(preferred)
    rng.shuffle(rest)
    for path in preferred + rest:
        new = _swap_op(rng, node, path)
        if new is not None:
            return new
    return None


def _hn_wrong_argument(rng: random.Random, node: Node, types: TypeTable) -> Optional[Node]:
    slots = [s for s in _literal_slots(node) if s[2] is not T.COLOR and s[2] is not T.RELATION]
    rng.shuffle(slots)
    for s in slots:
        new = _change_literal(rng, node, s)
        if new is not None:
            return new
    return _replace_subtree(rng, node, types, allow_root=False)


def _hn_wrong_order(rng: random.Random, node: Node, types: TypeTable) -> Optional[Node]:
    return _swap_order(rng, node, types)


def _hn_wrong_color(rng: random.Random, node: Node, types: TypeTable) -> Optional[Node]:
    slots = [s for s in _literal_slots(node) if s[2] is T.COLOR]
    rng.shuffle(slots)
    for s in slots:
        new = _change_literal(rng, node, s)
        if new is not None:
            return new
    return _insert_wrapper(rng, node, types, category="color")


def _hn_wrong_object(rng: random.Random, node: Node, types: TypeTable) -> Optional[Node]:
    paths = _op_paths(node, types, lambda p, n: types[p][0] in (T.OBJECT, T.OBJECT_SET))
    rng.shuffle(paths)
    for path in paths:
        new = _swap_op(rng, node, path)
        if new is not None:
            return new
    return _replace_subtree(rng, node, types, type_filter={T.OBJECT, T.OBJECT_SET})


def _hn_wrong_relation(rng: random.Random, node: Node, types: TypeTable) -> Optional[Node]:
    slots = [s for s in _literal_slots(node) if s[2] is T.RELATION]
    paths = _op_paths(node, types, lambda p, n: REGISTRY[n.op].category == "relation")
    choices: List[Callable[[], Optional[Node]]] = []
    choices += [lambda s=s: _change_literal(rng, node, s) for s in slots]
    choices += [lambda p=p: _swap_op(rng, node, p) for p in paths]
    rng.shuffle(choices)
    for ch in choices:
        new = ch()
        if new is not None:
            return new
    new = _replace_subtree(rng, node, types, type_filter={T.BOOLEAN})
    if new is not None:
        return new
    return _inject_relation(rng, node, types)


def _inject_relation(rng: random.Random, node: Node, types: TypeTable) -> Optional[Node]:
    """Relation confusion for programs without relations: an object ``x`` becomes its NEAREST / FARTHEST
    neighbour, or an object set ``s`` is filtered by a random relation to its largest member."""
    paths = [p for p, (t, _) in types.items() if t in (T.OBJECT, T.OBJECT_SET) and isinstance(node.get(p), Node)]
    rng.shuffle(paths)
    for path in paths:
        sub = node.get(path)
        assert isinstance(sub, Node)
        if types[path][0] is T.OBJECT:
            if sub.op == "OBJ":
                continue
            new_sub = Node(rng.choice(("NEAREST", "FARTHEST")), (Node("GET_COMPONENTS4", (Node("INPUT"),)), sub))
        else:
            rel = rng.choice(REGISTRY["FILTER"].literal_args[1])
            new_sub = Node("FILTER", (sub, rel, Node("SELECT_LARGEST", (sub,))))
        return node.replace(path, new_sub)
    return None


def _hn_under_complete(rng: random.Random, node: Node, types: TypeTable) -> Optional[Node]:
    new = _remove_wrapper(rng, node, types)
    if new is not None:
        return new
    return _replace_subtree(rng, node, types, depth_delta=-1)


def _hn_over_complete(rng: random.Random, node: Node, types: TypeTable) -> Optional[Node]:
    return _insert_wrapper(rng, node, types)


_HN: Dict[str, Callable[[random.Random, Node, TypeTable], Optional[Node]]] = {
    "wrong_primitive": _hn_wrong_primitive, "wrong_argument": _hn_wrong_argument, "wrong_order": _hn_wrong_order,
    "wrong_color": _hn_wrong_color, "wrong_object": _hn_wrong_object, "wrong_relation": _hn_wrong_relation,
    "under_complete": _hn_under_complete, "over_complete": _hn_over_complete,
}


# ======================================================================================= public API

#: Edit kinds of :func:`mutate`, in its internal order.
MUTATION_KINDS: Tuple[str, ...] = ("literal", "swap_op", "subtree", "insert_wrapper", "remove_wrapper", "order")


def mutate(rng: random.Random, node: Node, kind: Optional[str] = None) -> Node:
    """One random type-preserving edit (literal change, op swap, subtree replacement, wrapper insert/remove,
    argument reordering).  Returns ``node`` unchanged only when no valid edit was found.

    ``kind`` (one of :data:`MUTATION_KINDS`) tries that edit kind first and falls back to the others in random
    order; ``None`` draws the order at random (every kind equally likely first)."""
    types = _types(node)
    root_t = _out_type(node)
    if types is None or root_t is None:
        return node
    ops: List[Callable[[], Optional[Node]]] = [
        lambda: (lambda slots: _change_literal(rng, node, rng.choice(slots)) if slots else None)(_literal_slots(node)),
        lambda: (lambda paths: _swap_op(rng, node, rng.choice(paths)) if paths else None)(_op_paths(node, types)),
        lambda: _replace_subtree(rng, node, types),
        lambda: _insert_wrapper(rng, node, types),
        lambda: _remove_wrapper(rng, node, types),
        lambda: _swap_order(rng, node, types),
    ]
    if kind is None:
        rng.shuffle(ops)
    else:
        first = ops.pop(MUTATION_KINDS.index(kind))
        rng.shuffle(ops)
        ops.insert(0, first)
    for op in ops:
        for _ in range(3):
            new = op()
            if new is not None and new != node and _same_type(new, root_t):
                return new
    return node


def hard_negatives_typed(rng: random.Random, node: Node, k: int = 8) -> List[Tuple[str, Node]]:
    """``(negative_type, program)`` pairs covering the eight spec types where applicable (canonically distinct)."""
    types = _types(node)
    root_t = _out_type(node)
    if types is None or root_t is None:
        return []
    seen: Set[str] = {canonicalize(node).to_str()}
    out: List[Tuple[str, Node]] = []
    for _round in range(3):
        for name in HARD_NEGATIVE_TYPES:
            if len(out) >= k:
                return out
            for _ in range(6):
                cand = _HN[name](rng, node, types)
                if cand is None or not _same_type(cand, root_t):
                    continue
                key = canonicalize(cand).to_str()
                if key in seen:
                    continue
                seen.add(key)
                out.append((name, cand))
                break
    tries = 0
    while len(out) < k and tries < 40:
        tries += 1
        cand = mutate(rng, node)
        key = canonicalize(cand).to_str()
        if key in seen:
            continue
        seen.add(key)
        out.append(("mutation", cand))
    return out[:k]


def hard_negatives(rng: random.Random, node: Node, k: int = 8) -> List[Node]:
    """Up to ``k`` hard-negative programs covering the 8 spec types (see :func:`hard_negatives_typed`)."""
    return [n for _, n in hard_negatives_typed(rng, node, k)]


def crossover(rng: random.Random, a: Node, b: Node) -> Node:
    """Replace a random subtree of ``a`` by a type-compatible subtree of ``b``; returns ``a`` if none fits."""
    ta, tb = _types(a), _types(b)
    root_t = _out_type(a)
    if ta is None or tb is None or root_t is None:
        return a
    paths_a = [p for p in ta if p] or [()]  # prefer proper subtrees over the root
    rng.shuffle(paths_a)
    for path_a in paths_a[:8]:
        t, lam = ta[path_a]
        cands = [p for p, (tt, _) in tb.items() if tt is t]
        rng.shuffle(cands)
        for path_b in cands[:8]:
            sub = b.get(path_b)
            if not isinstance(sub, Node) or sub == a.get(path_a):
                continue
            if not path_a and not sub.args:  # never collapse the whole program to a leaf
                continue
            if sub.uses_obj() and not lam:
                continue
            new = a.replace(path_a, sub)
            if _same_type(new, root_t):
                return new
    return a
