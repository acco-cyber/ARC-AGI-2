"""Canonical AST forms (INTERFACES.md §1, ``canonicalize.py``).

Rewrites applied bottom-up to a fixpoint:

* D4 composition table: nested rotations / reflections collapse into one op (``ROTATE90∘ROTATE90 → ROTATE180``)
  or vanish when they compose to the identity; ``TRANSPOSE`` is spelled ``REFLECT_D1``.
* identity removal: ``MOVE(x,(0,0))``, ``SHIFT(g,(0,0))``, ``MAP_COLOR(g,a,a)``, ``SWAP_COLORS(g,a,a)``,
  ``REPLACE_BACKGROUND(g,0)``, ``TILE(g,1,1)``, ``REPEAT_*(g,1)``, ``APPLY_TO_EACH(s, OBJ)``;
* idempotents / overrides: ``RECOLOR(RECOLOR(o,a),b) → RECOLOR(o,b)``, same for COLOR_OBJECT, FRAME, FILL,
  REPLACE_BACKGROUND, ALIGN, SELECT_COLOR; ``MOVE∘MOVE`` offsets add (when the sum stays a literal offset);
  (``PERIODIC_REPEAT`` is deliberately NOT collapsed: its period detection is not transitive through zeros, so a
  second application can fill more cells);
* commutative / mirrored relations: ``TOUCHING`` and ``OVERLAPPING`` arguments sorted; ``RIGHT_OF(a,b) →
  LEFT_OF(b,a)``, ``BELOW → ABOVE``, ``CONTAINS → INSIDE``; ``SWAP_COLORS`` colours sorted;
* conditionals: ``IF(True,a,b) → a``, ``IF(False,a,b) → b``, ``IF(c,a,a) → a``; ``COMPOSE`` bodies are inlined.
* spec extensions (INTERFACES.md §1 "Spec extensions"): ``UPSCALE / DOWNSCALE / DOWNSCALE_ANY (g, 1) → g``;
  ``DOWNSCALE(_ANY)(UPSCALE(g, k), k) → g``; ``UPSCALE(UPSCALE(g, a), b) → UPSCALE(g, a·b)`` when ``a·b`` is a
  literal of the op; ``UPSCALE(g, COUNT_OBJECTS(SELECT_ALL(g))) → UPSCALE_NC(g)``;
  ``FILL_EMPTY_LINES(FILL_EMPTY_LINES(g, c), d) → FILL_EMPTY_LINES(g, c)`` for a non-zero literal ``c`` (no empty
  line survives the first pass); and the D4-equivariant extensions (the scalings, KRON_SELF, BBOX_FILL,
  FILL_EMPTY_LINES) move a D4 argument outside, ``op(D(g), lits) → D(op(g, lits))``, so it can merge with other
  D4 ops.  The move is made only when every other argument is a literal, so it never increases the depth.
  CONNECT_SAME and the panel ops are not D4-equivariant (fill priority, panel order) and are left alone.

No rule increases a program's depth.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from arcjepa.dsl.ast import Node
from arcjepa.dsl.primitives import REGISTRY, substitute_obj
from arcjepa.dsl.types import LEAF_OBJ

__all__ = ["canonicalize", "structural_signature", "GEOMETRIC_OPS", "compose_geometric", "D4_TABLE"]

#: Unary GRID -> GRID symmetry ops (D4 group); TRANSPOSE is an alias of REFLECT_D1.
GEOMETRIC_OPS: Tuple[str, ...] = ("ROTATE90", "ROTATE180", "ROTATE270", "REFLECT_H", "REFLECT_V", "REFLECT_D1",
                                  "REFLECT_D2")
_ALIASES = {"TRANSPOSE": "REFLECT_D1"}


def _build_d4_table() -> Dict[Tuple[str, str], Optional[str]]:
    probe = [[1, 2, 3], [4, 5, 6]]
    images = {name: REGISTRY[name].fn(probe) for name in GEOMETRIC_OPS}
    table: Dict[Tuple[str, str], Optional[str]] = {}
    for outer in GEOMETRIC_OPS:
        for inner in GEOMETRIC_OPS:
            res = REGISTRY[outer].fn(images[inner])
            if res == probe:
                table[(outer, inner)] = None
                continue
            matches = [n for n, img in images.items() if img == res]
            assert len(matches) == 1, (outer, inner, matches)
            table[(outer, inner)] = matches[0]
    return table


#: ``D4_TABLE[(outer, inner)]`` = the single op equal to ``outer(inner(x))`` or ``None`` for the identity.
D4_TABLE: Dict[Tuple[str, str], Optional[str]] = _build_d4_table()


def compose_geometric(outer: str, inner: str) -> Optional[str]:
    """Name of the op equal to ``outer∘inner`` (``None`` = identity); aliases resolved."""
    return D4_TABLE[(_ALIASES.get(outer, outer), _ALIASES.get(inner, inner))]


# ======================================================================================= rewrite rules

def _is_lit(x: object) -> bool:
    return not isinstance(x, Node)


def _sort_key(n: Node) -> str:
    return n.to_str()


def _rule_alias(n: Node) -> Optional[Node]:
    if n.op in _ALIASES:
        return Node(_ALIASES[n.op], n.args)
    return None


def _rule_geometric(n: Node) -> Optional[Node]:
    if n.op not in GEOMETRIC_OPS:
        return None
    inner = n.args[0]
    if isinstance(inner, Node) and inner.op in GEOMETRIC_OPS:
        composed = compose_geometric(n.op, inner.op)
        return inner.args[0] if composed is None else Node(composed, inner.args)  # type: ignore[return-value]
    return None


def _rule_identity(n: Node) -> Optional[Node]:
    op, a = n.op, n.args
    if op in ("MOVE", "SHIFT") and a[1] == (0, 0):
        return a[0]  # type: ignore[return-value]
    if op in ("MAP_COLOR", "SWAP_COLORS") and _is_lit(a[1]) and _is_lit(a[2]) and a[1] == a[2]:
        return a[0]  # type: ignore[return-value]
    if op == "REPLACE_BACKGROUND" and a[1] == 0:
        return a[0]  # type: ignore[return-value]
    if op == "TILE" and a[1] == 1 and a[2] == 1:
        return a[0]  # type: ignore[return-value]
    if op in ("REPEAT_X", "REPEAT_Y", "REPEAT_N") and a[1] == 1:
        return a[0]  # type: ignore[return-value]
    if op == "APPLY_TO_EACH" and isinstance(a[1], Node) and a[1].op == LEAF_OBJ:
        return a[0]  # type: ignore[return-value]
    return None


def _rule_idempotent(n: Node) -> Optional[Node]:
    op, a = n.op, n.args
    inner = a[0] if a and isinstance(a[0], Node) else None
    if inner is None:
        return None
    if op == "RECOLOR" and inner.op == "RECOLOR":
        return Node(op, (inner.args[0], a[1]))
    if op == "COLOR_OBJECT" and inner.op == "COLOR_OBJECT" and inner.args[1] == a[1]:
        return Node(op, (inner.args[0], a[1], a[2]))
    if op == "FRAME" and inner.op == "FRAME":
        return Node(op, (inner.args[0], a[1]))
    if op == "FILL" and inner.op == "FILL" and inner.args[1] == a[1]:
        return Node(op, (inner.args[0], a[1], a[2]))
    if (op == "REPLACE_BACKGROUND" and inner.op == "REPLACE_BACKGROUND" and isinstance(inner.args[1], int)
            and inner.args[1] != 0):  # literal only: a computed colour may evaluate to 0
        return inner
    if op == "ALIGN" and inner.op == "ALIGN" and inner.args[1] == a[1]:
        return inner
    if op == "SELECT_COLOR" and inner.op == "SELECT_COLOR" and inner.args[1] == a[1]:
        return inner
    if op == "MOVE" and inner.op == "MOVE" and isinstance(a[1], tuple) and isinstance(inner.args[1], tuple):
        dr = a[1][0] + inner.args[1][0]
        dc = a[1][1] + inner.args[1][1]
        if -3 <= dr <= 3 and -3 <= dc <= 3:
            return Node(op, (inner.args[0], (dr, dc)))
    return None


_MIRRORED = {"RIGHT_OF": "LEFT_OF", "BELOW": "ABOVE", "CONTAINS": "INSIDE"}
_COMMUTATIVE = {"TOUCHING", "OVERLAPPING"}


def _rule_relations(n: Node) -> Optional[Node]:
    op, a = n.op, n.args
    if op in _MIRRORED:
        return Node(_MIRRORED[op], (a[1], a[0]))
    if op in _COMMUTATIVE and isinstance(a[0], Node) and isinstance(a[1], Node):
        if _sort_key(a[0]) > _sort_key(a[1]):
            return Node(op, (a[1], a[0]))
    if op == "SWAP_COLORS" and _is_lit(a[1]) and _is_lit(a[2]) and a[1] > a[2]:  # type: ignore[operator]
        return Node(op, (a[0], a[2], a[1]))
    return None


def _rule_conditional(n: Node) -> Optional[Node]:
    op, a = n.op, n.args
    if op == "IF":
        if a[0] is True:
            return a[1]  # type: ignore[return-value]
        if a[0] is False:
            return a[2]  # type: ignore[return-value]
        if a[1] == a[2]:
            return a[1]  # type: ignore[return-value]
    if op == "COMPOSE" and isinstance(a[0], Node) and isinstance(a[1], Node):
        return substitute_obj(a[1], a[0])
    return None


#: Extensions whose GRID argument (slot 0) commutes with every D4 op: op(D(g), ...) == D(op(g, ...)).
_D4_EQUIVARIANT = frozenset({"UPSCALE", "DOWNSCALE", "DOWNSCALE_ANY", "UPSCALE_NC", "KRON_SELF", "BBOX_FILL",
                             "FILL_EMPTY_LINES"})
_DOWNSCALES = frozenset({"DOWNSCALE", "DOWNSCALE_ANY"})


def _is_int_lit(x: object) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _rule_extensions(n: Node) -> Optional[Node]:
    op, a = n.op, n.args
    prim = REGISTRY.get(op)
    if prim is None or not prim.extension:
        return None
    inner = a[0] if a and isinstance(a[0], Node) else None
    if op in ("UPSCALE", "DOWNSCALE", "DOWNSCALE_ANY") and _is_int_lit(a[1]) and a[1] == 1:
        return a[0]  # type: ignore[return-value]
    if inner is None:
        return None
    if op in _DOWNSCALES and inner.op == "UPSCALE" and inner.args[1] == a[1]:
        return inner.args[0]  # type: ignore[return-value]  (equal k: a literal, or the same pure expression)
    if op == "UPSCALE" and inner.op == "UPSCALE" and _is_int_lit(a[1]) and _is_int_lit(inner.args[1]):
        k = a[1] * inner.args[1]  # type: ignore[operator]
        if k in REGISTRY["UPSCALE"].literal_args.get(1, ()):
            return Node("UPSCALE", (inner.args[0], k))
    if (op == "UPSCALE" and isinstance(a[1], Node) and a[1].op == "COUNT_OBJECTS" and isinstance(a[1].args[0], Node)
            and a[1].args[0].op == "SELECT_ALL" and a[1].args[0].args[0] == a[0]):
        return Node("UPSCALE_NC", (a[0],))
    if (op == "FILL_EMPTY_LINES" and inner.op == "FILL_EMPTY_LINES" and _is_int_lit(inner.args[1])
            and inner.args[1] != 0):
        return inner
    if op in _D4_EQUIVARIANT and inner.op in GEOMETRIC_OPS and all(_is_lit(x) for x in a[1:]):
        return Node(inner.op, (Node(op, (inner.args[0],) + a[1:]),))
    return None


_RULES: List[Callable[[Node], Optional[Node]]] = [_rule_alias, _rule_geometric, _rule_identity, _rule_idempotent,
                                                 _rule_relations, _rule_conditional, _rule_extensions]


def _canon(node: Node, budget: List[int]) -> Node:
    if not node.args:
        return node
    args = tuple(_canon(a, budget) if isinstance(a, Node) else a for a in node.args)
    node = Node(node.op, args)
    changed = True
    while changed and budget[0] > 0:
        changed = False
        for rule in _RULES:
            new = rule(node)
            if new is not None and new != node:
                budget[0] -= 1
                node = new
                changed = True
                break
    return node


def canonicalize(node: Node) -> Node:
    """Return the canonical form of ``node`` (see module docstring); pure and idempotent."""
    budget = [10_000]
    cur = node
    for _ in range(16):
        new = _canon(cur, budget)
        if new == cur:
            return new
        cur = new
    return cur


def structural_signature(node: Node) -> str:
    """Operator skeleton with every literal replaced by ``_`` (used for search diversity clustering)."""
    if not node.args:
        return node.op
    parts = [node.op]
    for a in node.args:
        parts.append(structural_signature(a) if isinstance(a, Node) else "_")
    return "(" + " ".join(parts) + ")"
