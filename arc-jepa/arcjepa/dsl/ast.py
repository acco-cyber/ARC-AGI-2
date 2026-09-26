"""S-expression AST for DSL programs (INTERFACES.md §1, ``ast.py``).

Grammar of the textual form (``to_str`` / ``from_str`` are exact inverses):

* ``INPUT`` / ``OBJ``            leaf nodes (task input grid; bound object inside a PROGRAM body)
* ``(OP arg1 arg2 ...)``          application of primitive ``OP``
* ``3`` / ``-1``                  int literal (COLOR / INTEGER)
* ``True`` / ``False``            bool literal
* ``(1 0)``                       position offset literal ``(dr, dc)`` (head token is an int, so it can never be
                                  confused with an application)
* ``center`` / ``LEFT_OF``        bare string literals (anchor names, relation names)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator, List, Set, Tuple, Union

from arcjepa.dsl.types import LEAF_INPUT, LEAF_OBJ

__all__ = ["Node", "Literal", "Arg", "INPUT", "OBJ", "is_node", "literal_to_str"]

Literal = Union[int, str, bool, Tuple[int, int]]
Arg = Union["Node", int, str, bool, Tuple[int, int]]

_INT_RE = re.compile(r"^-?\d+$")
_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")


def is_node(x: Any) -> bool:
    """True when ``x`` is an AST node (as opposed to a raw literal)."""
    return isinstance(x, Node)


def literal_to_str(v: Any) -> str:
    """Render a raw literal value in S-expression form."""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return v
    if isinstance(v, tuple) and len(v) == 2:
        return f"({v[0]} {v[1]})"
    raise ValueError(f"unsupported literal {v!r}")


@dataclass(frozen=True)
class Node:
    """An AST node: an operator name plus a tuple of arguments (sub-nodes or raw literals)."""

    op: str
    args: Tuple[Arg, ...] = ()

    # ------------------------------------------------------------------ structure
    def is_leaf(self) -> bool:
        return not self.args

    def children(self) -> List["Node"]:
        """Sub-nodes only (literals are skipped)."""
        return [a for a in self.args if isinstance(a, Node)]

    def depth(self) -> int:
        """Number of nested operator levels; leaves (``INPUT``/``OBJ``) have depth 0."""
        if not self.args:
            return 0
        return 1 + max((a.depth() for a in self.args if isinstance(a, Node)), default=0)

    def size(self) -> int:
        """Number of AST nodes (leaves included, literals excluded)."""
        return 1 + sum(a.size() for a in self.args if isinstance(a, Node))

    def primitives(self) -> Set[str]:
        """Set of operator names used (leaf symbols excluded)."""
        out: Set[str] = set()
        if self.op not in (LEAF_INPUT, LEAF_OBJ):
            out.add(self.op)
        for a in self.args:
            if isinstance(a, Node):
                out |= a.primitives()
        return out

    def uses_obj(self) -> bool:
        """True when the ``OBJ`` leaf occurs anywhere in this subtree."""
        if self.op == LEAF_OBJ:
            return True
        return any(a.uses_obj() for a in self.args if isinstance(a, Node))

    def iter_nodes(self) -> Iterator[Tuple[Tuple[int, ...], "Node"]]:
        """Pre-order iteration of ``(path, node)`` over every sub-node (root path is ``()``)."""
        stack: List[Tuple[Tuple[int, ...], Node]] = [((), self)]
        while stack:
            path, node = stack.pop()
            yield path, node
            for i in range(len(node.args) - 1, -1, -1):
                a = node.args[i]
                if isinstance(a, Node):
                    stack.append((path + (i,), a))

    def paths(self) -> List[Tuple[int, ...]]:
        """Pre-order list of paths (tuples of argument indices) to every sub-node, root first."""
        return [p for p, _ in self.iter_nodes()]

    def literal_paths(self) -> List[Tuple[int, ...]]:
        """Paths to every raw literal argument (the last index addresses the literal slot)."""
        out: List[Tuple[int, ...]] = []
        for path, node in self.iter_nodes():
            for i, a in enumerate(node.args):
                if not isinstance(a, Node):
                    out.append(path + (i,))
        return out

    def get(self, path: Tuple[int, ...]) -> Arg:
        """Return the node or literal addressed by ``path``."""
        cur: Arg = self
        for i in path:
            if not isinstance(cur, Node):
                raise IndexError(f"path {path} descends into a literal")
            cur = cur.args[i]
        return cur

    def replace(self, path: Tuple[int, ...], new: Arg) -> "Node":
        """Return a copy with the node (or literal) at ``path`` replaced by ``new``."""
        if not path:
            if not isinstance(new, Node):
                raise TypeError("root replacement must be a Node")
            return new
        i, rest = path[0], path[1:]
        child = self.args[i]
        if rest:
            if not isinstance(child, Node):
                raise IndexError(f"path {path} descends into a literal")
            new_child: Arg = child.replace(rest, new)
        else:
            new_child = new
        args = self.args[:i] + (new_child,) + self.args[i + 1:]
        return Node(self.op, args)

    # ------------------------------------------------------------------ text form
    def to_str(self) -> str:
        if not self.args:
            return self.op
        parts = [self.op]
        for a in self.args:
            parts.append(a.to_str() if isinstance(a, Node) else literal_to_str(a))
        return "(" + " ".join(parts) + ")"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.to_str()

    @staticmethod
    def from_str(s: str) -> "Node":
        """Parse the S-expression produced by :meth:`to_str` (exact inverse)."""
        tokens = _TOKEN_RE.findall(s)
        if not tokens:
            raise ValueError("empty program string")
        pos = 0

        def parse() -> Arg:
            nonlocal pos
            if pos >= len(tokens):
                raise ValueError("unexpected end of program string")
            tok = tokens[pos]
            pos += 1
            if tok == ")":
                raise ValueError("unexpected ')'")
            if tok != "(":
                return _atom(tok)
            if pos >= len(tokens):
                raise ValueError("unterminated '('")
            head = tokens[pos]
            if _INT_RE.match(head):  # position literal (dr dc)
                pos += 1
                if pos >= len(tokens) or not _INT_RE.match(tokens[pos]):
                    raise ValueError("position literal needs two ints")
                second = int(tokens[pos])
                pos += 1
                if pos >= len(tokens) or tokens[pos] != ")":
                    raise ValueError("position literal must close with ')'")
                pos += 1
                return (int(head), second)
            if head in ("(", ")"):
                raise ValueError("operator name expected after '('")
            pos += 1
            args: List[Arg] = []
            while True:
                if pos >= len(tokens):
                    raise ValueError("unterminated '('")
                if tokens[pos] == ")":
                    pos += 1
                    break
                args.append(parse())
            return Node(head, tuple(args))

        result = parse()
        if pos != len(tokens):
            raise ValueError("trailing tokens after program")
        if not isinstance(result, Node):
            raise ValueError("program must be a node, not a literal")
        return result


def _atom(tok: str) -> Arg:
    if tok in (LEAF_INPUT, LEAF_OBJ):
        return Node(tok)
    if tok == "True":
        return True
    if tok == "False":
        return False
    if _INT_RE.match(tok):
        return int(tok)
    return tok


INPUT = Node(LEAF_INPUT)
OBJ = Node(LEAF_OBJ)
