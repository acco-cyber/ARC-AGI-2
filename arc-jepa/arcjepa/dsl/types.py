"""DSL value types and literal domains (INTERFACES.md §1).

``T`` enumerates the ten value types of the typed DSL.  Literal domains are the values that may appear
directly inside an AST as raw Python values; computed values of the same type may range wider (e.g. an
INTEGER produced by ``GET_AREA`` can exceed 9, a POSITION produced by ``GET_CENTROID`` is an absolute
``(r, c)`` pair).
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Tuple

try:  # the real parser Object wins whenever the parser module exists
    from arcjepa.parser.objects import Object  # type: ignore
except ImportError:  # pragma: no cover - depends on which modules exist
    from arcjepa.dsl._objects_fallback import Object  # type: ignore

__all__ = [
    "T", "Object", "COLORS", "INTEGERS", "POSITION_OFFSETS", "POSITION_ANCHORS", "POSITIONS", "BOOLEANS",
    "RELATIONS", "UNIT_DIRECTIONS", "LITERAL_TYPES", "LEAF_INPUT", "LEAF_OBJ", "is_literal_of",
]


class T(str, Enum):
    """The ten DSL value types."""

    GRID = "GRID"
    OBJECT_SET = "OBJECT_SET"
    OBJECT = "OBJECT"
    MASK = "MASK"
    COLOR = "COLOR"
    POSITION = "POSITION"
    INTEGER = "INTEGER"
    BOOLEAN = "BOOLEAN"
    RELATION = "RELATION"
    PROGRAM = "PROGRAM"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


#: Leaf symbols of the AST.  ``INPUT`` is the task input grid (type GRID); ``OBJ`` is the bound object inside a
#: PROGRAM (lambda) body (type OBJECT).
LEAF_INPUT = "INPUT"
LEAF_OBJ = "OBJ"

COLORS: Tuple[int, ...] = tuple(range(10))
INTEGERS: Tuple[int, ...] = tuple(range(10))
POSITION_OFFSETS: Tuple[Tuple[int, int], ...] = tuple((dr, dc) for dr in range(-3, 4) for dc in range(-3, 4))
POSITION_ANCHORS: Tuple[str, ...] = ("center", "top", "bottom", "left", "right")
POSITIONS: Tuple[Any, ...] = POSITION_OFFSETS + POSITION_ANCHORS
UNIT_DIRECTIONS: Tuple[Tuple[int, int], ...] = tuple(
    (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
)
BOOLEANS: Tuple[bool, ...] = (False, True)
#: RELATION literals: the eight binary spatial relations of the spec plus five attribute relations used by FILTER.
RELATIONS: Tuple[str, ...] = (
    "LEFT_OF", "RIGHT_OF", "ABOVE", "BELOW", "TOUCHING", "OVERLAPPING", "INSIDE", "CONTAINS",
    "SAME_COLOR", "SAME_SHAPE", "SAME_SIZE", "LARGER", "SMALLER",
)

#: Types whose values may be written as raw literals inside an AST.
LITERAL_TYPES = frozenset({T.COLOR, T.INTEGER, T.POSITION, T.BOOLEAN, T.RELATION})


def is_literal_of(value: Any, t: T) -> bool:
    """Return True when ``value`` is a raw Python literal in the literal domain of type ``t``."""
    if t is T.BOOLEAN:
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    if t is T.COLOR:
        return isinstance(value, int) and 0 <= value <= 9
    if t is T.INTEGER:
        return isinstance(value, int) and 0 <= value <= 9
    if t is T.POSITION:
        if isinstance(value, str):
            return value in POSITION_ANCHORS
        return (isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) and not isinstance(v, bool)
                                                                      and -3 <= v <= 3 for v in value))
    if t is T.RELATION:
        return isinstance(value, str) and value in RELATIONS
    return False
