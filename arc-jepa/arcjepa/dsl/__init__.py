"""Typed DSL for ARC-JEPA (72 spec primitives + structural helpers + spec extensions): AST, primitives,
interpreter, grammar, canonicalizer, mutations."""
from arcjepa.dsl.types import (BOOLEANS, COLORS, INTEGERS, LEAF_INPUT, LEAF_OBJ, POSITION_ANCHORS,
                               POSITION_OFFSETS, POSITIONS, RELATIONS, UNIT_DIRECTIONS, Object, T,
                               is_literal_of)
from arcjepa.dsl.ast import INPUT, OBJ, Node
from arcjepa.dsl.primitives import (CATEGORIES, EXTENSION_PRIMITIVES, REGISTRY, SPEC_PRIMITIVES,
                                    STRUCTURAL_PRIMITIVES, ExecContext, ExecError, Primitive, by_category,
                                    by_out_type, make_colored_object, make_object, pixel_map, same_signature)
from arcjepa.dsl.interpreter import DSLTypeError, evaluate, execute, infer_types, typecheck, value_type_ok
from arcjepa.dsl.grammar import (CATEGORY_PRIMS, DEPTH_MIX, can_host, enumerate_programs, expansions,
                                 random_expression, random_program, sample_depth)
from arcjepa.dsl.canonicalize import D4_TABLE, canonicalize, compose_geometric, structural_signature
from arcjepa.dsl.mutations import HARD_NEGATIVE_TYPES, crossover, hard_negatives, hard_negatives_typed, mutate

__all__ = [
    "T", "Object", "Node", "INPUT", "OBJ", "COLORS", "INTEGERS", "BOOLEANS", "POSITIONS", "POSITION_OFFSETS",
    "POSITION_ANCHORS", "RELATIONS", "UNIT_DIRECTIONS", "LEAF_INPUT", "LEAF_OBJ", "is_literal_of",
    "REGISTRY", "SPEC_PRIMITIVES", "STRUCTURAL_PRIMITIVES", "EXTENSION_PRIMITIVES", "CATEGORIES", "Primitive",
    "ExecContext", "ExecError",
    "by_out_type", "by_category", "same_signature", "make_object", "make_colored_object", "pixel_map",
    "DSLTypeError", "typecheck", "infer_types", "execute", "evaluate", "value_type_ok",
    "expansions", "random_program", "random_expression", "enumerate_programs", "sample_depth", "can_host",
    "DEPTH_MIX", "CATEGORY_PRIMS",
    "canonicalize", "structural_signature", "compose_geometric", "D4_TABLE",
    "mutate", "hard_negatives", "hard_negatives_typed", "crossover", "HARD_NEGATIVE_TYPES",
]
