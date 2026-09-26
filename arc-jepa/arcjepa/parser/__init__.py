"""Multi-hypothesis object parser: objects, segmentation hypotheses, relations, parse()."""
from .hypotheses import DEFAULT_ORDER, MAX_OBJECTS, default_hypothesis, parse
from .objects import FEATURE_DIM, FEATURE_NAMES, N_FEATURES, Object
from .relations import CONTINUOUS, REL_DIM, RELATIONS, relation_features, relation_matrix
from .segmentation import HYPOTHESES, all_hypotheses, order_objects, segment

__all__ = [
    "CONTINUOUS",
    "DEFAULT_ORDER",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "HYPOTHESES",
    "MAX_OBJECTS",
    "N_FEATURES",
    "Object",
    "REL_DIM",
    "RELATIONS",
    "all_hypotheses",
    "default_hypothesis",
    "order_objects",
    "parse",
    "relation_features",
    "relation_matrix",
    "segment",
]
