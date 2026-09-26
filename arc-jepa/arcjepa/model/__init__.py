"""ARC-JEPA neural model: encoders, EMA target, predictor, rule latent, program encoder, scorer, memory."""
from .arcjepa import ARCJEPA, EpisodeParts, encode_rows, split_episode_batch
from .cell_encoder import CellEncoder, neighbor_features
from .config import ModelConfig
from .jepa_encoder import GridJEPAEncoder, complete_grid_batch, make_grid_batch
from .losses import LossWeights, jepa_losses, variance_loss
from .memory import TransformationMemory
from .object_encoder import ObjectEncoder, ShapeCNN
from .predictor import TransformationPredictor
from .program_encoder import (SPEC_PRIMITIVE_NAMES, STRUCTURAL_OPS, ProgramEncoder, ProgramTokenizer, SimpleNode,
                              group_triples, max_nodes_for, parse_sexpr)
from .relation_encoder import RelationEncoder, select_pairs
from .rule_latent import RuleLatent
from .scorer import Scorer
from .target_encoder import EMATargetEncoder

__all__ = [
    "ARCJEPA", "EpisodeParts", "encode_rows", "split_episode_batch", "CellEncoder", "neighbor_features",
    "ModelConfig", "GridJEPAEncoder", "complete_grid_batch", "make_grid_batch", "LossWeights", "jepa_losses",
    "variance_loss", "TransformationMemory", "ObjectEncoder", "ShapeCNN", "TransformationPredictor",
    "SPEC_PRIMITIVE_NAMES", "STRUCTURAL_OPS", "ProgramEncoder", "ProgramTokenizer", "SimpleNode", "parse_sexpr",
    "group_triples", "max_nodes_for",
    "RelationEncoder", "select_pairs", "RuleLatent", "Scorer", "EMATargetEncoder",
]
