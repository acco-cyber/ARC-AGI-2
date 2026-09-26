"""Neural-guided symbolic program search for ARC-JEPA (FROZEN_SPEC "Search", INTERFACES.md §6).

Stages: type-constrained neural beam (:func:`beam_search`) -> exact verification (:mod:`.verifier`) -> local AST
repair (:func:`repair`) -> test-time rule-latent refinement (:func:`refine_rule_latent`) -> bounded A*
(:func:`astar_search`) and evolutionary (:func:`evolve`) fallbacks -> two diverse attempts (:func:`select_two`).
:func:`solve_task` runs the whole pipeline under a strict per-task wall-clock budget, with or without a model.
"""
from .verifier import (DEFAULT_TIMEOUT_S, TargetInfo, demo_error, demo_outputs, execute_safe, is_exact,
                       outputs_error, total_cells)
from .candidate import (ALPHA, BETA, GAMMA, Candidate, Prior, apply_prior, candidate_from_outputs, complexity,
                        dedup_candidates, demo_loss, make_candidate, merge_candidates, score_value, sort_candidates)
from .beam import TEMPLATES, ArgPool, PoolItem, SearchContext, Template, beam_search, task_palette
from .repair import diff_hints, local_edits, localise, repair
from .tta import refine_rule_latent, tta_loss
from .astar import astar_search
from .evolution import evolve
from .diversity import fallback_grids, predict_output_shape, select_two
from .memory_prior import MemoryPrior, NeuralPrior, episode_batch_for_task, rule_latent_for_task
from .solver import BUCKET_POLICY, SolveConfig, difficulty, difficulty_score, parse_task, solve_task

__all__ = [
    "DEFAULT_TIMEOUT_S", "TargetInfo", "demo_error", "demo_outputs", "execute_safe", "is_exact", "outputs_error",
    "total_cells",
    "ALPHA", "BETA", "GAMMA", "Candidate", "Prior", "apply_prior", "candidate_from_outputs", "complexity",
    "dedup_candidates", "demo_loss", "make_candidate", "merge_candidates", "score_value", "sort_candidates",
    "TEMPLATES", "ArgPool", "PoolItem", "SearchContext", "Template", "beam_search", "task_palette",
    "diff_hints", "local_edits", "localise", "repair",
    "refine_rule_latent", "tta_loss",
    "astar_search", "evolve",
    "fallback_grids", "predict_output_shape", "select_two",
    "MemoryPrior", "NeuralPrior", "episode_batch_for_task", "rule_latent_for_task",
    "BUCKET_POLICY", "SolveConfig", "difficulty", "difficulty_score", "parse_task", "solve_task",
]
