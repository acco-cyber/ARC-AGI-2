"""Evaluation: competition metric, per-task diagnostics, search statistics and error analysis (INTERFACES §8)."""
from .diagnostics import SPEC_KEYS, jsonable, read_diagnostics, task_diagnostics, write_diagnostics
from .error_analysis import analyze_errors, cell_accuracy, error_record, near_miss
from .evaluate import (attempts_to_submission, competition_score, evaluate, normalize_attempts, replay_solver,
                       score_task, solutions_from_tasks, solver_from_config)
from .search_stats import accuracy_by, accuracy_vs_nodes, describe, retrieval_at_k, search_nodes, summarize_search

__all__ = [
    "SPEC_KEYS", "jsonable", "read_diagnostics", "task_diagnostics", "write_diagnostics",
    "analyze_errors", "cell_accuracy", "error_record", "near_miss",
    "attempts_to_submission", "competition_score", "evaluate", "normalize_attempts", "replay_solver", "score_task",
    "solutions_from_tasks", "solver_from_config",
    "accuracy_by", "accuracy_vs_nodes", "describe", "retrieval_at_k", "search_nodes", "summarize_search",
]
