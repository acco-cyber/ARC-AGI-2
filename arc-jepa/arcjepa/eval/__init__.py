"""Evaluation: competition metric, per-task diagnostics, search statistics and error analysis (INTERFACES §8)."""
from .diagnostics import SPEC_KEYS, jsonable, read_diagnostics, task_diagnostics, write_diagnostics
from .error_analysis import analyze_errors, cell_accuracy, error_record, near_miss
from .evaluate import (attempts_to_submission, competition_score, evaluate, normalize_attempts, replay_solver,
                       score_task, solutions_from_tasks, solver_from_config)
from .search_stats import accuracy_by, accuracy_vs_nodes, describe, retrieval_at_k, search_nodes, summarize_search

#: names of :mod:`arcjepa.eval.hard180` exported lazily (PEP 562), so ``python -m arcjepa.eval.hard180`` does not
#: find the module pre-imported by its package
_HARD180_EXPORTS = ("FAILURE_CATEGORIES", "PER_TASK_FIELDS", "SPLIT_CHOICES", "cell_error", "failure_category",
                    "load_split_tasks", "regression_check", "summarize", "task_record")

__all__ = [
    "SPEC_KEYS", "jsonable", "read_diagnostics", "task_diagnostics", "write_diagnostics",
    "analyze_errors", "cell_accuracy", "error_record", "near_miss",
    "attempts_to_submission", "competition_score", "evaluate", "normalize_attempts", "replay_solver", "score_task",
    "solutions_from_tasks", "solver_from_config",
    "accuracy_by", "accuracy_vs_nodes", "describe", "retrieval_at_k", "search_nodes", "summarize_search",
    "run_hard180", *_HARD180_EXPORTS,
]


def __getattr__(name: str):  # PEP 562 lazy exports of the Hard-180 harness
    if name == "run_hard180" or name in _HARD180_EXPORTS:
        from . import hard180

        return hard180.run if name == "run_hard180" else getattr(hard180, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
