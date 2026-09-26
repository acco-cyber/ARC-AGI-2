"""Competition metric and task-level evaluation (INTERFACES §8).

The ARC Prize metric: for every task, the fraction of its test outputs for which ``attempt_1`` or ``attempt_2``
equals the truth exactly; the score is the mean of that fraction over tasks (a task missing from a submission
scores 0).

``evaluate(tasks, solver_fn)`` runs a solver over tasks with known test outputs and returns the metric plus a
per-family breakdown, search statistics, error analysis (near-miss cell fractions) and the spec's per-task
diagnostics. ``solver_fn(task)`` may return either the attempts list or ``(attempts, diagnostics)``, where an
attempt is ``(attempt_1, attempt_2)`` or ``{"attempt_1": g, "attempt_2": g}`` per test input -- exactly what
``arcjepa.search.solver.solve_task`` returns (use :func:`solver_from_config` to bind a model and config).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from arcjepa.core.types import Grid, Task

from .diagnostics import task_diagnostics, write_diagnostics
from .error_analysis import analyze_errors, error_record
from .search_stats import summarize_search

__all__ = ["score_task", "competition_score", "normalize_attempts", "attempts_to_submission", "evaluate",
           "solver_from_config", "replay_solver", "solutions_from_tasks"]

log = logging.getLogger(__name__)

AttemptPair = Tuple[Optional[Grid], Optional[Grid]]
SolverFn = Callable[[Task], Any]


def normalize_attempts(attempts: Any) -> List[AttemptPair]:
    """Attempts for a task as a list of ``(attempt_1, attempt_2)`` tuples (accepts tuples, lists or the
    submission's ``{"attempt_1", "attempt_2"}`` dicts; malformed entries become ``(None, None)``)."""
    out: List[AttemptPair] = []
    for a in attempts or []:
        if isinstance(a, Mapping):
            out.append((a.get("attempt_1"), a.get("attempt_2")))
        elif isinstance(a, (list, tuple)) and len(a) == 2 and all(x is None or isinstance(x, list) for x in a) \
                and not (a and isinstance(a[0], list) and a[0] and isinstance(a[0][0], int)):
            out.append((a[0], a[1]))
        else:
            out.append((None, None))
    return out


def _hit(attempt: AttemptPair, truth: Grid) -> bool:
    return bool(truth) and any(g is not None and g == truth for g in attempt)


def score_task(attempts: Any, truths: Sequence[Grid]) -> float:
    """Fraction of ``truths`` (one per test input) matched exactly by attempt_1 or attempt_2 of the same index."""
    if not truths:
        return 0.0
    att = normalize_attempts(attempts)
    hits = sum(1 for i, t in enumerate(truths) if i < len(att) and _hit(att[i], t))
    return hits / float(len(truths))


def competition_score(submission: Mapping[str, Any], solutions: Mapping[str, Sequence[Grid]]) -> float:
    """The competition metric: mean over the tasks of ``solutions`` (``{task_id: [truth per test input]}``) of
    :func:`score_task`; a task absent from ``submission`` scores 0."""
    if not solutions:
        return 0.0
    return sum(score_task(submission.get(tid, []), truths) for tid, truths in solutions.items()) / len(solutions)


def attempts_to_submission(attempts: Any) -> List[Dict[str, Grid]]:
    """Attempts of one task in the submission's list-of-dicts format."""
    return [{"attempt_1": a1, "attempt_2": a2} for a1, a2 in normalize_attempts(attempts)]


def solutions_from_tasks(tasks: Mapping[str, Task]) -> Dict[str, List[Grid]]:
    """``{task_id: [test outputs]}`` for the tasks whose test outputs are all known."""
    return {tid: [p.output for p in t.test] for tid, t in tasks.items() if t.test and all(p.output for p in t.test)}


def solver_from_config(model: Any = None, cfg: Any = None) -> SolverFn:
    """``solver_fn`` running :func:`arcjepa.search.solver.solve_task` with ``model`` (may be None) and ``cfg``."""
    from arcjepa.search.solver import solve_task

    def fn(task: Task) -> Any:
        return solve_task(task, model, cfg)

    return fn


def replay_solver(submission: Mapping[str, Any], diagnostics: Optional[Mapping[str, Mapping[str, Any]]] = None
                  ) -> SolverFn:
    """``solver_fn`` that returns precomputed attempts (e.g. from a parallel run) so :func:`evaluate` can score
    them; missing tasks yield no attempts."""
    diagnostics = diagnostics or {}

    def fn(task: Task) -> Any:
        return submission.get(task.task_id, []), dict(diagnostics.get(task.task_id, {}))

    return fn


def _family(task: Task, families: Optional[Mapping[str, str]]) -> str:
    if families and task.task_id in families:
        return str(families[task.task_id])
    try:
        from arcjepa.data.families import family_of

        return str(family_of(task))
    except Exception as exc:  # noqa: BLE001 - the breakdown is best-effort
        log.debug("family_of failed on %s: %s", task.task_id, exc)
        return "unknown"


def evaluate(tasks: Dict[str, Task], solver_fn: SolverFn, *, max_tasks: Optional[int] = None,
             families: Optional[Mapping[str, str]] = None, diagnostics_path: Optional[str] = None
             ) -> Dict[str, Any]:
    """Run ``solver_fn`` on every task (sorted by id, first ``max_tasks``) with known test outputs.

    Returns ``score`` (the competition metric over scored tasks), ``n_tasks``, ``n_test_outputs``,
    ``n_correct_outputs``, ``per_task`` scores, ``per_family`` ``{family: {n, score}}``, ``search_stats``,
    ``error_analysis``, ``diagnostics`` (per-task spec records), ``submission`` (the attempts in competition
    format), ``n_unscored`` (tasks without known outputs, solved but not scored), ``n_solver_errors`` and
    ``seconds``. A solver exception scores the task 0 and is recorded in its diagnostics. Writes the
    diagnostics JSON when ``diagnostics_path`` is given.
    """
    t_start = time.perf_counter()
    ids = sorted(tasks)
    if max_tasks is not None:
        ids = ids[: max(0, int(max_tasks))]
    per_task: Dict[str, float] = {}
    fam_scores: Dict[str, List[float]] = {}
    records: List[Dict[str, Any]] = []
    err_rows: List[Dict[str, Any]] = []
    submission: Dict[str, List[Dict[str, Grid]]] = {}
    n_out = n_ok = n_unscored = n_err = 0
    for tid in ids:
        task = tasks[tid]
        truths = [p.output for p in task.test]
        scored = bool(truths) and all(truths)
        t0 = time.perf_counter()
        error: Optional[str] = None
        diag: Mapping[str, Any] = {}
        try:
            res = solver_fn(task)
            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], Mapping):
                attempts, diag = res
            else:
                attempts = res
            att = normalize_attempts(attempts)
        except Exception as exc:  # noqa: BLE001 - one failing task must not stop the evaluation
            log.exception("solver failed on %s", tid)
            error, att = repr(exc), []
            n_err += 1
        ms = 1000.0 * (time.perf_counter() - t0)
        submission[tid] = [{"attempt_1": a1, "attempt_2": a2} for a1, a2 in att]
        fam = _family(task, families)
        if not scored:
            n_unscored += 1
            records.append(task_diagnostics(tid, [], diag, family=fam, inference_ms=diag.get("inference_ms") or ms,
                                            error=error))
            continue
        correct = [i < len(att) and _hit(att[i], t) for i, t in enumerate(truths)]
        s = sum(correct) / float(len(truths))
        per_task[tid] = s
        fam_scores.setdefault(fam, []).append(s)
        n_out += len(truths)
        n_ok += sum(correct)
        for i, t in enumerate(truths):
            a = att[i] if i < len(att) else (None, None)
            err_rows.append(error_record(tid, i, list(a), t))
        records.append(task_diagnostics(tid, correct, diag, family=fam,
                                        inference_ms=diag.get("inference_ms") or ms, error=error))
    score = sum(per_task.values()) / len(per_task) if per_task else 0.0
    result: Dict[str, Any] = {
        "score": score,
        "n_tasks": len(per_task),
        "n_test_outputs": n_out,
        "n_correct_outputs": n_ok,
        "n_unscored": n_unscored,
        "n_solver_errors": n_err,
        "per_task": per_task,
        "per_family": {f: {"n": len(v), "score": sum(v) / len(v)} for f, v in sorted(fam_scores.items())},
        "search_stats": summarize_search(records),
        "error_analysis": analyze_errors(err_rows),
        "diagnostics": records,
        "submission": submission,
        "seconds": round(time.perf_counter() - t_start, 3),
    }
    if diagnostics_path:
        summary = {k: v for k, v in result.items() if k not in ("diagnostics", "submission", "per_task")}
        write_diagnostics(records, diagnostics_path, summary)
    log.info("evaluated %d tasks: score %.4f (%d/%d outputs)", len(per_task), score, n_ok, n_out)
    return result
