"""Hard-180 acceptance harness (ARC-JEPA v2 brief, step H).

    python -m arcjepa.eval.hard180 --model <package dir | none> --budget 60 [--split hard180|val|train670]
        [--workers 6] [--max-tasks N] [--out reports] [--tag name] [--baseline reports/<tag>_h180.json]

Solves every task of a frozen split (``data/splits_670_150_180.json``, :mod:`arcjepa.data.splits`) with
:func:`arcjepa.search.solver.solve_task` under a wall budget of ``--budget`` seconds per task, in ``--workers``
spawned processes (the model package is loaded once per worker; CPU unless CUDA is available), and writes

* ``<out>/<tag>_h180_per_task.jsonl`` (``<tag>_<split>_per_task.jsonl`` for the other splits): one JSON row per
  task, appended as soon as the task finishes. The run is **resumable**: task ids already in the file are skipped
  (the rows must come from the same budget / model / solver overrides, otherwise the run refuses unless
  ``--fresh``).
* ``<out>/<tag>_h180.json`` (``<tag>_<split>.json``): the summary, recomputed from every row of the per-task file.
  Without ``--tag`` the names are ``h180.json`` and ``h180_per_task.jsonl``.

**Headline = task-level pass@2**: a task passes only when EVERY test output is matched exactly by attempt 1 or
attempt 2. Also reported: output pass@2, exact-fit-program rate, near-90 %-cell rate, runtime, failure categories,
per-family / per-bucket tables, and for Hard-180 the clean-150 (v1 holdout) and from-old-train-30 subsets. With
``--baseline`` the summary carries the regression gate of the brief: a configuration is rejected when its task
passes fall by more than 2 against the baseline on the same tasks.

Per-task fields (``PER_TASK_FIELDS``): ``task_pass``, ``output_pass`` / ``output_total``, ``exact_program_found``
(a search candidate fits every demo exactly -- ``n_exact > 0`` -- or the induced property table fired),
``best_cell_error`` (min over attempts and test outputs of the fraction of wrong cells; wrong shape = 1.0),
``difficulty`` (the solver's D score; bucket in ``bucket``), ``family`` (:func:`arcjepa.data.families.family_of`),
``search_nodes`` (beam expansions + A* nodes + evolution generations x population, or the solver's own
``search_nodes`` when it reports one), ``beam_expansions``, ``repair_rounds``, ``tta_steps``, ``candidate_count``
(``n_candidates``), ``runtime`` (wall seconds around ``solve_task``) and ``failure_category``:

* ``solved`` -- task_pass;
* ``exact_fit_wrong_on_test`` -- an exact demo fit exists but the task is not solved;
* ``wrong_shape`` -- some unsolved test output where no attempt has the right shape;
* ``near_miss_90`` -- every unsolved output has an attempt with >= 90 % of its cells right;
* ``no_candidate_close`` -- everything else.

Counters are read from the solver's diagnostics only (search behaviour is untouched); a counter the solver did not
report is recorded as ``null`` and listed in the row's ``missing_counters`` and the summary's notes.

The public-evaluation tasks are never loaded: tasks come from the HF ``train``/``val``/``test`` files of the
mirror (the 1,000 official training tasks) through :func:`arcjepa.data.hf_loader.load_tasks`.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from arcjepa.core.types import Grid, Task, validate_grid

log = logging.getLogger(__name__)

__all__ = [
    "SPLIT_CHOICES", "FAILURE_CATEGORIES", "PER_TASK_FIELDS", "COUNTER_FIELDS", "NEAR_MISS_ERROR",
    "REGRESSION_TOLERANCE", "TARGET_TASK_PASS", "output_paths", "split_task_ids", "load_split_tasks",
    "cell_error", "failure_category", "task_record", "summarize", "regression_check", "read_rows", "run", "main",
]

#: ``--split`` values -> keys of the split document
SPLIT_CHOICES: Dict[str, str] = {"hard180": "hard180", "val": "val", "train670": "train"}
FAILURE_CATEGORIES: Tuple[str, ...] = ("solved", "wrong_shape", "exact_fit_wrong_on_test", "near_miss_90",
                                       "no_candidate_close")
#: every per-task row carries at least these keys
PER_TASK_FIELDS: Tuple[str, ...] = (
    "task_id", "split", "subset", "task_pass", "output_pass", "output_total", "exact_program_found",
    "best_cell_error", "difficulty", "family", "search_nodes", "beam_expansions", "repair_rounds", "tta_steps",
    "candidate_count", "runtime", "failure_category",
)
#: search counters taken from the solver diagnostics (``null`` when the solver did not report them)
COUNTER_FIELDS: Tuple[str, ...] = ("search_nodes", "beam_expansions", "repair_rounds", "tta_steps",
                                   "candidate_count")
#: a test output is a near miss when some attempt has at most this fraction of wrong cells
NEAR_MISS_ERROR = 0.10
#: the brief's regression rule: reject a configuration whose task passes fall by more than this
REGRESSION_TOLERANCE = 2
#: the brief's target on Hard-180 (100 / 180 = 55.6 % task-level pass@2)
TARGET_TASK_PASS = 100
#: task ids of the 1,000 official training tasks live in these HF task files (eval_public is never opened)
TRAINING_HF_SPLITS: Tuple[str, ...] = ("train", "val", "test")
HARNESS_VERSION = 1

# per-worker state (model loaded once per process)
_WORKER: Dict[str, Any] = {}


# ============================================================================================ split / tasks

def output_paths(out_dir: str, split: str, tag: Optional[str] = None) -> Tuple[Path, Path]:
    """``(summary json, per-task jsonl)``: ``<tag>_h180.json`` / ``<tag>_h180_per_task.jsonl`` for Hard-180,
    ``<tag>_<split>.json`` / ``<tag>_<split>_per_task.jsonl`` otherwise (no ``<tag>_`` prefix without a tag)."""
    label = "h180" if split == "hard180" else split
    prefix = f"{tag}_" if tag else ""
    base = Path(out_dir)
    return base / f"{prefix}{label}.json", base / f"{prefix}{label}_per_task.jsonl"


def _split_doc(split_file: Optional[str]) -> Dict[str, Any]:
    from arcjepa.data.splits import load_split

    if split_file:
        from arcjepa.data.hf_loader import resolve_split_file

        return load_split(resolve_split_file(split_file))
    return load_split()


def split_task_ids(split: str, split_file: Optional[str] = None) -> List[str]:
    """Sorted task ids of ``split`` (``hard180`` / ``val`` / ``train670``) from the frozen split document."""
    if split not in SPLIT_CHOICES:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLIT_CHOICES)}")
    return sorted(_split_doc(split_file)[SPLIT_CHOICES[split]])


def load_split_tasks(split: str, *, root: Optional[str] = None, split_file: Optional[str] = None,
                     max_tasks: Optional[int] = None) -> Dict[str, Task]:
    """``{task_id: Task}`` for the first ``max_tasks`` sorted ids of ``split`` (all when ``None``).

    Tasks are read from the HF ``train``/``val``/``test`` task files only (never ``eval_public``)."""
    from arcjepa.data.hf_loader import load_tasks

    ids = split_task_ids(split, split_file)
    if max_tasks is not None:
        ids = ids[: max(0, int(max_tasks))]
    tasks = load_tasks(root, splits=TRAINING_HF_SPLITS)
    missing = [t for t in ids if t not in tasks]
    if missing:
        raise KeyError(f"{len(missing)} {split} ids are not training tasks of the mirror, e.g. {missing[:5]}")
    return {t: tasks[t] for t in ids}


# ============================================================================================ scoring helpers

def cell_error(pred: Any, truth: Grid) -> float:
    """Fraction of wrong cells of ``pred`` against ``truth``; 1.0 when ``pred`` is invalid or has another shape."""
    if not validate_grid(pred) or not validate_grid(truth):
        return 1.0
    if len(pred) != len(truth) or len(pred[0]) != len(truth[0]):
        return 1.0
    n = len(truth) * len(truth[0])
    bad = sum(1 for rp, rt in zip(pred, truth) for a, b in zip(rp, rt) if a != b)
    return bad / float(n)


def _shape_ok(pred: Any, truth: Grid) -> bool:
    return bool(validate_grid(pred) and validate_grid(truth) and len(pred) == len(truth)
                and len(pred[0]) == len(truth[0]))


def failure_category(correct: Sequence[bool], output_errors: Sequence[float], shape_ok: Sequence[bool],
                     exact_found: bool, has_attempts: bool = True) -> str:
    """Category of one task (see the module docstring); ``output_errors[i]`` = min wrong-cell fraction of the
    attempts for test output ``i`` and ``shape_ok[i]`` = some attempt has the right shape."""
    if correct and all(correct):
        return "solved"
    if exact_found:
        return "exact_fit_wrong_on_test"
    unsolved = [i for i, c in enumerate(correct) if not c] or list(range(len(output_errors)))
    if has_attempts and any(not shape_ok[i] for i in unsolved):
        return "wrong_shape"
    if has_attempts and unsolved and all(output_errors[i] <= NEAR_MISS_ERROR for i in unsolved):
        return "near_miss_90"
    return "no_candidate_close"


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def _int_or_none(v: Any) -> Optional[int]:
    x = _num(v)
    return None if x is None else int(x)


def _search_nodes(diag: Mapping[str, Any], evo_pop: int) -> Optional[int]:
    """The solver's ``search_nodes`` when present, else beam expansions + A* nodes + evolution generations x
    population; ``None`` when the solver reported none of these counters."""
    own = _int_or_none(diag.get("search_nodes"))
    if own is not None:
        return own
    parts = [_num(diag.get(k)) for k in ("beam_expansions", "astar_nodes", "evo_generations")]
    if all(p is None for p in parts):
        return None
    beam, astar, gens = (p or 0.0 for p in parts)
    return int(beam + astar + gens * max(1, int(evo_pop)))


def task_record(task: Task, attempts: Sequence[Tuple[Any, Any]], diag: Mapping[str, Any], *, runtime: float,
                split: str, family: Optional[str] = None, subset: Optional[str] = None,
                hardness: Optional[float] = None, evo_pop: int = 32, run_key: Optional[Mapping[str, Any]] = None,
                error: Optional[str] = None, keep_attempts: bool = True) -> Dict[str, Any]:
    """One per-task row from ``solve_task``'s attempts and diagnostics (read-only use of the diagnostics)."""
    from arcjepa.eval.diagnostics import jsonable

    truths = [p.output for p in task.test]
    att = [tuple(a) for a in (attempts or [])]
    correct: List[bool] = []
    errs: List[float] = []
    shapes: List[bool] = []
    for i, t in enumerate(truths):
        pair = att[i] if i < len(att) else (None, None)
        correct.append(bool(t) and any(g is not None and g == t for g in pair))
        errs.append(min((cell_error(g, t) for g in pair), default=1.0))
        shapes.append(any(_shape_ok(g, t) for g in pair))
    n_exact = _int_or_none(diag.get("n_exact"))
    induced = diag.get("induced") is not None
    exact_search = bool(n_exact and n_exact > 0)
    exact_found = exact_search or induced
    source = "+".join(s for s, on in (("search", exact_search), ("induce", induced)) if on) or None
    counters: Dict[str, Any] = {
        "search_nodes": _search_nodes(diag, evo_pop),
        "beam_expansions": _int_or_none(diag.get("beam_expansions")),
        "repair_rounds": _int_or_none(diag.get("repair_rounds")),
        "tta_steps": _int_or_none(diag.get("tta_steps")),
        "candidate_count": _int_or_none(diag.get("n_candidates")),
    }
    missing = [k for k in COUNTER_FIELDS if counters[k] is None]
    task_pass = bool(truths) and all(correct)
    rec: Dict[str, Any] = {
        "task_id": task.task_id,
        "split": split,
        "subset": subset,
        "task_pass": task_pass,
        "output_pass": int(sum(correct)),
        "output_total": len(truths),
        "correct": correct,
        "exact_program_found": bool(exact_found),
        "exact_program_source": source,
        "n_exact": n_exact,
        "best_cell_error": round(min(errs), 6) if errs else 1.0,
        "cell_error_per_output": [round(e, 6) for e in errs],
        "near_90": bool(errs) and all(e <= NEAR_MISS_ERROR for e in errs),
        "difficulty": _num(diag.get("difficulty")),
        "bucket": _int_or_none(diag.get("bucket")),
        "hardness": hardness,
        "family": family,
        **counters,
        "astar_nodes": _int_or_none(diag.get("astar_nodes")),
        "evo_generations": _int_or_none(diag.get("evo_generations")),
        "candidate_rank": _int_or_none(diag.get("candidate_rank")),
        "runtime": round(float(runtime), 3),
        "inference_ms": _num(diag.get("inference_ms")),
        "budget_s": _num(diag.get("budget_s")),
        "failure_category": failure_category(correct, errs, shapes, exact_found, has_attempts=bool(att)),
        "best_program": diag.get("best_program"),
        "best_source": diag.get("best_source"),
        "program_depth": _int_or_none(diag.get("program_depth")),
        "induced": diag.get("induced"),
        "selection": diag.get("selection"),
        "stages_ms": diag.get("stages"),
        "missing_counters": missing,
        "error": error or diag.get("error") or diag.get("skipped"),
        "run": dict(run_key or {}),
    }
    if keep_attempts:
        rec["attempts"] = [[a[0], a[1]] for a in att]
    return jsonable(rec)


# ============================================================================================ summary

def _rate(k: int, n: int) -> float:
    return round(k / float(n), 6) if n else 0.0


def _table(rows: Sequence[Mapping[str, Any]], n: Optional[int] = None) -> Dict[str, Any]:
    """Counts of one group of rows (``n`` = denominator, default the number of rows)."""
    n = len(rows) if n is None else n
    tp = sum(1 for r in rows if r.get("task_pass"))
    ex = sum(1 for r in rows if r.get("exact_program_found"))
    nm = sum(1 for r in rows if r.get("near_90"))
    return {"n": n, "n_done": len(rows), "task_pass": tp, "task_pass_rate": _rate(tp, n),
            "output_pass": sum(int(r.get("output_pass") or 0) for r in rows),
            "output_total": sum(int(r.get("output_total") or 0) for r in rows),
            "exact_program": ex, "exact_program_rate": _rate(ex, n), "near_90": nm, "near_90_rate": _rate(nm, n)}


def _describe(vals: Sequence[Any]) -> Optional[Dict[str, float]]:
    from arcjepa.eval.search_stats import describe

    d = describe(list(vals))
    return None if d is None else {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def summarize(rows: Sequence[Mapping[str, Any]], *, split: str, task_ids: Sequence[str],
              meta: Optional[Mapping[str, Any]] = None, subsets: Optional[Mapping[str, Sequence[str]]] = None
              ) -> Dict[str, Any]:
    """Summary of the rows of ``task_ids`` (rates over ``len(task_ids)``; tasks without a row count as failed and
    are listed in ``incomplete``)."""
    ids = list(task_ids)
    want = set(ids)
    by_id = {str(r["task_id"]): r for r in rows if str(r.get("task_id")) in want}
    done = [by_id[t] for t in ids if t in by_id]
    n = len(ids)
    head = _table(done, n)
    out_total_known = head["output_total"]
    cats = {c: 0 for c in FAILURE_CATEGORIES}
    for r in done:
        cats[str(r.get("failure_category"))] = cats.get(str(r.get("failure_category")), 0) + 1
    fams: Dict[str, List[Mapping[str, Any]]] = {}
    buckets: Dict[str, List[Mapping[str, Any]]] = {}
    for r in done:
        fams.setdefault(str(r.get("family")), []).append(r)
        buckets.setdefault(str(r.get("bucket")), []).append(r)
    missing_counters = sorted({k for r in done for k in (r.get("missing_counters") or [])})
    always_null = [k for k in COUNTER_FIELDS if done and all(r.get(k) is None for r in done)]
    summary: Dict[str, Any] = {
        "harness": "arcjepa.eval.hard180",
        "harness_version": HARNESS_VERSION,
        **dict(meta or {}),
        "split": split,
        "n_tasks": n,
        "n_done": len(done),
        "complete": len(done) == n,
        "incomplete": [t for t in ids if t not in by_id],
        "headline": {
            "metric": "task-level pass@2 (every test output solved by attempt 1 or 2)",
            "task_pass": head["task_pass"], "n": n, "rate": head["task_pass_rate"],
            "text": f"{head['task_pass']}/{n} = {100.0 * head['task_pass_rate']:.1f}%",
        },
        "task_pass@2": {"count": head["task_pass"], "n": n, "rate": head["task_pass_rate"]},
        "output_pass@2": {"count": head["output_pass"], "total": out_total_known,
                          "rate": _rate(head["output_pass"], out_total_known)},
        "exact_program": {"count": head["exact_program"], "n": n, "rate": head["exact_program_rate"]},
        "near_90_cell": {"count": head["near_90"], "n": n, "rate": head["near_90_rate"],
                         "definition": "every test output has an attempt with <= 10% wrong cells (solved included)"},
        "runtime_s": _describe([r.get("runtime") for r in done]),
        "failure_categories": cats,
        "per_family": {f: _table(v) for f, v in sorted(fams.items())},
        "per_bucket": {b: _table(v) for b, v in sorted(buckets.items())},
        "search": {k: _describe([r.get(k) for r in done])
                   for k in ("search_nodes", "beam_expansions", "astar_nodes", "evo_generations", "repair_rounds",
                             "tta_steps", "candidate_count")},
        "n_errors": sum(1 for r in done if r.get("error")),
        "n_overrun": sum(1 for r in done if _num(r.get("budget_s")) and _num(r.get("runtime"))
                         and float(r["runtime"]) > 1.1 * float(r["budget_s"])),
        "missing_counters": missing_counters,
        "task_pass_ids": sorted(t for t in by_id if by_id[t].get("task_pass")),
        "exact_fit_wrong_on_test_ids": sorted(t for t in by_id
                                              if by_id[t].get("failure_category") == "exact_fit_wrong_on_test"),
        "near_miss_90_ids": sorted(t for t in by_id if by_id[t].get("failure_category") == "near_miss_90"),
    }
    if subsets:
        summary["subsets"] = {name: _table([by_id[t] for t in sub if t in by_id], len(sub))
                              for name, sub in subsets.items()}
    if split == "hard180" and n == 180:
        summary["target"] = {"task_pass": TARGET_TASK_PASS, "n": 180, "met": head["task_pass"] >= TARGET_TASK_PASS,
                             "gap": TARGET_TASK_PASS - head["task_pass"]}
    notes: List[str] = []
    if always_null:
        notes.append("counters not reported by the solver diagnostics (recorded as null): " + ", ".join(always_null))
    elif missing_counters:
        notes.append("counters missing on some tasks (null in those rows): " + ", ".join(missing_counters))
    notes.append("search_nodes = solver 'search_nodes' when reported, else beam_expansions + astar_nodes + "
                 "evo_generations x evo_pop (read-only from the diagnostics)")
    notes.append("exact_program_found = a search candidate fits every demo exactly (n_exact > 0) or the induced "
                 "property table fired")
    if not summary["complete"]:
        notes.append(f"{len(summary['incomplete'])} tasks have no row yet; they count as failed in every rate")
    summary["notes"] = notes
    return summary


def regression_check(rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]],
                     task_ids: Sequence[str], tolerance: int = REGRESSION_TOLERANCE) -> Dict[str, Any]:
    """The brief's regression rule on the common tasks: ``accepted`` unless task passes fall by more than
    ``tolerance`` against the baseline; lists the lost and gained task ids."""
    want = set(task_ids)
    cur = {str(r["task_id"]): bool(r.get("task_pass")) for r in rows if str(r.get("task_id")) in want}
    base = {str(r["task_id"]): bool(r.get("task_pass")) for r in baseline_rows if str(r.get("task_id")) in want}
    common = sorted(set(cur) & set(base))
    c = sum(1 for t in common if cur[t])
    b = sum(1 for t in common if base[t])
    return {"n_common": len(common), "task_pass": c, "baseline_task_pass": b, "delta": c - b,
            "tolerance": tolerance, "accepted": (c - b) >= -tolerance,
            "lost": [t for t in common if base[t] and not cur[t]],
            "gained": [t for t in common if cur[t] and not base[t]]}


# ============================================================================================ rows I/O

def read_rows(path: Path) -> List[Dict[str, Any]]:
    """Rows of a per-task jsonl (a truncated last line from an interrupted run is ignored)."""
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                log.warning("ignoring a malformed line in %s", path)
    return rows


def _append_row(path: Path, row: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except OSError:
            pass


def _write_json(path: Path, doc: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    os.replace(tmp, path)


# ============================================================================================ workers

def _solve_config(package_meta: Optional[Mapping[str, Any]], budget: float, overrides: Mapping[str, Any]) -> Any:
    """The SolveConfig of a run: the package's search / tta / repair settings (``config.json``) when a model is
    used, the solver defaults otherwise; then the ``--set`` overrides and the per-task budget."""
    from arcjepa.search.solver import SolveConfig

    kw = dict(overrides or {})
    kw["per_task_seconds"] = float(budget)
    return SolveConfig.from_spec_yaml(dict(package_meta), **kw) if package_meta is not None else SolveConfig(**kw)


def _init_worker(model_dir: Optional[str], device: Optional[str], threads: Optional[int],
                 overrides: Mapping[str, Any], budget: float, split: str,
                 subsets: Mapping[str, str], hardness: Mapping[str, float], run_key: Mapping[str, Any]) -> None:
    """Load the model package once for this process and build its SolveConfig (errors are kept, not raised, so a
    spawned pool does not respawn a failing initializer forever; the first task then raises them)."""
    _WORKER.clear()
    _WORKER.update({"split": split, "subsets": dict(subsets), "hardness": dict(hardness), "run_key": dict(run_key)})
    try:
        model = None
        meta: Dict[str, Any] = {}
        if model_dir:
            import torch

            from arcjepa.model.arcjepa import ARCJEPA

            if threads:
                torch.set_num_threads(int(threads))
            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            model = ARCJEPA.load_package(model_dir, device=dev)
            meta = dict(getattr(model, "package_meta", None) or {})
            _WORKER["device"] = dev
        cfg = _solve_config(meta if model is not None else None, budget, overrides)
        cfg.memory = getattr(model, "memory", None) if model is not None else None
        _WORKER["model"] = model
        _WORKER["cfg"] = cfg
    except Exception as exc:  # noqa: BLE001 - reported by the first task of this worker
        log.exception("worker initialisation failed")
        _WORKER["init_error"] = repr(exc)


def _solve_one(task: Task) -> Dict[str, Any]:
    """Solve one task in this worker and return its row (raises when the worker could not initialise)."""
    if "init_error" in _WORKER:
        raise RuntimeError(f"worker initialisation failed: {_WORKER['init_error']}")
    from arcjepa.data.families import family_of
    from arcjepa.search.solver import solve_task

    cfg = _WORKER["cfg"]
    model = _WORKER.get("model")
    t0 = time.perf_counter()
    error: Optional[str] = None
    try:
        attempts, diag = solve_task(task, model, cfg)
    except Exception as exc:  # noqa: BLE001 - solve_task never raises by contract; record it if it does
        log.exception("solve_task raised on %s", task.task_id)
        attempts, diag, error = [], {}, repr(exc)
    runtime = time.perf_counter() - t0
    try:
        fam = family_of(task)
    except Exception:  # noqa: BLE001
        fam = "unknown"
    row = task_record(task, attempts, diag, runtime=runtime, split=_WORKER["split"], family=fam,
                      subset=_WORKER["subsets"].get(task.task_id), hardness=_WORKER["hardness"].get(task.task_id),
                      evo_pop=int(getattr(cfg, "evo_pop", 32)), run_key=_WORKER["run_key"], error=error)
    row["pid"] = os.getpid()
    row["model_loaded"] = model is not None
    row["device"] = _WORKER.get("device", "cpu")
    return row


@contextlib.contextmanager
def _child_thread_env(threads: int) -> Iterator[None]:
    """Cap BLAS / OpenMP threads of the spawned workers (restored afterwards)."""
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    old = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.setdefault(k, str(max(1, int(threads))))
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ============================================================================================ run

def _parse_overrides(items: Sequence[str]) -> Dict[str, Any]:
    """``key=value`` SolveConfig overrides (values parsed as YAML: ``false``, ``3``, ``[1, 2]``)."""
    import yaml

    from arcjepa.search.solver import SolveConfig

    names = set(SolveConfig.__dataclass_fields__)
    out: Dict[str, Any] = {}
    for it in items or ():
        if "=" not in it:
            raise ValueError(f"solver override {it!r} is not key=value")
        k, v = it.split("=", 1)
        k = k.strip()
        if k not in names or k in ("memory", "per_task_seconds"):
            raise ValueError(f"unknown / reserved SolveConfig field {k!r}")
        val = yaml.safe_load(v)
        out[k] = tuple(val) if isinstance(val, list) else val
    return out


def _run_key(model_dir: Optional[str], budget: float, overrides: Mapping[str, Any]) -> Dict[str, Any]:
    return {"model": (str(Path(model_dir).resolve()) if model_dir else None), "budget": float(budget),
            "overrides": {k: (list(v) if isinstance(v, tuple) else v) for k, v in sorted(overrides.items())}}


def _resolve_model(model: Optional[str]) -> Optional[str]:
    if model is None or str(model).strip().lower() in ("", "none", "null", "symbolic"):
        return None
    p = Path(model)
    if not (p / "config.json").is_file():
        raise FileNotFoundError(f"--model {model!r} is not a package dir (no config.json)")
    return str(p.resolve())


def run(split: str = "hard180", model: Optional[str] = None, budget: float = 60.0, workers: int = 6,
        max_tasks: Optional[int] = None, out: str = "reports", tag: Optional[str] = None, *,
        root: Optional[str] = None, split_file: Optional[str] = None, device: Optional[str] = None,
        threads: Optional[int] = None, overrides: Optional[Mapping[str, Any]] = None,
        baseline: Optional[str] = None, fresh: bool = False, summarize_only: bool = False,
        stall_s: Optional[float] = None) -> Dict[str, Any]:
    """Run (or resume) the harness and write the per-task jsonl + summary json; returns the summary."""
    if split not in SPLIT_CHOICES:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLIT_CHOICES)}")
    workers = max(1, int(workers))
    overrides = dict(overrides or {})
    model_dir = _resolve_model(model)
    doc = _split_doc(split_file)
    ids = sorted(doc[SPLIT_CHOICES[split]])
    if max_tasks is not None:
        ids = ids[: max(0, int(max_tasks))]
    subsets_full: Dict[str, List[str]] = {}
    subset_of: Dict[str, str] = {}
    hardness: Dict[str, float] = {}
    if split == "hard180":
        subsets_full = {"hard180_clean150": sorted(doc["hard180_clean150"]),
                        "hard180_from_old_train": sorted(doc["hard180_from_old_train"])}
        for name, sub in subsets_full.items():
            for t in sub:
                subset_of[t] = "clean150" if name == "hard180_clean150" else "from_old_train"
        hardness = {str(k): float(v) for k, v in (doc.get("hard180_scores") or {}).items()}
    sel = set(ids)
    subsets = {k: [t for t in v if t in sel] for k, v in subsets_full.items()}
    summary_path, rows_path = output_paths(out, split, tag)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    key = _run_key(model_dir, budget, overrides)

    if fresh and rows_path.exists() and not summarize_only:
        rows_path.unlink()
    rows = read_rows(rows_path)
    foreign = [r for r in rows if r.get("run") and r.get("run") != key]
    if foreign and not summarize_only:
        raise RuntimeError(f"{rows_path} holds {len(foreign)} rows from another configuration "
                           f"({foreign[0].get('run')} != {key}); use another --tag or --fresh")
    done = {str(r.get("task_id")) for r in rows}
    todo = [t for t in ids if t not in done]
    threads = int(threads) if threads else max(1, (os.cpu_count() or 2) // (2 * workers))
    t_start = time.perf_counter()
    stalled: List[str] = []
    if todo and not summarize_only:
        tasks = load_split_tasks(split, root=root, split_file=split_file)
        todo_tasks = [tasks[t] for t in todo]
        log.info("%s: %d tasks to solve (%d already done), budget %.1f s, %d worker(s), model %s",
                 split, len(todo_tasks), len(done & sel), budget, workers, model_dir or "none")
        initargs = (model_dir, device, threads, overrides, float(budget), split, subset_of, hardness, key)
        # the worker functions are taken from the importable module (not ``__main__`` under ``python -m``), so the
        # spawned processes unpickle them by module name
        from arcjepa.eval import hard180 as _mod

        if workers == 1 or len(todo_tasks) == 1:
            _mod._init_worker(*initargs)
            for i, task in enumerate(todo_tasks, 1):
                row = _mod._solve_one(task)
                _append_row(rows_path, row)
                _log_progress(i, len(todo_tasks), row)
        else:
            stall = float(stall_s) if stall_s else max(180.0, 3.0 * float(budget) + 120.0)
            ctx = mp.get_context("spawn")
            with _child_thread_env(threads):
                pool = ctx.Pool(min(workers, len(todo_tasks)), initializer=_mod._init_worker, initargs=initargs)
            finished: set = set()
            try:
                it = pool.imap_unordered(_mod._solve_one, todo_tasks, chunksize=1)
                for i in range(1, len(todo_tasks) + 1):
                    try:  # the first result also pays for spawning the workers and loading the package
                        row = it.next(timeout=stall + (300.0 if i == 1 else 0.0))
                    except mp.TimeoutError:
                        stalled = [t for t in todo if t not in finished]
                        log.error("no task finished for %.0f s; stopping the pool (%d tasks left for a resume)",
                                  stall, len(stalled))
                        break
                    finished.add(str(row["task_id"]))
                    _append_row(rows_path, row)
                    _log_progress(i, len(todo_tasks), row)
            finally:
                pool.terminate()  # only this run's own worker processes
                pool.join()
        rows = read_rows(rows_path)

    meta = {
        "tag": tag or None,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "split_file": str(split_file or "data/splits_670_150_180.json"),
        "hard180_sha256": doc.get("hard180_sha256"),
        "model": model_dir,
        "model_loaded": any(r.get("model_loaded") for r in rows if str(r.get("task_id")) in sel),
        "budget_s": float(budget),
        "workers": workers,
        "threads_per_worker": threads,
        "solver_overrides": key["overrides"],
        "max_tasks": max_tasks,
        "per_task_file": rows_path.name,
        "wall_s_this_invocation": round(time.perf_counter() - t_start, 1),
    }
    try:
        pmeta = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8")) if model_dir else None
        meta["solve_config"] = _solve_config(pmeta, budget, overrides).to_dict()
    except Exception as exc:  # noqa: BLE001 - informational only
        meta["solve_config"] = {"error": repr(exc)}
    summary = summarize(rows, split=split, task_ids=ids, meta=meta, subsets=subsets or None)
    if stalled:
        summary["stalled"] = stalled
        summary["notes"].append(f"the pool stalled; {len(stalled)} tasks were left for a resume")
    if baseline:
        bpath = Path(baseline)
        bdoc = json.loads(bpath.read_text(encoding="utf-8"))
        brows = read_rows(bpath.parent / str(bdoc.get("per_task_file") or ""))
        summary["regression"] = {"baseline": str(bpath), **regression_check(rows, brows, ids)}
    _write_json(summary_path, summary)
    log.info("%s: %s task-level pass@2 (%s); wrote %s and %s", split, summary["headline"]["text"],
             "complete" if summary["complete"] else f"{len(summary['incomplete'])} missing", summary_path, rows_path)
    return summary


def _log_progress(i: int, n: int, row: Mapping[str, Any]) -> None:
    log.info("[%d/%d] %s pass=%s %s %.1fs", i, n, row.get("task_id"), row.get("task_pass"),
             row.get("failure_category"), float(row.get("runtime") or 0.0))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(prog="python -m arcjepa.eval.hard180",
                                 description="Hard-180 task-level pass@2 harness (ARC-JEPA v2).")
    ap.add_argument("--model", default="none", help="package dir written by arcjepa.training.export, or 'none'")
    ap.add_argument("--budget", type=float, default=60.0, help="wall seconds per task")
    ap.add_argument("--split", choices=sorted(SPLIT_CHOICES), default="hard180")
    ap.add_argument("--workers", type=int, default=6, help="worker processes (the machine is shared: <= 6)")
    ap.add_argument("--max-tasks", type=int, default=None, help="first N sorted task ids only")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--tag", default=None, help="file prefix: <tag>_h180.json / <tag>_h180_per_task.jsonl")
    ap.add_argument("--root", default=None, help="data mirror root (default: arcjepa.data.resolve_root)")
    ap.add_argument("--split-file", default=None, help="split document (default data/splits_670_150_180.json)")
    ap.add_argument("--device", default=None, help="model device (default cuda when available, else cpu)")
    ap.add_argument("--threads", type=int, default=None, help="torch / BLAS threads per worker")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                    help="SolveConfig override for ablations, e.g. --set induce=false (repeatable)")
    ap.add_argument("--baseline", default=None, help="summary json of the accepted configuration (regression gate)")
    ap.add_argument("--fresh", action="store_true", help="discard existing rows of this tag/split first")
    ap.add_argument("--summarize-only", action="store_true", help="rebuild the summary from the existing rows")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("arcjepa.search").setLevel(logging.ERROR)
    s = run(args.split, args.model, args.budget, args.workers, args.max_tasks, args.out, args.tag, root=args.root,
            split_file=args.split_file, device=args.device, threads=args.threads,
            overrides=_parse_overrides(args.overrides), baseline=args.baseline, fresh=args.fresh,
            summarize_only=args.summarize_only)
    print(json.dumps({"headline": s["headline"], "output_pass@2": s["output_pass@2"],
                      "exact_program": s["exact_program"], "near_90_cell": s["near_90_cell"],
                      "failure_categories": s["failure_categories"], "subsets": s.get("subsets"),
                      "regression": s.get("regression"), "complete": s["complete"]}, indent=1))
    if s.get("regression") and not s["regression"]["accepted"]:
        return 3
    return 0 if s["complete"] else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
