"""Per-task diagnostics JSON of the spec (Evaluation / research tables).

Every record carries the spec keys ``correct, candidate_rank, program_depth, objects, hypotheses,
beam_expansions, repair_rounds, tta_steps, inference_ms, rule_retrieval_r8`` plus ``task_id``, ``family``,
``score`` (fraction of test outputs solved), ``n_test`` and the solver's extra fields (difficulty, bucket, stage
timings, best program, ...). Records are written as one JSON document ``{"summary": ..., "tasks": [...]}``.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = ["SPEC_KEYS", "task_diagnostics", "jsonable", "write_diagnostics", "read_diagnostics"]

#: the per-task diagnostics fields named by the spec
SPEC_KEYS = ("correct", "candidate_rank", "program_depth", "objects", "hypotheses", "beam_expansions",
             "repair_rounds", "tta_steps", "inference_ms", "rule_retrieval_r8")


def jsonable(v: Any) -> Any:
    """Recursively convert ``v`` into JSON-serialisable values (tuples -> lists, numpy scalars -> python,
    non-finite floats -> None, unknown objects -> ``repr``)."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, Mapping):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [jsonable(x) for x in v]
    item = getattr(v, "item", None)
    if callable(item):
        try:
            return jsonable(item())
        except (TypeError, ValueError):
            pass
    tolist = getattr(v, "tolist", None)
    if callable(tolist):
        return jsonable(tolist())
    return repr(v)


def task_diagnostics(task_id: str, correct: Sequence[bool], solver_diag: Optional[Mapping[str, Any]] = None, *,
                     family: Optional[str] = None, inference_ms: Optional[float] = None,
                     error: Optional[str] = None) -> Dict[str, Any]:
    """One per-task record: the spec keys (missing ones = ``None``), ``correct`` recomputed by the evaluator
    (list of bools per test output), ``score``, and every other solver diagnostic."""
    sd = dict(solver_diag or {})
    rec: Dict[str, Any] = {k: sd.get(k) for k in SPEC_KEYS}
    for k, v in sd.items():
        if k not in rec:
            rec[k] = v
    rec["task_id"] = task_id
    rec["correct"] = [bool(c) for c in correct]
    rec["n_test"] = len(rec["correct"])
    rec["score"] = (sum(rec["correct"]) / float(len(rec["correct"]))) if rec["correct"] else 0.0
    rec["family"] = family
    if inference_ms is not None:
        rec["inference_ms"] = round(float(inference_ms), 2)
    if error is not None:
        rec["error"] = error
    return jsonable(rec)


def write_diagnostics(records: Sequence[Mapping[str, Any]], path: str,
                      summary: Optional[Mapping[str, Any]] = None) -> str:
    """Write ``{"summary": summary, "tasks": records}`` to ``path`` (atomic rename); returns the path."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"summary": jsonable(summary or {}), "tasks": [jsonable(r) for r in records]}, fh, indent=1)
    os.replace(tmp, path)
    return path


def read_diagnostics(path: str) -> List[Dict[str, Any]]:
    """The task records of a file written by :func:`write_diagnostics` (a bare list is accepted too)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return list(data["tasks"] if isinstance(data, dict) else data)
