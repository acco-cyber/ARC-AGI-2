"""Search-efficiency statistics over per-task diagnostics (spec "search efficiency" table and plots)."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = ["NUMERIC_KEYS", "RETRIEVAL_KS", "describe", "search_nodes", "summarize_search", "retrieval_at_k",
           "accuracy_vs_nodes", "accuracy_by"]

#: diagnostics fields summarised by :func:`summarize_search`
NUMERIC_KEYS = ("inference_ms", "beam_expansions", "repair_rounds", "tta_steps", "astar_nodes", "evo_generations",
                "n_candidates", "n_exact", "program_depth", "objects", "hypotheses", "difficulty")
#: K values of the spec's Retrieval@K plot
RETRIEVAL_KS = (1, 4, 8, 16, 32, 64)


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def describe(values: Sequence[Any]) -> Optional[Dict[str, float]]:
    """``{n, mean, median, p90, max, sum}`` of the numeric entries of ``values`` (``None`` when there are none)."""
    vals = sorted(v for v in (_num(x) for x in values) if v is not None)
    if not vals:
        return None
    return {"n": len(vals), "mean": sum(vals) / len(vals), "median": _quantile(vals, 0.5),
            "p90": _quantile(vals, 0.9), "max": vals[-1], "sum": sum(vals)}


def search_nodes(diag: Mapping[str, Any]) -> int:
    """Search nodes spent on a task: beam expansions + A* nodes (+ evolution population x generations)."""
    n = int(_num(diag.get("beam_expansions")) or 0) + int(_num(diag.get("astar_nodes")) or 0)
    gens = _num(diag.get("evo_generations"))
    if gens:
        pop = _num((diag.get("policy") or {}).get("evo_pop") if isinstance(diag.get("policy"), Mapping) else None)
        n += int(gens * (pop or 32))
    return n


def _task_correct(d: Mapping[str, Any]) -> Optional[float]:
    s = _num(d.get("score"))
    if s is not None:
        return s
    c = d.get("correct")
    if isinstance(c, list) and c:
        return sum(1.0 for x in c if x) / len(c)
    if isinstance(c, bool):
        return float(c)
    return None


def retrieval_at_k(ranks: Sequence[Optional[int]], ks: Sequence[int] = RETRIEVAL_KS) -> Dict[str, float]:
    """``Retrieval@K`` = fraction of tasks whose correct program has rank <= K (``None`` rank = not found)."""
    n = len(ranks)
    return {f"retrieval@{k}": (sum(1 for r in ranks if r is not None and 1 <= r <= k) / n if n else 0.0)
            for k in ks}


def accuracy_vs_nodes(diags: Sequence[Mapping[str, Any]], base: float = 2.0) -> List[Dict[str, Any]]:
    """Accuracy grouped by ``floor(log_base(search nodes + 1))`` (the spec's accuracy vs log(search nodes))."""
    groups: Dict[int, List[float]] = {}
    for d in diags:
        c = _task_correct(d)
        if c is None:
            continue
        b = int(math.floor(math.log(search_nodes(d) + 1, base)))
        groups.setdefault(b, []).append(c)
    return [{"log_nodes_bin": b, "nodes_lo": int(base ** b) - 1, "n": len(v), "accuracy": sum(v) / len(v)}
            for b, v in sorted(groups.items())]


def accuracy_by(diags: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Dict[str, float]]:
    """Accuracy grouped by the value of ``key`` (e.g. ``bucket``, ``family``, ``best_source``)."""
    groups: Dict[str, List[float]] = {}
    for d in diags:
        c = _task_correct(d)
        if c is None:
            continue
        groups.setdefault(str(d.get(key)), []).append(c)
    return {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(groups.items())}


def summarize_search(diags: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Search-efficiency summary: distributions of the numeric diagnostics, total nodes, accuracy per difficulty
    bucket / node bin, Retrieval@K of the correct program (``candidate_rank``) and memory retrieval rate."""
    out: Dict[str, Any] = {"n_tasks": len(diags)}
    for k in NUMERIC_KEYS:
        s = describe([d.get(k) for d in diags])
        if s is not None:
            out[k] = s
    out["search_nodes"] = describe([search_nodes(d) for d in diags])
    out["accuracy_by_bucket"] = accuracy_by(diags, "bucket")
    out["accuracy_vs_nodes"] = accuracy_vs_nodes(diags)
    out["program_rank"] = retrieval_at_k([d.get("candidate_rank") if isinstance(d.get("candidate_rank"), int)
                                          else None for d in diags])
    r8 = [x for x in (_num(d.get("rule_retrieval_r8")) for d in diags) if x is not None]
    out["rule_retrieval_r8"] = (sum(r8) / len(r8)) if r8 else None
    out["n_errors"] = sum(1 for d in diags if d.get("error"))
    out["n_exact_found"] = sum(1 for d in diags if (_num(d.get("n_exact")) or 0) > 0)
    return out
