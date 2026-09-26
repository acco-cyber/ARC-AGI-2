"""Error analysis: how far wrong attempts are from the truth (near-miss cell fractions, shape errors)."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from arcjepa.core.types import Grid, validate_grid

__all__ = ["cell_accuracy", "near_miss", "error_record", "analyze_errors", "NEAR_MISS_THRESHOLDS", "ACC_BINS"]

#: cell-accuracy thresholds reported as "near misses"
NEAR_MISS_THRESHOLDS: Tuple[float, ...] = (0.9, 0.95, 0.99)
#: histogram bin edges for the best cell accuracy of wrong predictions (last bin is [0.99, 1.0))
ACC_BINS: Tuple[float, ...] = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99)


def cell_accuracy(pred: Optional[Grid], truth: Grid) -> float:
    """Fraction of cells of ``truth`` that ``pred`` gets right; 0.0 when shapes differ or ``pred`` is invalid."""
    if not validate_grid(pred) or not validate_grid(truth):
        return 0.0
    if len(pred) != len(truth) or len(pred[0]) != len(truth[0]):
        return 0.0
    n = len(truth) * len(truth[0])
    ok = sum(1 for rp, rt in zip(pred, truth) for a, b in zip(rp, rt) if a == b)
    return ok / float(n)


def near_miss(attempts: Sequence[Optional[Grid]], truth: Grid) -> Dict[str, Any]:
    """Compare the attempts for one test input with the truth.

    Returns ``exact`` (any attempt equal), ``shape_match`` (any attempt with the right shape), ``best_cell_acc``
    (max cell accuracy over attempts), ``wrong_cells`` (cells wrong in the best attempt, ``None`` on shape
    mismatch) and ``best_attempt`` (1-based index of the best attempt).
    """
    best, best_i = -1.0, 0
    shape_ok = False
    for i, a in enumerate(attempts):
        if validate_grid(a) and validate_grid(truth) and len(a) == len(truth) and len(a[0]) == len(truth[0]):
            shape_ok = True
        acc = cell_accuracy(a, truth)
        if acc > best:
            best, best_i = acc, i
    exact = any(validate_grid(a) and a == truth for a in attempts)
    n = len(truth) * len(truth[0]) if validate_grid(truth) else 0
    wrong = int(round((1.0 - best) * n)) if shape_ok and best >= 0 else None
    if exact:
        best, wrong = 1.0, 0
    return {"exact": bool(exact), "shape_match": bool(shape_ok), "best_cell_acc": max(0.0, best),
            "wrong_cells": wrong, "best_attempt": best_i + 1}


def error_record(task_id: str, test_index: int, attempts: Sequence[Optional[Grid]], truth: Grid) -> Dict[str, Any]:
    """:func:`near_miss` plus identifiers (one row of the per-output error table)."""
    rec = {"task_id": task_id, "test_index": int(test_index)}
    rec.update(near_miss(attempts, truth))
    return rec


def _bin_label(lo: float, hi: Optional[float]) -> str:
    return f"[{lo:.2f},{hi:.2f})" if hi is not None else f"[{lo:.2f},1.00)"


def analyze_errors(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate :func:`error_record` rows: exact / shape-mismatch / near-miss counts over all test outputs and a
    histogram of the best cell accuracy of the wrong ones."""
    n = len(records)
    wrong = [r for r in records if not r.get("exact")]
    shape_bad = [r for r in wrong if not r.get("shape_match")]
    shape_ok = [r for r in wrong if r.get("shape_match")]
    hist: Dict[str, int] = {}
    edges = list(ACC_BINS)
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else None
        hist[_bin_label(lo, hi)] = sum(1 for r in shape_ok
                                       if r["best_cell_acc"] >= lo and (hi is None or r["best_cell_acc"] < hi))
    out: Dict[str, Any] = {
        "n_outputs": n,
        "n_exact": n - len(wrong),
        "n_wrong": len(wrong),
        "n_shape_mismatch": len(shape_bad),
        "n_shape_match_wrong": len(shape_ok),
        "mean_cell_acc_wrong_shape_ok": (sum(r["best_cell_acc"] for r in shape_ok) / len(shape_ok)) if shape_ok else None,
        "cell_acc_histogram": hist,
    }
    for t in NEAR_MISS_THRESHOLDS:
        out[f"near_miss_{int(round(t * 100))}"] = sum(1 for r in shape_ok if r["best_cell_acc"] >= t)
    out["near_miss_task_ids"] = sorted({str(r.get("task_id")) for r in shape_ok if r["best_cell_acc"] >= 0.95})
    return out
