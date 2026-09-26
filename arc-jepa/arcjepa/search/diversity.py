"""Two diverse, always-valid attempts per test input (FROZEN_SPEC "Two outputs", INTERFACES §6).

``select_two(cands, test_input)``:

1. Every exact-fit candidate (``demo_err == 0``, best Score first) is run on the test input.  Candidates are
   clustered by their test output; each cluster records the structural signatures
   (:func:`arcjepa.dsl.canonicalize.structural_signature` of the canonical program) that produced it.  Clusters
   are ranked by their best Score, ties broken by the number of distinct structures that agree (a vote).
2. ``attempt_1`` = the best cluster's output.  ``attempt_2`` = the best cluster whose structures differ from the
   first cluster's (else any other exact cluster).
3. Missing attempts fall back to the best non-exact candidate output with the predicted output shape (when the
   demo pairs are given), then to the identity (test input) / a grid of the most common demo output shape filled
   with the demo outputs' dominant colour.  Every returned grid is validated; the two attempts differ whenever
   any different valid grid is available.
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from arcjepa.core.types import Grid, Pair, copy_grid, validate_grid
from arcjepa.dsl.canonicalize import canonicalize, structural_signature

from .candidate import Candidate, sort_candidates
from .verifier import execute_safe, grid_key

__all__ = ["select_two", "predict_output_shape", "fallback_grids"]

log = logging.getLogger(__name__)


def predict_output_shape(pairs: Sequence[Pair], test_input: Grid) -> Optional[Tuple[int, int]]:
    """Output shape implied by the demos: same as input, a consistent integer scaling, or a constant shape."""
    if not pairs or not validate_grid(test_input):
        return None
    ih, iw = len(test_input), len(test_input[0])
    shapes = [((len(p.input), len(p.input[0])), (len(p.output), len(p.output[0]))) for p in pairs
              if p.input and p.output]
    if not shapes:
        return None
    if all(i == o for i, o in shapes):
        return (ih, iw)
    fr = {(o[0] / i[0], o[1] / i[1]) for i, o in shapes}
    if len(fr) == 1:  # consistent scaling (checked first: it also explains a constant shape on constant inputs)
        a, b = next(iter(fr))
        h, w = ih * a, iw * b
        if abs(h - round(h)) < 1e-9 and abs(w - round(w)) < 1e-9 and 1 <= round(h) <= 30 and 1 <= round(w) <= 30:
            return (int(round(h)), int(round(w)))
    outs = {o for _, o in shapes}
    if len(outs) == 1:
        return next(iter(outs))
    return None


def fallback_grids(test_input: Grid, pairs: Optional[Sequence[Pair]] = None) -> List[Grid]:
    """Always-valid fallback attempts: identity (when plausible) and a demo-shaped constant / repeated grid."""
    out: List[Grid] = []
    valid_in = validate_grid(test_input)
    shape = predict_output_shape(pairs, test_input) if pairs else None
    if pairs:
        demo_outs = [p.output for p in pairs if validate_grid(p.output)]
        if demo_outs and all(o == demo_outs[0] for o in demo_outs):
            out.append(copy_grid(demo_outs[0]))  # constant-output task
    if valid_in and (shape is None or shape == (len(test_input), len(test_input[0]))):
        out.append(copy_grid(test_input))
    if pairs:
        demo_outs = [p.output for p in pairs if validate_grid(p.output)]
        if demo_outs:
            if shape is None:
                shape = Counter((len(o), len(o[0])) for o in demo_outs).most_common(1)[0][0]
            colours = Counter(v for o in demo_outs for row in o for v in row)
            fill = colours.most_common(1)[0][0] if colours else 0
            out.append([[int(fill)] * shape[1] for _ in range(shape[0])])
    if valid_in:
        out.append(copy_grid(test_input))
    out.append([[0]])
    return [g for g in out if validate_grid(g)]


def select_two(cands: Sequence[Candidate], test_input: Grid, *, pairs: Optional[Sequence[Pair]] = None,
               time_budget_s: Optional[float] = None, max_eval: int = 48) -> Tuple[Grid, Grid, Dict[str, Any]]:
    """Pick two attempts for ``test_input`` (see module docstring).  Never raises; outputs are valid grids.

    Optional keyword arguments: the task's demo ``pairs`` (enables output-shape prediction and demo-based
    fallbacks), a ``time_budget_s`` for executing candidates on the test input, and ``max_eval`` (candidates run
    per category).
    """
    t0 = time.perf_counter()
    deadline = None if time_budget_s is None else t0 + max(0.0, time_budget_s)
    ranked = sort_candidates(cands)
    shape = predict_output_shape(pairs, test_input) if pairs else None
    info: Dict[str, Any] = {"n_exact": 0, "n_clusters": 0, "attempt_1_program": None, "attempt_2_program": None,
                            "attempt_1_source": "fallback", "attempt_2_source": "fallback", "predicted_shape": shape}

    def run(c: Candidate) -> Optional[Grid]:
        if deadline is not None:
            left = deadline - time.perf_counter()
            if left <= 0:
                return None
            return execute_safe(c.program, test_input, timeout_s=min(0.1, left))
        return execute_safe(c.program, test_input)

    def late() -> bool:
        return deadline is not None and time.perf_counter() > deadline

    # ---------------------------------------------------------------- exact-fit clusters
    clusters: Dict[Any, Dict[str, Any]] = {}
    order: List[Any] = []
    exact = [c for c in ranked if c.demo_err == 0]
    info["n_exact"] = len(exact)
    if validate_grid(test_input):
        for c in exact[:max_eval]:
            if late():
                break
            out = run(c)
            if out is None or not validate_grid(out):
                continue
            k = grid_key(out)
            try:
                sig = structural_signature(canonicalize(c.program))
            except Exception:  # pragma: no cover - defensive
                sig = c.program.to_str()
            cl = clusters.get(k)
            if cl is None:
                clusters[k] = {"grid": out, "best": c, "score": c.score, "sigs": {sig}}
                order.append(k)
            else:
                cl["sigs"].add(sig)
    ranked_clusters = sorted(order, key=lambda k: (-clusters[k]["score"], -len(clusters[k]["sigs"])))
    info["n_clusters"] = len(ranked_clusters)

    a1: Optional[Grid] = None
    a2: Optional[Grid] = None
    if ranked_clusters:
        first = clusters[ranked_clusters[0]]
        a1 = first["grid"]
        info["attempt_1_program"] = first["best"].program.to_str()
        info["attempt_1_source"] = "exact"
        rest = [clusters[k] for k in ranked_clusters[1:]]
        pick = next((cl for cl in rest if not (cl["sigs"] & first["sigs"])), None) or (rest[0] if rest else None)
        if pick is not None:
            a2 = pick["grid"]
            info["attempt_2_program"] = pick["best"].program.to_str()
            info["attempt_2_source"] = "exact"

    # ---------------------------------------------------------------- non-exact fallbacks
    if (a1 is None or a2 is None) and validate_grid(test_input):
        tried = 0
        for c in ranked:
            if c.demo_err == 0 or tried >= max_eval or late():
                continue
            tried += 1
            out = run(c)
            if out is None or not validate_grid(out):
                continue
            if shape is not None and (len(out), len(out[0])) != shape:
                continue
            if a1 is None:
                a1 = out
                info["attempt_1_program"] = c.program.to_str()
                info["attempt_1_source"] = "near_miss"
            elif out != a1:
                a2 = out
                info["attempt_2_program"] = c.program.to_str()
                info["attempt_2_source"] = "near_miss"
                break
    fbs = fallback_grids(test_input, pairs)
    if a1 is None:
        a1 = fbs[0]
    if a2 is None:
        a2 = next((g for g in fbs if g != a1), a1)
    if not validate_grid(a1):
        a1 = fbs[0]
    if not validate_grid(a2):
        a2 = next((g for g in fbs if g != a1), fbs[0])
    info["distinct"] = a1 != a2
    info["select_ms"] = 1000.0 * (time.perf_counter() - t0)
    return copy_grid(a1), copy_grid(a2), info
