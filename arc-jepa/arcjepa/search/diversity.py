"""Two diverse, always-valid attempts per test input (FROZEN_SPEC "Two outputs", INTERFACES §6).

``select_two(cands, test_input)``:

1. Every exact-fit candidate (``demo_err == 0``, best Score first) is run on the test input.  Candidates are
   clustered by their test output; each cluster records the structural signatures
   (:func:`arcjepa.dsl.canonicalize.structural_signature` of the canonical program) that produced it.  Clusters
   are ranked by their best Score, ties broken by the number of distinct structures that agree (a vote).
2. ``attempt_1`` = the best cluster's output.  ``attempt_2`` = the best cluster whose structures differ from the
   first cluster's (else any other exact cluster).
3. Missing attempts are filled, in this order, by: the best non-exact (near-miss) candidate outputs with the
   predicted output shape (every shape when the demos imply none), the shape-inferred fills of
   :func:`fallback_grids` (constant demo output; the predicted shape filled with the most common, then the second
   most common demo output colour), near-miss outputs of another shape, exact clusters whose output is the
   identity, the identity itself and ``[[0]]``.

**The identity grid (the test input copied unchanged) is a last resort.**  No test output of the 1,000 public
training tasks equals its input (0 of 1,076, including the 7 tasks that have an identity demo pair), so an
attempt spent on the identity scores nothing.  The identity is ranked normally only for identity tasks, where
every demo output equals its input (:func:`identity_plausible`).  Every returned grid is validated; the two
attempts differ whenever any different valid grid is available.
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

__all__ = ["select_two", "predict_output_shape", "fallback_grids", "identity_plausible"]

log = logging.getLogger(__name__)


def _valid_pairs(pairs: Optional[Sequence[Pair]]) -> List[Pair]:
    return [p for p in (pairs or ()) if validate_grid(p.input) and validate_grid(p.output)]


def identity_plausible(pairs: Optional[Sequence[Pair]]) -> bool:
    """True only for identity tasks: there is at least one valid demo pair and every one maps its input to
    itself.  Otherwise the identity is (empirically) never the answer (see the module docstring)."""
    ps = _valid_pairs(pairs)
    return bool(ps) and all(p.input == p.output for p in ps)


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


def fallback_grids(test_input: Grid, pairs: Optional[Sequence[Pair]] = None, *,
                   last_resort: bool = True) -> List[Grid]:
    """Always-valid fallback attempts, most plausible first.

    1. the constant demo output (when every demo output is the same grid);
    2. the identity, for identity tasks only (:func:`identity_plausible`);
    3. the predicted output shape (else the most common demo output shape) filled with the most common, then the
       second most common demo output colour;
    4. with ``last_resort``: the identity and ``[[0]]``.

    The first two distinct grids equal ``fallback_attempts`` of ``arcjepa.utils.kaggle_submit_runner`` and of
    ``kaggle/validate_submission.py`` for the same task (tested).
    """
    out: List[Grid] = []
    valid_in = validate_grid(test_input)
    demo = _valid_pairs(pairs)
    demo_outs = [p.output for p in demo]
    if demo_outs and all(o == demo_outs[0] for o in demo_outs):
        out.append(copy_grid(demo_outs[0]))  # constant-output task
    if valid_in and identity_plausible(demo):
        out.append(copy_grid(test_input))
    if demo_outs:
        shape = predict_output_shape(demo, test_input) if valid_in else None
        if shape is None:
            shape = Counter((len(o), len(o[0])) for o in demo_outs).most_common(1)[0][0]
        colours = Counter(v for o in demo_outs for row in o for v in row).most_common(2)
        for c, _ in colours:
            out.append([[int(c)] * shape[1] for _ in range(shape[0])])
    if last_resort:
        if valid_in:
            out.append(copy_grid(test_input))
        out.append([[0]])
    return [g for g in out if validate_grid(g)]


def select_two(cands: Sequence[Candidate], test_input: Grid, *, pairs: Optional[Sequence[Pair]] = None,
               time_budget_s: Optional[float] = None, max_eval: int = 48) -> Tuple[Grid, Grid, Dict[str, Any]]:
    """Pick two attempts for ``test_input`` (see module docstring).  Never raises; outputs are valid grids.

    Optional keyword arguments: the task's demo ``pairs`` (enables output-shape prediction, the identity rule and
    demo-based fallbacks), a ``time_budget_s`` for executing candidates on the test input, and ``max_eval``
    (candidates run per category).
    """
    t0 = time.perf_counter()
    deadline = None if time_budget_s is None else t0 + max(0.0, time_budget_s)
    ranked = sort_candidates(cands)
    valid_in = validate_grid(test_input)
    demo = _valid_pairs(pairs)
    shape = predict_output_shape(demo, test_input) if demo and valid_in else None
    ident_ok = identity_plausible(demo)
    info: Dict[str, Any] = {"n_exact": 0, "n_clusters": 0, "attempt_1_program": None, "attempt_2_program": None,
                            "attempt_1_source": "fallback", "attempt_2_source": "fallback", "predicted_shape": shape,
                            "identity_demoted": False}

    def run(c: Candidate) -> Optional[Grid]:
        if deadline is not None:
            left = deadline - time.perf_counter()
            if left <= 0:
                return None
            return execute_safe(c.program, test_input, timeout_s=min(0.1, left))
        return execute_safe(c.program, test_input)

    def late() -> bool:
        return deadline is not None and time.perf_counter() > deadline

    def is_identity(g: Grid) -> bool:
        """An identity output that must be demoted (never for identity tasks)."""
        return valid_in and not ident_ok and g == test_input

    # ---------------------------------------------------------------- exact-fit clusters
    clusters: Dict[Any, Dict[str, Any]] = {}
    order: List[Any] = []
    exact = [c for c in ranked if c.demo_err == 0]
    info["n_exact"] = len(exact)
    if valid_in:
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
    good = [clusters[k] for k in ranked_clusters if not is_identity(clusters[k]["grid"])]
    ident_clusters = [clusters[k] for k in ranked_clusters if is_identity(clusters[k]["grid"])]

    # ---------------------------------------------------------------- ordered options: (grid, source, program)
    options: List[Tuple[Grid, str, Optional[str]]] = []
    if good:
        first = good[0]
        options.append((first["grid"], "exact", first["best"].program.to_str()))
        rest = good[1:]
        pick = next((cl for cl in rest if not (cl["sigs"] & first["sigs"])), None) or (rest[0] if rest else None)
        if pick is not None:
            options.append((pick["grid"], "exact", pick["best"].program.to_str()))

    def n_distinct() -> int:
        return len({grid_key(g) for g, _, _ in options})

    if n_distinct() < 2:
        other_shape: List[Tuple[Grid, str, Optional[str]]] = []
        if valid_in:
            tried = 0
            seen = {grid_key(g) for g, _, _ in options}
            for c in ranked:
                if c.demo_err == 0 or tried >= max_eval or late():
                    continue
                tried += 1
                out = run(c)
                if out is None or not validate_grid(out):
                    continue
                if is_identity(out):
                    info["identity_demoted"] = True
                    continue
                k = grid_key(out)
                if k in seen:
                    continue
                seen.add(k)
                if shape is not None and (len(out), len(out[0])) != shape:
                    other_shape.append((out, "near_miss_other_shape", c.program.to_str()))
                    continue
                options.append((out, "near_miss", c.program.to_str()))
                if n_distinct() >= 2:
                    break
        if n_distinct() < 2:
            options.extend((g, "fallback", None) for g in fallback_grids(test_input, demo, last_resort=False))
        if n_distinct() < 2:
            options.extend(other_shape)
    if ident_clusters:
        info["identity_demoted"] = True
        options.extend((cl["grid"], "exact_identity", cl["best"].program.to_str()) for cl in ident_clusters)
    if valid_in:
        options.append((copy_grid(test_input), "identity", None))
    options.append(([[0]], "fallback", None))
    options = [o for o in options if validate_grid(o[0])]

    a1, src1, prog1 = options[0]
    a2, src2, prog2 = next((o for o in options[1:] if o[0] != a1), options[0])
    info.update({"attempt_1_program": prog1, "attempt_1_source": src1,
                 "attempt_2_program": prog2, "attempt_2_source": src2})
    info["distinct"] = a1 != a2
    info["select_ms"] = 1000.0 * (time.perf_counter() - t0)
    return copy_grid(a1), copy_grid(a2), info
