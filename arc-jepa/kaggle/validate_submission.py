"""Validate (and build fallbacks for) an ARC Prize ``submission.json``.

Format (competition): ``{task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}`` with one dict per test input,
in test-input order; a grid is a non-empty rectangular list of lists of ints 0-9 with 1 <= H, W <= 30.

Only the standard library is used: the block between the ``INLINE`` markers is copied verbatim into the
inference notebook by ``kaggle/build_infer_nb.py`` so the notebook can validate and write fallback attempts
even when the ARC-JEPA code dataset is missing.

CLI::

    python kaggle/validate_submission.py submission.json [--challenges arc-agi_test_challenges.json]

exits 0 when the file is valid, 1 otherwise (errors on stdout).
"""
from __future__ import annotations

import argparse
import sys

# ---- BEGIN INLINE (stdlib only; embedded in kaggle/arc-jepa-infer.ipynb) ----
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

ATTEMPT_KEYS = ("attempt_1", "attempt_2")
MAX_GRID_SIDE = 30


def grid_error(g: Any) -> Optional[str]:
    """``None`` when ``g`` is a valid ARC grid, else a short reason."""
    if not isinstance(g, list) or not g:
        return "grid is not a non-empty list"
    if len(g) > MAX_GRID_SIDE:
        return "grid has %d rows (> %d)" % (len(g), MAX_GRID_SIDE)
    if not isinstance(g[0], list) or not g[0]:
        return "first row is not a non-empty list"
    w = len(g[0])
    if w > MAX_GRID_SIDE:
        return "grid has %d columns (> %d)" % (w, MAX_GRID_SIDE)
    for r, row in enumerate(g):
        if not isinstance(row, list) or len(row) != w:
            return "row %d is not a list of length %d" % (r, w)
        for v in row:
            if isinstance(v, bool) or not isinstance(v, int) or v < 0 or v > 9:
                return "row %d has a value that is not an int 0-9: %r" % (r, v)
    return None


def validate_submission(sub: Any, challenges: Optional[Dict[str, Any]] = None,
                        task_ids: Optional[Sequence[str]] = None) -> List[str]:
    """All format errors of ``sub`` (empty list = valid).

    With ``challenges`` (the challenges JSON dict) every task id must be present, no extra ids are allowed and
    each task must have exactly one attempts dict per test input. ``task_ids`` alone only checks presence.
    """
    errs: List[str] = []
    if not isinstance(sub, dict):
        return ["submission is not a JSON object"]
    if not sub:
        errs.append("submission is empty")
    expected: Dict[str, Optional[int]] = {}
    if challenges is not None:
        for tid, t in challenges.items():
            n = len(t.get("test", [])) if isinstance(t, dict) else None
            expected[str(tid)] = n
    for tid in task_ids or ():
        expected.setdefault(str(tid), None)
    for tid in expected:
        if tid not in sub:
            errs.append("task %s missing" % tid)
    if challenges is not None:
        for tid in sub:
            if tid not in expected:
                errs.append("task %s is not in the challenges" % tid)
    for tid, entries in sub.items():
        if not isinstance(tid, str):
            errs.append("task id %r is not a string" % (tid,))
        if not isinstance(entries, list) or not entries:
            errs.append("task %s: value is not a non-empty list" % tid)
            continue
        n = expected.get(str(tid))
        if n is not None and len(entries) != n:
            errs.append("task %s: %d attempt dicts for %d test inputs" % (tid, len(entries), n))
        for i, e in enumerate(entries):
            if not isinstance(e, dict):
                errs.append("task %s test %d: entry is not an object" % (tid, i))
                continue
            keys = set(e)
            missing = [k for k in ATTEMPT_KEYS if k not in keys]
            extra = sorted(str(k) for k in keys - set(ATTEMPT_KEYS))
            if missing:
                errs.append("task %s test %d: missing %s" % (tid, i, ",".join(missing)))
            if extra:
                errs.append("task %s test %d: unexpected keys %s" % (tid, i, ",".join(extra)))
            for k in ATTEMPT_KEYS:
                if k in e:
                    ge = grid_error(e[k])
                    if ge:
                        errs.append("task %s test %d %s: %s" % (tid, i, k, ge))
    return errs


def validate_file(path: str, challenges_path: Optional[str] = None) -> List[str]:
    """Errors of the submission file at ``path`` (JSON parse errors included)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            sub = json.load(fh)
    except (OSError, ValueError) as exc:
        return ["cannot read %s: %s" % (path, exc)]
    challenges = None
    if challenges_path:
        with open(challenges_path, "r", encoding="utf-8") as fh:
            challenges = json.load(fh)
    return validate_submission(sub, challenges)


def predicted_output_shape(demo: Sequence[Any], test_input: Any) -> Optional[Any]:
    """Output shape implied by valid demo (input, output) grids: same as input, a consistent integer scaling or a
    constant shape; else None (same rule as arcjepa.search.diversity.predict_output_shape)."""
    if not demo or grid_error(test_input) is not None:
        return None
    ih, iw = len(test_input), len(test_input[0])
    shapes = [((len(i), len(i[0])), (len(o), len(o[0]))) for i, o in demo]
    if all(a == b for a, b in shapes):
        return (ih, iw)
    fr = {(o[0] / i[0], o[1] / i[1]) for i, o in shapes}
    if len(fr) == 1:
        a, b = next(iter(fr))
        h, w = ih * a, iw * b
        if abs(h - round(h)) < 1e-9 and abs(w - round(w)) < 1e-9 and 1 <= round(h) <= 30 and 1 <= round(w) <= 30:
            return (int(round(h)), int(round(w)))
    outs = {o for _, o in shapes}
    return next(iter(outs)) if len(outs) == 1 else None


def fallback_attempts(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Always-valid attempts for every test input of a challenges task dict, without any search: the first two
    distinct grids of [constant demo output (all demo outputs equal); the identity for identity tasks (every demo
    output equals its input); the predicted output shape (else the most common demo output shape) filled with the
    most common and the second most common demo output colour; the identity; [[0]]]. The identity is a last
    resort: no test output of the 1,000 public training tasks equals its input. Same rule as
    arcjepa.utils.kaggle_submit_runner.fallback_attempts."""
    demo = [(p["input"], p["output"]) for p in task.get("train", []) or []
            if isinstance(p, dict) and grid_error(p.get("input")) is None and grid_error(p.get("output")) is None]
    outs = [o for _, o in demo]
    ident_ok = bool(demo) and all(i == o for i, o in demo)
    colours = [c for c, _ in Counter(v for o in outs for row in o for v in row).most_common(2)]
    common = Counter((len(o), len(o[0])) for o in outs).most_common(1)[0][0] if outs else None
    out: List[Dict[str, Any]] = []
    for tp in task.get("test", []) or [{}]:
        g = tp.get("input") if isinstance(tp, dict) else None
        valid_in = grid_error(g) is None
        opts: List[Any] = []
        if outs and all(o == outs[0] for o in outs):
            opts.append(outs[0])
        if valid_in and ident_ok:
            opts.append(g)
        if outs:
            shape = (predicted_output_shape(demo, g) if valid_in else None) or common
            opts.extend([[int(c)] * shape[1] for _ in range(shape[0])] for c in colours)
        if valid_in:
            opts.append(g)
        opts.append([[0]])
        a1 = opts[0]
        a2 = next((o for o in opts[1:] if o != a1), a1)
        out.append({"attempt_1": [list(r) for r in a1], "attempt_2": [list(r) for r in a2]})
    return out


def fallback_submission(challenges: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Fallback attempts for every task of ``challenges``."""
    return {str(tid): fallback_attempts(t if isinstance(t, dict) else {}) for tid, t in challenges.items()}


def repair_submission(sub: Any, challenges: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """A valid submission for ``challenges``: every valid attempt of ``sub`` is kept, everything missing or
    malformed is replaced by the fallback attempt."""
    fb = fallback_submission(challenges)
    sub = sub if isinstance(sub, dict) else {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for tid, fallback in fb.items():
        got = sub.get(tid)
        entries: List[Dict[str, Any]] = []
        for i, f in enumerate(fallback):
            e = got[i] if isinstance(got, list) and i < len(got) and isinstance(got[i], dict) else {}
            entries.append({k: (e[k] if k in e and grid_error(e[k]) is None else f[k]) for k in ATTEMPT_KEYS})
        out[tid] = entries
    return out


def score_submission(sub: Dict[str, Any], solutions: Dict[str, List[Any]]) -> float:
    """Competition metric (stdlib copy of ``arcjepa.eval.evaluate.competition_score``): mean over the tasks of
    ``solutions`` of the fraction of test outputs matched by attempt_1 or attempt_2; missing tasks score 0."""
    if not solutions:
        return 0.0
    total = 0.0
    for tid, truths in solutions.items():
        got = sub.get(tid) if isinstance(sub, dict) else None
        got = got if isinstance(got, list) else []
        hits = 0
        for i, t in enumerate(truths):
            e = got[i] if i < len(got) and isinstance(got[i], dict) else {}
            hits += int(bool(t) and any(e.get(k) == t for k in ATTEMPT_KEYS))
        total += hits / float(len(truths)) if truths else 0.0
    return total / len(solutions)


def write_json_atomic(obj: Any, path: str) -> None:
    """Write ``obj`` as JSON to ``path`` via a temporary file + rename (never leaves a half-written file)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    os.replace(tmp, path)
# ---- END INLINE ----


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point; returns the process exit code."""
    ap = argparse.ArgumentParser(description="Validate an ARC Prize submission.json")
    ap.add_argument("submission")
    ap.add_argument("--challenges", default=None, help="challenges JSON: checks ids and test-input counts")
    a = ap.parse_args(argv)
    errs = validate_file(a.submission, a.challenges)
    for e in errs[:50]:
        sys.stdout.write(e + "\n")
    if len(errs) > 50:
        sys.stdout.write("... %d more errors\n" % (len(errs) - 50))
    sys.stdout.write(("VALID" if not errs else "INVALID (%d errors)" % len(errs)) + "\n")
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
