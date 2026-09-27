"""Exact symbolic verification of DSL programs against demonstration pairs (INTERFACES.md §6).

``demo_error(prog, pairs)`` returns ``(E, cells)`` where ``E = sum_i 1[p(X_i) != Y_i]`` is the spec's exact demo
error and ``cells`` counts mismatched cells over all pairs (a failed execution or a shape mismatch counts every
cell of the expected output).  ``is_exact`` stops at the first mismatching pair.  Every execution goes through the
real interpreter (:func:`arcjepa.dsl.interpreter.execute`) with a short per-call timeout, so a verified program is
exactly what the submission will run.

**Search deadline.** Inside ``with search_deadline(t):`` (``t`` in ``time.perf_counter()`` seconds; the solver
opens one per task) every interpreter call made through :func:`execute_safe` / :func:`clipped_timeout` has its
timeout clipped to ``t`` (with a small floor, :data:`MIN_TIMEOUT_S`), so no single execution can run far past the
task's budget.  The deadline lives in a :class:`contextvars.ContextVar` (context-local, not shared between threads
or tasks); outside such a block nothing is clipped.
"""
from __future__ import annotations

import contextlib
import contextvars
import logging
import time
from collections import Counter
from itertools import chain
from typing import Iterator, List, Optional, Sequence, Tuple

from arcjepa.core.types import Grid, Pair, validate_grid
from arcjepa.dsl.ast import Node
from arcjepa.dsl.interpreter import execute
from arcjepa.dsl.primitives import ExecError

__all__ = ["DEFAULT_TIMEOUT_S", "MIN_TIMEOUT_S", "search_deadline", "clipped_timeout", "past_search_deadline",
           "execute_safe", "demo_outputs", "pair_cell_error", "outputs_error", "demo_error",
           "is_exact", "total_cells", "grid_key", "outputs_key", "color_hist", "hist_overlap", "pair_rank_loss",
           "TargetInfo"]

log = logging.getLogger(__name__)

#: Per-execution timeout used during search / verification (a legitimate program runs in a few ms on 30x30).
DEFAULT_TIMEOUT_S: float = 0.1
#: Floor of a deadline-clipped timeout: enough for a legitimate program, small enough to bound any overshoot.
MIN_TIMEOUT_S: float = 0.01

_SEARCH_DEADLINE: "contextvars.ContextVar[Optional[float]]" = contextvars.ContextVar("arcjepa_search_deadline",
                                                                                       default=None)


@contextlib.contextmanager
def search_deadline(deadline: Optional[float]) -> Iterator[None]:
    """Clip every interpreter call inside the block to ``deadline`` (``time.perf_counter()`` seconds).

    Nested blocks keep the earlier of the two deadlines; ``None`` leaves the current deadline unchanged.
    """
    cur = _SEARCH_DEADLINE.get()
    new = deadline if cur is None else (cur if deadline is None else min(cur, float(deadline)))
    token = _SEARCH_DEADLINE.set(new)
    try:
        yield
    finally:
        _SEARCH_DEADLINE.reset(token)


def clipped_timeout(timeout_s: float) -> float:
    """``timeout_s`` clipped to the time left before the current :func:`search_deadline` (floor
    :data:`MIN_TIMEOUT_S`); unchanged outside a deadline block."""
    dl = _SEARCH_DEADLINE.get()
    if dl is None:
        return timeout_s
    return max(min(MIN_TIMEOUT_S, timeout_s), min(timeout_s, dl - time.perf_counter()))


def past_search_deadline() -> bool:
    """True inside a :func:`search_deadline` block whose deadline has passed (results computed now may have been
    cut short by a clipped timeout and must not be cached as genuine failures)."""
    dl = _SEARCH_DEADLINE.get()
    return dl is not None and time.perf_counter() > dl


def execute_safe(prog: Node, grid: Grid, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> Optional[Grid]:
    """Run ``prog`` on ``grid``; return a valid Grid or ``None`` on any failure (never raises).  The timeout is
    clipped to the current :func:`search_deadline`, if any."""
    try:
        out = execute(prog, grid, timeout_s=clipped_timeout(timeout_s))
    except ExecError:
        return None
    except Exception as e:  # pragma: no cover - defensive: the interpreter should only raise ExecError
        log.debug("execute_safe: unexpected %s on %s", type(e).__name__, prog.to_str())
        return None
    return out if validate_grid(out) else None


def demo_outputs(prog: Node, grids: Sequence[Grid], *, timeout_s: float = DEFAULT_TIMEOUT_S) -> List[Optional[Grid]]:
    """Outputs of ``prog`` on every grid (``None`` where execution failed)."""
    return [execute_safe(prog, g, timeout_s=timeout_s) for g in grids]


def pair_cell_error(out: Optional[Grid], target: Grid) -> int:
    """Mismatched cells between ``out`` and ``target``; failure / shape mismatch counts all target cells."""
    th = len(target)
    tw = len(target[0]) if th else 0
    if out is None or len(out) != th or (th and len(out[0]) != tw):
        return th * tw
    if out == target:
        return 0
    return sum(1 for ro, rt in zip(out, target) for a, b in zip(ro, rt) if a != b)


def outputs_error(outs: Sequence[Optional[Grid]], pairs: Sequence[Pair]) -> Tuple[int, int]:
    """``(E, cells)`` for precomputed outputs (see module docstring)."""
    wrong = 0
    cells = 0
    for out, p in zip(outs, pairs):
        if out is not None and out == p.output:
            continue
        wrong += 1
        cells += pair_cell_error(out, p.output)
    return wrong, cells


def demo_error(prog: Node, pairs: Sequence[Pair], *, timeout_s: float = DEFAULT_TIMEOUT_S) -> Tuple[int, int]:
    """``(#pairs mismatched, #cells mismatched)``; shape mismatch or failure counts as all cells of that pair."""
    return outputs_error(demo_outputs(prog, [p.input for p in pairs], timeout_s=timeout_s), pairs)


def is_exact(prog: Node, pairs: Sequence[Pair], *, timeout_s: float = DEFAULT_TIMEOUT_S) -> bool:
    """True when ``prog`` reproduces every demo output exactly (stops at the first mismatch)."""
    for p in pairs:
        out = execute_safe(prog, p.input, timeout_s=timeout_s)
        if out is None or out != p.output:
            return False
    return True


def total_cells(pairs: Sequence[Pair]) -> int:
    """Number of cells over all expected outputs (normaliser of the cell error)."""
    return sum(len(p.output) * (len(p.output[0]) if p.output else 0) for p in pairs)


# ----------------------------------------------------------------------------------------------- search helpers

def grid_key(g: Grid) -> Tuple[Tuple[int, ...], ...]:
    """Hashable form of a grid."""
    return tuple(tuple(r) for r in g)


def outputs_key(outs: Sequence[Grid]) -> int:
    """Hash of a tuple of grids (observational-equivalence key used by the searches)."""
    return hash(tuple(grid_key(g) for g in outs))


def color_hist(g: Grid) -> Counter:
    """Colour histogram of a grid."""
    return Counter(chain.from_iterable(g))


def hist_overlap(a: Counter, b: Counter) -> float:
    """Histogram intersection normalised by the larger mass, in [0, 1]."""
    na = sum(a.values())
    nb = sum(b.values())
    if na == 0 or nb == 0:
        return 0.0
    inter = sum(min(v, b.get(k, 0)) for k, v in a.items())
    return inter / float(max(na, nb))


def _mismatch(a: Grid, b: Grid) -> int:
    return sum(1 for ra, rb in zip(a, b) for x, y in zip(ra, rb) if x != y)


def _block_frac(small: Grid, big: Grid, max_blocks: int = 16) -> Optional[float]:
    """Best mismatch fraction between ``small`` and the aligned blocks of ``big`` (None if shapes do not tile)."""
    h, w = len(small), len(small[0])
    bh, bw = len(big), len(big[0])
    if bh % h or bw % w:
        return None
    kr, kc = bh // h, bw // w
    if kr * kc > max_blocks:
        return None
    best = h * w
    for i in range(kr):
        for j in range(kc):
            bad = 0
            for r in range(h):
                ra = small[r]
                rb = big[i * h + r]
                off = j * w
                for c in range(w):
                    if ra[c] != rb[off + c]:
                        bad += 1
                if bad >= best:
                    break
            if bad < best:
                best = bad
                if best == 0:
                    return 0.0
    return best / float(h * w)


class TargetInfo:
    """Precomputed views of one expected output for :meth:`loss` (smooth ranking loss of partial solutions)."""

    def __init__(self, target: Grid) -> None:
        self.target = target
        self.h = len(target)
        self.w = len(target[0])
        self.cells = self.h * self.w
        self.hist = color_hist(target)
        t = target
        rot90 = [list(r) for r in zip(*t[::-1])]
        transpose = [list(r) for r in zip(*t)]
        rot270 = transpose[::-1]
        d2 = [r[::-1] for r in transpose[::-1]]
        #: inverse images of the target under the four shape-transposing symmetries (a later outer op fixes them)
        self.transposed: List[Grid] = [rot90, rot270, transpose, d2]

    def loss(self, out: Grid) -> float:
        """0 when exact; the mismatched-cell fraction in (0, 1] when shapes match; ``1 + 0.25 * x`` otherwise.

        ``x`` in [0, 1] grades a shape mismatch by the best of: histogram disagreement, block mismatch when one
        shape tiles the other (tiling / mirroring / cropping still to come), and mismatch against the transposed
        target views (a final rotation / transpose still to come).
        """
        h = len(out)
        w = len(out[0])
        if h == self.h and w == self.w:
            if out == self.target:
                return 0.0
            return _mismatch(out, self.target) / float(self.cells)
        x = 1.0 - hist_overlap(color_hist(out), self.hist)
        if x > 0.0:
            if h <= self.h and w <= self.w:
                bf = _block_frac(out, self.target)
            elif h >= self.h and w >= self.w:
                bf = _block_frac(self.target, out)
            else:
                bf = None
            if bf is not None and bf < x:
                x = bf
        if x > 0.0 and h == self.w and w == self.h:
            for v in self.transposed:
                f = _mismatch(out, v) / float(self.cells)
                if f < x:
                    x = f
        return 1.0 + 0.25 * x


def pair_rank_loss(out: Grid, target: Grid, info: Optional[TargetInfo] = None) -> float:
    """Smooth per-pair loss used only for ranking partial solutions (see :meth:`TargetInfo.loss`)."""
    return (info if info is not None else TargetInfo(target)).loss(out)
