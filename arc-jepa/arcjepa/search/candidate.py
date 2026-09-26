"""Search candidates and the spec's scoring rule (FROZEN_SPEC "Search", INTERFACES.md §6).

``Score(p) = alpha * s_neural - beta * L_demo - gamma * C(p)`` with alpha 1.0, beta 10.0, gamma 0.15.

v1 realisation of the terms:

* ``L_demo = E(p) + mismatched_cells / expected_cells`` where ``E(p)`` is the number of mismatching demo pairs
  (the spec's exact error).  It is 0 exactly when the program fits every demo, and with beta = 10 one wrong pair
  always costs more than any complexity difference, so exact programs rank first.
* ``C(p)`` = number of operator nodes plus number of literal arguments (leaves ``INPUT`` / ``OBJ`` are free).
* ``s_neural`` is whatever the prior returns (0 for the uniform prior used when no model is available).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from arcjepa.core.types import Grid, Pair
from arcjepa.dsl.ast import Node
from arcjepa.dsl.canonicalize import canonicalize
from arcjepa.dsl.types import LEAF_INPUT, LEAF_OBJ

from .verifier import DEFAULT_TIMEOUT_S, demo_outputs, outputs_error, total_cells

__all__ = ["ALPHA", "BETA", "GAMMA", "Prior", "Candidate", "complexity", "demo_loss", "score_value",
           "make_candidate", "candidate_from_outputs", "sort_candidates", "dedup_candidates", "merge_candidates",
           "apply_prior", "canonical_key"]

ALPHA: float = 1.0
BETA: float = 10.0
GAMMA: float = 0.15

#: A prior maps a batch of programs to neural compatibility scores (higher = more plausible).
Prior = Callable[[List[Node]], List[float]]


@dataclass
class Candidate:
    """One scored program.  The first seven fields are the INTERFACES contract; the rest are optional extras."""

    program: Node
    score: float
    demo_err: int
    cell_err: int
    neural: float
    complexity: int
    source: str
    loss: float = 0.0  # L_demo used in ``score``
    outputs: Optional[List[Optional[Grid]]] = field(default=None, repr=False, compare=False)
    meta: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def is_exact(self) -> bool:
        """True when the program reproduces every demo pair."""
        return self.demo_err == 0

    @property
    def key(self) -> str:
        """Canonical program string (dedup key)."""
        return canonical_key(self.program)

    def with_neural(self, neural: float, alpha: float = ALPHA, beta: float = BETA,
                    gamma: float = GAMMA) -> "Candidate":
        """Copy with a new neural score and the recomputed Score."""
        return replace(self, neural=float(neural),
                       score=score_value(neural, self.loss, self.complexity, alpha, beta, gamma))


def canonical_key(prog: Node) -> str:
    """Canonical S-expression of ``prog`` (falls back to the raw form if canonicalisation fails)."""
    try:
        return canonicalize(prog).to_str()
    except Exception:  # pragma: no cover - canonicalize is total on well-typed ASTs
        return prog.to_str()


def complexity(prog: Node) -> int:
    """C(p): operator nodes + literal arguments (leaves are free)."""
    n = 0
    for _, node in prog.iter_nodes():
        if node.op not in (LEAF_INPUT, LEAF_OBJ):
            n += 1
        for a in node.args:
            if not isinstance(a, Node):
                n += 1
    return n


def demo_loss(demo_err: int, cell_err: int, n_cells: int) -> float:
    """L_demo = E + cell_err / n_cells (0 iff exact)."""
    return float(demo_err) + float(cell_err) / float(max(1, n_cells))


def score_value(neural: float, loss: float, compl: float, alpha: float = ALPHA, beta: float = BETA,
                gamma: float = GAMMA) -> float:
    """Score(p) = alpha * s_neural - beta * L_demo - gamma * C(p)."""
    return alpha * float(neural) - beta * float(loss) - gamma * float(compl)


def candidate_from_outputs(prog: Node, outs: Sequence[Optional[Grid]], pairs: Sequence[Pair], *,
                           neural: float = 0.0, source: str = "", alpha: float = ALPHA, beta: float = BETA,
                           gamma: float = GAMMA, n_cells: Optional[int] = None) -> Candidate:
    """Build a Candidate from already computed demo outputs."""
    wrong, cells = outputs_error(outs, pairs)
    loss = demo_loss(wrong, cells, n_cells if n_cells is not None else total_cells(pairs))
    c = complexity(prog)
    return Candidate(program=prog, score=score_value(neural, loss, c, alpha, beta, gamma), demo_err=wrong,
                     cell_err=cells, neural=float(neural), complexity=c, source=source, loss=loss,
                     outputs=list(outs))


def make_candidate(prog: Node, pairs: Sequence[Pair], *, neural: float = 0.0, source: str = "",
                   alpha: float = ALPHA, beta: float = BETA, gamma: float = GAMMA,
                   timeout_s: float = DEFAULT_TIMEOUT_S, n_cells: Optional[int] = None) -> Candidate:
    """Execute ``prog`` on every demo input with the real interpreter and score it."""
    outs = demo_outputs(prog, [p.input for p in pairs], timeout_s=timeout_s)
    return candidate_from_outputs(prog, outs, pairs, neural=neural, source=source, alpha=alpha, beta=beta,
                                  gamma=gamma, n_cells=n_cells)


def sort_candidates(cands: Iterable[Candidate]) -> List[Candidate]:
    """Sorted by Score (desc); ties broken by fewer wrong pairs, lower complexity, then program text."""
    return sorted(cands, key=lambda c: (-c.score, c.demo_err, c.complexity, c.program.to_str()))


def dedup_candidates(cands: Iterable[Candidate]) -> List[Candidate]:
    """Keep the best-scoring candidate per canonical program."""
    best: Dict[str, Candidate] = {}
    for c in cands:
        k = c.key
        cur = best.get(k)
        if cur is None or (c.score, -c.complexity) > (cur.score, -cur.complexity):
            best[k] = c
    return sort_candidates(best.values())


def merge_candidates(*groups: Iterable[Candidate]) -> List[Candidate]:
    """Union of several candidate lists, deduplicated and sorted."""
    allc: List[Candidate] = []
    for g in groups:
        allc.extend(g)
    return dedup_candidates(allc)


def apply_prior(cands: Sequence[Candidate], prior: Optional[Prior], alpha: float = ALPHA, beta: float = BETA,
                gamma: float = GAMMA) -> List[Candidate]:
    """Re-score candidates with ``prior`` (unchanged when ``prior`` is None); returns a sorted list."""
    if prior is None or not cands:
        return sort_candidates(cands)
    scores = prior([c.program for c in cands])
    return sort_candidates(c.with_neural(s, alpha, beta, gamma) for c, s in zip(cands, scores))
