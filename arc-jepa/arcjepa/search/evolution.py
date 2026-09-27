"""Evolutionary fallback search / repair (FROZEN_SPEC "Evolutionary fallback", INTERFACES §6).

Population 32, 20 generations; each child is produced by type-preserving mutation (40 %), crossover (20 %) or
neural-guided mutation (40 %: several mutations of the parent are proposed and the one the prior likes best is
kept; with the uniform prior the proposal with the lowest demo loss is kept instead).  Fitness is the spec's
``F = -10 E_demo + s_neural - 0.15 C(p)`` with ``E_demo`` realised as L_demo (wrong pairs + cell fraction, see
:mod:`arcjepa.search.candidate`).  Parents are chosen by size-3 tournaments; the best eighth of the population is
carried over unchanged (elitism).  Every program is verified exactly with the interpreter and evaluations are
memoised by canonical form.  A generation keeps at most one program per demo behaviour (its outputs on the
demos), like the beam's observational-equivalence merge: no-op wrappers of a member would otherwise crowd the
population out.  Guided proposals are stratified by edit kind (one literal change, one op swap, ...).
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Union

from arcjepa.core.types import Pair
from arcjepa.dsl.ast import Node
from arcjepa.dsl.grammar import random_program
from arcjepa.dsl.mutations import MUTATION_KINDS, crossover, mutate

from .candidate import (ALPHA, BETA, GAMMA, Candidate, Prior, canonical_key, make_candidate, sort_candidates)
from .verifier import past_search_deadline, total_cells

__all__ = ["evolve"]

log = logging.getLogger(__name__)


class _Timeout(Exception):
    pass


def evolve(pairs: Sequence[Pair], seeds: Sequence[Union[Node, Candidate]], prior: Optional[Prior], pop: int = 32,
           gens: int = 20, p_mut: float = 0.4, p_cross: float = 0.2, p_neural: float = 0.4,
           time_budget_s: float = 10.0, *, rng: Optional[random.Random] = None, alpha: float = ALPHA,
           beta: float = BETA, gamma: float = GAMMA, n_proposals: int = 6, init_depths: Sequence[int] = (1, 2, 3),
           early_stop: bool = True, max_results: int = 64, stats: Optional[Dict[str, Any]] = None
           ) -> List[Candidate]:
    """Run the evolutionary search; returns the best distinct candidates found, sorted by Score (fitness).

    ``seeds`` may be programs or candidates (e.g. beam / repair near-misses); the population is topped up with
    random typed programs.  Extra keyword arguments are optional (``rng`` for determinism, scoring weights,
    proposal count, initial depths, ``early_stop`` after the generation that found an exact program,
    ``max_results`` and a ``stats`` dict).
    """
    t0 = time.perf_counter()
    deadline = t0 + 0.95 * max(0.0, float(time_budget_s))
    rng = rng if rng is not None else random.Random(0)
    pairs = list(pairs)
    st: Dict[str, Any] = stats if stats is not None else {}
    st.update({"evo_generations": 0, "evo_evaluations": 0, "evo_timed_out": False})
    if not pairs or pop <= 0:
        return []
    n_cells = total_cells(pairs)
    cache: Dict[str, Candidate] = {}
    neural_cache: Dict[str, float] = {}

    def neural_of(progs: List[Node]) -> List[float]:
        if prior is None:
            return [0.0] * len(progs)
        keys = [p.to_str() for p in progs]
        missing = [p for p, k in zip(progs, keys) if k not in neural_cache]
        if missing:
            for p, s in zip(missing, prior(missing)):
                neural_cache[p.to_str()] = float(s)
        return [neural_cache[k] for k in keys]

    def evaluate(prog: Node, source: str = "evolution") -> Optional[Candidate]:
        if time.perf_counter() > deadline:
            raise _Timeout
        key = canonical_key(prog)
        c = cache.get(key)
        if c is None:
            c = make_candidate(prog, pairs, neural=neural_of([prog])[0], source=source, alpha=alpha, beta=beta,
                               gamma=gamma, n_cells=n_cells)
            if past_search_deadline():
                raise _Timeout  # executions may have been cut short by the solver's deadline: do not cache
            cache[key] = c
            st["evo_evaluations"] += 1
        if all(o is None for o in (c.outputs or [None])):
            return None  # never executes: useless as a parent
        return c

    def tournament(popl: List[Candidate]) -> Candidate:
        k = min(3, len(popl))
        return max(rng.sample(popl, k), key=lambda c: c.score)

    population: List[Candidate] = []
    try:
        for s in seeds:
            prog = s.program if isinstance(s, Candidate) else s
            if isinstance(prog, Node):
                c = evaluate(prog, "seed")
                if c is not None:
                    population.append(c)
            if len(population) >= pop:
                break
        tries = 0
        while len(population) < pop and tries < 8 * pop:
            tries += 1
            c = evaluate(random_program(rng, rng.choice(list(init_depths))), "evolution")
            if c is not None:
                population.append(c)
        if not population:
            return sort_candidates(cache.values())[:max_results]
        n_elite = max(1, pop // 8)
        for _gen in range(int(gens)):
            population = sort_candidates(population)
            if early_stop and population[0].demo_err == 0:
                break
            nxt: List[Candidate] = list(population[:n_elite])
            behaviours = {_behaviour(c) for c in nxt}
            attempts = 0
            while len(nxt) < pop and attempts < 4 * pop:
                attempts += 1
                u = rng.random()
                parent = tournament(population)
                if u < p_mut:
                    child = mutate(rng, parent.program)
                elif u < p_mut + p_cross:
                    child = crossover(rng, parent.program, tournament(population).program)
                elif u < p_mut + p_cross + p_neural:
                    child = _guided_mutation(rng, parent.program, n_proposals, prior, neural_of, evaluate)
                else:
                    child = random_program(rng, rng.choice(list(init_depths)))
                c = evaluate(child)
                if c is None:
                    continue
                b = _behaviour(c)
                if b in behaviours:  # observationally equivalent to a member (e.g. a no-op wrapper): skip it
                    continue
                behaviours.add(b)
                nxt.append(c)
            population = nxt
            st["evo_generations"] += 1
    except _Timeout:
        st["evo_timed_out"] = True
    st["evo_seconds"] = time.perf_counter() - t0
    return sort_candidates(cache.values())[:max_results]


def _behaviour(c: Candidate) -> Any:
    """Observational-equivalence key of a candidate: its demo outputs (failed executions as ``None``)."""
    return tuple(None if o is None else tuple(tuple(r) for r in o) for o in (c.outputs or ()))


def _guided_mutation(rng: random.Random, prog: Node, n: int, prior: Optional[Prior], neural_of: Any,
                     evaluate: Any) -> Node:
    """Propose ``n`` mutations; keep the prior's favourite (or the lowest-loss one under the uniform prior).

    Proposals are stratified by edit kind (:data:`arcjepa.dsl.mutations.MUTATION_KINDS`, cycled from a random
    start), so every guided step tries a literal change, an op swap, a subtree replacement, ... instead of
    ``n`` edits of whatever kinds a uniform draw happens to repeat."""
    props: List[Node] = []
    seen = set()
    start = rng.randrange(len(MUTATION_KINDS))
    for i in range(max(1, n)):
        m = mutate(rng, prog, MUTATION_KINDS[(start + i) % len(MUTATION_KINDS)])
        k = m.to_str()
        if k not in seen:
            seen.add(k)
            props.append(m)
    if len(props) == 1:
        return props[0]
    if prior is not None:
        scores = neural_of(props)
        return props[max(range(len(props)), key=lambda i: scores[i])]
    best, best_loss = props[0], float("inf")
    for m in props:
        c = evaluate(m)
        if c is not None and c.loss < best_loss:
            best, best_loss = m, c.loss
    return best
