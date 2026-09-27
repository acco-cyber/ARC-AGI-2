"""Per-task solver: difficulty buckets, budget policy and the full search pipeline (INTERFACES §6).

Pipeline for one task (FROZEN_SPEC "Search")::

    parse -> difficulty bucket -> induced property -> colour table (:mod:`arcjepa.search.induce`)
          -> [rule latent r_task, memory seeds, neural prior]   (only with a model)
          -> neural beam -> repair -> TTA (+ re-guided beam) -> A* fallback -> evolutionary repair
          -> select_two per test input (+ the induced prediction: attempt 1 when the search has no exact fit for
             that test input, attempt 2 otherwise)

Budget policy by difficulty bucket (spec): D0 beam 32 / 1 repair round; D1 64 / 2; D2 128 / 4 + TTA;
D3 128 / 4 + TTA + A* fallback + evolutionary repair.  Stages after the beam run only while no exact program is
known.  **Escalation** (v1 addition, ``cfg.escalate``): when the planned stages end without an exact program and
at least 15 % of the budget is left, the unused fallbacks (A*, evolution) run in the remaining time.

The wall-clock budget ``cfg.per_task_seconds`` is enforced by one deadline shared by every stage (each stage
receives ``min(planned share, time left - reserve)`` and returns within it; when parsing and the model stage
leave less time than the planned shares add up to, every planned share is scaled down by the same factor,
``diagnostics["share_scale"]``, so the later stages are not starved); the reserve pays for executing
candidates on the test inputs.  Every interpreter call of the search stages runs inside
:func:`arcjepa.search.verifier.search_deadline` (``deadline - reserve``), so its per-call timeout is clipped to
the time left and no single execution can overrun the budget; the parser statistics stop after 20 % of the
budget.  Fallback attempts exist from the start, so ``solve_task`` always returns two valid grids per test input,
even on internal errors (recorded in ``diagnostics["error"]``).

Works with ``model=None``: the prior is uniform (s_neural = 0), TTA and memory retrieval are skipped.
"""
from __future__ import annotations

import contextlib
import logging
import math
import random
import time
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

from arcjepa.core.types import Grid, Pair, Task, copy_grid, validate_grid

from .astar import astar_search
from .beam import ArgPool, beam_search, task_palette
from .candidate import Candidate, apply_prior, merge_candidates, sort_candidates
from .diversity import fallback_grids, predict_output_shape, select_two
from .evolution import evolve
from .induce import induce_recolor, induced_candidate
from .memory_prior import MemoryPrior, NeuralPrior, rule_latent_for_task
from .repair import repair
from .tta import refine_rule_latent
from .verifier import execute_safe, is_exact, search_deadline

__all__ = ["SolveConfig", "difficulty", "difficulty_score", "parse_task", "solve_task", "BUCKET_POLICY"]

log = logging.getLogger(__name__)

#: Stage plan per difficulty bucket: (beam width, repair rounds, tta, astar, evolution).
BUCKET_POLICY: Dict[int, Dict[str, Any]] = {
    0: {"beam_width": 32, "repair_rounds": 1, "tta": False, "astar": False, "evolution": False},
    1: {"beam_width": 64, "repair_rounds": 2, "tta": False, "astar": False, "evolution": False},
    2: {"beam_width": 128, "repair_rounds": 4, "tta": True, "astar": False, "evolution": False},
    3: {"beam_width": 128, "repair_rounds": 4, "tta": True, "astar": True, "evolution": True},
}
#: Planned share of the per-task budget for each stage, by bucket.
_BEAM_SHARE = {0: 0.55, 1: 0.5, 2: 0.4, 3: 0.3}
_REPAIR_SHARE = 0.15
_TTA_SHARE = {2: 0.15, 3: 0.1}
_ASTAR_SHARE = 0.15
_EVO_SHARE = 0.1
#: the parser statistics stop starting new segmentations after this share of the budget
_PARSE_SHARE = 0.2
#: cap on the induced-table stage (measured worst case ~0.1 s on the 850 train + val tasks)
_INDUCE_SHARE = 0.1


@dataclass
class SolveConfig:
    """Every spec search knob plus the per-task wall-clock budget."""

    # search (spec v1 yaml: search.*)
    beam_width: int = 64
    max_depth: int = 6
    top_primitives: int = 8
    astar_max_nodes: int = 50_000
    alpha: float = 1.0
    beta: float = 10.0
    gamma: float = 0.15
    # test-time refinement (tta.*)
    tta_enabled: bool = True
    tta_steps: int = 8
    tta_lr: float = 0.05
    tta_anchor: float = 0.1
    # repair (repair.*)
    repair_enabled: bool = True
    repair_max_rounds: int = 4
    # memory (memory.*)
    memory_top_k: int = 16
    # outputs (outputs.*)
    num_candidates: int = 2
    # evolutionary fallback
    evo_pop: int = 32
    evo_gens: int = 20
    evo_p_mut: float = 0.4
    evo_p_cross: float = 0.2
    evo_p_neural: float = 0.4
    # budget policy
    per_task_seconds: float = 30.0
    bucket_beam_width: Tuple[int, int, int, int] = (32, 64, 128, 128)
    bucket_repair_rounds: Tuple[int, int, int, int] = (1, 2, 4, 4)
    #: D thresholds between buckets = quartiles of D over the 800 public training tasks (about 25 % per bucket)
    difficulty_thresholds: Tuple[float, float, float] = (0.38, 0.47, 0.59)
    escalate: bool = True
    #: run the induced property -> colour table stage (arcjepa.search.induce)
    induce: bool = True
    use_neural_prior: bool = True
    seed: int = 0
    #: optional TransformationMemory (not serialised)
    memory: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly dict (the memory object is omitted and never copied)."""
        return {f.name: (list(v) if isinstance(v, tuple) else v)
                for f in fields(self) if f.name != "memory" for v in [getattr(self, f.name)]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SolveConfig":
        """Build from a flat dict (unknown keys ignored)."""
        names = {f.name for f in fields(cls)}
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
        return cls(**kw)

    @classmethod
    def from_spec_yaml(cls, y: Dict[str, Any], **overrides: Any) -> "SolveConfig":
        """Build from the spec's v1 yaml layout (``search`` / ``tta`` / ``repair`` / ``memory`` / ``outputs``)."""
        s, t, r = y.get("search", {}) or {}, y.get("tta", {}) or {}, y.get("repair", {}) or {}
        kw: Dict[str, Any] = {
            "beam_width": s.get("beam_width", 64), "max_depth": s.get("max_depth", 6),
            "top_primitives": s.get("top_primitives", 8), "astar_max_nodes": s.get("astar_max_nodes", 50_000),
            "tta_enabled": t.get("enabled", True), "tta_steps": t.get("steps", 8), "tta_lr": t.get("lr", 0.05),
            "tta_anchor": t.get("anchor_weight", 0.1), "repair_enabled": r.get("enabled", True),
            "repair_max_rounds": r.get("max_rounds", 4),
            "memory_top_k": (y.get("memory", {}) or {}).get("top_k", 16),
            "num_candidates": (y.get("outputs", {}) or {}).get("num_candidates", 2),
        }
        kw.update(overrides)
        return cls(**kw)


# ============================================================================================ difficulty

def _partition_key(objs: Sequence[Any]) -> frozenset:
    return frozenset(frozenset(o.cells) for o in objs)


def parse_task(task: Task, *, max_grids: int = 4, deadline: Optional[float] = None) -> Dict[str, Any]:
    """Parser statistics used by :func:`difficulty`: objects per grid and agreement between segmentation
    hypotheses (hypotheses yielding identical partitions are grouped) on up to ``max_grids`` demo inputs.

    ``deadline`` (``time.perf_counter()`` seconds) bounds the parse: the default hypothesis of the first grid is
    always computed, after that no further hypothesis or grid is started once the deadline has passed (the result
    then carries ``"truncated": True``)."""
    grids = [p.input for p in task.train][:max_grids]
    n_objects: List[int] = []
    groups: List[List[int]] = []
    truncated = False
    try:
        from arcjepa.parser import HYPOTHESES, segment
        hyps: Tuple[str, ...] = tuple(HYPOTHESES)

        def seg(g: Grid, h: str) -> List[Any]:
            return segment(g, h)
        default = "cc4"
    except Exception:  # parser module unavailable: DSL components as three hypotheses
        from arcjepa.dsl.primitives import REGISTRY
        hyps = ("GET_COMPONENTS4", "GET_COMPONENTS8", "SELECT_ALL")

        def seg(g: Grid, h: str) -> List[Any]:
            return REGISTRY[h].fn(g)
        default = "GET_COMPONENTS4"
    order = (default,) + tuple(h for h in hyps if h != default)  # the default hypothesis first
    for g in grids:
        if not validate_grid(g):
            continue
        if n_objects and deadline is not None and time.perf_counter() > deadline:
            truncated = True
            break
        parts: Dict[frozenset, int] = {}
        n_default = 0
        for h in order:
            if h != default and deadline is not None and time.perf_counter() > deadline:
                truncated = True
                break
            try:
                objs = seg(g, h)
            except Exception:
                continue
            if h == default:
                n_default = len(objs)
            k = _partition_key(objs)
            parts[k] = parts.get(k, 0) + 1
        n_objects.append(n_default)
        groups.append(sorted(parts.values(), reverse=True))
        if truncated:
            break
    out: Dict[str, Any] = {"n_objects": n_objects, "hypothesis_groups": groups, "n_hypotheses": len(hyps)}
    if truncated:
        out["truncated"] = True
    return out


def _shape_changed(p: Pair) -> bool:
    return (len(p.input), len(p.input[0])) != (len(p.output), len(p.output[0]))


def difficulty_score(task: Task, parsed: Optional[Dict[str, Any]] = None) -> Tuple[float, Dict[str, float]]:
    """D = 0.25 H(P) + 0.2 N_obj + 0.2 N_seg + 0.2 composition + 0.15 ambiguity, every term in [0, 1].

    * H(P): normalised entropy of the distribution of hypotheses over distinct segmentations (uniform over the
      parser's hypotheses; a learned P(S_i | X) can replace it);
    * N_obj: mean default-hypothesis object count / 30 (clipped);
    * N_seg: (distinct segmentations - 1) / (hypotheses - 1);
    * composition: mean of [shape changes], [palette changes], [> 30 % of cells change] over the demos;
    * ambiguity: (4 - #demos) / 3 clipped, + 0.5 when the demos imply no consistent output shape.
    """
    parsed = parsed if parsed is not None else parse_task(task)
    nh = max(2, int(parsed.get("n_hypotheses", 10)))
    hs, segs = [], []
    for grp in parsed.get("hypothesis_groups", []):
        tot = float(sum(grp)) or 1.0
        ent = -sum((g / tot) * math.log(g / tot) for g in grp if g > 0)
        hs.append(ent / math.log(nh))
        segs.append((len(grp) - 1) / float(nh - 1))
    h_term = sum(hs) / len(hs) if hs else 0.5
    n_seg = sum(segs) / len(segs) if segs else 0.5
    nobj = parsed.get("n_objects", [])
    n_obj = min(1.0, (sum(nobj) / len(nobj)) / 30.0) if nobj else 0.5
    comp_terms = []
    for p in task.train:
        if not (validate_grid(p.input) and validate_grid(p.output)):
            continue
        shape = _shape_changed(p)
        pal = {v for r in p.input for v in r} != {v for r in p.output for v in r}
        changed = 0.0
        if not shape:
            cells = len(p.input) * len(p.input[0])
            changed = sum(1 for ri, ro in zip(p.input, p.output) for a, b in zip(ri, ro) if a != b) / float(cells)
        comp_terms.append((float(shape) + float(pal) + float(changed > 0.3)) / 3.0)
    composition = sum(comp_terms) / len(comp_terms) if comp_terms else 0.5
    ambiguity = max(0.0, (4 - len(task.train)) / 3.0)
    test_in = task.test[0].input if task.test else (task.train[0].input if task.train else [[0]])
    if predict_output_shape(task.train, test_in) is None:
        ambiguity += 0.5
    ambiguity = min(1.0, ambiguity)
    terms = {"H": h_term, "N_obj": n_obj, "N_seg": n_seg, "composition": composition, "ambiguity": ambiguity}
    d = 0.25 * h_term + 0.2 * n_obj + 0.2 * n_seg + 0.2 * composition + 0.15 * ambiguity
    return float(d), terms


def difficulty(task: Task, parsed: Optional[Dict[str, Any]] = None,
               thresholds: Sequence[float] = (0.38, 0.47, 0.59)) -> int:
    """Difficulty bucket 0-3 of ``task`` (see :func:`difficulty_score`).  The default thresholds are the
    quartiles of D over the 800 public training tasks, so each bucket holds about a quarter of them."""
    d, _ = difficulty_score(task, parsed)
    return int(sum(d >= t for t in thresholds))


# ============================================================================================ solver

def _has_exact(cands: Sequence[Candidate]) -> bool:
    return any(c.demo_err == 0 for c in cands)


def _place_induced(g: Optional[Grid], a1: Grid, a2: Grid, info: Dict[str, Any]) -> Tuple[Grid, Grid]:
    """Insert the induced prediction ``g``: attempt 2 when attempt 1 comes from an exact DSL fit, else attempt 1
    (the old attempt 1 moves to attempt 2).  ``info`` (select_two's) is updated in place."""
    if g is None or not validate_grid(g):
        return a1, a2
    if info.get("attempt_1_source") == "exact":
        if g != a1:
            info.update({"attempt_2_source": "induce", "distinct": True})
            return a1, g
        return a1, a2
    second, src2 = (a1, info.get("attempt_1_source")) if a1 != g else (a2, info.get("attempt_2_source"))
    info.update({"attempt_1_source": "induce", "attempt_2_source": src2, "distinct": second != g})
    return g, second


def solve_task(task: Task, model: Optional[Any], cfg: Optional[SolveConfig] = None
               ) -> Tuple[List[Tuple[Grid, Grid]], Dict[str, Any]]:
    """Solve one task: ``([(attempt_1, attempt_2) per test input], diagnostics)``.

    Always returns exactly one pair of valid grids per test input within ``cfg.per_task_seconds`` (+10 %).
    Diagnostics follow the spec's per-task JSON (correct, candidate_rank, program_depth, objects, hypotheses,
    beam_expansions, repair_rounds, tta_steps, inference_ms, rule_retrieval_r8) plus difficulty, bucket, stage
    timings and the best program.
    """
    cfg = cfg if cfg is not None else SolveConfig()
    t_start = time.perf_counter()
    budget = max(0.05, float(cfg.per_task_seconds))
    deadline = t_start + budget
    reserve = min(0.2 * budget, 0.02 + 0.04 * budget)
    tests = list(task.test)
    pairs = [p for p in task.train if validate_grid(p.input) and validate_grid(p.output)]
    diag: Dict[str, Any] = {
        "task_id": task.task_id, "correct": None, "candidate_rank": None, "program_depth": None, "objects": None,
        "hypotheses": None, "beam_expansions": 0, "repair_rounds": 0, "tta_steps": 0, "inference_ms": 0.0,
        "rule_retrieval_r8": None, "difficulty": None, "bucket": None, "n_candidates": 0, "n_exact": 0,
        "best_program": None, "stages": {}, "budget_s": budget,
    }

    # fallback attempts exist from the very start
    attempts: List[Tuple[Grid, Grid]] = []
    for tp in tests:
        fbs = fallback_grids(tp.input, pairs)
        a1 = fbs[0]
        a2 = next((g for g in fbs if g != a1), a1)
        attempts.append((a1, a2))

    def left() -> float:
        return deadline - reserve - time.perf_counter()

    def stage(name: str, t0: float) -> None:
        diag["stages"][name] = round(1000.0 * (time.perf_counter() - t0), 2)

    cands: List[Candidate] = []
    induced: Optional[Candidate] = None
    # every interpreter call of the search stages is clipped to the search deadline (budget minus the reserve that
    # pays for the selection); the selection itself runs up to the full deadline
    search_block = contextlib.ExitStack()
    try:
        if not pairs or not tests:
            raise _Skip("no usable demo pairs" if not pairs else "no test inputs")
        search_block.enter_context(search_deadline(deadline - reserve))
        rng = random.Random(cfg.seed)
        t0 = time.perf_counter()
        parsed = parse_task(task, deadline=t_start + _PARSE_SHARE * budget)
        if parsed.get("truncated"):
            diag["parse_truncated"] = True
        dscore, terms = difficulty_score(task, parsed)
        bucket = int(sum(dscore >= t for t in cfg.difficulty_thresholds))
        policy = dict(BUCKET_POLICY[bucket])
        policy["beam_width"] = int(cfg.bucket_beam_width[bucket])
        policy["repair_rounds"] = min(int(cfg.bucket_repair_rounds[bucket]), int(cfg.repair_max_rounds))
        diag.update({"difficulty": round(dscore, 4), "difficulty_terms": {k: round(v, 4) for k, v in terms.items()},
                     "bucket": bucket, "policy": policy})
        nobj = parsed.get("n_objects", [])
        diag["objects"] = round(sum(nobj) / len(nobj), 2) if nobj else 0
        groups = parsed.get("hypothesis_groups", [])
        diag["hypotheses"] = round(sum(len(g) for g in groups) / len(groups), 2) if groups else 0
        pal, out_pal = task_palette(pairs, [tp.input for tp in tests])
        pool = ArgPool([p.input for p in pairs], pal, out_palette=out_pal,
                       deadline=time.perf_counter() + max(0.0, 0.2 * left()))
        diag["pool"] = pool.summary()
        stage("parse", t0)

        # ------------------------------------------------------------ induced property -> colour / keep table
        if cfg.induce and left() > 0.02:
            t0 = time.perf_counter()
            test_inputs = [tp.input for tp in tests]
            try:
                rule = induce_recolor(pairs, test_inputs,
                                      deadline=time.perf_counter() + max(0.0, min(_INDUCE_SHARE * budget, left())))
                if rule is not None:
                    induced = induced_candidate(rule, pairs, test_inputs)
            except Exception as e:  # the search still runs
                log.warning("induction failed on %s: %s", task.task_id, e)
                diag["induce_error"] = repr(e)
            if induced is not None:
                diag["induced"] = {"program": induced.program.to_str(), "table_size": induced.meta["table_size"]}
                for i, (a1, a2) in enumerate(attempts):  # kept even if a later stage fails
                    attempts[i] = _place_induced(induced.meta["test_outputs"][i], a1, a2, {})
            stage("induce", t0)

        # ------------------------------------------------------------ model: rule latent, prior, memory seeds
        prior: Optional[NeuralPrior] = None
        r_task = None
        seeds: List[Any] = []
        if model is not None and left() > 0.05:
            t0 = time.perf_counter()
            try:
                r_task = rule_latent_for_task(model, task)
                if cfg.use_neural_prior:
                    prior = NeuralPrior(model, r_task)
                if cfg.memory is not None:
                    seeds = MemoryPrior(cfg.memory, model).seeds_for(r_task, k=cfg.memory_top_k)
                    if seeds:
                        diag["rule_retrieval_r8"] = float(any(is_exact(s, pairs) for s in seeds[:8]))
            except Exception as e:  # the symbolic search still runs
                log.warning("model stage failed on %s: %s", task.task_id, e)
                diag["model_error"] = repr(e)
            stage("model", t0)

        kw = dict(alpha=cfg.alpha, beta=cfg.beta, gamma=cfg.gamma)

        # planned stage shares are fractions of the budget; when parsing and the model stage leave less than the
        # planned total, every planned stage is scaled down alike (instead of the first ones eating the rest)
        tta_planned = bool(policy["tta"] and cfg.tta_enabled and prior is not None)
        planned = (_BEAM_SHARE[bucket] + (_REPAIR_SHARE if cfg.repair_enabled else 0.0)
                   + (_TTA_SHARE.get(bucket, 0.1) if tta_planned else 0.0)
                   + (_ASTAR_SHARE if policy["astar"] else 0.0) + (_EVO_SHARE if policy["evolution"] else 0.0))
        # (10 % of the time left is kept for the un-interruptible neural prior calls that end a stage late)
        scale = max(0.0, min(1.0, 0.9 * left() / max(1e-9, planned * budget)))
        diag["share_scale"] = round(scale, 3)
        unit = budget * scale  # seconds per unit of planned share

        # ------------------------------------------------------------ stage 1: neural beam
        t0 = time.perf_counter()
        st: Dict[str, Any] = {}
        b = min(_BEAM_SHARE[bucket] * unit, left())
        if b > 0.01:
            cands = beam_search(pairs, prior=prior, width=policy["beam_width"], max_depth=cfg.max_depth,
                                top_primitives=cfg.top_primitives, time_budget_s=b, seeds=seeds, pool=pool,
                                stats=st, **kw)
        diag["beam_expansions"] = int(st.get("beam_expansions", 0))
        diag["beam_levels"] = int(st.get("beam_levels", 0))
        stage("beam", t0)

        # ------------------------------------------------------------ stage 2: repair
        if cfg.repair_enabled and cands and not _has_exact(cands) and left() > 0.02:
            t0 = time.perf_counter()
            st = {}
            cands = repair(cands, pairs, rounds=policy["repair_rounds"], rng=rng, prior=prior,
                           time_budget_s=min(_REPAIR_SHARE * unit, left()), pool=pool, stats=st, **kw)
            diag["repair_rounds"] = int(st.get("repair_rounds", 0))
            stage("repair", t0)

        # ------------------------------------------------------------ stage 3: TTA (+ re-guided beam)
        if (policy["tta"] and cfg.tta_enabled and model is not None and r_task is not None and prior is not None
                and cands and left() > 0.02):
            t0 = time.perf_counter()
            try:
                r_ref = refine_rule_latent(model, r_task, pairs, cands, steps=cfg.tta_steps, lr=cfg.tta_lr,
                                           anchor=cfg.tta_anchor)
                diag["tta_steps"] = int(cfg.tta_steps)
                prior = prior.with_rule(r_ref)
                cands = apply_prior(cands, prior, cfg.alpha, cfg.beta, cfg.gamma)
                # the TTA share pays for the refinement too: the re-guided beam gets what is left of it
                b = min(_TTA_SHARE.get(bucket, 0.1) * unit - (time.perf_counter() - t0), left())
                if not _has_exact(cands) and b > 0.05:
                    more = beam_search(pairs, prior=prior, width=policy["beam_width"], max_depth=cfg.max_depth,
                                       top_primitives=cfg.top_primitives, time_budget_s=b,
                                       seeds=[c.program for c in cands[:8]], pool=pool, **kw)
                    cands = merge_candidates(cands, more)
            except Exception as e:
                log.warning("TTA failed on %s: %s", task.task_id, e)
                diag["tta_error"] = repr(e)
            stage("tta", t0)

        # ------------------------------------------------------------ stages 4-5: fallbacks (+ escalation)
        ran: set = set()

        def run_astar(share: float) -> None:
            nonlocal cands
            t1 = time.perf_counter()
            sta: Dict[str, Any] = {}
            more = astar_search(pairs, prior, max_nodes=cfg.astar_max_nodes, time_budget_s=min(share, left()),
                                max_depth=cfg.max_depth, pool=pool, seeds=[c.program for c in cands[:4]],
                                stats=sta, **kw)
            cands = merge_candidates(cands, more)
            diag["astar_nodes"] = int(sta.get("astar_nodes", 0))
            ran.add("astar")
            stage("astar", t1)

        def run_evolution(share: float) -> None:
            nonlocal cands
            t1 = time.perf_counter()
            ste: Dict[str, Any] = {}
            more = evolve(pairs, [c for c in cands[:cfg.evo_pop // 2]], prior, pop=cfg.evo_pop, gens=cfg.evo_gens,
                          p_mut=cfg.evo_p_mut, p_cross=cfg.evo_p_cross, p_neural=cfg.evo_p_neural,
                          time_budget_s=min(share, left()), rng=rng, stats=ste, **kw)
            cands = merge_candidates(cands, more)
            diag["evo_generations"] = int(ste.get("evo_generations", 0))
            ran.add("evolution")
            stage("evolution", t1)

        if policy["astar"] and not _has_exact(cands) and left() > 0.05:
            run_astar(_ASTAR_SHARE * unit)
        if policy["evolution"] and not _has_exact(cands) and left() > 0.05:
            run_evolution(_EVO_SHARE * unit)
        if cfg.escalate and not _has_exact(cands) and left() > 0.15 * budget:
            todo = [n for n in ("astar", "evolution") if n not in ran]
            for i, name in enumerate(todo):
                share = left() / (len(todo) - i)
                if share <= 0.05 or _has_exact(cands):
                    break
                (run_astar if name == "astar" else run_evolution)(share)

        # ------------------------------------------------------------ two attempts per test input
        search_block.close()
        t0 = time.perf_counter()
        cands = sort_candidates(cands)
        diag["n_candidates"] = len(cands)
        diag["n_exact"] = sum(1 for c in cands if c.demo_err == 0)
        if cands:
            diag["best_program"] = cands[0].program.to_str()
            diag["best_source"] = cands[0].source
            diag["program_depth"] = cands[0].program.depth()
        selections = []
        with search_deadline(deadline):
            for i, tp in enumerate(tests):
                share = max(0.005, (deadline - time.perf_counter()) * 0.9 / max(1, len(tests) - i))
                a1, a2, info = select_two(cands, tp.input, pairs=pairs, time_budget_s=share)
                if induced is not None:
                    a1, a2 = _place_induced(induced.meta["test_outputs"][i], a1, a2, info)
                attempts[i] = (a1, a2)
                selections.append({k: info.get(k) for k in ("attempt_1_source", "attempt_2_source", "n_clusters",
                                                             "distinct", "identity_demoted")})
        diag["selection"] = selections
        stage("select", t0)
    except _Skip as e:
        diag["skipped"] = str(e)
    except Exception as e:  # never fail a task: keep the fallback attempts
        log.exception("solver error on task %s", task.task_id)
        diag["error"] = repr(e)
    finally:
        search_block.close()

    # ------------------------------------------------------------ diagnostics against known answers
    attempts = [(a1 if validate_grid(a1) else [[0]], a2 if validate_grid(a2) else [[0]]) for a1, a2 in attempts]
    if tests and all(tp.output for tp in tests):
        diag["correct"] = [bool(tp.output == a1 or tp.output == a2) for tp, (a1, a2) in zip(tests, attempts)]
        diag["score"] = sum(diag["correct"]) / float(len(tests))
        target = tests[0].output
        with search_deadline(deadline):
            for rank, c in enumerate(cands[:16], start=1):
                if deadline - time.perf_counter() < 0.02:
                    break
                out = execute_safe(c.program, tests[0].input,
                                   timeout_s=min(0.05, max(0.001, deadline - time.perf_counter())))
                if out is not None and out == target:
                    diag["candidate_rank"] = rank
                    break
    diag["inference_ms"] = round(1000.0 * (time.perf_counter() - t_start), 2)
    return [(copy_grid(a1), copy_grid(a2)) for a1, a2 in attempts], diag


class _Skip(Exception):
    """Internal: nothing to search (the fallback attempts are returned)."""
