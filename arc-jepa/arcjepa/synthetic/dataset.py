"""Synthetic program-generated ARC tasks: sampling, degeneracy filtering, compositional split, JSONL writer.

INTERFACES.md §3 (``dataset.py``):

* :class:`SynthTask` — one program-generated task (canonical program string, 3..6 demonstration pairs, depth,
  primitives, category, difficulty, adversarial flag).
* :func:`make_task` — sample one task; returns ``None`` when it is degenerate (identity on every pair, constant
  outputs, the same output for every input, ``ExecError``, non-deterministic re-execution, a side > 30).
* :func:`generate` — write ``n`` tasks as JSONL rows ``{task_id, program, pairs:[{input,output}], depth,
  primitives, category, difficulty, adversarial, split}`` with ``multiprocessing`` workers.  The output is a
  function of ``(n, seed, split_rule)`` only: work is cut into fixed-size chunks with their own seeds, so the
  number of workers does not change a single byte.
* Compositional split: a task goes to ``val_comp`` iff its primitive set contains BOTH members of one of the
  :data:`HELDOUT_COMPOSITIONS` pairs; otherwise it is ``train``.  Every primitive stays seen in training, only
  these compositions are unseen.  Synthetic programs are never randomly split.

CLI::

    python -m arcjepa.synthetic.dataset --n N --out PATH --seed S --workers W [--split-rule compositional]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from arcjepa.core.types import Grid, MAX_SIDE, Pair, Task, validate_grid
from arcjepa.dsl.ast import Node
from arcjepa.dsl.canonicalize import canonicalize
from arcjepa.dsl.interpreter import ExecError, execute
from arcjepa.dsl.primitives import REGISTRY
from arcjepa.dsl.types import T
from arcjepa.synthetic.generators import STYLE_WEIGHTS, random_input_grid, random_palette, shaped_input_grid
from arcjepa.synthetic.program_sampler import sample_category, sample_depth, sample_program

__all__ = [
    "SynthTask", "HELDOUT_COMPOSITIONS", "SPLIT_RULES", "EXEC_TIMEOUT_S", "CHUNK_SIZE", "make_task",
    "make_task_with_reason", "input_shape_hint", "generate", "split_of", "difficulty_score", "count_cc4", "task_difficulty", "degeneracy_reason", "execute_pairs",
    "program_colors", "task_to_row", "row_to_task", "iter_rows", "load_jsonl", "to_arc_task", "main",
]

log = logging.getLogger(__name__)

#: Held-out primitive PAIRS: a task whose primitive set contains both members of any pair is ``val_comp``.
#: (Spec example: ROTATE+RECOLOR / MOVE+COPY / FILTER+MOVE stay in train, ROTATE+COPY+RECOLOR is validation.)
#: Pairs were picked from measured co-occurrence rates so that val_comp is ~8-10 % of the stream and spans
#: object, geometry, pattern, colour, counting and relational compositions.
HELDOUT_COMPOSITIONS: Tuple[Tuple[str, str], ...] = (
    ("ROTATE90", "COPY"),
    ("REFLECT_H", "RECOLOR"),
    ("FRAME", "OUTLINE"),
    ("COPY", "MERGE"),
    ("ALIGN", "MERGE"),
    ("MIRROR_TILE", "SWAP_COLORS"),
    ("PERIODIC_REPEAT", "REFLECT_V"),
    ("COLOR_BY_POSITION", "MAP_COLOR"),
    ("ARGMAX_SIZE", "FILL"),
    ("COUNT_OBJECTS", "REPEAT_X"),
    ("NEAREST", "MERGE"),
)
SPLIT_RULES: Tuple[str, ...] = ("compositional", "none")
#: Wall-clock limit per program execution inside the generator (the step budget is the main guard).
EXEC_TIMEOUT_S: float = 0.25
#: Tasks per work chunk; each chunk has its own seed so output is independent of the worker count.
CHUNK_SIZE: int = 250
#: make_task attempts per (depth, category) slot before the slot is re-drawn.
_MAX_SLOT_TRIES: int = 400

PairsLike = Sequence[Pair]


@dataclass
class SynthTask:
    """One synthetic program-generated task (INTERFACES.md §3)."""

    task_id: str
    program: str
    pairs: List[Pair]
    depth: int
    primitives: List[str]
    category: str
    difficulty: float
    adversarial: bool

    def node(self) -> Node:
        """The program as an AST node."""
        return Node.from_str(self.program)


# ======================================================================================= helpers

def program_colors(prog: Node) -> List[int]:
    """COLOR literals appearing in ``prog`` (in pre-order, de-duplicated)."""
    out: List[int] = []

    def walk(n: Node) -> None:
        prim = REGISTRY.get(n.op)
        for i, a in enumerate(n.args):
            if isinstance(a, Node):
                walk(a)
            elif prim is not None and i < len(prim.arg_types) and prim.arg_types[i] is T.COLOR \
                    and isinstance(a, int) and not isinstance(a, bool) and a not in out:
                out.append(a)

    walk(prog)
    return out


def execute_pairs(prog: Node, inputs: Sequence[Grid]) -> Optional[List[Grid]]:
    """Execute ``prog`` on every input; ``None`` on any ExecError or invalid output."""
    outs: List[Grid] = []
    for g in inputs:
        try:
            outs.append(execute(prog, g, timeout_s=EXEC_TIMEOUT_S))
        except ExecError:
            return None
    return outs


def _uniform(g: Grid) -> bool:
    first = g[0][0]
    return all(v == first for row in g for v in row)


def degeneracy_reason(prog: Node, pairs: PairsLike, *, recheck: Union[bool, int] = 1) -> Optional[str]:
    """Return why ``pairs`` generated by ``prog`` form a degenerate task, or ``None`` when the task is usable.

    Reasons: ``too_few_pairs`` (< 2), ``oversize`` (a side > 30 / invalid grid), ``duplicate_inputs``,
    ``identity`` (output == input on every pair), ``constant_output`` (every output is a single colour),
    ``same_output`` (all outputs identical), ``nondeterministic`` (re-execution differs).

    ``recheck`` = how many pairs (from the end) are re-executed for the determinism check: ``True`` = all,
    ``False``/0 = none, an int = that many.  The generator re-executes one pair (throughput); the test-suite
    re-executes every pair of every task.
    """
    if len(pairs) < 2:
        return "too_few_pairs"
    for p in pairs:
        if not validate_grid(p.input) or not validate_grid(p.output):
            return "oversize"
        if len(p.output) > MAX_SIDE or len(p.output[0]) > MAX_SIDE:
            return "oversize"  # pragma: no cover - validate_grid already enforces this
    ins = [p.input for p in pairs]
    if any(ins[i] == ins[j] for i in range(len(ins)) for j in range(i)):
        return "duplicate_inputs"
    if all(p.output == p.input for p in pairs):
        return "identity"
    if all(_uniform(p.output) for p in pairs):
        return "constant_output"
    first = pairs[0].output
    if all(p.output == first for p in pairs[1:]):
        return "same_output"
    n_check = len(pairs) if recheck is True else int(recheck)
    if n_check > 0:
        chk = list(pairs)[-n_check:]
        again = execute_pairs(prog, [p.input for p in chk])
        if again is None or any(a != p.output for a, p in zip(again, chk)):
            return "nondeterministic"
    return None


def split_of(primitives: Sequence[str], split_rule: str = "compositional") -> str:
    """``val_comp`` iff the primitive set contains both members of a :data:`HELDOUT_COMPOSITIONS` pair."""
    if split_rule == "none":
        return "train"
    if split_rule != "compositional":
        raise ValueError(f"unknown split_rule {split_rule!r}; expected one of {SPLIT_RULES}")
    ps = set(primitives)
    for a, b in HELDOUT_COMPOSITIONS:
        if a in ps and b in ps:
            return "val_comp"
    return "train"


def difficulty_score(depth: int, n_objects: float, palette_size: int, adversarial: bool) -> float:
    """Difficulty in [0, 1]: 0.45·depth + 0.20·objects + 0.15·palette + 0.20·adversarial (each term in [0, 1]).

    ``depth`` 1..6 maps to 0..1, ``n_objects`` (mean cc4 objects per input) saturates at 10, ``palette_size``
    (distinct non-background colours over the task) maps 1..9 to 0..1.
    """
    d = min(max((depth - 1) / 5.0, 0.0), 1.0)
    o = min(max(n_objects / 10.0, 0.0), 1.0)
    p = min(max((palette_size - 1) / 8.0, 0.0), 1.0)
    a = 1.0 if adversarial else 0.0
    return round(min(1.0, 0.45 * d + 0.20 * o + 0.15 * p + 0.20 * a), 4)


def _background(g: Grid) -> int:
    """Most common colour of ``g`` (ties prefer 0), the same rule as ``perturbations.task_background``."""
    counts = Counter(v for row in g for v in row)
    return max(counts, key=lambda c: (counts[c], c == 0))


def count_cc4(g: Grid, background: int = 0) -> int:
    """Number of same-colour 4-connected non-background components (= ``len(segment(g, "cc4"))``, faster)."""
    h, w = len(g), len(g[0])
    flat = [v for row in g for v in row]
    seen = [False] * (h * w)
    n = 0
    for s in range(h * w):
        if seen[s] or flat[s] == background:
            continue
        n += 1
        col = flat[s]
        seen[s] = True
        stack = [s]
        while stack:
            i = stack.pop()
            r, c = divmod(i, w)
            if r > 0 and not seen[i - w] and flat[i - w] == col:
                seen[i - w] = True
                stack.append(i - w)
            if r < h - 1 and not seen[i + w] and flat[i + w] == col:
                seen[i + w] = True
                stack.append(i + w)
            if c > 0 and not seen[i - 1] and flat[i - 1] == col:
                seen[i - 1] = True
                stack.append(i - 1)
            if c < w - 1 and not seen[i + 1] and flat[i + 1] == col:
                seen[i + 1] = True
                stack.append(i + 1)
    return n


def task_difficulty(pairs: PairsLike, depth: int, adversarial: bool) -> float:
    """:func:`difficulty_score` computed from the task's grids (cc4 object count, palette size)."""
    n_obj = 0.0
    colours = set()
    for p in pairs:
        bg = _background(p.input)
        n_obj += count_cc4(p.input, bg)
        colours.update(*p.input, *p.output)
        colours.discard(bg)
    n_obj /= max(1, len(pairs))
    return difficulty_score(depth, n_obj, len(colours), adversarial)


# ======================================================================================= task sampling

def _choose_style(rng: random.Random) -> str:
    styles = list(STYLE_WEIGHTS)
    return rng.choices(styles, weights=[STYLE_WEIGHTS[s] for s in styles])[0]


def _int_literal(prog: Node, op: str) -> Optional[int]:
    """The INTEGER literal of the first ``op`` node of ``prog`` (pre-order), if it is a raw literal."""
    for _, n in prog.iter_nodes():
        if n.op == op and len(n.args) > 1 and isinstance(n.args[1], int) and not isinstance(n.args[1], bool):
            return int(n.args[1])
    return None


def input_shape_hint(prog: Node) -> Optional[Tuple[str, Optional[int]]]:
    """``(hint, k)`` for :func:`arcjepa.synthetic.generators.shaped_input_grid` when ``prog`` contains a spec
    extension that random ARC-like inputs almost never satisfy, else ``None`` (inputs are drawn as usual).

    Priority: KRON_SELF (sides <= 5) > DOWNSCALE / DOWNSCALE_ANY (block-structured, k from the literal) > panel
    ops (separated panels; a PANEL_OVERLAY order literal ``o`` needs more than ``o / 2`` panels, so k is that
    minimum panel count) > UPSCALE / UPSCALE_NC (small sides).
    """
    ops = prog.primitives()
    if "KRON_SELF" in ops:
        return "kron", None
    if "DOWNSCALE" in ops:
        return "blocks", _int_literal(prog, "DOWNSCALE")
    if "DOWNSCALE_ANY" in ops:
        return "blocks_any", _int_literal(prog, "DOWNSCALE_ANY")
    if "PANEL_BOOL" in ops or "PANEL_OVERLAY" in ops:
        order = _int_literal(prog, "PANEL_OVERLAY")
        return "panels", (None if order is None else min(4, order // 2 + 1))
    if "UPSCALE" in ops:
        return "small", _int_literal(prog, "UPSCALE")
    if "UPSCALE_NC" in ops:
        return "small", 4
    return None


def make_task_with_reason(rng: random.Random, *, n_pairs: Tuple[int, int] = (3, 6),
                          category: Optional[str] = None, depth: Optional[int] = None,
                          program: Union[Node, str, None] = None, task_id: Optional[str] = None
                          ) -> Tuple[Optional[SynthTask], str]:
    """:func:`make_task` that also returns the rejection reason (``"ok"`` on success)."""
    cat = category if category is not None else sample_category(rng)
    if program is None:
        target = depth if depth is not None else sample_depth(rng)
        prog = sample_program(rng, cat, depth=target, max_tries=2)
        if prog.depth() != target:
            return None, "depth_mismatch"
    else:
        prog = canonicalize(Node.from_str(program) if isinstance(program, str) else program)
    if prog.depth() < 1:
        return None, "trivial_program"
    tid = task_id if task_id is not None else f"syn-{rng.getrandbits(48):012x}"

    bg = 0 if rng.random() < 0.92 else rng.randint(1, 9)
    lits = [c for c in program_colors(prog) if c != bg][:3]
    palette = random_palette(rng, bg, include=lits)
    style = _choose_style(rng)
    hint = input_shape_hint(prog)
    lo, hi = n_pairs
    k = rng.randint(lo, hi)
    pairs: List[Pair] = []
    fails = 0
    while len(pairs) < k:
        if hint is None:
            g = random_input_grid(rng, style=style, palette=palette, background=bg)
        else:
            g = shaped_input_grid(rng, hint[0], palette=palette, background=bg, style=style, k=hint[1])
        try:
            out = execute(prog, g, timeout_s=EXEC_TIMEOUT_S)
        except ExecError:
            fails += 1
            if not pairs or fails > k:  # a program that fails on the very first input is dropped at once
                return None, "exec_error"
            continue
        if any(p.input == g for p in pairs):
            fails += 1
            if fails > k:
                return None, "duplicate_inputs"
            continue
        pairs.append(Pair(g, out))
        if len(pairs) == 2:  # early exit: programs that are identity / constant on two random inputs are dropped
            early = degeneracy_reason(prog, pairs, recheck=0)
            if early in ("identity", "constant_output"):
                return None, early
    if len(pairs) < lo:
        return None, "too_few_pairs"  # pragma: no cover - loop exits only with k pairs
    reason = degeneracy_reason(prog, pairs)
    if reason is not None:
        return None, reason
    task = SynthTask(task_id=tid, program=prog.to_str(), pairs=pairs, depth=prog.depth(),
                     primitives=sorted(prog.primitives()), category=cat, difficulty=0.0,
                     adversarial=cat == "adversarial")
    if task.adversarial:
        from arcjepa.synthetic.perturbations import add_distractors, ambiguous_segmentation
        u = rng.random()
        if u < 0.45:
            task = add_distractors(rng, task)
        elif u < 0.8:
            task = ambiguous_segmentation(rng, task)
        else:
            task = ambiguous_segmentation(rng, add_distractors(rng, task))
    task.difficulty = task_difficulty(task.pairs, task.depth, task.adversarial)
    return task, "ok"


def make_task(rng: random.Random, *, n_pairs: Tuple[int, int] = (3, 6), category: Optional[str] = None,
              depth: Optional[int] = None, program: Union[Node, str, None] = None,
              task_id: Optional[str] = None) -> Optional[SynthTask]:
    """Sample one synthetic task, or ``None`` when the draw is degenerate.

    Args:
        rng: source of randomness (the task is a deterministic function of the rng state).
        n_pairs: inclusive range for the number of demonstration pairs.
        category: sample category (drawn from ``CATEGORY_MIX`` when ``None``).
        depth: program depth (drawn from ``DEPTH_MIX`` when ``None``); the canonical program must have exactly
            this depth, otherwise the draw is rejected.
        program: use this program (Node or S-expression) instead of sampling one.
        task_id: explicit id (random hex id from ``rng`` when ``None``).

    Returns ``None`` for: ExecError on the inputs, identity on every pair, constant (single-colour) outputs,
    identical outputs for all inputs, non-deterministic re-execution, a side > 30, or a canonical depth mismatch.
    """
    return make_task_with_reason(rng, n_pairs=n_pairs, category=category, depth=depth, program=program,
                                 task_id=task_id)[0]


# ======================================================================================= (de)serialisation

def task_to_row(task: SynthTask, split: str) -> Dict[str, Any]:
    """JSONL row for ``task`` (INTERFACES.md field order)."""
    return {
        "task_id": task.task_id,
        "program": task.program,
        "pairs": [{"input": p.input, "output": p.output} for p in task.pairs],
        "depth": task.depth,
        "primitives": list(task.primitives),
        "category": task.category,
        "difficulty": task.difficulty,
        "adversarial": bool(task.adversarial),
        "split": split,
    }


def row_to_task(row: Dict[str, Any]) -> SynthTask:
    """Inverse of :func:`task_to_row` (the ``split`` field is dropped)."""
    return SynthTask(task_id=row["task_id"], program=row["program"],
                     pairs=[Pair(p["input"], p["output"]) for p in row["pairs"]], depth=int(row["depth"]),
                     primitives=list(row["primitives"]), category=row["category"],
                     difficulty=float(row["difficulty"]), adversarial=bool(row["adversarial"]))


def iter_rows(path: str, split: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Stream JSONL rows from ``path`` (optionally only those of one ``split``)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if split is None or row.get("split") == split:
                yield row


def load_jsonl(path: str, split: Optional[str] = None) -> List[SynthTask]:
    """Load every task of ``path`` (optionally only one ``split``)."""
    return [row_to_task(r) for r in iter_rows(path, split)]


def to_arc_task(task: SynthTask, n_test: int = 1) -> Task:
    """View a synthetic task as a core :class:`Task` (the last ``n_test`` pairs become the test pairs)."""
    n_test = max(1, min(n_test, len(task.pairs) - 1))
    return Task(task_id=task.task_id, train=list(task.pairs[:-n_test]), test=list(task.pairs[-n_test:]))


# ======================================================================================= generation

def _chunk_rng(seed: int, chunk: int) -> random.Random:
    return random.Random(f"arcjepa-synthetic:{seed}:{chunk}")


def _gen_chunk(args: Tuple[int, int, int, str, Tuple[int, int]]) -> Tuple[List[str], Dict[str, int]]:
    """Worker: generate ``count`` tasks for chunk ``chunk`` -> (JSONL lines, counters incl. ``compute_us`` (wall)
    and ``cpu_us`` (this process's CPU time))."""
    seed, chunk, count, split_rule, n_pairs = args
    t0 = time.perf_counter()
    c0 = time.process_time()
    rng = _chunk_rng(seed, chunk)
    stats: Counter = Counter()
    lines: List[str] = []
    for i in range(count):
        tid = f"syn{seed}-{chunk:05d}-{i:04d}"
        depth, cat = sample_depth(rng), sample_category(rng)
        tries = 0
        while True:
            task, reason = make_task_with_reason(rng, n_pairs=n_pairs, category=cat, depth=depth, task_id=tid)
            stats["attempts"] += 1
            if task is not None:
                break
            stats[f"reject_{reason}"] += 1
            tries += 1
            if tries >= _MAX_SLOT_TRIES:  # pragma: no cover - never observed; keeps the loop total
                log.warning("slot depth=%d category=%s re-drawn after %d failures", depth, cat, tries)
                stats["slot_redrawn"] += 1
                depth, cat, tries = sample_depth(rng), sample_category(rng), 0
        split = split_of(task.primitives, split_rule)
        stats[split] += 1
        stats[f"depth_{task.depth}"] += 1
        stats[f"category_{task.category}"] += 1
        stats["adversarial"] += int(task.adversarial)
        lines.append(json.dumps(task_to_row(task, split), separators=(",", ":")))
    stats["compute_us"] += int((time.perf_counter() - t0) * 1e6)
    stats["cpu_us"] += int((time.process_time() - c0) * 1e6)
    return lines, dict(stats)


def generate(n: int, out_path: str, seed: int, workers: int = 1, split_rule: str = "compositional", *,
             n_pairs: Tuple[int, int] = (3, 6), chunk_size: int = CHUNK_SIZE) -> Dict[str, int]:
    """Generate ``n`` synthetic tasks into the JSONL file ``out_path``.

    Output is deterministic in ``(n, seed, split_rule, n_pairs, chunk_size)`` and independent of ``workers``.

    Returns counters: ``n``, ``train``, ``val_comp``, ``attempts``, ``reject_<reason>``, ``depth_<d>``,
    ``category_<c>``, ``adversarial``, ``elapsed_ms`` (wall), ``compute_ms`` (summed worker wall time),
    ``cpu_ms`` (summed worker CPU time), ``tasks_per_s`` (wall) and ``tasks_per_s_per_worker`` (n / summed worker
    CPU seconds: the per-core throughput, independent of how many other processes share the machine; falls back
    to the worker wall time when the CPU clock reports nothing).
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if split_rule not in SPLIT_RULES:
        raise ValueError(f"unknown split_rule {split_rule!r}; expected one of {SPLIT_RULES}")
    chunk_size = max(1, int(chunk_size))
    n_chunks = math.ceil(n / chunk_size) if n else 0
    jobs = [(seed, c, min(chunk_size, n - c * chunk_size), split_rule, tuple(n_pairs)) for c in range(n_chunks)]
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    t0 = time.perf_counter()
    totals: Counter = Counter()
    workers = max(1, int(workers))
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        if workers == 1 or len(jobs) <= 1:
            results: Iterator[Tuple[List[str], Dict[str, int]]] = map(_gen_chunk, jobs)
            for lines, st in results:
                f.write("".join(line + "\n" for line in lines))
                totals.update(st)
        else:
            with mp.get_context().Pool(processes=min(workers, len(jobs))) as pool:
                for lines, st in pool.imap(_gen_chunk, jobs):
                    f.write("".join(line + "\n" for line in lines))
                    totals.update(st)
    elapsed = time.perf_counter() - t0
    out: Dict[str, int] = {"n": n, "train": 0, "val_comp": 0}
    out.update({k: int(v) for k, v in totals.items() if k not in ("compute_us", "cpu_us")})
    compute_s = totals.get("compute_us", 0) / 1e6
    cpu_s = totals.get("cpu_us", 0) / 1e6
    out["elapsed_ms"] = int(elapsed * 1000)
    out["compute_ms"] = int(compute_s * 1000)
    out["cpu_ms"] = int(cpu_s * 1000)
    out["tasks_per_s"] = int(n / elapsed) if elapsed > 0 else 0
    per_worker_s = cpu_s if cpu_s > 0 else compute_s
    out["tasks_per_s_per_worker"] = int(n / per_worker_s) if per_worker_s > 0 else 0
    log.info("generated %d tasks -> %s (%s)", n, out_path, out)
    return out


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, int]:
    """CLI entry point: ``python -m arcjepa.synthetic.dataset --n N --out PATH --seed S --workers W``."""
    ap = argparse.ArgumentParser(description="Generate synthetic ARC-JEPA tasks as JSONL.")
    ap.add_argument("--n", type=int, required=True, help="number of tasks")
    ap.add_argument("--out", type=str, required=True, help="output JSONL path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--split-rule", type=str, default="compositional", choices=SPLIT_RULES)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    stats = generate(a.n, a.out, a.seed, workers=a.workers, split_rule=a.split_rule, chunk_size=a.chunk_size)
    sys.stdout.write(json.dumps(stats, sort_keys=True) + "\n")
    return stats


if __name__ == "__main__":
    # Re-import under the package name so pool workers pickle ``arcjepa.synthetic.dataset._gen_chunk``.
    from arcjepa.synthetic.dataset import main as _main

    _main()
