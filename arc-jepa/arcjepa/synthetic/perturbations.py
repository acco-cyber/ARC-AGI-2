"""Adversarial perturbations of synthetic tasks (INTERFACES.md §3, ``perturbations.py``).

Both functions perturb the task's INPUT grids and recompute every output by re-executing the task's program, so
the task stays exactly consistent with its program.  If the perturbed task would be degenerate (ExecError,
identity, constant or identical outputs, non-determinism), the original task is returned unchanged.

* :func:`add_distractors` — distractor pixels / objects in a colour absent from the task, and look-alike shapes
  (a copy of an existing object's shape in another colour).
* :func:`ambiguous_segmentation` — diagonal "bridge" pixels that join objects under 8-connectivity but not
  under 4-connectivity, so cc4 and cc8 segmentations disagree.
"""
from __future__ import annotations

import dataclasses
import random
from collections import Counter
from typing import TYPE_CHECKING, Callable, List, Optional, Set, Tuple

from arcjepa.core.types import Grid, Pair
from arcjepa.dsl.ast import Node
from arcjepa.parser.segmentation import segment
from arcjepa.synthetic.generators import place_shape

if TYPE_CHECKING:  # pragma: no cover
    from arcjepa.synthetic.dataset import SynthTask

__all__ = ["add_distractors", "ambiguous_segmentation", "perturb_inputs", "task_background"]

Cell = Tuple[int, int]


def task_background(task: "SynthTask") -> int:
    """Background colour of a task: the most common colour over all input cells (ties prefer 0)."""
    counts = Counter(v for p in task.pairs for row in p.input for v in row)
    return max(counts, key=lambda c: (counts[c], c == 0))


def perturb_inputs(task: "SynthTask", fn: Callable[[Grid], Optional[Grid]]) -> "SynthTask":
    """Apply ``fn`` to every input, re-execute the program and return the new task (or ``task`` if degenerate).

    ``fn`` returns a new grid or ``None`` (input left unchanged).  The result keeps the task's metadata, is marked
    ``adversarial=True`` when at least one input changed, and its ``difficulty`` is left for the caller to update.
    """
    from arcjepa.synthetic.dataset import degeneracy_reason, execute_pairs

    new_inputs: List[Grid] = []
    changed = False
    for p in task.pairs:
        g = fn(p.input)
        if g is None or g == p.input:
            new_inputs.append(p.input)
        else:
            new_inputs.append(g)
            changed = True
    if not changed:
        return task
    prog = Node.from_str(task.program)
    outs = execute_pairs(prog, new_inputs)
    if outs is None:
        return task
    pairs = [Pair(i, o) for i, o in zip(new_inputs, outs)]
    if degeneracy_reason(prog, pairs) is not None:
        return task
    return dataclasses.replace(task, pairs=pairs, adversarial=True)


def _free_cells(g: Grid, bg: int) -> Set[Cell]:
    return {(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v != bg}


def add_distractors(rng: random.Random, task: "SynthTask") -> "SynthTask":
    """Add 1..3 distractors per input: single pixels / small blobs in an unused colour, or look-alike shapes.

    Distractors are placed with one cell of clearance from existing content so they do not merge into objects.
    """
    used = {v for p in task.pairs for g in (p.input, p.output) for row in g for v in row}
    spare = [c for c in range(1, 10) if c not in used]
    look_alike = rng.random() < 0.5
    bg = task_background(task)

    def perturb(g: Grid) -> Optional[Grid]:
        h, w = len(g), len(g[0])
        out = [list(row) for row in g]
        occupied = _free_cells(g, bg)
        objs = segment(g, "cc4", background=bg) if look_alike else []
        palette = sorted({v for row in g for v in row} - {bg})
        for _ in range(rng.randint(1, 3)):
            if objs:
                src = rng.choice(objs)
                r0, c0 = src.bbox[0], src.bbox[1]
                rel = {(r - r0, c - c0) for r, c in src.cells}
                others = [c for c in palette if c != src.primary_color] or spare or [src.primary_color]
                col = rng.choice(others)
            else:
                size = rng.random()
                rel = {(0, 0)} if size < 0.6 else {(0, 0), (0, 1)} if size < 0.8 else {(0, 0), (1, 0)}
                col = rng.choice(spare) if spare else rng.choice([c for c in range(10) if c != bg])
            if col == bg:
                continue
            cells = place_shape(rng, occupied, rel, h, w, gap=1, tries=12)
            if cells is None:
                continue
            for r, c in cells:
                out[r][c] = col
            occupied |= cells
        return out

    return perturb_inputs(task, perturb)


def ambiguous_segmentation(rng: random.Random, task: "SynthTask") -> "SynthTask":
    """Add 1..3 diagonal bridge pixels per input so cc4 and cc8 segmentations disagree.

    A bridge pixel of colour ``col`` is placed at ``(r+dr, c+dc)`` diagonal to a foreground cell ``(r, c)`` of
    the same colour while both shared orthogonal neighbours stay background: it is a separate cc4 object but
    joins the object under cc8.
    """
    bg = task_background(task)

    def perturb(g: Grid) -> Optional[Grid]:
        h, w = len(g), len(g[0])
        out = [list(row) for row in g]
        fg = sorted(_free_cells(g, bg))
        if not fg:
            return None
        added = 0
        want = rng.randint(1, 3)
        for _ in range(want * 12):
            if added >= want:
                break
            r, c = rng.choice(fg)
            dr, dc = rng.choice(((1, 1), (1, -1), (-1, 1), (-1, -1)))
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if out[nr][nc] != bg or out[r + dr][c] != bg or out[r][c + dc] != bg:
                continue
            out[nr][nc] = out[r][c]
            added += 1
        return out if added else None

    return perturb_inputs(task, perturb)
