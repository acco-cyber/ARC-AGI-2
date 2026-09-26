"""Core ARC data types shared by every ARC-JEPA module. Do not change signatures without updating INTERFACES.md."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Grid = List[List[int]]

PAD_ID = 10
MAX_SIDE = 30
N_COLORS = 10


@dataclass(frozen=True)
class Pair:
    input: Grid
    output: Grid


@dataclass
class Task:
    task_id: str
    train: List[Pair]
    test: List[Pair]  # test[i].output == [] when the answer is unknown (competition rerun)

    @property
    def all_pairs(self) -> List[Pair]:
        return self.train + [p for p in self.test if p.output]


@dataclass
class Episode:
    episode_id: str
    task_id: str
    split: str
    context: List[Pair]
    test_input: Grid
    target_output: Optional[Grid]
    source: str = "arc"
    meta: Dict[str, Any] = field(default_factory=dict)


def grid_shape(g: Grid) -> Tuple[int, int]:
    return (len(g), len(g[0]) if g else 0)


def validate_grid(g: Any) -> bool:
    if not isinstance(g, list) or not g or not isinstance(g[0], list) or not g[0]:
        return False
    h, w = len(g), len(g[0])
    if h > MAX_SIDE or w > MAX_SIDE:
        return False
    for row in g:
        if not isinstance(row, list) or len(row) != w:
            return False
        for c in row:
            if not isinstance(c, int) or isinstance(c, bool) or c < 0 or c > 9:
                return False
    return True


def grids_equal(a: Optional[Grid], b: Optional[Grid]) -> bool:
    return a is not None and b is not None and a == b


def copy_grid(g: Grid) -> Grid:
    return [list(r) for r in g]


def task_from_json(task_id: str, d: Dict[str, Any]) -> Task:
    train = [Pair(p["input"], p["output"]) for p in d["train"]]
    test = [Pair(p["input"], p.get("output") or []) for p in d["test"]]
    return Task(task_id=task_id, train=train, test=test)


def task_to_json(t: Task) -> Dict[str, Any]:
    return {
        "train": [{"input": p.input, "output": p.output} for p in t.train],
        "test": [{"input": p.input, **({"output": p.output} if p.output else {})} for p in t.test],
    }


def episode_from_json(d: Dict[str, Any]) -> Episode:
    return Episode(
        episode_id=d["episode_id"],
        task_id=d["task_id"],
        split=d.get("split", ""),
        context=[Pair(p["input"], p["output"]) for p in d["context"]],
        test_input=d["test_input"],
        target_output=d.get("target_output"),
        source=d.get("source", d.get("kind", "arc")),
        meta={k: v for k, v in d.items() if k not in ("context", "test_input", "target_output")},
    )
