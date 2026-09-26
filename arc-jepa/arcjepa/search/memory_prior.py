"""Model-facing priors for the search: memory-retrieved seed programs and the neural program prior.

* :class:`MemoryPrior` (INTERFACES §6): ``seeds_for(r_task, k=16)`` queries the transformation memory (cosine KNN
  over rule latents) and returns the retrieved programs as typed GRID ASTs, optionally re-ranked by the model's
  scorer.
* :class:`NeuralPrior`: a cached callable ``List[Node] -> List[float]`` returning the model's s_neural for a fixed
  rule latent, clipped to ``[-4, 4]``: the largest possible neural difference (8) stays below the cost of one
  wrong demo pair (beta = 10), so exact programs always outrank non-exact ones.
* :func:`rule_latent_for_task`: r_task for a :class:`~arcjepa.core.types.Task` (tensorises the demos with the data
  module when available, else a grid-only episode batch).
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from arcjepa.core.types import MAX_SIDE, PAD_ID, Episode, Grid, Task
from arcjepa.dsl.ast import Node
from arcjepa.dsl.interpreter import typecheck
from arcjepa.dsl.types import T

__all__ = ["MemoryPrior", "NeuralPrior", "rule_latent_for_task", "episode_batch_for_task"]

log = logging.getLogger(__name__)


# ============================================================================================ neural prior

class NeuralPrior:
    """``prior(programs) -> scores``: the model's s_neural against a fixed rule latent (batched, memoised)."""

    def __init__(self, model: Any, r_task: Tensor, *, batch_size: int = 256, clip: float = 4.0,
                 cache_size: int = 200_000) -> None:
        self.model = model
        self.r = r_task.detach().reshape(1, -1)
        self.batch_size = int(batch_size)
        self.clip = float(clip)
        self.cache_size = int(cache_size)
        self._cache: "OrderedDict[str, float]" = OrderedDict()
        self.calls = 0

    def with_rule(self, r_task: Tensor) -> "NeuralPrior":
        """A new prior for another rule latent (e.g. after test-time refinement)."""
        return NeuralPrior(self.model, r_task, batch_size=self.batch_size, clip=self.clip, cache_size=self.cache_size)

    def __call__(self, programs: List[Node]) -> List[float]:
        keys = [p.to_str() for p in programs]
        todo_idx = [i for i, k in enumerate(keys) if k not in self._cache]
        if todo_idx:
            self.calls += 1
            model = self.model
            was_training = bool(getattr(model, "training", False))
            model.eval()
            try:
                dev = next(model.parameters()).device
                r = self.r.to(dev)
                with torch.no_grad():
                    for s in range(0, len(todo_idx), self.batch_size):
                        chunk = todo_idx[s:s + self.batch_size]
                        progs = [programs[i] for i in chunk]
                        try:
                            scores = model.score_programs(r, progs).reshape(-1).float().cpu().numpy()
                        except Exception as e:  # untokenisable program etc.: neutral score
                            log.debug("neural prior failed on a batch (%s); using 0", e)
                            scores = np.zeros(len(chunk), dtype=np.float32)
                        for i, v in zip(chunk, scores):
                            self._cache[keys[i]] = float(np.clip(v, -self.clip, self.clip))
            finally:
                if was_training:
                    model.train()
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return [self._cache.get(k, 0.0) for k in keys]


# ============================================================================================ memory prior

class MemoryPrior:
    """Seed programs retrieved from a :class:`~arcjepa.model.memory.TransformationMemory` by rule latent."""

    def __init__(self, memory: Any, model: Any = None) -> None:
        self.memory = memory
        self.model = model
        self.last_query: List[Dict[str, Any]] = []

    def seeds_for(self, r_task: Any, k: int = 16) -> List[Node]:
        """Top-``k`` retrieved programs that parse and type-check to GRID (memory order, or scorer order when a
        model is attached).  Returns ``[]`` for an empty / missing memory."""
        self.last_query = []
        if self.memory is None or k <= 0 or len(self.memory) == 0:
            return []
        r = r_task.detach().reshape(-1).cpu().numpy() if isinstance(r_task, Tensor) else np.asarray(r_task).reshape(-1)
        try:
            recs = self.memory.query(r, k=k)
        except Exception as e:
            log.warning("memory query failed: %s", e)
            return []
        self.last_query = recs
        progs: List[Node] = []
        seen = set()
        for rec in recs:
            try:
                node = Node.from_str(str(rec["program"]))
                if typecheck(node) is not T.GRID:
                    continue
            except (ValueError, TypeError, KeyError):
                continue
            key = node.to_str()
            if key not in seen:
                seen.add(key)
                progs.append(node)
        if self.model is not None and len(progs) > 1 and isinstance(r_task, Tensor):
            try:
                scores = NeuralPrior(self.model, r_task)(progs)
                progs = [p for _, p in sorted(zip(scores, progs), key=lambda x: -x[0])]
            except Exception as e:  # pragma: no cover - keep memory order
                log.debug("seed re-ranking failed: %s", e)
        return progs


# ============================================================================================ rule latent

def _grid_tensor(g: Optional[Grid]) -> Tensor:
    t = torch.full((MAX_SIDE, MAX_SIDE), PAD_ID, dtype=torch.long)
    if g:
        t[:len(g), :len(g[0])] = torch.tensor(g, dtype=torch.long)
    return t


def episode_batch_for_task(task: Task, *, test_index: int = 0, max_ctx: int = 10, parser: Any = None,
                           with_objects: bool = True) -> Dict[str, Tensor]:
    """A one-episode batch (leading batch axis 1) with the task's demos as context and one test input."""
    test_in = task.test[test_index].input if task.test else task.train[0].input
    ep = Episode(episode_id=f"{task.task_id}:{test_index}", task_id=task.task_id, split="test",
                 context=list(task.train)[:max_ctx], test_input=test_in, target_output=None)
    if with_objects:
        try:
            from arcjepa.data.tensorize import collate, encode_episode
            return collate([encode_episode(ep, parser, max_ctx)])
        except Exception as e:  # data module missing or failing: grid-only batch
            log.debug("encode_episode unavailable (%s); using a grid-only batch", e)
    k = max(1, len(ep.context))
    ctx_in = torch.full((1, k, MAX_SIDE, MAX_SIDE), PAD_ID, dtype=torch.long)
    ctx_out = torch.full((1, k, MAX_SIDE, MAX_SIDE), PAD_ID, dtype=torch.long)
    ctx_mask = torch.zeros(1, k, dtype=torch.bool)
    for i, p in enumerate(ep.context):
        ctx_in[0, i] = _grid_tensor(p.input)
        ctx_out[0, i] = _grid_tensor(p.output)
        ctx_mask[0, i] = True
    return {"ctx_in": ctx_in, "ctx_out": ctx_out, "ctx_mask": ctx_mask, "test_in": _grid_tensor(test_in)[None],
            "target": _grid_tensor(None)[None]}


def rule_latent_for_task(model: Any, task: Task, *, parser: Any = None, max_ctx: int = 10,
                         with_objects: bool = True) -> Tensor:
    """r_task Float[rule_dim] for ``task`` (no gradient, eval mode, on the model's device)."""
    batch = episode_batch_for_task(task, max_ctx=max_ctx, parser=parser, with_objects=with_objects)
    dev = next(model.parameters()).device
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.no_grad():
            r = model.rule_from_episode(batch)
    finally:
        if was_training:
            model.train()
    return r[0].detach()
