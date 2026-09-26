"""Grid <-> tensor conversion, episode encoding, batching and the ``EpisodeDataset`` wrapper.

Every grid is padded to ``MAX_SIDE x MAX_SIDE`` (30 x 30) with :data:`arcjepa.core.types.PAD_ID` (10) in the
bottom/right; the real cells are recovered by :func:`tensor_to_grid`, which makes ``grid_to_tensor`` invertible.

:func:`encode_episode` produces the dictionary consumed by the model (INTERFACES §4)::

    ctx_in    Long[K, 30, 30]      demonstration inputs, PAD-filled beyond the episode's demos
    ctx_out   Long[K, 30, 30]      demonstration outputs
    ctx_mask  Bool[K]              True for real demonstrations
    test_in   Long[30, 30]
    target    Long[30, 30]         all PAD when the target output is unknown
    obj_feats Float[K+1, 64, 32]   objects of the K context inputs (slots 0..K-1) and the test input (slot K)
    obj_crops Int8[K+1, 64, 30, 30]  object colour crops, PAD_ID outside the object
    obj_mask  Bool[K+1, 64]
    rel_feats Float[K+1, 64, 64, 24]
    n_ctx     Long[]               number of real demonstrations (extra key, convenient for pooling)

``K`` is ``max_ctx`` (10 by default) so that a batch of episodes stacks without further padding.

The object tensors come from a *parser*: any callable with the signature of
``arcjepa.parser.hypotheses.parse(grid, hypothesis=None, max_objects=64) -> (objects, feats[N,32],
rels[N,N,24])`` whose objects expose ``crop(pad_to=30)`` (or ``cells``). When no parser is passed,
:func:`resolve_parser` imports the real one if the parser module exists and otherwise uses
:func:`zero_parser`, which yields zero objects (all-False ``obj_mask``) so the data pipeline runs before the
parser module lands.
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from arcjepa.core.types import MAX_SIDE, PAD_ID, Episode, Grid, Pair

logger = logging.getLogger(__name__)

MAX_OBJECTS = 64
OBJ_FEAT_DIM = 32
REL_FEAT_DIM = 24
DEFAULT_MAX_CTX = 10

ParserFn = Callable[..., Tuple[Sequence[Any], np.ndarray, np.ndarray]]

GRID_KEYS: Tuple[str, ...] = ("ctx_in", "ctx_out", "test_in", "target", "obj_crops")
_PAD_VALUE: Dict[str, Any] = {"ctx_in": PAD_ID, "ctx_out": PAD_ID, "test_in": PAD_ID, "target": PAD_ID,
                              "obj_crops": PAD_ID, "ctx_mask": False, "obj_mask": False}


# --------------------------------------------------------------------------------------------------------------
# Grid <-> tensor
# --------------------------------------------------------------------------------------------------------------


def grid_to_tensor(g: Optional[Grid], side: int = MAX_SIDE) -> torch.Tensor:
    """Encode a grid as ``LongTensor[side, side]`` filled with ``PAD_ID`` outside the grid.

    ``None`` or an empty grid gives an all-PAD tensor. Grids larger than ``side`` raise ``ValueError``.
    """
    t = torch.full((side, side), PAD_ID, dtype=torch.long)
    if not g:
        return t
    h, w = len(g), len(g[0])
    if h > side or w > side:
        raise ValueError(f"grid {h}x{w} exceeds the {side}x{side} canvas")
    t[:h, :w] = torch.as_tensor(np.asarray(g, dtype=np.int64))
    return t


def grid_mask(g: Optional[Grid], side: int = MAX_SIDE) -> torch.Tensor:
    """``BoolTensor[side, side]``: True on the real cells of ``g`` (all False for ``None``/empty)."""
    m = torch.zeros((side, side), dtype=torch.bool)
    if g:
        m[: len(g), : len(g[0])] = True
    return m


def tensor_to_grid(t: torch.Tensor) -> Grid:
    """Inverse of :func:`grid_to_tensor`: strip the PAD border and return a ``Grid`` (``[]`` if all PAD)."""
    a = t.detach().cpu().to(torch.long).numpy()
    if a.ndim != 2:
        raise ValueError("expected a 2-D tensor")
    valid_rows = np.nonzero(a[:, 0] != PAD_ID)[0]
    if valid_rows.size == 0:
        return []
    h = int(valid_rows.max()) + 1
    valid_cols = np.nonzero(a[0, :] != PAD_ID)[0]
    w = int(valid_cols.max()) + 1
    return a[:h, :w].astype(int).tolist()


def mask_to_shape(m: torch.Tensor) -> Tuple[int, int]:
    """(H, W) of the real region encoded by a :func:`grid_mask` tensor (``(0, 0)`` when empty)."""
    if not bool(m.any()):
        return (0, 0)
    return int(m[:, 0].sum().item()), int(m[0, :].sum().item())


def pair_to_tensors(p: Pair) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(input, output)`` tensors of a demonstration pair."""
    return grid_to_tensor(p.input), grid_to_tensor(p.output)


# --------------------------------------------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------------------------------------------


def zero_parser(grid: Grid, hypothesis: Optional[str] = None, max_objects: int = MAX_OBJECTS):
    """Stub parser: no objects. Returns ``([], zeros[0,32], zeros[0,0,24])``."""
    return [], np.zeros((0, OBJ_FEAT_DIM), dtype=np.float32), np.zeros((0, 0, REL_FEAT_DIM), dtype=np.float32)


def resolve_parser(parser: Optional[ParserFn] = None) -> ParserFn:
    """Return ``parser`` itself, else ``arcjepa.parser.hypotheses.parse`` if importable, else :func:`zero_parser`."""
    if parser is not None:
        return parser
    try:
        mod = importlib.import_module("arcjepa.parser.hypotheses")
        fn = getattr(mod, "parse")
        return fn
    except Exception as exc:  # ImportError or a half-implemented module: degrade gracefully
        logger.debug("parser unavailable (%s); using zero_parser", exc)
        return zero_parser


def _object_crop(obj: Any, grid: np.ndarray, pad_to: int = MAX_SIDE) -> np.ndarray:
    """Colour crop of an object as ``int8[pad_to, pad_to]`` with ``PAD_ID`` outside the object's cells."""
    crop_fn = getattr(obj, "crop", None)
    if callable(crop_fn):
        c = np.asarray(crop_fn(pad_to), dtype=np.int8)
        if c.shape == (pad_to, pad_to):
            return c
    out = np.full((pad_to, pad_to), PAD_ID, dtype=np.int8)
    cells = getattr(obj, "cells", None)
    if not cells:
        return out
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    for r, c in cells:
        rr, cc = r - r0, c - c0
        if 0 <= rr < pad_to and 0 <= cc < pad_to:
            out[rr, cc] = int(grid[r, c])
    return out


def encode_objects(grid: Grid, parser: ParserFn, max_objects: int = MAX_OBJECTS) -> Dict[str, torch.Tensor]:
    """Parse one grid and return ``obj_feats[64,32]``, ``obj_crops[64,30,30]``, ``obj_mask[64]``, ``rel_feats[64,64,24]``."""
    feats = torch.zeros((max_objects, OBJ_FEAT_DIM), dtype=torch.float32)
    crops = torch.full((max_objects, MAX_SIDE, MAX_SIDE), PAD_ID, dtype=torch.int8)
    mask = torch.zeros((max_objects,), dtype=torch.bool)
    rels = torch.zeros((max_objects, max_objects, REL_FEAT_DIM), dtype=torch.float32)
    if not grid:
        return {"obj_feats": feats, "obj_crops": crops, "obj_mask": mask, "rel_feats": rels}
    objs, f, r = parser(grid, None, max_objects)
    n = min(len(objs), max_objects)
    if n == 0:
        return {"obj_feats": feats, "obj_crops": crops, "obj_mask": mask, "rel_feats": rels}
    f = np.asarray(f, dtype=np.float32)
    r = np.asarray(r, dtype=np.float32)
    d = min(f.shape[1], OBJ_FEAT_DIM) if f.ndim == 2 else 0
    if d:
        feats[:n, :d] = torch.from_numpy(np.ascontiguousarray(f[:n, :d]))
    if r.ndim == 3:
        rd = min(r.shape[2], REL_FEAT_DIM)
        rels[:n, :n, :rd] = torch.from_numpy(np.ascontiguousarray(r[:n, :n, :rd]))
    ga = np.asarray(grid, dtype=np.int16)
    for i in range(n):
        crops[i] = torch.from_numpy(_object_crop(objs[i], ga))
    mask[:n] = True
    return {"obj_feats": feats, "obj_crops": crops, "obj_mask": mask, "rel_feats": rels}


# --------------------------------------------------------------------------------------------------------------
# Episode encoding
# --------------------------------------------------------------------------------------------------------------


def encode_episode(ep: Episode, parser: Optional[ParserFn] = None, max_ctx: int = DEFAULT_MAX_CTX) -> Dict[str, torch.Tensor]:
    """Tensorise an episode (see the module docstring for keys and shapes).

    Demonstrations beyond ``max_ctx`` are dropped (the first ``max_ctx`` are kept). The test input occupies
    object slot ``max_ctx`` (the last one); context inputs occupy slots ``0 .. n_ctx-1``.
    """
    parse = resolve_parser(parser)
    ctx = list(ep.context[:max_ctx])
    k = len(ctx)
    ctx_in = torch.full((max_ctx, MAX_SIDE, MAX_SIDE), PAD_ID, dtype=torch.long)
    ctx_out = torch.full((max_ctx, MAX_SIDE, MAX_SIDE), PAD_ID, dtype=torch.long)
    ctx_mask = torch.zeros((max_ctx,), dtype=torch.bool)
    for i, p in enumerate(ctx):
        ctx_in[i] = grid_to_tensor(p.input)
        ctx_out[i] = grid_to_tensor(p.output)
        ctx_mask[i] = True

    n_slots = max_ctx + 1
    obj_feats = torch.zeros((n_slots, MAX_OBJECTS, OBJ_FEAT_DIM), dtype=torch.float32)
    obj_crops = torch.full((n_slots, MAX_OBJECTS, MAX_SIDE, MAX_SIDE), PAD_ID, dtype=torch.int8)
    obj_mask = torch.zeros((n_slots, MAX_OBJECTS), dtype=torch.bool)
    rel_feats = torch.zeros((n_slots, MAX_OBJECTS, MAX_OBJECTS, REL_FEAT_DIM), dtype=torch.float32)
    grids: List[Tuple[int, Grid]] = [(i, p.input) for i, p in enumerate(ctx)] + [(max_ctx, ep.test_input)]
    for slot, g in grids:
        o = encode_objects(g, parse)
        obj_feats[slot] = o["obj_feats"]
        obj_crops[slot] = o["obj_crops"]
        obj_mask[slot] = o["obj_mask"]
        rel_feats[slot] = o["rel_feats"]

    return {
        "ctx_in": ctx_in,
        "ctx_out": ctx_out,
        "ctx_mask": ctx_mask,
        "test_in": grid_to_tensor(ep.test_input),
        "target": grid_to_tensor(ep.target_output),
        "obj_feats": obj_feats,
        "obj_crops": obj_crops,
        "obj_mask": obj_mask,
        "rel_feats": rel_feats,
        "n_ctx": torch.tensor(k, dtype=torch.long),
    }


def _pad_dim0(t: torch.Tensor, n: int, fill: Any) -> torch.Tensor:
    if t.dim() == 0 or t.shape[0] == n:
        return t
    pad = torch.full((n - t.shape[0],) + tuple(t.shape[1:]), fill, dtype=t.dtype)
    return torch.cat([t, pad], dim=0)


def collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of :func:`encode_episode` dictionaries into batched tensors ``[B, ...]``.

    Episodes encoded with different ``max_ctx`` are padded along their first dimension (PAD_ID for grids
    and crops, False for masks, 0 for features) to the largest size in the batch before stacking.
    """
    if not batch:
        raise ValueError("empty batch")
    keys = list(batch[0].keys())
    out: Dict[str, torch.Tensor] = {}
    for key in keys:
        ts = [b[key] for b in batch]
        if ts[0].dim() > 0 and len({t.shape[0] for t in ts}) > 1:
            n = max(t.shape[0] for t in ts)
            fill = _PAD_VALUE.get(key, 0)
            ts = [_pad_dim0(t, n, fill) for t in ts]
        out[key] = torch.stack(ts, dim=0)
    return out


class EpisodeDataset(Dataset):
    """``torch.utils.data.Dataset`` of tensorised episodes (``__getitem__`` returns :func:`encode_episode`)."""

    def __init__(self, episodes: Sequence[Episode], parser: Optional[ParserFn] = None, max_ctx: int = DEFAULT_MAX_CTX):
        self.episodes: List[Episode] = list(episodes)
        self.parser: ParserFn = resolve_parser(parser)
        self.max_ctx = int(max_ctx)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return encode_episode(self.episodes[idx], self.parser, self.max_ctx)

    @property
    def episode_ids(self) -> List[str]:
        return [ep.episode_id for ep in self.episodes]

    @staticmethod
    def collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Alias of :func:`collate` for ``DataLoader(collate_fn=...)``."""
        return collate(batch)
