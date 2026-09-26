"""Hierarchical grid JEPA encoder.

Spec §Model: z_cell (256) from the cell encoder, z_object (256) from the object encoder, z_global (384) fused
from the cell / object / relation summaries, concatenated and projected to z_X in R^512. The three summaries
concatenate to 896 (not the spec's 1024), so the projection is an MLP 896 -> fusion_hidden (1024) -> 512.

Grid batch dict keys (all but ``grid`` optional):
    grid Long[B,30,30], mask Bool[B,30,30] (default ``grid != PAD_ID``), obj_crops Int[B,64,30,30],
    obj_feats Float[B,64,32], obj_mask Bool[B,64], rel_feats Float[B,64,64,24].
Grids without object information are encoded with zero objects (the object / relation summaries then come from
their learned summary tokens only).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor, nn

from .cell_encoder import CellEncoder
from .config import ModelConfig
from .object_encoder import ObjectEncoder
from .relation_encoder import RelationEncoder

GRID_BATCH_KEYS = ("grid", "mask", "obj_crops", "obj_feats", "obj_mask", "rel_feats")
CROP_DTYPE = torch.int8  # the dtype of ``arcjepa.data.tensorize`` object crops


def make_grid_batch(grid: Tensor, mask: Optional[Tensor] = None, obj_crops: Optional[Tensor] = None,
                    obj_feats: Optional[Tensor] = None, obj_mask: Optional[Tensor] = None,
                    rel_feats: Optional[Tensor] = None) -> Dict[str, Tensor]:
    """Assemble a grid batch dict, dropping ``None`` entries."""
    d = {"grid": grid, "mask": mask, "obj_crops": obj_crops, "obj_feats": obj_feats, "obj_mask": obj_mask,
         "rel_feats": rel_feats}
    return {k: v for k, v in d.items() if v is not None}


def complete_grid_batch(batch: Dict[str, Tensor], cfg: ModelConfig) -> Dict[str, Tensor]:
    """Return a copy of ``batch`` with every key of ``GRID_BATCH_KEYS`` present (defaults for missing ones)."""
    grid = batch["grid"].long()
    b = grid.shape[0]
    dev = grid.device
    out = dict(batch)
    out["grid"] = grid
    if out.get("mask") is None:
        out["mask"] = grid != cfg.pad_id
    out["mask"] = out["mask"].bool()
    n, s = cfg.max_objects, cfg.max_side
    if out.get("obj_mask") is None:
        if out.get("obj_feats") is not None:
            out["obj_mask"] = torch.ones(b, out["obj_feats"].shape[1], dtype=torch.bool, device=dev)
        else:
            out["obj_mask"] = torch.zeros(b, n, dtype=torch.bool, device=dev)
    out["obj_mask"] = out["obj_mask"].bool()
    n = out["obj_mask"].shape[1]
    if out.get("obj_feats") is None:
        out["obj_feats"] = torch.zeros(b, n, cfg.obj_feat_dim, device=dev)
    if out.get("obj_crops") is None:
        out["obj_crops"] = torch.full((b, n, s, s), cfg.pad_id, dtype=CROP_DTYPE, device=dev)
    if out.get("rel_feats") is None:
        out["rel_feats"] = torch.zeros(b, n, n, cfg.rel_feat_dim, device=dev)
    return out


def concat_grid_batches(batches: Sequence[Dict[str, Tensor]], cfg: ModelConfig) -> Dict[str, Tensor]:
    """Complete every grid batch and concatenate them along the row axis (one encoder pass for all)."""
    done = [complete_grid_batch(b_, cfg) for b_ in batches]
    out: Dict[str, Tensor] = {}
    for key in GRID_BATCH_KEYS:
        parts = [d[key] for d in done]
        if key == "obj_crops":  # colours 0..10 fit int8; ShapeCNN upcasts only the valid crops
            parts = [p.to(CROP_DTYPE) for p in parts]
        out[key] = torch.cat(parts, dim=0)
    return out


def split_encoding(enc: Dict[str, Tensor], sizes: Sequence[int]) -> List[Dict[str, Tensor]]:
    """Inverse of ``concat_grid_batches`` for encoder outputs: split every tensor into row blocks."""
    pieces = {key: torch.split(v, list(sizes), dim=0) for key, v in enc.items() if torch.is_tensor(v)}
    return [{key: pieces[key][i] for key in pieces} for i in range(len(sizes))]


class GridJEPAEncoder(nn.Module):
    """Grid (+ objects, relations) -> hierarchical latent.

    forward(batch) -> {z Float[B,jepa_dim], z_global Float[B,global_dim], obj_tokens Float[B,64,object_dim],
                       obj_mask Bool[B,64], obj_pooled, rel_pooled Float[B,relation_dim], cell_pooled
                       Float[B,cell_dim]} (+ cell_tokens when ``return_cell_tokens``).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cell = CellEncoder(cfg)
        self.objects = ObjectEncoder(cfg)
        self.relations = RelationEncoder(cfg)
        self.global_mlp = nn.Sequential(
            nn.Linear(cfg.cell_dim + cfg.object_dim + cfg.relation_dim, cfg.global_dim), nn.GELU(),
            nn.Linear(cfg.global_dim, cfg.global_dim))
        self.global_norm = nn.LayerNorm(cfg.global_dim)
        self.fusion = nn.Sequential(
            nn.Linear(cfg.cell_dim + cfg.object_dim + cfg.global_dim, cfg.fusion_hidden), nn.GELU(),
            nn.Linear(cfg.fusion_hidden, cfg.jepa_dim))
        self.out_norm = nn.LayerNorm(cfg.jepa_dim)

    def forward(self, batch: Dict[str, Tensor], *, return_cell_tokens: bool = False) -> Dict[str, Tensor]:
        """Encode a grid batch dict (see module docstring for keys)."""
        cfg = self.cfg
        b_ = complete_grid_batch(batch, cfg)
        cell_tokens, cell_pooled = self.cell(b_["grid"], b_["mask"])
        obj_tokens, obj_pooled = self.objects(b_["obj_crops"], b_["obj_feats"], b_["obj_mask"])
        _, rel_pooled = self.relations(obj_tokens, b_["rel_feats"], b_["obj_mask"], return_tokens=False)
        z_global = self.global_norm(self.global_mlp(torch.cat([cell_pooled, obj_pooled, rel_pooled], dim=-1)))
        z = self.out_norm(self.fusion(torch.cat([cell_pooled, obj_pooled, z_global], dim=-1)))
        out = {"z": z, "z_global": z_global, "obj_tokens": obj_tokens, "obj_mask": b_["obj_mask"],
               "obj_pooled": obj_pooled, "rel_pooled": rel_pooled, "cell_pooled": cell_pooled}
        if return_cell_tokens:
            out["cell_tokens"] = cell_tokens
        return out
