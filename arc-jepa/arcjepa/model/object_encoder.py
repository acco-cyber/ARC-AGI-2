"""Object encoder.

Spec §Model / §Object parser: per object, a CNN shape encoder (crop <= 30x30 -> 3x3 conv -> 3x3 conv -> global
average pool -> 128) is concatenated with the 32 deterministic features (160) and projected to 256; 4 pre-norm
transformer blocks (d 256) contextualise up to 64 objects. A learned grid token is prepended so that the pooled
summary is always defined (also for grids without any parsed object).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from arcjepa.utils.model_blocks import TransformerStack, padding_mask_from_valid

from .config import ModelConfig


class ShapeCNN(nn.Module):
    """One-hot colour crop -> 3x3 conv -> GELU -> 3x3 conv -> GELU -> global average pool -> ``shape_dim``."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_in = cfg.n_colors + 1  # colours + PAD (outside the object)
        self.conv1 = nn.Conv2d(self.n_in, cfg.shape_conv_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(cfg.shape_conv_dim, cfg.shape_dim, 3, padding=1)

    def forward(self, crops: Tensor) -> Tensor:
        """crops Int[N, H, W] -> Float[N, shape_dim]."""
        x = F.one_hot(crops.long().clamp(0, self.n_in - 1), self.n_in).to(torch.float32).permute(0, 3, 1, 2)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        return x.mean(dim=(2, 3))


class ObjectEncoder(nn.Module):
    """Objects of one grid -> object tokens + pooled object summary.

    forward(crops Int[B,64,30,30], feats Float[B,64,32], mask Bool[B,64]) ->
        (obj_tokens Float[B,64,object_dim], pooled Float[B,object_dim]).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.object_dim
        self.shape = ShapeCNN(cfg)
        self.in_proj = nn.Linear(cfg.shape_dim + cfg.obj_feat_dim, d)
        self.grid_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.grid_token, std=0.02)
        self.blocks = TransformerStack(d, cfg.object_layers, cfg.heads, cfg.ffn_dim, cfg.dropout)
        self.pool_norm = nn.LayerNorm(d)

    def forward(self, crops: Tensor, feats: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Encode padded object slots.

        Args:
            crops: Int[B, N, S, S] colour crops (``pad_id`` outside the object).
            feats: Float[B, N, obj_feat_dim] deterministic features.
            mask: Bool[B, N] slot validity (defaults to all valid).
        Returns:
            obj_tokens Float[B, N, object_dim] (zeros at invalid slots) and pooled Float[B, object_dim].
        """
        b, n = feats.shape[:2]
        if mask is None:
            mask = torch.ones(b, n, dtype=torch.bool, device=feats.device)
        mask = mask.bool()
        shape_all = feats.new_zeros(b, n, self.cfg.shape_dim)
        if bool(mask.any()):
            shape_all[mask] = self.shape(crops[mask])
        tok = self.in_proj(torch.cat([shape_all, feats.to(shape_all.dtype)], dim=-1))
        # crop the slot axis to the last valid slot in the batch (objects are padded at the end)
        last = int((mask.long() * torch.arange(1, n + 1, device=mask.device)).max().item())
        nc = max(1, last)
        seq = torch.cat([self.grid_token.expand(b, 1, -1), tok[:, :nc]], dim=1)
        valid = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=mask.device), mask[:, :nc]], dim=1)
        h = self.blocks(seq, padding_mask_from_valid(valid))
        obj_tokens = tok.new_zeros(b, n, tok.shape[-1])
        obj_tokens[:, :nc] = h[:, 1:] * mask[:, :nc].unsqueeze(-1).to(h.dtype)
        pooled = self.pool_norm(h[:, 0])
        return obj_tokens, pooled
