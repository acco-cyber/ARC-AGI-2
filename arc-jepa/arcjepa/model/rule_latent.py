"""Task rule latent.

Spec §Model: r_i = MLP([z_X; z_Y; z_Y - z_X]) 1536 -> 1024 -> 512 -> 256 per demonstration pair, then
r_task = AttentionPool(r_1..r_n) in R^256 (learned query, masked softmax over the demonstrations).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from arcjepa.utils.model_blocks import make_mlp

from .config import ModelConfig


class RuleLatent(nn.Module):
    """forward(z_x Float[B,K,jepa_dim], z_y Float[B,K,jepa_dim], k_mask Bool[B,K]) -> Float[B,rule_dim]."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        r = cfg.rule_dim
        self.mlp = make_mlp([3 * cfg.jepa_dim, *cfg.rule_hidden, r])
        self.demo_norm = nn.LayerNorm(r)
        self.query = nn.Parameter(torch.zeros(r))
        nn.init.normal_(self.query, std=0.02)
        self.key = nn.Linear(r, r)
        self.value = nn.Linear(r, r)
        self.out_norm = nn.LayerNorm(r)

    def per_demo(self, z_x: Tensor, z_y: Tensor) -> Tensor:
        """Per-demonstration rule latents Float[B, K, rule_dim] (no pooling)."""
        return self.demo_norm(self.mlp(torch.cat([z_x, z_y, z_y - z_x], dim=-1)))

    def pool(self, r_demo: Tensor, k_mask: Optional[Tensor] = None) -> Tensor:
        """Attention-pool Float[B, K, rule_dim] with a learned query; masked demos get zero weight.

        Rows whose mask is entirely False fall back to a uniform average (finite output, no NaN).
        """
        b, k, _ = r_demo.shape
        if k_mask is None:
            k_mask = torch.ones(b, k, dtype=torch.bool, device=r_demo.device)
        k_mask = k_mask.bool()
        empty = ~k_mask.any(dim=1)
        if bool(empty.any()):
            k_mask = k_mask | empty.unsqueeze(1)
        scores = (self.key(r_demo) * self.query).sum(-1) / math.sqrt(r_demo.shape[-1])  # [B,K]
        scores = scores.masked_fill(~k_mask, float("-inf"))
        attn = torch.softmax(scores, dim=1)
        return self.out_norm((attn.unsqueeze(-1) * self.value(r_demo)).sum(1))

    def forward(self, z_x: Tensor, z_y: Tensor, k_mask: Optional[Tensor] = None) -> Tensor:
        """Task rule latent from K demonstration (input, output) latent pairs."""
        return self.pool(self.per_demo(z_x, z_y), k_mask)
