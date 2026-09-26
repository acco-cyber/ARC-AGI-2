"""Transformation predictor.

Spec §Model: 6 pre-norm blocks (d 512, 8 heads, FFN 2048); input z_X + task context (r_task) + target mask
info -> z_hat_Y in R^512, plus per-slot object predictions and a pooled relation prediction. The sequence is
[Z token, R token, REL query token, 64 object slot tokens]; slots without an object carry a learned "empty slot"
embedding so the object mask (the target-mask information) is part of the input.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from arcjepa.utils.model_blocks import TransformerStack

from .config import ModelConfig


class TransformationPredictor(nn.Module):
    """forward(z_x Float[B,jepa_dim], r_task Float[B,rule_dim], obj_tokens_x Float[B,64,object_dim], obj_mask
    Bool[B,64]) -> {z_hat Float[B,jepa_dim], obj_hat Float[B,64,object_dim], rel_hat Float[B,relation_dim]}."""

    N_TYPES = 4  # z, r, rel-query, object slot

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.predictor_dim
        self.z_in = nn.Linear(cfg.jepa_dim, d)
        self.r_in = nn.Linear(cfg.rule_dim, d)
        self.obj_in = nn.Linear(cfg.object_dim, d)
        self.type_emb = nn.Embedding(self.N_TYPES, d)
        self.slot_emb = nn.Embedding(cfg.max_objects, d)
        self.empty_slot = nn.Parameter(torch.zeros(d))
        self.rel_query = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.empty_slot, std=0.02)
        nn.init.normal_(self.rel_query, std=0.02)
        self.blocks = TransformerStack(d, cfg.predictor_layers, cfg.heads, cfg.predictor_ffn_dim, cfg.dropout)
        self.z_out = nn.Linear(d, cfg.jepa_dim)
        self.obj_out = nn.Linear(d, cfg.object_dim)
        self.rel_out = nn.Linear(d, cfg.relation_dim)

    def forward(self, z_x: Tensor, r_task: Tensor, obj_tokens_x: Optional[Tensor] = None,
                obj_mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Predict the target-grid latents from the input latent and the task rule latent.

        Args:
            z_x: Float[B, jepa_dim] input-grid latent.
            r_task: Float[B, rule_dim] task rule latent.
            obj_tokens_x: Float[B, N, object_dim] input objects (``None`` -> all slots empty).
            obj_mask: Bool[B, N] (``None`` -> all given slots valid).
        Returns:
            dict with ``z_hat`` Float[B, jepa_dim], ``obj_hat`` Float[B, N, object_dim], ``rel_hat``
            Float[B, relation_dim].
        """
        b = z_x.shape[0]
        d = self.cfg.predictor_dim
        n = self.cfg.max_objects if obj_tokens_x is None else obj_tokens_x.shape[1]
        dev = z_x.device
        if obj_tokens_x is None:
            obj_tokens_x = z_x.new_zeros(b, n, self.cfg.object_dim)
            obj_mask = torch.zeros(b, n, dtype=torch.bool, device=dev)
        if obj_mask is None:
            obj_mask = torch.ones(b, n, dtype=torch.bool, device=dev)
        types = self.type_emb.weight  # [4, d]
        slots = torch.where(obj_mask.bool().unsqueeze(-1), self.obj_in(obj_tokens_x), self.empty_slot.view(1, 1, d))
        slots = slots + self.slot_emb(torch.arange(n, device=dev)).unsqueeze(0) + types[3]
        z_tok = (self.z_in(z_x) + types[0]).unsqueeze(1)
        r_tok = (self.r_in(r_task) + types[1]).unsqueeze(1)
        rel_tok = (self.rel_query + types[2]).view(1, 1, d).expand(b, 1, d)
        h = self.blocks(torch.cat([z_tok, r_tok, rel_tok, slots], dim=1))
        return {"z_hat": self.z_out(h[:, 0]), "obj_hat": self.obj_out(h[:, 3:]), "rel_hat": self.rel_out(h[:, 2])}
