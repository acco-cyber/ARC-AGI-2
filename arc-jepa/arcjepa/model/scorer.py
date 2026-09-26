"""Neural program scorer: s_neural = MLP([r_task; z_p; r_task * z_p]) -> R (spec §Model)."""
from __future__ import annotations

from torch import Tensor, nn
import torch

from arcjepa.utils.model_blocks import make_mlp

from .config import ModelConfig


class Scorer(nn.Module):
    """forward(r Float[B,rule_dim], z_p Float[B,program_dim]) -> Float[B].

    ``z_p`` may also be Float[B, N, program_dim] (N candidate programs per task), giving Float[B, N].
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.program_dim, cfg.rule_dim) if cfg.program_dim != cfg.rule_dim else nn.Identity()
        self.mlp = make_mlp([3 * cfg.rule_dim, *cfg.scorer_hidden, 1])

    def forward(self, r: Tensor, z_p: Tensor) -> Tensor:
        """Score program latents against a rule latent (higher = more compatible)."""
        z_p = self.proj(z_p)
        if z_p.dim() == 3 and r.dim() == 2:
            r = r.unsqueeze(1).expand_as(z_p)
        return self.mlp(torch.cat([r, z_p, r * z_p], dim=-1)).squeeze(-1)
