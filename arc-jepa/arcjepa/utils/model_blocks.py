"""Shared neural building blocks for ``arcjepa.model``.

Every encoder in the model package shares one transformer-stack implementation built from plain
``nn.TransformerEncoderLayer`` (pre-norm, batch-first, GELU, dropout 0 by default, no flash-attention
dependency), one MLP factory and masked-pooling helpers. Torch-only; no other dependencies.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor, nn


class TransformerStack(nn.Module):
    """A stack of pre-norm ``nn.TransformerEncoderLayer`` blocks followed by a final LayerNorm.

    Args:
        dim: model width.
        layers: number of blocks.
        heads: attention heads (must divide ``dim``).
        ffn_dim: feed-forward hidden width.
        dropout: dropout probability (the spec uses 0).
    """

    def __init__(self, dim: int, layers: int, heads: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} must be divisible by heads {heads}")
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(dim, heads, dim_feedforward=ffn_dim, dropout=dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(layers))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Apply the stack.

        Args:
            x: Float[B, N, D] token sequence.
            key_padding_mask: Bool[B, N] with ``True`` = ignore this key (torch convention), or ``None``.
        Returns:
            Float[B, N, D] contextualised tokens (LayerNorm applied).
        """
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=key_padding_mask)
        return self.norm(x)


def make_mlp(dims: Sequence[int], *, final_activation: bool = False) -> nn.Sequential:
    """Build ``Linear -> GELU -> ... -> Linear`` over the given widths (GELU between layers only)."""
    if len(dims) < 2:
        raise ValueError("make_mlp needs at least an input and an output width")
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or final_activation:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


def masked_mean(x: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """Mean of ``x[B, N, D]`` over the N axis restricted to ``mask[B, N]`` (True = keep). Empty rows give zeros."""
    m = mask.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp_min(eps)


def padding_mask_from_valid(valid: Tensor) -> Tensor:
    """Turn a validity mask (True = real token) into a key-padding mask (True = ignore).

    Rows without any valid token keep their first key unmasked so that softmax attention never sees an
    all-masked row (which produces NaNs); callers still discard those rows via their own masks.
    """
    kpm = ~valid.bool()
    all_masked = kpm.all(dim=1)
    if bool(all_masked.any()):
        kpm = kpm.clone()
        kpm[all_masked, 0] = False
    return kpm


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    """Number of parameters in ``module`` (optionally only those with ``requires_grad``)."""
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))
