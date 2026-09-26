"""Cell-level grid encoder.

Spec §Model: cell embedding = colour 32 + row 32 + column 32 + neighbour statistics 32 = 128 -> 256, followed by
12 pre-norm transformer blocks (d 256, 8 heads, FFN 1024, dropout 0). Grids are 30x30 padded tensors with
``PAD_ID`` outside the valid region; the encoder crops the batch to its largest valid H x W before attention
(exactly equivalent for valid cells, much cheaper for small grids) and scatters the tokens back to 900 slots.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from arcjepa.utils.model_blocks import TransformerStack, masked_mean, padding_mask_from_valid

from .config import ModelConfig

N_NEIGHBOR_FEATS = 16
_SHIFTS8: Tuple[Tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


def _shift(x: Tensor, dr: int, dc: int) -> Tensor:
    """out[..., r, c] = x[..., r + dr, c + dc] (zero outside), for dr, dc in {-1, 0, 1}."""
    h, w = x.shape[-2:]
    xp = F.pad(x, (1, 1, 1, 1))
    return xp[..., 1 + dr:1 + dr + h, 1 + dc:1 + dc + w]


def neighbor_features(grid: Tensor, mask: Tensor, n_colors: int = 10, max_side: int = 30) -> Tensor:
    """Deterministic per-cell neighbourhood statistics (16 features in [0, 1]).

    Features: same-colour 4-neighbours / 4, same-colour 8-neighbours / 8, non-zero 8-neighbours / 8, distinct
    colours in the 3x3 window / 10, is-nonzero, 4 grid-border flags (up/down/left/right), row and column
    fraction inside the valid region, valid H / 30, valid W / 30, fraction of the row and of the column sharing
    the cell's colour, and the cell colour's global frequency. Invalid cells are all-zero.

    Args:
        grid: Long[B, H, W] colours (PAD outside the valid region).
        mask: Bool[B, H, W] validity.
    Returns:
        Float[B, H, W, 16].
    """
    b, h, w = grid.shape
    valid = mask.to(torch.float32)
    g = grid.long().clamp(0, n_colors)
    oh = F.one_hot(g, n_colors + 1).to(torch.float32) * valid.unsqueeze(-1)  # [B,H,W,C]
    ohc = oh.permute(0, 3, 1, 2)  # [B,C,H,W]
    same4 = torch.zeros_like(valid)
    same8 = torch.zeros_like(valid)
    nz8 = torch.zeros_like(valid)
    nb_max = ohc[:, :n_colors].clone()
    border = []
    for k, (dr, dc) in enumerate(_SHIFTS8):
        sh = _shift(ohc, dr, dc)
        same = (ohc * sh).sum(1)
        same8 = same8 + same
        nz8 = nz8 + sh[:, 1:n_colors].sum(1)
        nb_max = torch.maximum(nb_max, sh[:, :n_colors])
        if k < 4:
            same4 = same4 + same
            border.append(valid * (1.0 - sh.sum(1)))
    ndistinct = nb_max.sum(1)
    is_nonzero = oh[..., 1:n_colors].sum(-1)
    h_valid = mask.any(2).sum(1).clamp_min(1).to(torch.float32)  # [B]
    w_valid = mask.any(1).sum(1).clamp_min(1).to(torch.float32)
    rows = torch.arange(h, device=grid.device, dtype=torch.float32).view(1, h, 1).expand(b, h, w)
    cols = torch.arange(w, device=grid.device, dtype=torch.float32).view(1, 1, w).expand(b, h, w)
    row_frac = rows / (h_valid - 1).clamp_min(1).view(b, 1, 1)
    col_frac = cols / (w_valid - 1).clamp_min(1).view(b, 1, 1)
    h_frac = (h_valid / max_side).view(b, 1, 1).expand(b, h, w)
    w_frac = (w_valid / max_side).view(b, 1, 1).expand(b, h, w)
    row_same = (oh * oh.sum(2, keepdim=True)).sum(-1) / w_valid.view(b, 1, 1)
    col_same = (oh * oh.sum(1, keepdim=True)).sum(-1) / h_valid.view(b, 1, 1)
    color_freq = (oh * oh.sum((1, 2), keepdim=True)).sum(-1) / (h_valid * w_valid).view(b, 1, 1)
    feats = torch.stack(
        [same4 / 4.0, same8 / 8.0, nz8 / 8.0, ndistinct / n_colors, is_nonzero, *border,
         row_frac, col_frac, h_frac, w_frac, row_same, col_same, color_freq], dim=-1)
    return feats * valid.unsqueeze(-1)


def valid_extent(mask: Tensor) -> Tuple[int, int]:
    """Smallest (H, W) crop of a Bool[B, H, W] mask that contains every valid cell of the batch (at least 1x1)."""
    rows = torch.nonzero(mask.any(2).any(0), as_tuple=False)
    cols = torch.nonzero(mask.any(1).any(0), as_tuple=False)
    hc = int(rows.max().item()) + 1 if rows.numel() else 1
    wc = int(cols.max().item()) + 1 if cols.numel() else 1
    return hc, wc


class CellEncoder(nn.Module):
    """Grid -> cell tokens + pooled cell summary.

    forward(grid Long[B,30,30], mask Bool[B,30,30]) -> (tokens Float[B,900,cell_dim], pooled Float[B,cell_dim]).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        p = cfg.cell_part_dim
        self.color_emb = nn.Embedding(cfg.n_colors + 1, p)  # colours 0..9 plus PAD
        self.row_emb = nn.Embedding(cfg.max_side, p)
        self.col_emb = nn.Embedding(cfg.max_side, p)
        self.neigh_proj = nn.Linear(N_NEIGHBOR_FEATS, p)
        self.in_proj = nn.Linear(cfg.cell_embed_dim, cfg.cell_dim)
        self.blocks = TransformerStack(cfg.cell_dim, cfg.cell_layers, cfg.heads, cfg.ffn_dim, cfg.dropout)
        self.pool_norm = nn.LayerNorm(cfg.cell_dim)

    def embed(self, grid: Tensor, mask: Tensor) -> Tensor:
        """Cell embedding Float[B, H, W, cell_dim] for an (already cropped) grid."""
        b, h, w = grid.shape
        g = grid.long().clamp(0, self.cfg.n_colors)
        col = self.color_emb(g)
        rows = self.row_emb(torch.arange(h, device=grid.device)).view(1, h, 1, -1).expand(b, h, w, -1)
        cols = self.col_emb(torch.arange(w, device=grid.device)).view(1, 1, w, -1).expand(b, h, w, -1)
        neigh = self.neigh_proj(neighbor_features(grid, mask, self.cfg.n_colors, self.cfg.max_side))
        return self.in_proj(torch.cat([col, rows, cols, neigh], dim=-1))

    def forward(self, grid: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Encode a batch of padded grids.

        Args:
            grid: Long[B, S, S] colours with ``pad_id`` outside the valid region (S = max_side).
            mask: Bool[B, S, S] validity; defaults to ``grid != pad_id``.
        Returns:
            tokens Float[B, S*S, cell_dim] (zeros at padded cells) and pooled Float[B, cell_dim] (masked mean +
            LayerNorm).
        """
        grid = grid.long()
        if mask is None:
            mask = grid != self.cfg.pad_id
        mask = mask.bool()
        b, s, s2 = grid.shape
        hc, wc = valid_extent(mask)
        g_c, m_c = grid[:, :hc, :wc], mask[:, :hc, :wc]
        x = self.embed(g_c, m_c).reshape(b, hc * wc, -1)
        flat_mask = m_c.reshape(b, hc * wc)
        h = self.blocks(x, padding_mask_from_valid(flat_mask))
        h = h * flat_mask.unsqueeze(-1).to(h.dtype)
        tokens = h.new_zeros(b, s, s2, h.shape[-1])
        tokens[:, :hc, :wc] = h.view(b, hc, wc, -1)
        pooled = self.pool_norm(masked_mean(h, flat_mask))
        return tokens.reshape(b, s * s2, -1), pooled
