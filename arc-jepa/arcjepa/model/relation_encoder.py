"""Relation encoder.

Spec §Model: 4 pre-norm blocks (d 256) over pair tokens built from (o_i, o_j, e_ij) with e_ij in R^64 embedded
from the 24 parser relation features. The pair projection is linear, so the pair token decomposes as
W_i o_i + W_j o_j + W_e e_ij and the full 64x64 pair grid is materialised only when ``return_tokens`` is set.
Attention runs over at most ``cfg.max_pairs`` (512) ordered pairs per grid, evenly subsampled from the valid
pairs (i != j, both slots valid) in row-major order; a learned summary token gives the pooled output.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from arcjepa.utils.model_blocks import TransformerStack, padding_mask_from_valid

from .config import ModelConfig


def select_pairs(pair_valid: Tensor, max_pairs: int) -> Tuple[Tensor, Tensor]:
    """Pick <= ``max_pairs`` valid ordered pairs per sample, deterministically.

    Args:
        pair_valid: Bool[B, N, N].
        max_pairs: cap per sample; when exceeded the valid pairs (row-major) are split into ``max_pairs`` equal
            rank buckets and the first pair of each bucket is kept, which covers every object uniformly, needs no
            RNG and is fully vectorised.
    Returns:
        sel Long[B, P] flat indices ``i * N + j`` (0 where unused) and sel_mask Bool[B, P].
    """
    b, n, _ = pair_valid.shape
    flat = pair_valid.reshape(b, n * n)
    count = flat.sum(1, keepdim=True)  # [B,1] valid pairs per sample
    rank = flat.long().cumsum(1) - 1  # 0-based rank of each valid pair (row-major)
    # even subsample: split the valid ranks into ``max_pairs`` equal buckets and keep the first pair of each
    denom = count.clamp_min(1)
    keep_sub = (rank * max_pairs) // denom != ((rank - 1) * max_pairs) // denom
    keep = flat & torch.where(count > max_pairs, keep_sub, torch.ones_like(flat))
    slot = keep.long().cumsum(1) - 1  # position of each kept pair in the compact list
    p = max(1, int(keep.sum(1).max().item()))
    sel = torch.zeros(b, p, dtype=torch.long, device=pair_valid.device)
    sel_mask = torch.zeros(b, p, dtype=torch.bool, device=pair_valid.device)
    bidx, pos = keep.nonzero(as_tuple=True)
    sel[bidx, slot[bidx, pos]] = pos
    sel_mask[bidx, slot[bidx, pos]] = True
    return sel, sel_mask


class RelationEncoder(nn.Module):
    """Object tokens + pair features -> pair tokens + pooled relation summary.

    forward(obj_tokens Float[B,64,object_dim], rel_feats Float[B,64,64,24], mask Bool[B,64]) ->
        (rel_tokens Float[B,64,64,relation_dim] or None, pooled Float[B,relation_dim]).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.relation_dim
        self.edge = nn.Sequential(nn.Linear(cfg.rel_feat_dim, cfg.edge_dim), nn.GELU())
        self.proj_i = nn.Linear(cfg.object_dim, d)
        self.proj_j = nn.Linear(cfg.object_dim, d)
        self.proj_e = nn.Linear(cfg.edge_dim, d)
        self.summary_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.summary_token, std=0.02)
        self.blocks = TransformerStack(d, cfg.relation_layers, cfg.heads, cfg.ffn_dim, cfg.dropout)
        self.pool_norm = nn.LayerNorm(d)

    def forward(self, obj_tokens: Tensor, rel_feats: Optional[Tensor], mask: Optional[Tensor] = None, *,
                return_tokens: bool = True) -> Tuple[Optional[Tensor], Tensor]:
        """Encode object pairs.

        Args:
            obj_tokens: Float[B, N, object_dim].
            rel_feats: Float[B, N, N, rel_feat_dim] (zeros if ``None``).
            mask: Bool[B, N] object validity (defaults to all valid).
            return_tokens: materialise the full Float[B, N, N, relation_dim] pair grid (contextualised where the
                pair was attended, base projection elsewhere, zeros at invalid pairs).
        Returns:
            (rel_tokens or None, pooled Float[B, relation_dim]).
        """
        b, n, _ = obj_tokens.shape
        dev = obj_tokens.device
        if mask is None:
            mask = torch.ones(b, n, dtype=torch.bool, device=dev)
        mask = mask.bool()
        if rel_feats is None:
            rel_feats = obj_tokens.new_zeros(b, n, n, self.cfg.rel_feat_dim)
        e = self.edge(rel_feats.to(obj_tokens.dtype))
        hi = self.proj_i(obj_tokens)
        hj = self.proj_j(obj_tokens)
        eye = torch.eye(n, dtype=torch.bool, device=dev)
        pair_valid = mask[:, :, None] & mask[:, None, :] & ~eye
        sel, sel_mask = select_pairs(pair_valid, self.cfg.max_pairs)
        p = sel.shape[1]
        i_idx, j_idx = sel // n, sel % n
        bidx = torch.arange(b, device=dev).unsqueeze(1).expand(b, p)
        base_sel = hi[bidx, i_idx] + hj[bidx, j_idx] + self.proj_e(e[bidx, i_idx, j_idx])
        seq = torch.cat([self.summary_token.expand(b, 1, -1), base_sel], dim=1)
        valid = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=dev), sel_mask], dim=1)
        h = self.blocks(seq, padding_mask_from_valid(valid))
        pooled = self.pool_norm(h[:, 0])
        if not return_tokens:
            return None, pooled
        refined = h[:, 1:]
        full = hi[:, :, None, :] + hj[:, None, :, :] + self.proj_e(e)
        full = full * pair_valid.unsqueeze(-1).to(full.dtype)
        # under bf16 autocast `full` (linear outputs) is bf16 while `refined` (LayerNorm) is fp32: index_put
        # needs equal dtypes
        full = full.index_put((bidx[sel_mask], i_idx[sel_mask], j_idx[sel_mask]), refined[sel_mask].to(full.dtype))
        return full, pooled
