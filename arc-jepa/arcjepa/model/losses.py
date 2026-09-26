"""JEPA training losses (spec §Losses).

L_g   = ||z_hat_Y - sg(z_Y)||^2                (weight 1.0)   -- mean-squared over the 512 dims
L_o   = mean_i ||obj_hat_i - sg(obj_Y,i)||^2   (1.0)          -- over the target grid's valid object slots
L_r   = ||rel_hat - sg(rel_Y)||^2               (0.5)          -- pooled relation summary (predictor contract)
L_prog = 1 - cos(r_task, z_p)                   (0.5)
L_rank = max(0, 0.2 - s(r, p+) + s(r, p-))      (0.5)          -- averaged over the provided negatives
L_var = (1/d) sum_j max(0, gamma - std_j(z))    (0.05, gamma 1.0) -- over every online grid latent in the batch

Prediction structure (v1 realisation): with K >= 2 context pairs the rule latent used to predict demo k is the
leave-one-out pool over the other demos; the test pair (when the target is known) is predicted from the full
rule latent. Squared errors are averaged over feature dims (MSE) rather than summed, which only rescales the
spec's weights by a constant. All grids of a batch go through the online encoder in one pass and through the
target encoder in one pass (fewer kernel launches).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from .arcjepa import ARCJEPA, TokenBatch, encode_rows
from .jepa_encoder import concat_grid_batches, split_encoding
from .program_encoder import ProgramTokenizer
from .target_encoder import EMATargetEncoder


@dataclass
class LossWeights:
    """Loss weights (field names mirror the ``jepa:`` block of the v1 yaml)."""

    global_loss: float = 1.0
    object_loss: float = 1.0
    relation_loss: float = 0.5
    program_loss: float = 0.5
    ranking_loss: float = 0.5
    variance_loss: float = 0.05
    margin: float = 0.2
    var_gamma: float = 1.0
    var_eps: float = 1e-4
    leave_one_out: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LossWeights":
        known = set(asdict(cls()).keys())
        return cls(**{k: v for k, v in d.items() if k in known})


def variance_loss(z: Tensor, gamma: float = 1.0, eps: float = 1e-4) -> Tensor:
    """VICReg-style variance term (1/d) sum_j relu(gamma - std_j) over the batch axis of Float[M, d]."""
    if z.shape[0] < 2:
        return z.new_zeros(())
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    m = mask.to(x.dtype)
    return (x * m).sum() / m.sum().clamp_min(1.0)


def _pad_tokens(tokens: Tensor, mask: Tensor, length: int) -> Tuple[Tensor, Tensor]:
    extra = length - tokens.shape[-1]
    if extra <= 0:
        return tokens, mask
    tokens = torch.cat([tokens, tokens.new_full((*tokens.shape[:-1], extra), ProgramTokenizer.PAD)], dim=-1)
    mask = torch.cat([mask, torch.zeros((*mask.shape[:-1], extra), dtype=torch.bool, device=mask.device)], dim=-1)
    return tokens, mask


def jepa_losses(model: ARCJEPA, target_encoder: EMATargetEncoder, batch: Dict[str, Tensor],
                programs_pos: Optional[TokenBatch] = None, programs_neg: Optional[TokenBatch] = None,
                weights: LossWeights = LossWeights()) -> Dict[str, Tensor]:
    """Compute every spec loss on an episode batch (layout: ``arcjepa.model.arcjepa`` docstring).

    Args:
        model: the online network.
        target_encoder: EMA copy of ``model.encoder`` providing stop-gradient targets.
        batch: episode batch.
        programs_pos: one exact program per episode (any form accepted by ``model.as_token_batch``); rows whose
            token mask is empty are skipped.
        programs_neg: N hard negatives per episode (``Long[B, N, L]`` tokens or a nested sequence).
        weights: loss weights and margins.
    Returns:
        dict with ``L_g, L_o, L_r, L_prog, L_rank, L_var, total`` plus diagnostics ``collapse_std`` (mean
        per-dim std of the online latents) and ``n_pred`` (number of predicted pairs).
    """
    cfg = model.cfg
    parts = model.split_episode(batch)
    b, k = parts.batch_size, parts.n_ctx
    ctx_mask = parts.ctx_mask
    dev = ctx_mask.device
    valid_c = ctx_mask.reshape(b * k)
    zero = torch.zeros((), device=dev)
    has_test = parts.test_in is not None
    has_target = parts.target is not None and bool(parts.target_valid.any())

    # ---- one online encoder pass over context inputs, context outputs and the test input
    dicts = [parts.ctx_in, parts.ctx_out] + ([parts.test_in] if has_test else [])
    valids = [valid_c, valid_c] + ([torch.ones(b, dtype=torch.bool, device=dev)] if has_test else [])
    sizes = [b * k, b * k] + ([b] if has_test else [])
    enc_all = model.encode_rows(concat_grid_batches(dicts, cfg), torch.cat(valids))
    pieces = split_encoding(enc_all, sizes)
    enc_xc, enc_yc = pieces[0], pieces[1]
    enc_xt = pieces[2] if has_test else None
    z_xc = enc_xc["z"].view(b, k, -1)
    z_yc = enc_yc["z"].view(b, k, -1)

    # ---- rule latents (full + leave-one-out)
    r_demo = model.rule_latent.per_demo(z_xc, z_yc)  # [B,K,R]
    r_full = model.rule_latent.pool(r_demo, ctx_mask)  # [B,R]
    if weights.leave_one_out and k > 1:
        eye = torch.eye(k, dtype=torch.bool, device=dev)
        loo_mask = ctx_mask[:, None, :] & ~eye  # [B,K,K]: row k keeps every demo but k
        r_loo = model.rule_latent.pool(r_demo.unsqueeze(1).expand(b, k, k, -1).reshape(b * k, k, -1),
                                       loo_mask.reshape(b * k, k)).view(b, k, -1)
        single = (ctx_mask.sum(1) <= 1).view(b, 1, 1)
        r_loo = torch.where(single, r_full.unsqueeze(1).expand(b, k, -1), r_loo)
    else:
        r_loo = r_full.unsqueeze(1).expand(b, k, -1)

    # ---- one target encoder pass over context outputs (+ the known target), stop-gradient
    with torch.no_grad():
        t_dicts = [parts.ctx_out] + ([parts.target] if has_target else [])
        t_valids = [valid_c] + ([parts.target_valid] if has_target else [])
        t_sizes = [b * k] + ([b] if has_target else [])
        tgt_all = encode_rows(target_encoder.encoder, concat_grid_batches(t_dicts, cfg), torch.cat(t_valids), cfg)
        t_pieces = split_encoding(tgt_all, t_sizes)
        tgt_yc = t_pieces[0]
        tgt_yt = t_pieces[1] if has_target else None

    # ---- one predictor pass over every (input, rule, target) triple
    pz: List[Tensor] = []; pr: List[Tensor] = []; po: List[Tensor] = []; pm: List[Tensor] = []
    tz: List[Tensor] = []; to: List[Tensor] = []; tm: List[Tensor] = []; tr: List[Tensor] = []
    if bool(valid_c.any()):
        pz.append(z_xc.reshape(b * k, -1)[valid_c]); pr.append(r_loo.reshape(b * k, -1)[valid_c])
        po.append(enc_xc["obj_tokens"][valid_c]); pm.append(enc_xc["obj_mask"][valid_c])
        tz.append(tgt_yc["z"][valid_c]); to.append(tgt_yc["obj_tokens"][valid_c])
        tm.append(tgt_yc["obj_mask"][valid_c]); tr.append(tgt_yc["rel_pooled"][valid_c])
    if has_test and has_target:
        tv = parts.target_valid
        pz.append(enc_xt["z"][tv]); pr.append(r_full[tv]); po.append(enc_xt["obj_tokens"][tv]); pm.append(enc_xt["obj_mask"][tv])
        tz.append(tgt_yt["z"][tv]); to.append(tgt_yt["obj_tokens"][tv]); tm.append(tgt_yt["obj_mask"][tv]); tr.append(tgt_yt["rel_pooled"][tv])
    if pz:
        pred = model.predictor(torch.cat(pz), torch.cat(pr), torch.cat(po), torch.cat(pm))
        z_t, o_t, o_m, r_t = torch.cat(tz), torch.cat(to), torch.cat(tm), torch.cat(tr)
        l_g = F.mse_loss(pred["z_hat"], z_t)
        l_o = _masked_mean(((pred["obj_hat"] - o_t) ** 2).mean(-1), o_m) if bool(o_m.any()) else zero
        l_r = F.mse_loss(pred["rel_hat"], r_t)
        n_pred = int(z_t.shape[0])
    else:
        l_g = l_o = l_r = zero
        n_pred = 0

    # ---- program alignment and ranking (positives and negatives share one program-encoder pass)
    l_prog = zero
    l_rank = zero
    if programs_pos is not None:
        tok_p, m_p = model.as_token_batch(programs_pos)
        pos_valid = m_p.any(-1)
        if programs_neg is not None:
            tok_n, m_n = model.as_token_batch(programs_neg, nested=True)
            if tok_n.dim() == 2:  # a single negative per task
                tok_n, m_n = tok_n.unsqueeze(1), m_n.unsqueeze(1)
            bn, n, ln = tok_n.shape
            length = max(tok_p.shape[1], ln)
            tok_p, m_p = _pad_tokens(tok_p, m_p, length)
            tok_n, m_n = _pad_tokens(tok_n, m_n, length)
            z_all = model.program_encoder(torch.cat([tok_p, tok_n.reshape(bn * n, length)]),
                                          torch.cat([m_p, m_n.reshape(bn * n, length)]))
            z_p, z_n = z_all[:tok_p.shape[0]], z_all[tok_p.shape[0]:].view(bn, n, -1)
            s_pos = model.scorer(r_full, z_p)  # [B]
            s_neg = model.scorer(r_full, z_n)  # [B,N]
            neg_valid = m_n.any(-1) & pos_valid.unsqueeze(1)
            l_rank = _masked_mean(F.relu(weights.margin - s_pos.unsqueeze(1) + s_neg), neg_valid)
        else:
            z_p = model.program_encoder(tok_p, m_p)
        # scorer.proj is the identity when program_dim == rule_dim (the spec: both 256)
        l_prog = _masked_mean(1.0 - F.cosine_similarity(r_full, model.scorer.proj(z_p), dim=-1), pos_valid)

    # ---- variance (anti-collapse) over every online grid latent
    online_valid = torch.cat(valids)
    z_cat = enc_all["z"][online_valid]
    l_var = variance_loss(z_cat, weights.var_gamma, weights.var_eps)
    collapse_std = torch.sqrt(z_cat.var(dim=0, unbiased=False) + weights.var_eps).mean() if z_cat.shape[0] > 1 else zero

    total = (weights.global_loss * l_g + weights.object_loss * l_o + weights.relation_loss * l_r
             + weights.program_loss * l_prog + weights.ranking_loss * l_rank + weights.variance_loss * l_var)
    return {"L_g": l_g, "L_o": l_o, "L_r": l_r, "L_prog": l_prog, "L_rank": l_rank, "L_var": l_var, "total": total,
            "collapse_std": collapse_std.detach(), "n_pred": torch.tensor(float(n_pred), device=dev)}
