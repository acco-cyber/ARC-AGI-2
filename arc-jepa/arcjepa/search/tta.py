"""Test-time refinement of the task rule latent (FROZEN_SPEC "Test-time latent refinement", INTERFACES §6).

The spec optimises ``r_task`` from ``r_0`` with frozen weights for 8 steps at lr 0.05 on
``L_TTA = L_demo + 0.1 * ||r - r_0||^2``.  L_demo (exact demo error of programs) is not differentiable in ``r``;
the **v1 realisation** makes it differentiable as the *softmax-weighted expected demo error over the current
candidate set*::

    w_i(r)   = softmax_i( scorer(r, z_{p_i}) )          (program latents z_p computed once, no grad)
    L_demo(r) = sum_i w_i(r) * L_i                       (L_i = E(p_i) + cell_err_i / cells, constants)
    L_TTA(r)  = L_demo(r) + anchor * ||r - r_0||^2

and takes plain gradient-descent steps on ``r`` only (``torch.autograd.grad``; model parameters are never
updated and receive no ``.grad``).  Minimising it moves ``r`` towards latents under which the scorer prefers
programs that fit the demonstrations, which then re-ranks / re-guides the search.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

import torch
from torch import Tensor

from arcjepa.core.types import Pair

from .candidate import Candidate, demo_loss
from .verifier import total_cells

__all__ = ["refine_rule_latent", "tta_loss"]

log = logging.getLogger(__name__)


def tta_loss(scores: Tensor, losses: Tensor, r: Tensor, r0: Tensor, anchor: float) -> Tensor:
    """``sum softmax(scores) * losses + anchor * ||r - r0||^2`` (scalar)."""
    w = torch.softmax(scores, dim=-1)
    return (w * losses).sum() + anchor * ((r - r0) ** 2).sum()


def refine_rule_latent(model: Any, r0: Tensor, pairs: Sequence[Pair], candidates: Sequence[Candidate], steps: int = 8,
                       lr: float = 0.05, anchor: float = 0.1, *, max_candidates: int = 64,
                       history: Optional[List[float]] = None) -> Tensor:
    """Refine ``r0`` (Float[rule_dim] or Float[1, rule_dim]) against the candidate set; returns a tensor of the
    same shape (detached).  ``r0`` is returned unchanged when there is no model, no candidate, or the candidate
    losses are all equal (zero gradient of the data term).  ``history`` (optional list) receives the loss value
    before every step and after the last one.
    """
    base = r0.detach().clone()
    if model is None or not candidates or steps <= 0:
        return base
    cands = list(candidates)[:max_candidates]
    n_cells = total_cells(pairs)
    losses_py = [demo_loss(c.demo_err, c.cell_err, n_cells) for c in cands]
    try:
        device = next(model.parameters()).device
    except StopIteration:  # pragma: no cover - parameter-free model
        device = base.device
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.no_grad():
            tokens, mask = model.as_token_batch([c.program for c in cands])
            z_p = model.encode_programs(tokens, mask).float()  # [N, program_dim]
        r_init = base.reshape(-1).to(device=device, dtype=torch.float32)
        losses = torch.tensor(losses_py, dtype=torch.float32, device=device)
        r = r_init.clone()
        for _ in range(int(steps)):
            with torch.enable_grad():
                rv = r.clone().requires_grad_(True)
                scores = model.scorer(rv.unsqueeze(0), z_p.unsqueeze(0)).reshape(-1)
                loss = tta_loss(scores, losses, rv, r_init, anchor)
                (grad,) = torch.autograd.grad(loss, [rv])
            if history is not None:
                history.append(float(loss.detach()))
            r = (rv - lr * grad).detach()
        if history is not None:
            with torch.no_grad():
                scores = model.scorer(r.unsqueeze(0), z_p.unsqueeze(0)).reshape(-1)
                history.append(float(tta_loss(scores, losses, r, r_init, anchor)))
    finally:
        if was_training:
            model.train()
    return r.reshape(base.shape).to(dtype=base.dtype, device=base.device)
