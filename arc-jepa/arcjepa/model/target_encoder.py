"""EMA target encoder (spec §Model: tau 0.996 -> 0.9995 on a cosine schedule, no gradient)."""
from __future__ import annotations

import copy
import math
import weakref
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn

from .jepa_encoder import GridJEPAEncoder


class EMATargetEncoder(nn.Module):
    """Exponential-moving-average copy of the online ``GridJEPAEncoder``.

    The copy is always in eval mode and never receives gradients. ``update(step, total_steps)`` moves its
    parameters towards the online encoder with tau(step) = tau_end - (tau_end - tau_start) * (1 + cos(pi *
    step / total_steps)) / 2, i.e. a cosine ramp from ``tau_start`` to ``tau_end``.
    """

    def __init__(self, online: GridJEPAEncoder, tau_start: float = 0.996, tau_end: float = 0.9995) -> None:
        super().__init__()
        if not (0.0 < tau_start <= tau_end < 1.0):
            raise ValueError("need 0 < tau_start <= tau_end < 1")
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.encoder = copy.deepcopy(online)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        self._online_ref = weakref.ref(online)
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long), persistent=True)
        self.register_buffer("tau_current", torch.tensor(self.tau_start), persistent=True)

    def train(self, mode: bool = True) -> "EMATargetEncoder":
        """Keep the target copy in eval mode regardless of the parent's mode."""
        super().train(mode)
        self.encoder.eval()
        return self

    def tau(self, step: int, total_steps: int) -> float:
        """Cosine schedule value at ``step`` of ``total_steps`` (clamped to [tau_start, tau_end])."""
        if total_steps <= 0:
            return self.tau_end
        frac = min(max(step / float(total_steps), 0.0), 1.0)
        return self.tau_end - (self.tau_end - self.tau_start) * (1.0 + math.cos(math.pi * frac)) / 2.0

    @torch.no_grad()
    def update(self, step: int, total_steps: int, online: Optional[GridJEPAEncoder] = None) -> float:
        """EMA step: target <- tau * target + (1 - tau) * online. Buffers are copied. Returns tau used."""
        src = online if online is not None else self._online_ref()
        if src is None:
            raise RuntimeError("online encoder no longer alive; pass it explicitly to update()")
        tau = self.tau(step, total_steps)
        params_t = list(self.encoder.parameters())
        params_o = [p.detach() for p in src.parameters()]
        if len(params_t) != len(params_o):
            raise RuntimeError("online / target parameter lists differ in length")
        torch._foreach_mul_(params_t, tau)
        torch._foreach_add_(params_t, params_o, alpha=1.0 - tau)
        for b_t, b_o in zip(self.encoder.buffers(), src.buffers()):
            b_t.copy_(b_o)
        self.num_updates += 1
        self.tau_current.fill_(tau)
        return tau

    @torch.no_grad()
    def copy_from_online(self, online: Optional[GridJEPAEncoder] = None) -> None:
        """Hard re-synchronisation of the target with the online encoder."""
        src = online if online is not None else self._online_ref()
        if src is None:
            raise RuntimeError("online encoder no longer alive; pass it explicitly")
        self.encoder.load_state_dict(src.state_dict())

    @torch.no_grad()
    def forward(self, batch: Dict[str, Tensor], **kwargs: Any) -> Dict[str, Tensor]:
        """Target encoding of a grid batch (same dict contract as ``GridJEPAEncoder``), no gradient."""
        return self.encoder(batch, **kwargs)
