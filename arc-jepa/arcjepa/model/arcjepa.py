"""Composite ARC-JEPA model: grid encoder + rule latent + predictor + program encoder + scorer.

Episode batch layout (from ``arcjepa.data.tensorize.collate``; every tensor carries a leading batch axis B):
    ctx_in Long[B,K,30,30], ctx_out Long[B,K,30,30], ctx_mask Bool[B,K], test_in Long[B,30,30],
    target Long[B,30,30] (all PAD when unknown),
    obj_feats Float[B,K+1,64,32], obj_crops Int[B,K+1,64,30,30], obj_mask Bool[B,K+1,64],
    rel_feats Float[B,K+1,64,64,24]   -- objects of the K context INPUTS and (index K) the test input,
    out_obj_feats / out_obj_crops / out_obj_mask / out_rel_feats [B,K+1,...]  -- optional objects of the K
    context OUTPUTS and (index K) the target; outputs are encoded object-free when absent.
Grid masks are derived from ``grid != PAD_ID``. A plain grid batch dict (see ``jepa_encoder``) is accepted
wherever a single grid is encoded.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from arcjepa.core.types import PAD_ID

from .config import ModelConfig
from .jepa_encoder import GridJEPAEncoder, concat_grid_batches, split_encoding
from .predictor import TransformationPredictor
from .program_encoder import ProgramEncoder, ProgramTokenizer
from .rule_latent import RuleLatent
from .scorer import Scorer

TokenBatch = Union[Tuple[Tensor, Tensor], Dict[str, Tensor], Tensor, Sequence[Any]]

# Offline package layout (written by ``arcjepa.training.export``, read by ``ARCJEPA.load_package``).
PACKAGE_FORMAT = "arcjepa-package-v1"
CONFIG_FILE = "config.json"
VOCAB_FILE = "vocab.json"
WEIGHTS_SAFETENSORS_FILE = "model.safetensors"
WEIGHTS_PT_FILE = "model.pt"
MEMORY_NPZ_FILE = "program_memory.npz"
PROGRAMS_FILE = "programs.json"


def load_package_memory(pkg_dir: Union[str, "os.PathLike[str]"]) -> Optional[Any]:
    """``TransformationMemory`` from ``program_memory.npz`` + ``programs.json`` of a package (None if absent)."""
    import numpy as np

    from .memory import TransformationMemory

    pkg = Path(pkg_dir)
    if not (pkg / MEMORY_NPZ_FILE).is_file() or not (pkg / PROGRAMS_FILE).is_file():
        return None
    doc = json.loads((pkg / PROGRAMS_FILE).read_text(encoding="utf-8"))
    with np.load(pkg / MEMORY_NPZ_FILE) as z:
        latents = np.asarray(z["latents"], dtype=np.float32)
    records = list(doc.get("records", []))
    if latents.shape[0] != len(records):
        raise ValueError("program_memory.npz / programs.json record count mismatch")
    mem = TransformationMemory(int(doc.get("dim", latents.shape[1] if latents.ndim == 2 else 256)))
    for vec, rec in zip(latents, records):
        rec = dict(rec)
        mem.add(vec, rec.pop("program", ""), int(rec.pop("complexity", 0)), str(rec.pop("family", "unknown")), **rec)
    return mem


@dataclass
class EpisodeParts:
    """An episode batch split into grid batches (context rows flattened to B*K)."""

    ctx_mask: Tensor  # Bool[B, K]
    ctx_in: Dict[str, Tensor]  # grid batch with B*K rows
    ctx_out: Dict[str, Tensor]  # grid batch with B*K rows
    test_in: Optional[Dict[str, Tensor]]  # grid batch with B rows
    target: Optional[Dict[str, Tensor]]  # grid batch with B rows
    target_valid: Optional[Tensor]  # Bool[B]

    @property
    def batch_size(self) -> int:
        return int(self.ctx_mask.shape[0])

    @property
    def n_ctx(self) -> int:
        return int(self.ctx_mask.shape[1])


def _objects_slice(batch: Dict[str, Tensor], prefix: str, k: int, which: str) -> Optional[Dict[str, Tensor]]:
    crops, feats, om = (batch.get(prefix + "obj_crops"), batch.get(prefix + "obj_feats"), batch.get(prefix + "obj_mask"))
    if crops is None or feats is None or om is None:
        return None
    rf = batch.get(prefix + "rel_feats")
    avail = om.shape[1]
    if which == "ctx":
        if avail < k:
            return None

        def sl(t: Tensor) -> Tensor:
            return t[:, :k].reshape(-1, *t.shape[2:])
    else:
        if avail <= k:
            return None

        def sl(t: Tensor) -> Tensor:
            return t[:, k]
    out = {"obj_crops": sl(crops), "obj_feats": sl(feats), "obj_mask": sl(om)}
    if rf is not None:
        out["rel_feats"] = sl(rf)
    return out


def split_episode_batch(batch: Dict[str, Tensor], pad_id: int = PAD_ID) -> EpisodeParts:
    """Split an episode batch (module docstring layout) into per-grid batches."""
    ctx_in = batch["ctx_in"].long()
    ctx_out = batch["ctx_out"].long()
    b, k = ctx_in.shape[:2]
    ctx_mask = batch.get("ctx_mask")
    if ctx_mask is None:
        ctx_mask = (ctx_in != pad_id).flatten(2).any(-1)
    ctx_mask = ctx_mask.bool()

    def grid_dict(grid: Tensor, objs: Optional[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        d = {"grid": grid, "mask": grid != pad_id}
        if objs:
            d.update(objs)
        return d

    parts_in = grid_dict(ctx_in.reshape(b * k, *ctx_in.shape[2:]), _objects_slice(batch, "", k, "ctx"))
    parts_out = grid_dict(ctx_out.reshape(b * k, *ctx_out.shape[2:]), _objects_slice(batch, "out_", k, "ctx"))
    test = None
    if batch.get("test_in") is not None:
        test = grid_dict(batch["test_in"].long(), _objects_slice(batch, "", k, "test"))
    target, target_valid = None, None
    if batch.get("target") is not None:
        tgt = batch["target"].long()
        target_valid = (tgt != pad_id).flatten(1).any(-1)
        target = grid_dict(tgt, _objects_slice(batch, "out_", k, "test"))
    return EpisodeParts(ctx_mask, parts_in, parts_out, test, target, target_valid)


def empty_encoding(cfg: ModelConfig, m: int, device: torch.device, n_objects: Optional[int] = None) -> Dict[str, Tensor]:
    """Zero encoder outputs for ``m`` rows (used when no row of a grid batch is valid)."""
    n = n_objects if n_objects is not None else cfg.max_objects
    z = torch.zeros
    return {"z": z(m, cfg.jepa_dim, device=device), "z_global": z(m, cfg.global_dim, device=device),
            "obj_tokens": z(m, n, cfg.object_dim, device=device),
            "obj_mask": torch.zeros(m, n, dtype=torch.bool, device=device),
            "obj_pooled": z(m, cfg.object_dim, device=device), "rel_pooled": z(m, cfg.relation_dim, device=device),
            "cell_pooled": z(m, cfg.cell_dim, device=device)}


def encode_rows(encoder: nn.Module, grid_batch: Dict[str, Tensor], valid: Tensor, cfg: ModelConfig) -> Dict[str, Tensor]:
    """Run ``encoder`` on the valid rows of a grid batch only; invalid rows get zeros in every output."""
    m = int(valid.shape[0])
    valid = valid.bool()
    if bool(valid.all()):
        return encoder(grid_batch)
    dev = grid_batch["grid"].device
    om = grid_batch.get("obj_mask")
    if not bool(valid.any()):
        return empty_encoding(cfg, m, dev, None if om is None else int(om.shape[1]))
    sub = {key: v[valid] for key, v in grid_batch.items() if torch.is_tensor(v)}
    out_sub = encoder(sub)
    out: Dict[str, Tensor] = {}
    for key, v in out_sub.items():
        full = v.new_zeros((m, *v.shape[1:]))
        full[valid] = v
        out[key] = full
    return out


class ARCJEPA(nn.Module):
    """The full ARC-JEPA network (the EMA target encoder is owned by the training loop, not by this module)."""

    def __init__(self, cfg: Optional[ModelConfig] = None, tokenizer: Optional[ProgramTokenizer] = None) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else ModelConfig.v1()
        self.tokenizer = tokenizer if tokenizer is not None else ProgramTokenizer(max_depth_tokens=self.cfg.program_depth_tokens)
        self.encoder = GridJEPAEncoder(self.cfg)
        self.predictor = TransformationPredictor(self.cfg)
        self.rule_latent = RuleLatent(self.cfg)
        self.program_encoder = ProgramEncoder(self.cfg, self.tokenizer)
        self.scorer = Scorer(self.cfg)

    # ---------------------------------------------------------------- utilities
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Total parameter count (excludes any EMA target encoder)."""
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))

    def split_episode(self, batch: Dict[str, Tensor]) -> EpisodeParts:
        return split_episode_batch(batch, self.cfg.pad_id)

    def encode_rows(self, grid_batch: Dict[str, Tensor], valid: Tensor) -> Dict[str, Tensor]:
        return encode_rows(self.encoder, grid_batch, valid, self.cfg)

    # ---------------------------------------------------------------- grids
    def encode_grid_full(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Full encoder output for a grid batch dict, or for the test input of an episode batch."""
        if "grid" in batch:
            return self.encoder(batch)
        parts = self.split_episode(batch)
        if parts.test_in is None:
            raise KeyError("batch has neither 'grid' nor 'test_in'")
        return self.encoder(parts.test_in)

    def encode_grid(self, batch: Dict[str, Tensor]) -> Tensor:
        """z_X Float[B, jepa_dim] for a grid batch (or an episode batch's test input)."""
        return self.encode_grid_full(batch)["z"]

    def encode_context(self, batch: Union[Dict[str, Tensor], EpisodeParts]) -> Dict[str, Any]:
        """Encode the K context pairs of an episode batch with the online encoder.

        Returns ``z_x`` / ``z_y`` Float[B, K, jepa_dim], ``ctx_mask`` Bool[B, K], and the flattened encoder
        dicts ``enc_x`` / ``enc_y`` (B*K rows, zeros where the context slot is padding).
        """
        parts = batch if isinstance(batch, EpisodeParts) else self.split_episode(batch)
        b, k = parts.batch_size, parts.n_ctx
        valid = parts.ctx_mask.reshape(b * k)
        both = self.encode_rows(concat_grid_batches([parts.ctx_in, parts.ctx_out], self.cfg), torch.cat([valid, valid]))
        enc_x, enc_y = split_encoding(both, [b * k, b * k])
        return {"z_x": enc_x["z"].view(b, k, -1), "z_y": enc_y["z"].view(b, k, -1), "ctx_mask": parts.ctx_mask,
                "enc_x": enc_x, "enc_y": enc_y, "parts": parts}

    def rule_from_episode(self, batch: Dict[str, Tensor]) -> Tensor:
        """r_task Float[B, rule_dim] from the context pairs of an episode batch."""
        ctx = self.encode_context(batch)
        return self.rule_latent(ctx["z_x"], ctx["z_y"], ctx["ctx_mask"])

    # ---------------------------------------------------------------- prediction
    def predict_full(self, z_x: Tensor, r: Tensor, obj_tokens_x: Optional[Tensor] = None,
                     obj_mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        return self.predictor(z_x, r, obj_tokens_x, obj_mask)

    def predict(self, z_x: Tensor, r: Tensor, obj_tokens_x: Optional[Tensor] = None,
                obj_mask: Optional[Tensor] = None) -> Tensor:
        """z_hat_Y Float[B, jepa_dim] from the input latent and the rule latent."""
        return self.predict_full(z_x, r, obj_tokens_x, obj_mask)["z_hat"]

    # ---------------------------------------------------------------- programs
    def tokenize(self, programs: Sequence[Any]) -> Tuple[Tensor, Tensor]:
        """Nodes / S-expressions -> (tokens Long[N, L], mask Bool[N, L]) on the model device."""
        return self.tokenizer.encode_batch(programs, max_len=self.cfg.max_program_len, device=self.device)

    def as_token_batch(self, programs: TokenBatch, nested: bool = False) -> Tuple[Tensor, Tensor]:
        """Normalise the many accepted program-batch forms to ``(tokens, mask)``.

        Accepted: ``(tokens, mask)`` tensors, ``{"tokens", "mask"}``, a tokens tensor (mask = ``!= PAD``), a
        sequence of programs (Node / S-expression / pre-encoded id list). With ``nested=True`` a sequence of
        sequences (N candidates per task) yields Long[B, N, L] with all-PAD rows for missing candidates.
        """
        if isinstance(programs, dict):
            tokens = programs["tokens"]
            mask = programs.get("mask")
            return tokens.long(), (tokens != ProgramTokenizer.PAD) if mask is None else mask.bool()
        if torch.is_tensor(programs):
            return programs.long(), programs != ProgramTokenizer.PAD
        if isinstance(programs, (tuple, list)) and len(programs) == 2 and torch.is_tensor(programs[0]) \
                and torch.is_tensor(programs[1]):
            return programs[0].long(), programs[1].bool()
        seqs = list(programs)
        if not nested:
            ids = [self._ids(p) for p in seqs]
            return self.tokenizer.pad_batch(ids, max_len=self.cfg.max_program_len, device=self.device)
        per_task: List[List[List[int]]] = [[self._ids(p) for p in cands] for cands in seqs]
        n = max(1, max((len(c) for c in per_task), default=1))
        length = max(1, max((len(ids) for c in per_task for ids in c), default=1))
        length = min(length, self.cfg.max_program_len)
        tokens = torch.full((len(per_task), n, length), ProgramTokenizer.PAD, dtype=torch.long, device=self.device)
        mask = torch.zeros(len(per_task), n, length, dtype=torch.bool, device=self.device)
        for i, cands in enumerate(per_task):
            for j, ids in enumerate(cands):
                ids = ids[:length]
                tokens[i, j, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
                mask[i, j, :len(ids)] = True
        return tokens, mask

    def _ids(self, program: Any) -> List[int]:
        if isinstance(program, (list, tuple)) and all(isinstance(x, int) and not isinstance(x, bool) for x in program):
            return list(program)
        return self.tokenizer.encode(program)

    def encode_programs(self, tokens: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """z_p Float[N, program_dim] for padded token streams."""
        return self.program_encoder(tokens, mask)

    def score_programs(self, r: Tensor, token_batch: TokenBatch) -> Tensor:
        """Neural compatibility scores.

        ``r`` Float[B, rule_dim] (or Float[rule_dim]) with tokens Long[B, L] -> Float[B]; a single rule latent
        with tokens Long[N, L] -> Float[N]; tokens Long[B, N, L] -> Float[B, N].
        """
        tokens, mask = self.as_token_batch(token_batch, nested=False)
        if r.dim() == 1:
            r = r.unsqueeze(0)
        if tokens.dim() == 3:
            b, n, length = tokens.shape
            z_p = self.program_encoder(tokens.reshape(b * n, length), mask.reshape(b * n, length)).view(b, n, -1)
            return self.scorer(r, z_p)
        z_p = self.program_encoder(tokens, mask)
        if r.shape[0] == 1 and z_p.shape[0] != 1:
            r = r.expand(z_p.shape[0], -1)
        return self.scorer(r, z_p)

    # ---------------------------------------------------------------- offline package
    @classmethod
    def load_package(cls, path: Union[str, "os.PathLike[str]"], device: Union[str, torch.device] = "cpu",
                     load_memory: bool = True) -> "ARCJEPA":
        """Load a package written by ``arcjepa.training.export`` (config.json, vocab.json, model.safetensors or
        model.pt, program_memory.npz + programs.json). Returns the model in eval mode on ``device``; the
        transformation memory (or ``None``) is attached as ``model.memory`` and the parsed config.json as
        ``model.package_meta``."""
        pkg = Path(path)
        meta = json.loads((pkg / CONFIG_FILE).read_text(encoding="utf-8"))
        cfg = ModelConfig.from_dict(meta.get("model", meta))
        vocab = pkg / meta.get("vocab", VOCAB_FILE)
        tokenizer = ProgramTokenizer.load(vocab) if vocab.is_file() else None
        model = cls(cfg, tokenizer)
        weights = meta.get("weights") or (WEIGHTS_SAFETENSORS_FILE if (pkg / WEIGHTS_SAFETENSORS_FILE).is_file()
                                          else WEIGHTS_PT_FILE)
        wpath = pkg / weights
        if wpath.suffix == ".safetensors":
            from safetensors.torch import load_model

            load_model(model, str(wpath), strict=True)
        else:
            model.load_state_dict(torch.load(wpath, map_location="cpu", weights_only=True))
        model.to(device)
        model.eval()
        model.memory = load_package_memory(pkg) if load_memory else None
        model.package_meta = meta
        return model

    # ---------------------------------------------------------------- convenience forward
    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Rule latent from the context, test-input latent and predicted target latent."""
        parts = self.split_episode(batch)
        ctx = self.encode_context(parts)
        r = self.rule_latent(ctx["z_x"], ctx["z_y"], ctx["ctx_mask"])
        out: Dict[str, Tensor] = {"r_task": r}
        if parts.test_in is not None:
            enc = self.encoder(parts.test_in)
            pred = self.predictor(enc["z"], r, enc["obj_tokens"], enc["obj_mask"])
            out.update({"z_x": enc["z"], "z_hat": pred["z_hat"], "obj_hat": pred["obj_hat"], "rel_hat": pred["rel_hat"]})
        return out
