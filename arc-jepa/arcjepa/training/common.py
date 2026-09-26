"""Shared training infrastructure for ARC-JEPA stages A-D.

Contents
--------
* config loading (``inherit:`` chains + ``a.b.c=value`` overrides), seeding, stable seed derivation;
* distributed helpers: ``torchrun`` process-per-GPU with manual gradient all-reduce, single-process fallback;
* bf16 autocast (CUDA only), AdamW with per-module learning-rate groups, warmup + cosine schedule;
* JSONL metrics logger, atomic resumable checkpoints, wall-clock budgets;
* episode datasets over synthetic JSONL files (lazy, byte-offset indexed) and real ARC episodes, with optional
  object tensors for the output grids (``out_obj_*`` keys understood by ``arcjepa.model``);
* a deterministic, rank-sharded, resumable epoch batch sampler with optional weighted sampling without
  replacement (family oversampling that never replicates an episode inside an epoch);
* program retrieval@k on a held-out synthetic program set;
* :func:`run_stage`, the generic optimisation loop every stage module delegates to.
"""
from __future__ import annotations

import contextlib
import copy
import json
import logging
import math
import os
import random
import re
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from arcjepa.core.types import Episode, Pair
from arcjepa.data.tensorize import collate as data_collate
from arcjepa.data.tensorize import encode_episode, encode_objects, resolve_parser
from arcjepa.model.arcjepa import ARCJEPA
from arcjepa.model.config import ModelConfig
from arcjepa.model.losses import LossWeights, jepa_losses
from arcjepa.model.target_encoder import EMATargetEncoder

log = logging.getLogger(__name__)

STAGES: Tuple[str, ...] = ("A", "B", "C", "D")
STAGE_NAMES: Dict[str, str] = {"A": "pretrain_jepa", "B": "train_real", "C": "program_align", "D": "hard_negatives"}
MODEL_MODULES: Tuple[str, ...] = ("encoder", "predictor", "rule_latent", "program_encoder", "scorer")
OUT_OBJECT_KEYS: Tuple[str, ...] = ("out_obj_feats", "out_obj_crops", "out_obj_mask", "out_rel_feats")
PathLike = Union[str, Path]


# ============================================================================================ config

def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge (``over`` wins; nested dicts are merged, everything else replaced)."""
    out: Dict[str, Any] = copy.deepcopy(dict(base))
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_yaml(path: Path, seen: Tuple[str, ...] = ()) -> Dict[str, Any]:
    key = str(path.resolve())
    if key in seen:
        raise ValueError(f"config inheritance cycle through {path}")
    with open(path, "r", encoding="utf-8") as fh:
        d = yaml.safe_load(fh) or {}
    parents = d.pop("inherit", None)
    if not parents:
        return d
    if isinstance(parents, str):
        parents = [parents]
    merged: Dict[str, Any] = {}
    for p in parents:
        pp = Path(p)
        if not pp.is_absolute():
            pp = path.parent / pp
        merged = deep_merge(merged, _read_yaml(pp, seen + (key,)))
    return deep_merge(merged, d)


def set_by_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    """``set_by_path(cfg, "stages.A.max_steps", 5)`` creating intermediate dicts as needed."""
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        if not isinstance(cur.get(k), dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def load_config(path: PathLike, overrides: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Load a YAML config, resolving ``inherit:`` (path or list, relative to the file) and ``key.path=value``
    overrides (values parsed as YAML, so ``5``, ``1e-4``, ``null`` and ``[1, 8]`` work)."""
    cfg = _read_yaml(Path(path))
    for ov in overrides or ():
        if "=" not in ov:
            raise ValueError(f"override {ov!r} is not of the form key.path=value")
        k, v = ov.split("=", 1)
        set_by_path(cfg, k.strip(), yaml.safe_load(v))
    cfg.setdefault("config_path", str(path))
    return cfg


def cfg_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Nested lookup ``cfg_get(cfg, "training.lr", 1e-4)`` (``None`` values count as missing)."""
    cur: Any = cfg
    for k in dotted.split("."):
        if not isinstance(cur, Mapping) or cur.get(k) is None:
            return default
        cur = cur[k]
    return cur


# ============================================================================================ seeding

def seed_for(*parts: Any) -> int:
    """Stable 31-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    return zlib.crc32(repr(parts).encode("utf-8")) & 0x7FFFFFFF


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


# ============================================================================================ distributed

@dataclass
class DistInfo:
    """Process-group information. ``enabled`` is True only under ``torchrun`` with WORLD_SIZE > 1."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    enabled: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def pick_device(pref: str = "auto", local_rank: int = 0) -> torch.device:
    """``auto`` -> ``cuda:<local_rank>`` when available else CPU; ``cpu``/``cuda`` force."""
    if pref == "cpu" or (pref == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        log.warning("device %r requested but CUDA is unavailable; using CPU", pref)
        return torch.device("cpu")
    return torch.device(f"cuda:{local_rank}")


def init_distributed(device_pref: str = "auto") -> DistInfo:
    """Initialise ``torch.distributed`` when launched by ``torchrun`` (WORLD_SIZE > 1), else a plain process."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = pick_device(device_pref, local_rank)
    if world <= 1:
        return DistInfo(0, 1, 0, device, False)
    import torch.distributed as tdist

    if device.type == "cuda":
        torch.cuda.set_device(device)
    if not tdist.is_initialized():
        tdist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    log.info("distributed: rank %d/%d on %s", rank, world, device)
    return DistInfo(rank, world, local_rank, device, True)


def cleanup_distributed(dist: DistInfo) -> None:
    """Destroy the process group (no-op for the fallback)."""
    if dist.enabled:
        import torch.distributed as tdist

        if tdist.is_initialized():
            tdist.destroy_process_group()


def barrier(dist: DistInfo) -> None:
    """Synchronise all ranks (no-op for the fallback)."""
    if dist.enabled:
        import torch.distributed as tdist

        tdist.barrier()


def broadcast_module(module: nn.Module, dist: DistInfo, src: int = 0) -> None:
    """Copy ``module``'s parameters and buffers from rank ``src`` to every rank."""
    if not dist.enabled:
        return
    import torch.distributed as tdist

    with torch.no_grad():
        for t in list(module.parameters()) + list(module.buffers()):
            tdist.broadcast(t.data, src)


def all_reduce_grads(params: Sequence[nn.Parameter], dist: DistInfo) -> None:
    """Average gradients over ranks in one flattened collective (missing grads count as zeros).

    This replaces the DDP wrapper: ``jepa_losses`` calls sub-modules directly instead of ``model.forward``, which
    DDP's reducer does not track; an explicit bucketless all-reduce keeps every rank's update identical.
    """
    if not dist.enabled or not params:
        return
    import torch.distributed as tdist

    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch.cat([g.reshape(-1).float() for g in grads])
    tdist.all_reduce(flat)
    flat /= dist.world_size
    off = 0
    for p, g in zip(params, grads):
        n = g.numel()
        new = flat[off:off + n].view_as(g).to(g.dtype)
        if p.grad is None:
            p.grad = new
        else:
            p.grad.copy_(new)
        off += n


def all_reduce_flag(flag: bool, dist: DistInfo) -> bool:
    """Logical OR of a boolean over ranks (so every rank stops at the same step)."""
    if not dist.enabled:
        return flag
    import torch.distributed as tdist

    t = torch.tensor([1.0 if flag else 0.0], device=dist.device)
    tdist.all_reduce(t, op=tdist.ReduceOp.MAX)
    return bool(t.item() > 0)


# ============================================================================================ numerics

def autocast(device: torch.device, precision: str = "bf16") -> contextlib.AbstractContextManager:
    """bf16 autocast on CUDA when ``precision == "bf16"``; a null context otherwise (CPU runs in fp32)."""
    if device.type == "cuda" and str(precision).lower() in ("bf16", "bfloat16"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def lr_multiplier(step: int, total: int, warmup: int, min_ratio: float = 0.0) -> float:
    """Linear warmup over ``warmup`` steps then cosine decay to ``min_ratio`` at ``total``."""
    total = max(1, int(total))
    warmup = max(0, min(int(warmup), total - 1))
    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup)
    frac = min(1.0, max(0.0, (step - warmup) / float(max(1, total - warmup))))
    return float(min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * frac)))


# ============================================================================================ model / optimiser

def model_config_from(cfg: Mapping[str, Any]) -> ModelConfig:
    """``ModelConfig`` from the ``model:`` block: ``preset`` (v1 | v1_wide | tiny) then explicit overrides."""
    mcfg = dict(cfg.get("model") or {})
    preset = str(mcfg.pop("preset", "v1"))
    if not hasattr(ModelConfig, preset):
        raise ValueError(f"unknown model preset {preset!r}")
    base = getattr(ModelConfig, preset)().to_dict()
    jepa = cfg.get("jepa") or {}
    for k in ("ema_start", "ema_end"):
        if k in jepa and k not in mcfg:
            mcfg[k] = jepa[k]
    base.update(mcfg)
    return ModelConfig.from_dict(base)


def build_model(cfg: Mapping[str, Any]) -> ARCJEPA:
    """Instantiate the ARC-JEPA network described by ``cfg`` (seeded by the caller)."""
    return ARCJEPA(model_config_from(cfg))


def build_target(model: ARCJEPA) -> EMATargetEncoder:
    """EMA target encoder of ``model.encoder`` with the config's tau schedule."""
    return EMATargetEncoder(model.encoder, model.cfg.ema_start, model.cfg.ema_end)


def loss_weights(cfg: Mapping[str, Any], stage_cfg: Mapping[str, Any]) -> LossWeights:
    """Loss weights: the ``jepa:`` block overridden by the stage's ``loss:`` block."""
    d = dict(cfg.get("jepa") or {})
    d.update(stage_cfg.get("loss") or {})
    return LossWeights.from_dict(d)


def configure_trainable(model: ARCJEPA, stage_cfg: Mapping[str, Any], base_lr: float, weight_decay: float
                        ) -> Tuple[List[Dict[str, Any]], List[nn.Parameter]]:
    """Parameter groups for a stage.

    ``train_modules`` ("all" or a list of top-level module names) get ``base_lr``; the other modules get
    ``base_lr * core_lr_mult`` and are frozen (``requires_grad=False``) when that multiplier is 0. 1-D tensors
    (biases, norms, embeddings' scales) are excluded from weight decay.
    """
    train_modules = stage_cfg.get("train_modules", "all")
    if train_modules in (None, "all"):
        train_modules = list(MODEL_MODULES)
    core_mult = float(stage_cfg.get("core_lr_mult", 0.0 if train_modules != list(MODEL_MODULES) else 1.0))
    groups: Dict[Tuple[str, bool], List[nn.Parameter]] = {}
    trainable: List[nn.Parameter] = []
    for name, module in model.named_children():
        is_head = name in train_modules
        mult = 1.0 if is_head else core_mult
        for p in module.parameters():
            p.requires_grad_(mult > 0.0)
            if mult <= 0.0:
                continue
            trainable.append(p)
            groups.setdefault(("head" if is_head else "core", p.ndim >= 2), []).append(p)
    out: List[Dict[str, Any]] = []
    for (kind, decay), params in sorted(groups.items()):
        lr = base_lr * (1.0 if kind == "head" else core_mult)
        out.append({"params": params, "lr": lr, "base_lr": lr, "weight_decay": weight_decay if decay else 0.0,
                    "name": f"{kind}_{'decay' if decay else 'nodecay'}"})
    return out, trainable


def build_optimizer(model: ARCJEPA, cfg: Mapping[str, Any], stage_cfg: Mapping[str, Any]
                    ) -> Tuple[torch.optim.Optimizer, List[nn.Parameter]]:
    """AdamW (betas, eps, wd from ``training:``; lr from the stage) over the stage's trainable parameters."""
    tr = cfg.get("training") or {}
    lr = float(stage_cfg.get("lr", tr.get("lr", 1.5e-4)))
    wd = float(stage_cfg.get("weight_decay", tr.get("weight_decay", 0.05)))
    groups, trainable = configure_trainable(model, stage_cfg, lr, wd)
    betas = tuple(float(b) for b in tr.get("betas", (0.9, 0.95)))
    opt = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=float(tr.get("eps", 1e-8)), weight_decay=wd)
    return opt, trainable


# ============================================================================================ logging / checkpoints

def _jsonable(v: Any) -> Any:
    if isinstance(v, Tensor):
        return float(v.detach().float().mean().item()) if v.numel() else None
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


class MetricsLogger:
    """Append-only JSONL metrics file (one object per line; written by the main rank only)."""

    def __init__(self, path: PathLike, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.t0 = time.time()
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Mapping[str, Any]) -> None:
        """Write one record (``wall`` = seconds since logger creation is added)."""
        if not self.enabled:
            return
        rec = {k: _jsonable(v) for k, v in record.items()}
        rec.setdefault("wall", round(time.time() - self.t0, 3))
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def read(self) -> List[Dict[str, Any]]:
        """All records written so far (empty when the file does not exist)."""
        if not self.path.is_file():
            return []
        with open(self.path, "r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


class Checkpointer:
    """Atomic ``last.pt`` (plus ``stage_<X>.pt`` snapshots) under a directory; only the main rank writes."""

    def __init__(self, directory: PathLike, enabled: bool = True) -> None:
        self.dir = Path(directory)
        self.enabled = bool(enabled)
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def last_path(self) -> Path:
        return self.dir / "last.pt"

    def save(self, state: Mapping[str, Any], name: str = "last.pt") -> Optional[Path]:
        """Write ``state`` to ``<dir>/<name>`` via a temporary file + ``os.replace``."""
        if not self.enabled:
            return None
        path = self.dir / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(dict(state), tmp)
        os.replace(tmp, path)
        return path

    def load(self, map_location: Union[str, torch.device] = "cpu") -> Optional[Dict[str, Any]]:
        """Load ``last.pt`` if present (``None`` otherwise)."""
        if not self.last_path.is_file():
            return None
        return torch.load(self.last_path, map_location=map_location, weights_only=False)


class TimeBudget:
    """Wall-clock budget measured from construction (``seconds <= 0`` is already expired)."""

    def __init__(self, seconds: float) -> None:
        self.seconds = float(seconds)
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0

    def remaining(self) -> float:
        return self.seconds - self.elapsed()

    def expired(self) -> bool:
        return self.remaining() <= 0.0


def setup_logging(out_dir: Optional[PathLike] = None, main: bool = True, level: int = logging.INFO) -> None:
    """Root logging to stderr and ``<out_dir>/train.log`` (main rank at ``level``, other ranks at WARNING)."""
    root = logging.getLogger()
    root.setLevel(level if main else logging.WARNING)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
    if out_dir is not None and main:
        path = str(Path(out_dir) / "train.log")
        if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == os.path.abspath(path)
                   for h in root.handlers):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)


def remove_file_logging(out_dir: PathLike) -> None:
    """Detach and close the ``train.log`` handler added by :func:`setup_logging` (lets tests delete ``out``)."""
    target = os.path.abspath(str(Path(out_dir) / "train.log"))
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == target:
            root.removeHandler(h)
            h.close()


# ============================================================================================ synthetic store

_SPLIT_RE = re.compile(r'"split"\s*:\s*"([A-Za-z_]+)"')


class SynthStore:
    """Lazy random access to a synthetic JSONL file (``arcjepa.synthetic.generate`` rows) by byte offsets.

    Keeps only ``int64`` offsets and split labels in memory, so million-task files fit every rank.
    """

    def __init__(self, path: PathLike, max_rows: Optional[int] = None) -> None:
        self.path = str(path)
        offsets: List[int] = []
        splits: List[str] = []
        with open(self.path, "rb") as fh:
            pos = 0
            for line in fh:
                if line.strip():
                    tail = line[-160:].decode("utf-8", "ignore")
                    m = _SPLIT_RE.findall(tail)
                    split = m[-1] if m else str(json.loads(line).get("split", "train"))
                    offsets.append(pos)
                    splits.append(split)
                    if max_rows is not None and len(offsets) >= max_rows:
                        break
                pos += len(line)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.splits = splits

    def __len__(self) -> int:
        return int(self.offsets.shape[0])

    def row(self, i: int) -> Dict[str, Any]:
        """Decoded JSON row ``i``."""
        with open(self.path, "rb") as fh:
            fh.seek(int(self.offsets[i]))
            return json.loads(fh.readline())

    def task(self, i: int) -> Any:
        """Row ``i`` as an ``arcjepa.synthetic.SynthTask``."""
        from arcjepa.synthetic.dataset import row_to_task

        return row_to_task(self.row(i))

    def indices(self, split: str) -> List[int]:
        return [i for i, s in enumerate(self.splits) if s == split]


def prepare_synthetic(cfg: Mapping[str, Any], out_dir: PathLike, dist: DistInfo) -> Tuple[SynthStore, List[int], List[int]]:
    """Locate or generate the synthetic task file; return ``(store, train_indices, heldout_indices)``.

    The held-out retrieval set is the compositional ``val_comp`` split (capped at ``heldout_max``); when it has
    fewer than ``heldout_min`` rows, the last train rows top it up and are removed from training.
    """
    scfg = cfg.get("synthetic") or {}
    path = scfg.get("path")
    n = int(scfg.get("n_tasks", 200))
    seed = int(scfg.get("seed", 1))
    path = str(path) if path else str(Path(out_dir) / f"synthetic_{n}_{seed}.jsonl")
    if dist.is_main and not os.path.isfile(path):
        from arcjepa.synthetic.dataset import generate

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = path + ".part"
        stats = generate(n, tmp, seed=seed, workers=int(scfg.get("workers", 1)),
                         n_pairs=tuple(scfg.get("n_pairs", (3, 6))))
        os.replace(tmp, path)
        log.info("generated %d synthetic tasks into %s (%s)", n, path, stats)
    barrier(dist)  # every rank, unconditionally: the file exists for all ranks after this point
    store = SynthStore(path, max_rows=scfg.get("max_rows"))
    train = store.indices("train")
    held = store.indices("val_comp")[: int(scfg.get("heldout_max", 256))]
    need = int(scfg.get("heldout_min", 16)) - len(held)
    if need > 0 and len(train) > need + 1:
        held = held + train[-need:]
        train = train[:-need]
    log.info("synthetic store %s: %d rows, %d train, %d held-out", path, len(store), len(train), len(held))
    return store, train, held


# ============================================================================================ episodes / tensors

def synth_episode(task: Any, test_index: int = -1) -> Episode:
    """Episode view of a synthetic task: pair ``test_index`` is the (known) test pair, the rest the context."""
    pairs: List[Pair] = list(task.pairs)
    ti = test_index % len(pairs)
    ctx = pairs[:ti] + pairs[ti + 1:]
    return Episode(episode_id=f"{task.task_id}:{ti}", task_id=task.task_id, split="synthetic", context=ctx,
                   test_input=pairs[ti].input, target_output=pairs[ti].output, source="synthetic",
                   meta={"category": getattr(task, "category", "")})


def encode_item(ep: Episode, parser: Any, max_ctx: int, out_objects: bool = True) -> Dict[str, Tensor]:
    """``encode_episode`` plus (optionally) ``out_obj_*`` tensors for the context outputs (slots 0..K-1) and
    the target (slot ``max_ctx``)."""
    item = encode_episode(ep, parser, max_ctx)
    if not out_objects:
        return item
    n_slots = max_ctx + 1
    grids = [(i, p.output) for i, p in enumerate(ep.context[:max_ctx])]
    if ep.target_output:
        grids.append((max_ctx, ep.target_output))
    empty = encode_objects([], parser)
    enc = {slot: encode_objects(g, parser) for slot, g in grids}
    for key, src in (("out_obj_feats", "obj_feats"), ("out_obj_crops", "obj_crops"), ("out_obj_mask", "obj_mask"),
                     ("out_rel_feats", "rel_feats")):
        item[key] = torch.stack([enc.get(s, empty)[src] for s in range(n_slots)])
    return item


def collate_items(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack tensor keys with ``arcjepa.data.collate``; list keys (``program``, ``negatives``) stay lists."""
    tensor_keys = [k for k, v in items[0].items() if torch.is_tensor(v)]
    out: Dict[str, Any] = data_collate([{k: it[k] for k in tensor_keys} for it in items])
    if "program" in items[0]:
        out["programs_pos"] = [it["program"] for it in items]
    if "negatives" in items[0]:
        out["programs_neg"] = [it["negatives"] for it in items]
    return out


def to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move every tensor of ``batch`` to ``device`` (non-blocking); other values are kept."""
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def pad_negatives(rng: random.Random, node: Any, k: int) -> List[str]:
    """``k`` hard-negative S-expressions for ``node`` (``hard_negatives`` topped up with ``mutate``)."""
    from arcjepa.dsl.canonicalize import canonicalize
    from arcjepa.dsl.mutations import hard_negatives, mutate

    try:
        pos_key = canonicalize(node).to_str()
    except Exception:  # noqa: BLE001 - canonicalisation is a nicety here
        pos_key = node.to_str()
    negs = [n.to_str() for n in hard_negatives(rng, node, k)]
    seen = set(negs) | {pos_key, node.to_str()}
    tries = 0
    while len(negs) < k and tries < 8 * k:
        tries += 1
        m = mutate(rng, node).to_str()
        if m not in seen:
            seen.add(m)
            negs.append(m)
    return negs[:k]


class SynthEpisodeDataset(Dataset):
    """Synthetic tasks as training episodes (random test pair per epoch), with program labels and optional hard
    negatives. Deterministic in ``(seed, epoch, index)``."""

    def __init__(self, store: SynthStore, indices: Sequence[int], parser: Any = None, *, max_ctx: int = 5,
                 out_objects: bool = True, with_program: bool = False, negatives: int = 0, seed: int = 0) -> None:
        self.store = store
        self.indices = list(indices)
        self.parser = resolve_parser(parser)
        self.max_ctx = int(max_ctx)
        self.out_objects = bool(out_objects)
        self.with_program = bool(with_program)
        self.negatives = int(negatives)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        task = self.store.task(self.indices[i])
        rng = random.Random(seed_for(self.seed, self.epoch, i))
        ep = synth_episode(task, rng.randrange(len(task.pairs)))
        item: Dict[str, Any] = encode_item(ep, self.parser, self.max_ctx, self.out_objects)
        if self.with_program or self.negatives:
            item["program"] = task.program
        if self.negatives:
            item["negatives"] = pad_negatives(rng, task.node(), self.negatives)
        return item


class RealEpisodeDataset(Dataset):
    """Real ARC episodes (context + known target) as training items."""

    def __init__(self, episodes: Sequence[Episode], parser: Any = None, *, max_ctx: int = 10,
                 out_objects: bool = True) -> None:
        self.episodes = list(episodes)
        self.parser = resolve_parser(parser)
        self.max_ctx = int(max_ctx)
        self.out_objects = bool(out_objects)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return encode_item(self.episodes[i], self.parser, self.max_ctx, self.out_objects)


class EpochBatchSampler:
    """Deterministic per-epoch micro-batches, sharded over ranks, with equal batch counts on every rank.

    With ``weights`` an epoch draws ``epoch_size`` distinct indices by weighted sampling *without replacement*
    (oversampling that never repeats an item inside an epoch); otherwise a permutation (truncated to
    ``epoch_size``).
    """

    def __init__(self, n: int, batch_size: int, *, seed: int = 0, rank: int = 0, world_size: int = 1,
                 weights: Optional[Sequence[float]] = None, epoch_size: Optional[int] = None) -> None:
        if n <= 0:
            raise ValueError("empty dataset")
        self.n = int(n)
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed)
        self.rank, self.world = int(rank), max(1, int(world_size))
        self.weights = None if weights is None else torch.as_tensor(list(weights), dtype=torch.double)
        n_draw = self.n if self.weights is None else int((self.weights > 0).sum().item())
        self.epoch_size = min(int(epoch_size), n_draw) if epoch_size else n_draw

    def indices(self, epoch: int) -> List[int]:
        g = torch.Generator().manual_seed(seed_for(self.seed, "epoch", epoch))
        if self.weights is None:
            idx = torch.randperm(self.n, generator=g)[: self.epoch_size]
        else:
            idx = torch.multinomial(self.weights, self.epoch_size, replacement=False, generator=g)
        return [int(i) for i in idx.tolist()]

    def per_rank(self) -> int:
        return max(1, self.epoch_size // self.world)

    def batches(self, epoch: int) -> List[List[int]]:
        """This rank's micro-batches for ``epoch`` (identical count on all ranks)."""
        idx = self.indices(epoch)
        per = self.per_rank()
        idx = (idx * (1 + (per * self.world) // max(1, len(idx))))[: per * self.world]
        mine = idx[self.rank::self.world]
        nb = max(1, len(mine) // self.batch_size)
        return [mine[b * self.batch_size:(b + 1) * self.batch_size] for b in range(nb)]

    def __len__(self) -> int:
        return max(1, self.per_rank() // self.batch_size)


# ============================================================================================ retrieval

def _chunks(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for s in range(0, len(seq), size):
        yield seq[s:s + size]


@torch.no_grad()
def rule_latents(model: ARCJEPA, episodes: Sequence[Episode], parser: Any, device: torch.device, *,
                 max_ctx: int = 10, batch_size: int = 16, precision: str = "fp32") -> Tensor:
    """r_task Float[N, rule_dim] (fp32, CPU) for episodes; output grids carry no objects (inference layout)."""
    was = model.training
    model.eval()
    parser = resolve_parser(parser)
    outs: List[Tensor] = []
    try:
        for chunk in _chunks(list(episodes), batch_size):
            batch = collate_items([encode_item(ep, parser, max_ctx, out_objects=False) for ep in chunk])
            with autocast(device, precision):
                r = model.rule_from_episode(to_device(batch, device))
            outs.append(r.float().cpu())
    finally:
        model.train(was)
    if not outs:
        return torch.zeros(0, model.cfg.rule_dim)
    return torch.cat(outs)


@torch.no_grad()
def retrieval_at_k(model: ARCJEPA, tasks: Sequence[Any], parser: Any, device: torch.device, *,
                   ks: Sequence[int] = (1, 8), max_ctx: int = 5, batch_size: int = 16,
                   precision: str = "fp32") -> Dict[str, float]:
    """Program retrieval@k on held-out synthetic tasks.

    Each task's rule latent (context = all pairs but the last) scores every distinct held-out program with the
    model's scorer; a hit at k means the task's own program (string identity) is among the top k.
    """
    tasks = list(tasks)
    if not tasks:
        return {f"retrieval@{k}": float("nan") for k in ks}
    was = model.training
    model.eval()
    try:
        r = rule_latents(model, [synth_episode(t, -1) for t in tasks], parser, device, max_ctx=max_ctx,
                         batch_size=batch_size, precision=precision).to(device)
        programs = sorted({t.program for t in tasks})
        pid = {p: i for i, p in enumerate(programs)}
        zs: List[Tensor] = []
        for chunk in _chunks(programs, 256):
            tok, mask = model.tokenize(list(chunk))
            with autocast(device, precision):
                zs.append(model.encode_programs(tok, mask).float())
        z_p = torch.cat(zs)
        n, p = r.shape[0], z_p.shape[0]
        scores = torch.empty(n, p, device=device)
        for s in range(0, n, 32):
            rr = r[s:s + 32]
            b = rr.shape[0]
            with autocast(device, precision):
                sc = model.scorer(rr.unsqueeze(1).expand(b, p, -1).reshape(b * p, -1),
                                  z_p.unsqueeze(0).expand(b, p, -1).reshape(b * p, -1))
            scores[s:s + b] = sc.float().view(b, p)
        target = torch.tensor([pid[t.program] for t in tasks], device=device)
        true_score = scores.gather(1, target.view(-1, 1))
        rank = (scores > true_score).sum(1)  # 0-based rank of the true program (ties favour it)
        out = {f"retrieval@{k}": float((rank < k).float().mean().item()) for k in ks}
        out.update({"retrieval_n": float(n), "retrieval_programs": float(p)})
        return out
    finally:
        model.train(was)


# ============================================================================================ stage loop

@dataclass
class TrainContext:
    """Everything a stage needs; created by ``train_all`` (or a stage module's own CLI)."""

    cfg: Dict[str, Any]
    model: ARCJEPA
    target: EMATargetEncoder
    device: torch.device
    dist: DistInfo
    out_dir: Path
    metrics: MetricsLogger
    ckpt: Checkpointer
    parser: Any = None
    store: Optional[SynthStore] = None
    synth_train: List[int] = field(default_factory=list)
    synth_heldout: List[int] = field(default_factory=list)
    completed: List[str] = field(default_factory=list)
    resume: Optional[Dict[str, Any]] = None
    global_step: int = 0
    budget: Optional[TimeBudget] = None
    last_ckpt_time: float = field(default_factory=time.time)
    summaries: Dict[str, Any] = field(default_factory=dict)

    @property
    def precision(self) -> str:
        return str(cfg_get(self.cfg, "training.precision", "bf16"))

    def heldout_tasks(self) -> List[Any]:
        if self.store is None:
            return []
        return [self.store.task(i) for i in self.synth_heldout]

    def state(self, stage: Optional[str], stage_state: Optional[Dict[str, Any]] = None,
              optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """Checkpoint payload."""
        return {"version": 1, "model": self.model.state_dict(), "target": self.target.state_dict(),
                "optimizer": optimizer.state_dict() if optimizer is not None else None, "stage": stage,
                "stage_state": stage_state or {}, "completed": list(self.completed), "global_step": self.global_step,
                "model_config": self.model.cfg.to_dict(), "config": self.cfg, "saved_at": time.time()}


@dataclass
class StageSpec:
    """One stage: its config block, dataset (+ optional sampling weights) and what the loss uses."""

    name: str
    cfg: Dict[str, Any]
    dataset: Dataset
    weights: Optional[Sequence[float]] = None
    use_programs: bool = False
    use_negatives: bool = False
    val_fn: Optional[Callable[["TrainContext"], Dict[str, float]]] = None


def _stage_steps(cfg: Mapping[str, Any], spec: StageSpec, sampler: EpochBatchSampler, accum: int) -> Tuple[int, int]:
    spe = max(1, len(sampler) // accum)
    epochs = float(spec.cfg.get("epochs", 1))
    planned = max(1, int(math.ceil(epochs * spe)))
    if spec.cfg.get("max_steps"):
        planned = min(planned, int(spec.cfg["max_steps"]))
    warm_ep = float(spec.cfg.get("warmup_epochs", cfg_get(cfg, "training.warmup_epochs", 0)))
    warm_frac = min(0.5, warm_ep / epochs) if epochs > 0 else 0.0
    return planned, int(round(warm_frac * planned))


def evaluate_retrieval(ctx: TrainContext, stage: str) -> Dict[str, float]:
    """Retrieval@k on the held-out synthetic set (main rank computes; other ranks wait)."""
    res: Dict[str, float] = {}
    if ctx.dist.is_main and ctx.synth_heldout:
        ks = [int(k) for k in cfg_get(ctx.cfg, "eval.retrieval_k", [1, 8])]
        res = retrieval_at_k(ctx.model, ctx.heldout_tasks(), ctx.parser, ctx.device, ks=ks,
                             max_ctx=int(cfg_get(ctx.cfg, "synthetic.max_ctx", 5)),
                             batch_size=int(cfg_get(ctx.cfg, "eval.batch_size", 16)), precision=ctx.precision)
        ctx.metrics.log({"kind": "eval", "stage": stage, "global_step": ctx.global_step, **res})
    barrier(ctx.dist)
    return res


def run_stage(ctx: TrainContext, spec: StageSpec, budget_s: float) -> Dict[str, Any]:
    """Optimise one stage for at most ``budget_s`` seconds (or its planned steps), resumably.

    Per optimizer step: autocast forward of :func:`jepa_losses`, backward (``accum_steps`` micro-batches),
    cross-rank gradient averaging, clipping, AdamW, EMA target update (when the encoder is trainable), JSONL
    logging every ``log_every`` steps, retrieval eval every ``eval.every_steps`` in program stages, and a
    checkpoint every ``checkpoint_minutes``. The stage is recorded as completed at the end (also when it stops
    on its time budget). Returns a summary dict.
    """
    cfg, model, dev, dist = ctx.cfg, ctx.model, ctx.device, ctx.dist
    tr = cfg.get("training") or {}
    name = spec.name
    t_start = time.time()
    deadline = t_start + max(0.0, float(budget_s))
    accum = max(1, int(spec.cfg.get("accum_steps", 1)))
    bs = int(spec.cfg.get("batch_size", 8))
    sampler = EpochBatchSampler(len(spec.dataset), bs, seed=seed_for(cfg.get("seed", 0), name), rank=dist.rank,
                                world_size=dist.world_size, weights=spec.weights,
                                epoch_size=spec.cfg.get("epoch_size"))
    planned, warmup = _stage_steps(cfg, spec, sampler, accum)
    opt, trainable = build_optimizer(model, cfg, spec.cfg)
    weights = loss_weights(cfg, spec.cfg)
    ema_on = any(p.requires_grad for p in model.encoder.parameters())
    clip = float(tr.get("grad_clip", 1.0))
    min_ratio = float(tr.get("min_lr_ratio", 0.0))
    log_every = max(1, int(spec.cfg.get("log_every", tr.get("log_every", 10))))
    ckpt_every_s = 60.0 * float(tr.get("checkpoint_minutes", 20))
    eval_every = int(cfg_get(cfg, "eval.every_steps", 0) or 0)
    out_obj_drop = float(cfg_get(cfg, "data.out_objects_drop", 0.0))
    num_workers = int(tr.get("num_workers", 0))
    drop_rng = random.Random(seed_for(cfg.get("seed", 0), name, "drop", dist.rank))

    step, epoch, offset, total = 0, 0, 0, planned
    rs = ctx.resume
    if rs and rs.get("stage") == name and rs.get("optimizer") is not None:
        st = rs.get("stage_state") or {}
        step, epoch, offset = int(st.get("step", 0)), int(st.get("epoch", 0)), int(st.get("offset", 0))
        total = int(st.get("total", planned))
        try:
            opt.load_state_dict(rs["optimizer"])
        except (ValueError, KeyError) as exc:
            log.warning("stage %s: optimizer state not restored (%s)", name, exc)
        log.info("stage %s: resuming at step %d/%d (epoch %d, batch %d)", name, step, total, epoch, offset)
    ctx.resume = None

    def stage_state() -> Dict[str, Any]:
        return {"step": step, "epoch": epoch, "offset": offset, "total": total, "planned": planned}

    model.train()
    opt.zero_grad(set_to_none=True)
    agg: Dict[str, float] = {}
    n_agg, micro, skipped = 0, 0, 0
    t_first: Optional[float] = None
    steps_this_run = 0
    last_log_t = time.time()
    min_steps = max(0, int(spec.cfg.get("min_steps", 1)))  # every stage takes >= min_steps optimizer steps

    def out_of_time() -> bool:
        return time.time() >= deadline and step >= min_steps

    stop = step >= total or out_of_time()
    log.info("stage %s (%s): %d items, batch %d x accum %d x world %d, planned %d steps, budget %.0fs",
             name, STAGE_NAMES.get(name, name), len(spec.dataset), bs, accum, dist.world_size, total, budget_s)
    while not stop:
        if hasattr(spec.dataset, "set_epoch"):
            spec.dataset.set_epoch(epoch)
        batches = sampler.batches(epoch)
        if offset >= len(batches):
            epoch, offset = epoch + 1, 0
            continue
        loader = DataLoader(spec.dataset, batch_sampler=batches[offset:], collate_fn=collate_items,
                            num_workers=num_workers, pin_memory=dev.type == "cuda",
                            persistent_workers=False)
        for batch in loader:
            offset += 1
            if out_obj_drop > 0 and drop_rng.random() < out_obj_drop:
                for k in OUT_OBJECT_KEYS:
                    batch.pop(k, None)
            pos = batch.pop("programs_pos", None) if spec.use_programs else None
            neg = batch.pop("programs_neg", None) if spec.use_negatives else None
            batch.pop("programs_pos", None)
            batch.pop("programs_neg", None)
            batch = to_device(batch, dev)
            with autocast(dev, ctx.precision):
                out = jepa_losses(model, ctx.target, batch, programs_pos=pos, programs_neg=neg, weights=weights)
            loss = out["total"].float()
            if not torch.isfinite(loss):
                skipped += 1
                opt.zero_grad(set_to_none=True)
                micro = 0
                log.warning("stage %s step %d: non-finite loss skipped", name, step)
                continue
            (loss / accum).backward()
            micro += 1
            for k in ("total", "L_g", "L_o", "L_r", "L_prog", "L_rank", "L_var", "collapse_std"):
                agg[k] = agg.get(k, 0.0) + float(out[k].detach().float().item())
            n_agg += 1
            if micro < accum:
                continue
            micro = 0
            all_reduce_grads(trainable, dist)
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, clip) if clip > 0 else torch.zeros(())
            mult = lr_multiplier(step, total, warmup, min_ratio)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * mult
            opt.step()
            opt.zero_grad(set_to_none=True)
            if ema_on:
                ctx.target.update(step, total, online=model.encoder)
            step += 1
            steps_this_run += 1
            ctx.global_step += 1
            now = time.time()
            if t_first is None:
                t_first = now
            elif steps_this_run >= 4:
                # shrink the schedule so the cosine reaches its end inside the time budget
                per = (now - t_first) / max(1, steps_this_run - 1)
                allowed = step + int(max(0.0, deadline - now) / max(per, 1e-6))
                if allowed < total:
                    total = max(step + 1, allowed)
            if step % log_every == 0 or step >= total:
                rec = {"kind": "train", "stage": name, "step": step, "total": total, "epoch": epoch,
                       "global_step": ctx.global_step, "lr": max(g["lr"] for g in opt.param_groups),
                       "grad_norm": float(gnorm), "sec_per_step": (now - last_log_t) / log_every,
                       "skipped": skipped}
                for k, v in agg.items():
                    key = "collapse_std" if k == "collapse_std" else "loss_" + k.replace("L_", "").lower()
                    rec[key] = v / max(1, n_agg)
                ctx.metrics.log(rec)
                agg, n_agg, last_log_t = {}, 0, now
            if eval_every and spec.use_programs and step % eval_every == 0 and step < total:
                evaluate_retrieval(ctx, name)
            want_stop = step >= total or out_of_time()
            stop = all_reduce_flag(want_stop, dist)
            if not stop and time.time() - ctx.last_ckpt_time >= ckpt_every_s:
                ctx.ckpt.save(ctx.state(name, stage_state(), opt))
                ctx.last_ckpt_time = time.time()
            if stop:
                break
        if not stop and offset >= len(batches):
            epoch, offset = epoch + 1, 0
    summary: Dict[str, Any] = {"stage": name, "steps": step, "steps_this_run": steps_this_run,
                               "total": total, "planned": planned,
                               "epochs_done": epoch, "seconds": round(time.time() - t_start, 2),
                               "skipped": skipped, "stopped_on_time": step < total}
    if spec.val_fn is not None:
        summary.update(spec.val_fn(ctx))
    summary.update(evaluate_retrieval(ctx, name))
    ctx.completed.append(name)
    ctx.summaries[name] = summary
    ctx.metrics.log({"kind": "stage_end", "global_step": ctx.global_step, **summary})
    ctx.ckpt.save(ctx.state(None))
    if cfg_get(cfg, "training.keep_stage_checkpoints", False):
        ctx.ckpt.save(ctx.state(None), name=f"stage_{name}.pt")
    ctx.last_ckpt_time = time.time()
    for p in model.parameters():
        p.requires_grad_(True)
    log.info("stage %s done: %s", name, summary)
    return summary


@torch.no_grad()
def eval_loss(ctx: TrainContext, dataset: Dataset, stage: str, max_items: int = 64, batch_size: int = 8,
              prefix: str = "val") -> Dict[str, float]:
    """Mean JEPA losses over the first ``max_items`` items of ``dataset`` (main rank; no gradient)."""
    res: Dict[str, float] = {}
    if ctx.dist.is_main and len(dataset) > 0:
        model = ctx.model
        was = model.training
        model.eval()
        weights = loss_weights(ctx.cfg, {})
        n = min(len(dataset), int(max_items))
        sums: Dict[str, float] = {}
        cnt = 0
        try:
            for s in range(0, n, batch_size):
                batch = collate_items([dataset[i] for i in range(s, min(n, s + batch_size))])
                batch.pop("programs_pos", None)
                batch.pop("programs_neg", None)
                with autocast(ctx.device, ctx.precision):
                    out = jepa_losses(model, ctx.target, to_device(batch, ctx.device), weights=weights)
                for k in ("total", "L_g", "L_o", "L_r", "L_var"):
                    sums[k] = sums.get(k, 0.0) + float(out[k].float().item())
                cnt += 1
        finally:
            model.train(was)
        res = {f"{prefix}_loss_{k.replace('L_', '').lower()}": v / max(1, cnt) for k, v in sums.items()}
        ctx.metrics.log({"kind": "eval", "stage": stage, "global_step": ctx.global_step, **res})
    barrier(ctx.dist)
    return res
