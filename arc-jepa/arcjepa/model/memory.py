"""Transformation memory: numpy cosine KNN over (rule latent, program, complexity, family) records.

Spec §Model: ``rules.faiss``-style KNN, K = 16, built only from synthetic + training tasks. Implemented with
numpy (no faiss dependency): latents are L2-normalised and scored by dot product.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

log = logging.getLogger(__name__)


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), eps)


class _HybridLoad:
    """Descriptor so that both ``TransformationMemory.load(path)`` and ``mem.load(path)`` work."""

    def __get__(self, obj: Optional["TransformationMemory"], owner: type):
        def _load(path: Union[str, Path]) -> "TransformationMemory":
            target = obj if obj is not None else owner()
            return target._load_into(path)
        return _load


class TransformationMemory:
    """KNN memory of rule latents with attached programs.

    ``add(rule_latent np[dim], program: str, complexity: int, family: str)``, ``query(r, k=16)`` -> list of dicts
    ``{index, score, program, complexity, family, ...meta}`` sorted by cosine similarity, ``save(path)`` writes
    ``<base>.npz`` (latents) + ``<base>.json`` (records), ``load(path)`` reads them back.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = int(dim)
        self._rows: List[np.ndarray] = []
        self._matrix: Optional[np.ndarray] = None
        self.records: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.records)

    # ---------------------------------------------------------------- building
    def add(self, rule_latent: Any, program: str, complexity: int, family: str, **meta: Any) -> int:
        """Append one record; returns its index."""
        v = _to_numpy(rule_latent).reshape(-1)
        if not self._rows and v.shape[0] != self.dim:
            self.dim = int(v.shape[0])
        if v.shape[0] != self.dim:
            raise ValueError(f"rule latent has dim {v.shape[0]}, memory expects {self.dim}")
        self._rows.append(_unit(v))
        self._matrix = None
        rec = {"program": str(program), "complexity": int(complexity), "family": str(family)}
        rec.update(meta)
        self.records.append(rec)
        return len(self.records) - 1

    def add_many(self, latents: Any, programs: Sequence[str], complexities: Sequence[int],
                 families: Sequence[str]) -> None:
        """Append several records (``latents`` is [N, dim])."""
        mat = _to_numpy(latents).reshape(len(programs), -1)
        for i in range(mat.shape[0]):
            self.add(mat[i], programs[i], complexities[i], families[i])

    def clear(self) -> None:
        self._rows, self._matrix, self.records = [], None, []

    def matrix(self) -> np.ndarray:
        """All unit latents as Float32[N, dim] (empty -> [0, dim])."""
        if self._matrix is None:
            self._matrix = np.stack(self._rows).astype(np.float32) if self._rows else np.zeros((0, self.dim), np.float32)
        return self._matrix

    # ---------------------------------------------------------------- querying
    def query(self, r: Any, k: int = 16, *, family: Optional[str] = None) -> List[Dict[str, Any]]:
        """Top-``k`` records by cosine similarity to ``r`` (optionally restricted to one family)."""
        if not self.records or k <= 0:
            return []
        q = _unit(_to_numpy(r).reshape(-1))
        if q.shape[0] != self.dim:
            raise ValueError(f"query has dim {q.shape[0]}, memory expects {self.dim}")
        scores = self.matrix() @ q
        if family is not None:
            keep = np.array([rec["family"] == family for rec in self.records])
            scores = np.where(keep, scores, -np.inf)
        k = min(k, len(self.records))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        out = []
        for idx in top:
            if not np.isfinite(scores[idx]):
                continue
            rec = dict(self.records[int(idx)])
            rec["index"] = int(idx)
            rec["score"] = float(scores[idx])
            out.append(rec)
        return out

    def programs(self) -> List[str]:
        return [rec["program"] for rec in self.records]

    # ---------------------------------------------------------------- persistence
    @staticmethod
    def _base(path: Union[str, Path]) -> Path:
        """Strip a trailing ``.npz`` / ``.json`` only (dotted stems such as ``memory.v1`` are kept intact)."""
        p = Path(path)
        if p.suffix in (".npz", ".json"):
            p = p.with_suffix("")
        return p

    def save(self, path: Union[str, Path]) -> None:
        """Write ``<base>.npz`` (key ``latents``) and ``<base>.json`` (``{"dim", "records"}``)."""
        base = self._base(path)
        base.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(base) + ".npz", latents=self.matrix())
        Path(str(base) + ".json").write_text(json.dumps({"dim": self.dim, "records": self.records}),
                                             encoding="utf-8")
        log.info("saved %d memory records to %s.{npz,json}", len(self.records), base)

    def _load_into(self, path: Union[str, Path]) -> "TransformationMemory":
        base = self._base(path)
        with np.load(str(base) + ".npz") as z:
            latents = np.asarray(z["latents"], dtype=np.float32)
        meta = json.loads(Path(str(base) + ".json").read_text(encoding="utf-8"))
        self.dim = int(meta.get("dim", latents.shape[1] if latents.ndim == 2 else self.dim))
        self.records = list(meta["records"])
        self._rows = [latents[i] for i in range(latents.shape[0])]
        self._matrix = None
        if len(self._rows) != len(self.records):
            raise ValueError("latent / record count mismatch in memory files")
        return self

    load = _HybridLoad()
