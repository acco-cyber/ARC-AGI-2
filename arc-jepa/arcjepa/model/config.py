"""Model configuration for ARC-JEPA (dims from ``docs/FROZEN_SPEC.md`` §Model and the v1 yaml).

``ModelConfig.v1()`` is the spec's v1 yaml (cell 12x256/8/1024, object 4x256, relation 4x256, predictor 6x512/8,
program 6x256, z 512, r 256). The v1 yaml has a single ``ffn_dim: 1024`` and the spec budgets the model at
~15-30 M (INTERFACES: 10-40 M), so the predictor FFN is 1024 in ``v1()`` (~39.5 M parameters).
``ModelConfig.v1_wide()`` uses the predictor FFN 2048 of the spec's §Model prose (~45.8 M, over budget).
``ModelConfig.tiny()`` is the CPU test preset (d 64, 2 layers everywhere).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Tuple, Union

from arcjepa.core.types import MAX_SIDE, N_COLORS, PAD_ID


@dataclass
class ModelConfig:
    """All architectural hyper-parameters of the ARC-JEPA model.

    Widths: ``cell_dim``/``object_dim``/``relation_dim`` are the three encoder widths, ``global_dim`` the fused
    global summary, ``jepa_dim`` the grid latent ``z_X``, ``rule_dim`` the task rule latent ``r_task``,
    ``program_dim`` the program latent ``z_p`` and ``predictor_dim`` the predictor width.
    """

    name: str = "v1"
    # widths
    cell_dim: int = 256
    object_dim: int = 256
    relation_dim: int = 256
    global_dim: int = 384
    jepa_dim: int = 512
    rule_dim: int = 256
    program_dim: int = 256
    predictor_dim: int = 512
    # depths
    cell_layers: int = 12
    object_layers: int = 4
    relation_layers: int = 4
    predictor_layers: int = 6
    program_layers: int = 6
    # attention / FFN
    heads: int = 8
    ffn_dim: int = 1024
    predictor_ffn_dim: int = 1024  # v1 yaml ffn_dim; the §Model prose's 2048 is ``v1_wide()``
    dropout: float = 0.0
    # cell embedding: colour + row + column + neighbour stats, each ``cell_part_dim`` wide (4 x 32 = 128 -> 256)
    cell_part_dim: int = 32
    max_side: int = MAX_SIDE
    n_colors: int = N_COLORS
    pad_id: int = PAD_ID
    # objects
    max_objects: int = 64
    obj_feat_dim: int = 32
    shape_conv_dim: int = 64
    shape_dim: int = 128
    # relations
    rel_feat_dim: int = 24
    edge_dim: int = 64
    max_pairs: int = 512
    # hierarchical fusion [cell; object; global] -> fusion_hidden -> jepa_dim
    fusion_hidden: int = 1024
    # rule latent MLP [z_x; z_y; z_y - z_x] (3 * jepa_dim) -> rule_hidden -> rule_dim
    rule_hidden: Tuple[int, int] = (1024, 512)
    # scorer MLP [r; z_p; r * z_p] (3 * rule_dim) -> scorer_hidden -> 1
    scorer_hidden: Tuple[int, int] = (512, 256)
    # programs: token-stream cap, depth vocabulary, per-node embedding [primitive; type; depth] -> program_dim
    max_program_len: int = 192
    program_depth_tokens: int = 16
    program_sym_dim: int = 128
    program_aux_dim: int = 64
    # episodes
    max_ctx: int = 10
    # EMA target encoder
    ema_start: float = 0.996
    ema_end: float = 0.9995
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rule_hidden = tuple(int(x) for x in self.rule_hidden)  # type: ignore[assignment]
        self.scorer_hidden = tuple(int(x) for x in self.scorer_hidden)  # type: ignore[assignment]
        self.validate()

    # ------------------------------------------------------------------ derived
    @property
    def cell_embed_dim(self) -> int:
        """Width of the concatenated cell embedding before projection (4 parts)."""
        return 4 * self.cell_part_dim

    @property
    def n_grid_tokens(self) -> int:
        return self.max_side * self.max_side

    def validate(self) -> None:
        """Raise ``ValueError`` on inconsistent settings (head divisibility, positive sizes)."""
        for name in ("cell_dim", "object_dim", "relation_dim", "program_dim", "predictor_dim"):
            d = getattr(self, name)
            if d % self.heads != 0:
                raise ValueError(f"{name}={d} is not divisible by heads={self.heads}")
        for f in fields(self):
            if f.type in ("int", int) and f.name not in ("pad_id",) and getattr(self, f.name) <= 0:
                raise ValueError(f"{f.name} must be positive")
        if not (0.0 < self.ema_start <= self.ema_end < 1.0):
            raise ValueError("need 0 < ema_start <= ema_end < 1")

    # ------------------------------------------------------------------ presets
    @classmethod
    def v1(cls) -> "ModelConfig":
        """The spec's v1 yaml dims (cell 12x256, object 4x256, relation 4x256, predictor 6x512/8/1024, ...)."""
        return cls(name="v1")

    @classmethod
    def v1_wide(cls) -> "ModelConfig":
        """``v1`` with the §Model prose's predictor FFN 2048 (about 45.8 M parameters, above the spec budget)."""
        return cls(name="v1_wide", predictor_ffn_dim=2048)

    @classmethod
    def tiny(cls) -> "ModelConfig":
        """CPU test preset: d 64, 2 layers everywhere, 4 heads."""
        return cls(
            name="tiny",
            cell_dim=64, object_dim=64, relation_dim=64, global_dim=96, jepa_dim=64, rule_dim=64,
            program_dim=64, predictor_dim=64,
            cell_layers=2, object_layers=2, relation_layers=2, predictor_layers=2, program_layers=2,
            heads=4, ffn_dim=128, predictor_ffn_dim=128,
            cell_part_dim=8, shape_conv_dim=16, shape_dim=32, edge_dim=16, max_pairs=128,
            fusion_hidden=128, rule_hidden=(128, 64), scorer_hidden=(128, 64), max_program_len=96,
            program_sym_dim=32, program_aux_dim=16,
        )

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["rule_hidden"] = list(self.rule_hidden)
        d["scorer_hidden"] = list(self.scorer_hidden)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in known}
        unknown = {k: v for k, v in d.items() if k not in known}
        if unknown:
            kwargs.setdefault("extra", {}).update(unknown)
        return cls(**kwargs)

    def save_json(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "ModelConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
