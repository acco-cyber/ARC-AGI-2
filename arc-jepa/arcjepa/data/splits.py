"""ARC-JEPA v2 task split: Train-670 / Val-150 / Hard-180 (frozen, sha256-pinned).

The v2 brief re-splits the 1,000 official training tasks into 670 train / 150 validation / 180 Hard-Holdout. The
split is derived deterministically from the v1 split ``data/splits_700_150_150.json`` (seed 20260926) and the task
JSON only (no solver runs):

* **Hard-180** = all 150 ids of the v1 ``holdout`` split (never trained on, never inspected by a design step)
  + the 30 hardest ids of the v1 ``train`` split by the pre-registered hardness score below.
* **Val-150** = the v1 ``val`` ids unchanged (the development set).
* **Train-670** = the v1 ``train`` ids minus the 30 moved to Hard-180.

Hardness (computed over the solver-visible part of a task: train pairs + test inputs; test outputs are not used)::

    h = 0.30 z(max grid area) + 0.15 z(#colours) + 0.15 [#test inputs > 1] + 0.15 [output shape not constant
        across the train pairs] + 0.10 [output shape differs from input shape in some train pair]
        + 0.15 z(-#train pairs)                                   (fewer demonstrations = harder)

``z`` standardises over the 700 v1 train ids (population standard deviation; ``z = 0`` when the deviation is 0).
The 30 ids with the largest ``h`` move (ties broken by task id, ``h`` rounded to 12 decimals first).

The frozen document ``data/splits_670_150_180.json`` carries the id lists, the hardness score and features of every
Hard-180 task, the method text and ``hard180_sha256`` = sha256 of ``"\\n".join(sorted(hard180))`` (UTF-8).
:func:`load_split` verifies that digest against :data:`HARD180_SHA256` on every load, so the Hard-180 ids cannot
change silently. Nothing here reads the public-evaluation tasks: tasks are loaded from the HF ``train``/``val``/
``test`` files of the mirror only.

CLI::

    python -m arcjepa.data.splits --check          # rebuild from the v1 split + task JSON and compare to the file
    python -m arcjepa.data.splits --write [--force] # (re)write data/splits_670_150_180.json (refuses to overwrite)
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from arcjepa.core.types import Grid, Task

logger = logging.getLogger(__name__)

__all__ = [
    "HARD180_SEED", "HARD180_SHA256", "SPLIT_FILENAME", "DEFAULT_SPLIT_PATH", "OLD_SPLIT_FILENAME", "SPLIT_SIZES",
    "SPLIT_ALIASES", "HARDNESS_WEIGHTS", "N_FROM_OLD_TRAIN", "SplitIntegrityError", "hard180_sha256",
    "hardness_features", "hardness_scores", "build_split", "write_split", "load_split", "split_ids",
    "verify_split",
]

#: pre-registered seed of the v2 split (the construction itself is deterministic and uses no randomness)
HARD180_SEED = 20260927
SPLIT_FILENAME = "splits_670_150_180.json"
OLD_SPLIT_FILENAME = "splits_700_150_150.json"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT_PATH = PACKAGE_ROOT / "data" / SPLIT_FILENAME
OLD_SPLIT_PATH = PACKAGE_ROOT / "data" / OLD_SPLIT_FILENAME
SPLIT_SIZES: Dict[str, int] = {"train": 670, "val": 150, "hard180": 180}
N_FROM_OLD_TRAIN = 30
#: the HF task files that hold the 1,000 official training tasks (``eval_public`` is never opened here)
TRAINING_HF_SPLITS: Tuple[str, ...] = ("train", "val", "test")

#: sha256 of "\n".join(sorted(hard180 ids)) -- pinned; also pinned by tests/test_hard180.py
HARD180_SHA256 = "109f247cbc0ccf729e1a66d0747bcb2fde5e68c3715a90f76a271e94d447f9db"

HARDNESS_WEIGHTS: Dict[str, float] = {
    "z_max_area": 0.30,
    "z_n_colors": 0.15,
    "multi_test": 0.15,
    "out_shape_varies": 0.15,
    "shape_changes": 0.10,
    "z_neg_n_demos": 0.15,
}

#: split-name aliases accepted by :func:`split_ids`
SPLIT_ALIASES: Dict[str, str] = {
    "train": "train", "train670": "train", "train_670": "train",
    "val": "val", "val150": "val", "val_150": "val",
    "hard180": "hard180", "h180": "hard180", "hard_180": "hard180",
    "hard180_clean150": "hard180_clean150", "clean150": "hard180_clean150",
    "hard180_from_old_train": "hard180_from_old_train", "from_old_train": "hard180_from_old_train",
}

METHOD_TEXT = (
    "Pre-registered 2026-09-27 (seed 20260927; the construction is deterministic). Source: the v1 split "
    "data/splits_700_150_150.json (1,000 official ARC-AGI-2 training tasks; seed 20260926). Hard-180 = all 150 "
    "v1 'holdout' ids + the 30 v1 'train' ids with the largest hardness h, ties broken by task id (h rounded to "
    "12 decimals). Val-150 = the v1 'val' ids unchanged. Train-670 = v1 'train' minus the 30. Hardness is "
    "computed ONLY from the task JSON (no solver runs), over the solver-visible grids (train inputs, train "
    "outputs, test inputs; test outputs unused): h = 0.30*z(max grid area) + 0.15*z(#distinct colours) + "
    "0.15*[#test inputs > 1] + 0.15*[train output shapes not all equal] + 0.10*[some train pair has output "
    "shape != input shape] + 0.15*z(-#train pairs). z standardises over the 700 v1 train ids (population std; "
    "z = 0 when std = 0). hard180_sha256 = sha256 of '\\n'.join(sorted(hard180)) in UTF-8. The 120 public "
    "evaluation tasks are not part of any list and were not read."
)


class SplitIntegrityError(AssertionError):
    """Raised when a split document fails its integrity checks (sha256, sizes, disjointness)."""


# ============================================================================================ hashing / features

def hard180_sha256(ids: Sequence[str]) -> str:
    """sha256 hex digest of ``"\\n".join(sorted(ids))`` (UTF-8): the pinned identity of the Hard-180 id list."""
    return hashlib.sha256("\n".join(sorted(str(t) for t in ids)).encode("utf-8")).hexdigest()


_list_sha256 = hard180_sha256  # same digest rule for the recorded v1 id lists


def _visible_grids(task: Task) -> List[Grid]:
    grids: List[Grid] = []
    for p in task.train:
        grids.append(p.input)
        grids.append(p.output)
    for p in task.test:
        grids.append(p.input)
    return [g for g in grids if g and g[0]]


def _shape(g: Grid) -> Tuple[int, int]:
    return (len(g), len(g[0]) if g else 0)


def hardness_features(task: Task) -> Dict[str, Any]:
    """Raw hardness features of ``task`` from its JSON only (train pairs + test inputs, never test outputs).

    ``max_area`` (largest H*W), ``n_colors`` (distinct colours, 0 included), ``n_test`` (test inputs),
    ``out_shape_varies`` (train output shapes not all equal), ``shape_changes`` (some train pair changes shape),
    ``n_demos`` (train pairs).
    """
    grids = _visible_grids(task)
    max_area = max((len(g) * len(g[0]) for g in grids), default=0)
    colors = {int(v) for g in grids for row in g for v in row}
    out_shapes = {_shape(p.output) for p in task.train}
    shape_changes = any(_shape(p.input) != _shape(p.output) for p in task.train)
    return {
        "max_area": int(max_area),
        "n_colors": int(len(colors)),
        "n_test": int(len(task.test)),
        "out_shape_varies": bool(len(out_shapes) > 1),
        "shape_changes": bool(shape_changes),
        "n_demos": int(len(task.train)),
    }


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var)


def _z(v: float, mean: float, std: float) -> float:
    return 0.0 if std <= 0.0 else (v - mean) / std


def hardness_scores(features: Mapping[str, Mapping[str, Any]], reference_ids: Sequence[str]
                    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """``(h per id, reference statistics)`` for every id in ``features``.

    The z-scores standardise over ``reference_ids`` (the 700 v1 train ids): ``stats = {feature: {mean, std}}``
    for ``max_area``, ``n_colors`` and ``neg_n_demos``.
    """
    ref = [features[t] for t in reference_ids]
    stats: Dict[str, Dict[str, float]] = {}
    for key, fn in (("max_area", lambda f: float(f["max_area"])), ("n_colors", lambda f: float(f["n_colors"])),
                    ("neg_n_demos", lambda f: -float(f["n_demos"]))):
        m, s = _mean_std([fn(f) for f in ref])
        stats[key] = {"mean": m, "std": s}
    w = HARDNESS_WEIGHTS
    out: Dict[str, float] = {}
    for tid, f in features.items():
        h = (w["z_max_area"] * _z(float(f["max_area"]), **stats["max_area"])
             + w["z_n_colors"] * _z(float(f["n_colors"]), **stats["n_colors"])
             + w["multi_test"] * float(int(f["n_test"]) > 1)
             + w["out_shape_varies"] * float(bool(f["out_shape_varies"]))
             + w["shape_changes"] * float(bool(f["shape_changes"]))
             + w["z_neg_n_demos"] * _z(-float(f["n_demos"]), **stats["neg_n_demos"]))
        out[tid] = round(h, 12)
    return out, stats


# ============================================================================================ build

def _read_json(path: Union[str, os.PathLike]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_split(root: Optional[str] = None, old_split_path: Optional[Union[str, os.PathLike]] = None,
                tasks: Optional[Mapping[str, Task]] = None) -> Dict[str, Any]:
    """Construct the v2 split document from the v1 split file and the task JSON (deterministic).

    ``tasks`` may be given (id -> Task for at least the 850 v1 train + holdout ids); otherwise the tasks are
    read from the ``train``/``val``/``test`` files of the mirror at ``root`` (never ``eval_public``).
    """
    old_path = Path(old_split_path) if old_split_path else OLD_SPLIT_PATH
    old = _read_json(old_path)
    old_train = sorted(str(t) for t in old["train"])
    old_val = sorted(str(t) for t in old["val"])
    old_hold = sorted(str(t) for t in old["holdout"])
    if (len(old_train), len(old_val), len(old_hold)) != (700, 150, 150):
        raise SplitIntegrityError(f"{old_path} is not the 700/150/150 v1 split")
    if tasks is None:
        from arcjepa.data.hf_loader import load_tasks

        tasks = load_tasks(root, splits=TRAINING_HF_SPLITS)
    missing = [t for t in old_train + old_hold if t not in tasks]
    if missing:
        raise SplitIntegrityError(f"{len(missing)} split ids are missing from the task files, e.g. {missing[:5]}")

    feats = {t: hardness_features(tasks[t]) for t in old_train + old_hold}
    scores, stats = hardness_scores(feats, old_train)
    ranked = sorted(old_train, key=lambda t: (-scores[t], t))
    moved = sorted(ranked[:N_FROM_OLD_TRAIN])
    moved_set = set(moved)
    train = [t for t in old_train if t not in moved_set]
    hard = sorted(old_hold + moved)
    old_fams = {str(k): str(v) for k, v in (old.get("families") or {}).items()}

    def fam_counts(ids: Sequence[str]) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for t in ids:
            f = old_fams.get(t, "unknown")
            c[f] = c.get(f, 0) + 1
        return dict(sorted(c.items()))

    doc: Dict[str, Any] = {
        "name": "splits_670_150_180",
        "version": 1,
        "seed": HARD180_SEED,
        "created": "2026-09-27",
        "method": METHOD_TEXT,
        "source_split_file": f"data/{OLD_SPLIT_FILENAME}",
        "source_split_seed": old.get("seed"),
        "source_split_sha256": {"train": _list_sha256(old_train), "val": _list_sha256(old_val),
                                "holdout": _list_sha256(old_hold)},
        "hardness_weights": dict(HARDNESS_WEIGHTS),
        "hardness_reference": {"population": "v1 train (700 ids)", **stats},
        "hardness_cutoff": {"rank30_score": scores[ranked[N_FROM_OLD_TRAIN - 1]],
                            "rank31_score": scores[ranked[N_FROM_OLD_TRAIN]]},
        "sizes": {"train": len(train), "val": len(old_val), "hard180": len(hard),
                  "hard180_clean150": len(old_hold), "hard180_from_old_train": len(moved)},
        "hard180_sha256": hard180_sha256(hard),
        "sha256_rule": "sha256 of '\\n'.join(sorted(hard180)) encoded as UTF-8",
        "family_counts": {"train": fam_counts(train), "val": fam_counts(old_val), "hard180": fam_counts(hard)},
        "train": train,
        "val": old_val,
        "hard180": hard,
        "hard180_clean150": old_hold,
        "hard180_from_old_train": moved,
        "hard180_scores": {t: scores[t] for t in hard},
        "hard180_features": {t: feats[t] for t in hard},
        "families": {t: old_fams.get(t, "unknown") for t in sorted(train + old_val + hard)},
    }
    verify_split(doc, pinned=None)
    return doc


def write_split(doc: Mapping[str, Any], path: Optional[Union[str, os.PathLike]] = None, *,
                force: bool = False) -> str:
    """Write the split document (LF line endings, indent 1). Refuses to overwrite unless ``force``."""
    p = Path(path) if path else DEFAULT_SPLIT_PATH
    if p.exists() and not force:
        raise FileExistsError(f"{p} exists; the Hard-180 split is frozen (pass force=True to rewrite)")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    os.replace(tmp, p)
    return str(p)


# ============================================================================================ load / verify

def verify_split(doc: Mapping[str, Any], pinned: Optional[str] = HARD180_SHA256) -> None:
    """Integrity checks: sizes 670/150/180, pairwise disjoint, Hard-180 = clean150 + from_old_train (150 + 30),
    the recorded digest equals the recomputed one and, when ``pinned`` is given, equals ``pinned``."""
    try:
        train, val, hard = list(doc["train"]), list(doc["val"]), list(doc["hard180"])
        clean, moved = list(doc["hard180_clean150"]), list(doc["hard180_from_old_train"])
    except KeyError as exc:
        raise SplitIntegrityError(f"split document lacks {exc}") from exc
    sizes = (len(train), len(val), len(hard))
    if sizes != (SPLIT_SIZES["train"], SPLIT_SIZES["val"], SPLIT_SIZES["hard180"]):
        raise SplitIntegrityError(f"split sizes {sizes} != (670, 150, 180)")
    for name, ids in (("train", train), ("val", val), ("hard180", hard)):
        if len(set(ids)) != len(ids):
            raise SplitIntegrityError(f"duplicate ids in {name}")
    if set(train) & set(val) or set(train) & set(hard) or set(val) & set(hard):
        raise SplitIntegrityError("train / val / hard180 overlap")
    if len(clean) != 150 or len(moved) != N_FROM_OLD_TRAIN or set(clean) | set(moved) != set(hard) \
            or set(clean) & set(moved):
        raise SplitIntegrityError("hard180 != hard180_clean150 (150) + hard180_from_old_train (30)")
    digest = hard180_sha256(hard)
    if doc.get("hard180_sha256") != digest:
        raise SplitIntegrityError(f"recorded hard180_sha256 {doc.get('hard180_sha256')} != recomputed {digest}")
    if pinned is not None and digest != pinned:
        raise SplitIntegrityError(f"Hard-180 ids changed: sha256 {digest} != pinned {pinned}")


def load_split(path: Optional[Union[str, os.PathLike]] = None, *, verify: bool = True) -> Dict[str, Any]:
    """Read the v2 split document (default ``data/splits_670_150_180.json``) and return it as a dict.

    Keys: ``train`` (670), ``val`` (150), ``hard180`` (180), ``hard180_clean150``, ``hard180_from_old_train``,
    ``hard180_scores``, ``hard180_features``, ``families``, ``method``, ``hard180_sha256``, ... The integrity
    checks of :func:`verify_split` always run; ``verify=True`` (default) also asserts the digest equals the pinned
    :data:`HARD180_SHA256`.
    """
    p = Path(path) if path else DEFAULT_SPLIT_PATH
    doc = _read_json(p)
    if "eval_public" in doc:
        raise SplitIntegrityError("a split document must not carry eval_public ids")
    verify_split(doc, pinned=HARD180_SHA256 if verify else None)
    return doc


def split_ids(name: str, path: Optional[Union[str, os.PathLike]] = None) -> List[str]:
    """Sorted ids of one split: ``train``/``train670``, ``val``/``val150``, ``hard180``/``h180``,
    ``hard180_clean150``/``clean150`` or ``hard180_from_old_train``/``from_old_train``."""
    key = SPLIT_ALIASES.get(str(name).lower())
    if key is None:
        raise KeyError(f"unknown split {name!r}; expected one of {sorted(SPLIT_ALIASES)}")
    return sorted(load_split(path)[key])


# ============================================================================================ CLI

def _main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - thin CLI
    import argparse

    ap = argparse.ArgumentParser(description="Build / check the frozen Train-670 / Val-150 / Hard-180 split.")
    ap.add_argument("--root", default=None, help="data mirror root (default: resolve_root)")
    ap.add_argument("--out", default=str(DEFAULT_SPLIT_PATH))
    ap.add_argument("--write", action="store_true", help="write the split document")
    ap.add_argument("--force", action="store_true", help="overwrite an existing document")
    ap.add_argument("--check", action="store_true", help="rebuild and compare with the existing document")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    doc = build_split(args.root)
    logger.info("hard180_sha256 %s (pinned %s)", doc["hard180_sha256"], HARD180_SHA256)
    logger.info("moved from v1 train: %s", " ".join(doc["hard180_from_old_train"]))
    if args.check:
        cur = _read_json(args.out)
        same = all(cur.get(k) == doc[k] for k in ("train", "val", "hard180", "hard180_from_old_train"))
        logger.info("rebuild matches %s: %s", args.out, same)
        return 0 if same else 1
    if args.write:
        logger.info("wrote %s", write_split(doc, args.out, force=args.force))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
