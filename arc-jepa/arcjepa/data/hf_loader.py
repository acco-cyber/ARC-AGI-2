"""Loaders for the local JSONL mirror of the HF dataset ``koushikz1/arc-agi-2-jepa-episodes``.

Layout of the mirror (identical to the HF repository)::

    <root>/tasks/{train,val,test,eval_public}.jsonl
    <root>/episodes/{train,val,test,eval_public}.jsonl
    <root>/episodes_aug/{train,val,test}.jsonl
    <root>/arcgen_fresh/{train,val,test}.jsonl
    <root>/sdg_hard/train.jsonl
    <root>/counterfactual/{train,val,test,eval_public}.jsonl
    <root>/rule_programs/{train,val,test}.jsonl
    <root>/splits.json

The root is resolved from (in order) an explicit argument, the ``ARCJEPA_DATA`` environment variable, the
local Windows mirror and the Kaggle dataset mount. Only when ``ARCJEPA_HF=1`` is set *and* no local root is
found does the loader fall back to the ``datasets`` library.

The 120 public-evaluation tasks (``eval_public``) are a benchmark only. ``load_episodes`` raises
:class:`EvalPublicGuardError` (a subclass of ``AssertionError``) for that split unless the caller passes
``allow_eval_public=True`` explicitly; ``load_tasks`` returns them by default because it is the task
catalogue (1,120 rows on the dataset card), and :func:`training_task_ids` exposes the 1,000 training ids that
every training loader is restricted to.

The deterministic family-balanced 700 / 150 / 150 re-split of the 1,000 training tasks (seed 20260926) is
computed by :func:`resplit_700_150_150` and persisted by :func:`ensure_resplit` to
``<root>/splits_700_150_150.json`` and ``<package root>/data/splits_700_150_150.json``.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from arcjepa.core.types import Episode, Pair, Task, episode_from_json, task_from_json

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------------------------

HF_REPO_ID = "koushikz1/arc-agi-2-jepa-episodes"
DEFAULT_LOCAL_ROOT = r"E:\Claude code\arc2\dataset\hf_arc2_episodes"
KAGGLE_ROOT = "/kaggle/input/datasets/poby7722/arc-agi-2-jepa-episodes"
ENV_ROOT = "ARCJEPA_DATA"
ENV_USE_HF = "ARCJEPA_HF"

TASK_CONFIG = "tasks"
EPISODE_CONFIGS: Tuple[str, ...] = ("episodes", "episodes_aug", "arcgen_fresh", "sdg_hard")
TABLE_CONFIGS: Tuple[str, ...] = ("counterfactual", "rule_programs")
ALL_CONFIGS: Tuple[str, ...] = (TASK_CONFIG,) + EPISODE_CONFIGS + TABLE_CONFIGS

HF_SPLITS: Tuple[str, ...] = ("train", "val", "test", "eval_public")
EVAL_PUBLIC = "eval_public"
TRAINING_SOURCE_SPLIT = "training"  # ``source_split`` value of the 1,000 official training tasks

RESPLIT_SEED = 20260926
RESPLIT_SIZES: Dict[str, int] = {"train": 700, "val": 150, "holdout": 150}
RESPLIT_NAMES: Tuple[str, ...] = ("train", "val", "holdout")
SPLITS_FILENAME = "splits_700_150_150.json"
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DATA_DIR = PACKAGE_ROOT / "data"

# Card counts (docs README / stats.json of the mirror) used by the tests.
CARD_COUNTS: Dict[str, int] = {
    "tasks": 1120,
    "episodes": 6077,
    "episodes_aug": 43072,
    "arcgen_fresh": 10464,
    "sdg_hard": 1824,
    "counterfactual": 6077,
    "rule_programs": 891,
}


class EvalPublicGuardError(AssertionError):
    """Raised when a training loader is asked for the ``eval_public`` benchmark split."""


class DataRootNotFound(FileNotFoundError):
    """Raised when no local mirror of the dataset can be located."""


# --------------------------------------------------------------------------------------------------------------
# Root resolution and JSONL reading
# --------------------------------------------------------------------------------------------------------------


def _looks_like_root(path: str) -> bool:
    return os.path.isfile(os.path.join(path, TASK_CONFIG, "train.jsonl"))


def candidate_roots(root: Optional[str] = None) -> List[str]:
    """Return the ordered list of directories probed for the mirror (explicit, env, local, Kaggle)."""
    cands: List[str] = []
    if root:
        cands.append(str(root))
    env = os.environ.get(ENV_ROOT)
    if env:
        cands.append(env)
    cands.append(DEFAULT_LOCAL_ROOT)
    cands.append(KAGGLE_ROOT)
    # Kaggle datasets are sometimes wrapped in one extra folder level.
    for base in (KAGGLE_ROOT, "/kaggle/input"):
        for nested in sorted(glob.glob(os.path.join(base, "*", TASK_CONFIG, "train.jsonl"))):
            cands.append(os.path.dirname(os.path.dirname(nested)))
    seen: set = set()
    out: List[str] = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def resolve_root(root: Optional[str] = None) -> str:
    """Return the first existing mirror root; raise :class:`DataRootNotFound` when none is available."""
    for cand in candidate_roots(root):
        if _looks_like_root(cand):
            return cand
    raise DataRootNotFound(
        "No local mirror of %s found. Probed: %s (set %s to the folder that contains tasks/train.jsonl)"
        % (HF_REPO_ID, candidate_roots(root), ENV_ROOT)
    )


def has_local_root(root: Optional[str] = None) -> bool:
    """True when :func:`resolve_root` would succeed."""
    try:
        resolve_root(root)
        return True
    except DataRootNotFound:
        return False


def jsonl_path(root: str, config: str, split: str) -> str:
    """Path of ``<root>/<config>/<split>.jsonl`` (not checked for existence)."""
    return os.path.join(root, config, f"{split}.jsonl")


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Yield one decoded JSON object per non-empty line of ``path``."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def available_splits(root: str, config: str) -> List[str]:
    """Splits present on disk for ``config`` in card order."""
    return [s for s in HF_SPLITS if os.path.isfile(jsonl_path(root, config, s))]


def _rows_from_hf(config: str, split: str) -> List[Dict[str, Any]]:
    """Fetch rows through the ``datasets`` library (only when ``ARCJEPA_HF=1``)."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise DataRootNotFound("ARCJEPA_HF=1 but the `datasets` library is not installed") from exc
    ds = load_dataset(HF_REPO_ID, config, split=split)
    return [dict(r) for r in ds]


def load_rows(root: Optional[str], config: str, split: str) -> List[Dict[str, Any]]:
    """Load the raw rows of one ``config``/``split`` (local mirror first, HF only when ``ARCJEPA_HF=1``)."""
    if config not in ALL_CONFIGS:
        raise ValueError(f"unknown config {config!r}; expected one of {ALL_CONFIGS}")
    try:
        r = resolve_root(root)
    except DataRootNotFound:
        if os.environ.get(ENV_USE_HF) == "1":
            logger.info("no local mirror; loading %s/%s from the HF hub", config, split)
            return _rows_from_hf(config, split)
        raise
    path = jsonl_path(r, config, split)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{config}/{split} is not part of the mirror at {r}")
    rows = list(iter_jsonl(path))
    logger.debug("loaded %d rows from %s", len(rows), path)
    return rows


# --------------------------------------------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------------------------------------------


def load_task_rows(root: Optional[str] = None, splits: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Raw ``tasks`` rows (with the per-task statistics of the card) keyed by task id.

    ``splits`` defaults to every split present, including ``eval_public`` (the catalogue has 1,120 rows).
    """
    r = None
    try:
        r = resolve_root(root)
    except DataRootNotFound:
        if os.environ.get(ENV_USE_HF) != "1":
            raise
    if splits is None:
        splits = available_splits(r, TASK_CONFIG) if r is not None else list(HF_SPLITS)
    out: Dict[str, Dict[str, Any]] = {}
    for split in splits:
        for row in load_rows(root, TASK_CONFIG, split):
            tid = str(row["task_id"])
            if tid in out:
                raise ValueError(f"duplicate task id {tid} across task splits")
            out[tid] = row
    return out


def load_tasks(root: Optional[str] = None, splits: Optional[Sequence[str]] = None) -> Dict[str, Task]:
    """Load the official tasks (1,000 training + 120 public evaluation) as :class:`Task` objects.

    Test pairs keep their outputs (the mirror stores them). Pass ``splits`` to restrict, e.g.
    ``splits=("train", "val", "test")`` for the 1,000 training tasks, or use :func:`load_training_tasks`.
    """
    rows = load_task_rows(root, splits)
    return {tid: task_from_json(tid, row) for tid, row in rows.items()}


def task_splits(root: Optional[str] = None) -> Dict[str, str]:
    """Map task id -> HF split label (``train``/``val``/``test``/``eval_public``)."""
    return {tid: str(row["split"]) for tid, row in load_task_rows(root).items()}


def training_task_ids(root: Optional[str] = None) -> List[str]:
    """Sorted ids of the 1,000 official training tasks (``source_split == "training"``)."""
    rows = load_task_rows(root)
    ids = sorted(tid for tid, row in rows.items() if row.get("source_split", "") == TRAINING_SOURCE_SPLIT)
    if not ids:  # defensive: fall back to "everything that is not eval_public"
        ids = sorted(tid for tid, row in rows.items() if row.get("split") != EVAL_PUBLIC)
    return ids


def load_training_tasks(root: Optional[str] = None) -> Dict[str, Task]:
    """The 1,000 training tasks only; asserts that no ``eval_public`` task leaks through."""
    rows = load_task_rows(root)
    out: Dict[str, Task] = {}
    for tid, row in rows.items():
        if row.get("split") == EVAL_PUBLIC:
            continue
        out[tid] = task_from_json(tid, row)
    for tid, row in rows.items():
        if tid in out and row.get("split") == EVAL_PUBLIC:
            raise EvalPublicGuardError(f"eval_public task {tid} in a training loader")
    return out


def load_eval_public_tasks(root: Optional[str] = None) -> Dict[str, Task]:
    """The 120 public evaluation tasks (final benchmark only; never for training or tuning)."""
    return load_tasks(root, splits=(EVAL_PUBLIC,))


# --------------------------------------------------------------------------------------------------------------
# Episodes and auxiliary tables
# --------------------------------------------------------------------------------------------------------------


def load_episodes(
    root: Optional[str] = None,
    config: str = "episodes",
    split: str = "train",
    *,
    allow_eval_public: bool = False,
) -> List[Episode]:
    """Load ``<root>/<config>/<split>.jsonl`` as :class:`Episode` objects.

    ``config`` must be one of :data:`EPISODE_CONFIGS`. The ``eval_public`` split is guarded: requesting it
    without ``allow_eval_public=True`` raises :class:`EvalPublicGuardError` so that no training code path
    can consume the benchmark by accident. Row fields other than the grids are kept in ``Episode.meta``.
    """
    if config not in EPISODE_CONFIGS:
        raise ValueError(f"config {config!r} is not an episode config; expected one of {EPISODE_CONFIGS}")
    if split == EVAL_PUBLIC and not allow_eval_public:
        raise EvalPublicGuardError(
            "eval_public is the locked benchmark split and is never returned by a training loader; "
            "pass allow_eval_public=True from evaluation code only"
        )
    episodes = [episode_from_json(row) for row in load_rows(root, config, split)]
    if not allow_eval_public:
        for ep in episodes:
            if ep.split == EVAL_PUBLIC:
                raise EvalPublicGuardError(f"episode {ep.episode_id} carries split=eval_public")
    return episodes


def load_counterfactuals(root: Optional[str] = None, split: str = "train", *, allow_eval_public: bool = False) -> List[Dict[str, Any]]:
    """Rows of the ``counterfactual`` config (``episode_id, task_id, split, target_shape, negatives``)."""
    if split == EVAL_PUBLIC and not allow_eval_public:
        raise EvalPublicGuardError("eval_public counterfactuals are benchmark-only")
    return load_rows(root, "counterfactual", split)


def load_rule_programs(root: Optional[str] = None, split: str = "train") -> Dict[str, Dict[str, Any]]:
    """ARC-GEN generator sources keyed by task id (``task_id, split, language, source, license, ...``)."""
    return {str(r["task_id"]): r for r in load_rows(root, "rule_programs", split)}


def episodes_from_task(task: Task, *, kind: str = "canonical") -> List[Episode]:
    """One canonical episode per test input of ``task`` (context = its train pairs)."""
    eps: List[Episode] = []
    for i, tp in enumerate(task.test):
        eps.append(
            Episode(
                episode_id=f"{task.task_id}_{kind}_{i}",
                task_id=task.task_id,
                split="",
                context=list(task.train),
                test_input=tp.input,
                target_output=tp.output if tp.output else None,
                source="arc",
                meta={"kind": kind, "target_index": len(task.train) + i, "n_context": len(task.train)},
            )
        )
    return eps


def read_hf_splits(root: Optional[str] = None) -> Dict[str, Any]:
    """The dataset's own ``splits.json`` (800/100/100 + eval_public id lists)."""
    with open(os.path.join(resolve_root(root), "splits.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------------------------------------------
# Deterministic family-balanced 700 / 150 / 150 re-split
# --------------------------------------------------------------------------------------------------------------


def _largest_remainder(quotas: Mapping[str, float], total: int, capacity: Mapping[str, int]) -> Dict[str, int]:
    """Integer allocation of ``total`` items to keys proportional to ``quotas`` (Hamilton method).

    Each key receives at most ``capacity[key]``. Ties on fractional parts are broken by key name so the result
    is deterministic.
    """
    alloc = {k: min(int(q), capacity[k]) for k, q in quotas.items()}
    remaining = total - sum(alloc.values())
    order = sorted(quotas.keys(), key=lambda k: (-(quotas[k] - int(quotas[k])), k))
    while remaining > 0:
        progressed = False
        for k in order:
            if remaining == 0:
                break
            if alloc[k] < capacity[k]:
                alloc[k] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError("insufficient capacity for the requested allocation")
    return alloc


def resplit_700_150_150(
    task_ids: Sequence[str],
    families: Mapping[str, str],
    seed: int = RESPLIT_SEED,
    sizes: Optional[Mapping[str, int]] = None,
) -> Dict[str, List[str]]:
    """Family-balanced, deterministic task-level split ``{"train": [...], "val": [...], "holdout": [...]}``.

    For 1,000 ids the sizes are 700 / 150 / 150 (spec). For another number ``N`` the val and holdout sizes
    are ``round(0.15 N)`` each and train takes the rest, unless ``sizes`` overrides them. Within every family
    the val and holdout quotas are the family's proportional share, rounded with the largest-remainder
    method, so the family mix of each split matches the pool as closely as integers allow. Ids are shuffled
    per family with ``random.Random(seed)`` after sorting, so the output is independent of input order.
    ``families`` maps task id -> family name (missing ids get family ``"unknown"``).
    """
    ids = sorted(set(str(t) for t in task_ids))
    n = len(ids)
    if sizes is None:
        if n == 1000:
            sizes = dict(RESPLIT_SIZES)
        else:
            n_val = int(round(0.15 * n))
            n_hold = int(round(0.15 * n))
            sizes = {"train": n - n_val - n_hold, "val": n_val, "holdout": n_hold}
    if sum(sizes.values()) != n:
        raise ValueError(f"split sizes {dict(sizes)} do not sum to the number of task ids {n}")

    by_family: Dict[str, List[str]] = {}
    for tid in ids:
        by_family.setdefault(str(families.get(tid, "unknown")), []).append(tid)
    rng = random.Random(seed)
    for fam in sorted(by_family):
        rng.shuffle(by_family[fam])

    counts = {fam: len(v) for fam, v in by_family.items()}
    val_alloc = _largest_remainder({f: counts[f] * sizes["val"] / n for f in counts}, sizes["val"], counts)
    hold_cap = {f: counts[f] - val_alloc[f] for f in counts}
    hold_alloc = _largest_remainder({f: counts[f] * sizes["holdout"] / n for f in counts}, sizes["holdout"], hold_cap)

    out: Dict[str, List[str]] = {name: [] for name in RESPLIT_NAMES}
    for fam in sorted(by_family):
        pool = by_family[fam]
        nv, nh = val_alloc[fam], hold_alloc[fam]
        out["val"].extend(pool[:nv])
        out["holdout"].extend(pool[nv:nv + nh])
        out["train"].extend(pool[nv + nh:])
    for name in RESPLIT_NAMES:
        out[name].sort()
    assert [len(out[k]) for k in RESPLIT_NAMES] == [sizes[k] for k in RESPLIT_NAMES]
    return out


def splits_payload(
    splits: Mapping[str, Sequence[str]],
    families: Mapping[str, str],
    seed: int = RESPLIT_SEED,
) -> Dict[str, Any]:
    """The JSON document written to ``splits_700_150_150.json`` (id lists + family bookkeeping)."""
    fam_counts: Dict[str, Dict[str, int]] = {}
    for name in RESPLIT_NAMES:
        c: Dict[str, int] = {}
        for tid in splits[name]:
            f = families.get(tid, "unknown")
            c[f] = c.get(f, 0) + 1
        fam_counts[name] = dict(sorted(c.items()))
    return {
        "seed": seed,
        "rule": "task-level; the 1000 official training tasks -> 700 train / 150 val (model selection) / "
                "150 locked holdout, family-balanced (largest remainder per family); 120 public evaluation "
                "tasks are never included",
        "sizes": {name: len(splits[name]) for name in RESPLIT_NAMES},
        "family_counts": fam_counts,
        "families": {tid: families.get(tid, "unknown") for name in RESPLIT_NAMES for tid in splits[name]},
        **{name: list(splits[name]) for name in RESPLIT_NAMES},
    }


def write_splits(payload: Mapping[str, Any], path: str) -> str:
    """Write the split document to ``path`` (parent directories are created); returns ``path``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
        fh.write("\n")
    return path


def read_splits(path: str) -> Dict[str, List[str]]:
    """Read a split document and return only the id lists ``{"train", "val", "holdout"}``."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    out = {name: list(doc[name]) for name in RESPLIT_NAMES}
    if EVAL_PUBLIC in doc:
        raise EvalPublicGuardError("a training split file must not carry eval_public ids")
    return out


def ensure_resplit(
    root: Optional[str] = None,
    *,
    package_dir: Optional[str] = None,
    seed: int = RESPLIT_SEED,
    force: bool = False,
    families: Optional[Mapping[str, str]] = None,
) -> Dict[str, List[str]]:
    """Compute (or reuse) the 700/150/150 re-split and persist it under the data root and ``<pkg>/data``.

    If both files already exist and ``force`` is False the package copy (the committed source of record) is
    read back; otherwise the split is recomputed from :func:`training_task_ids` and
    :func:`arcjepa.data.families.family_of` (or the given ``families``), written to both locations and
    returned.
    """
    r = resolve_root(root)
    pkg_dir = Path(package_dir) if package_dir else PACKAGE_DATA_DIR
    root_path = os.path.join(r, SPLITS_FILENAME)
    pkg_path = str(pkg_dir / SPLITS_FILENAME)
    if not force and os.path.isfile(root_path) and os.path.isfile(pkg_path):
        splits = read_splits(pkg_path)
        if [len(splits[k]) for k in RESPLIT_NAMES] == [RESPLIT_SIZES[k] for k in RESPLIT_NAMES]:
            if read_splits(root_path) != splits:  # self-heal a stale root copy from the source of record
                logger.warning("%s disagrees with %s; rewriting it from the package copy", root_path, pkg_path)
                try:
                    with open(pkg_path, "r", encoding="utf-8") as fh:
                        write_splits(json.load(fh), root_path)
                except OSError as exc:
                    logger.warning("could not rewrite %s: %s", root_path, exc)
            return splits
    ids = training_task_ids(r)
    if families is None:
        from arcjepa.data.families import family_of

        tasks = load_training_tasks(r)
        families = {tid: family_of(tasks[tid]) for tid in ids}
    splits = resplit_700_150_150(ids, families, seed=seed)
    payload = splits_payload(splits, families, seed=seed)
    for path in (root_path, pkg_path):
        try:
            write_splits(payload, path)
            logger.info("wrote %s", path)
        except OSError as exc:  # read-only mounts (Kaggle input) are tolerated
            logger.warning("could not write %s: %s", path, exc)
    return splits


def load_resplit(root: Optional[str] = None) -> Dict[str, List[str]]:
    """Read ``splits_700_150_150.json`` from the package ``data/`` dir, else the data root, else compute it."""
    for path in (str(PACKAGE_DATA_DIR / SPLITS_FILENAME),):
        if os.path.isfile(path):
            return read_splits(path)
    try:
        r = resolve_root(root)
        p = os.path.join(r, SPLITS_FILENAME)
        if os.path.isfile(p):
            return read_splits(p)
    except DataRootNotFound:
        raise
    return ensure_resplit(root)


def filter_episodes_by_tasks(episodes: Iterable[Episode], task_ids: Iterable[str]) -> List[Episode]:
    """Keep the episodes whose ``task_id`` is in ``task_ids`` (used to apply the re-split to any config)."""
    keep = set(task_ids)
    return [ep for ep in episodes if ep.task_id in keep]


def _main() -> None:  # pragma: no cover - thin CLI
    import argparse

    ap = argparse.ArgumentParser(description="Compute and write the 700/150/150 family-balanced re-split.")
    ap.add_argument("--root", default=None)
    ap.add_argument("--seed", type=int, default=RESPLIT_SEED)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    splits = ensure_resplit(args.root, seed=args.seed, force=args.force)
    logger.info("sizes: %s", {k: len(v) for k, v in splits.items()})


if __name__ == "__main__":  # pragma: no cover
    _main()
