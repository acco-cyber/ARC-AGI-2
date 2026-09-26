"""Tests for arcjepa.data (loader counts, re-split, families, tensorisation). CPU, < 60 s."""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arcjepa.core.types import PAD_ID, Episode, Pair, Task, validate_grid  # noqa: E402
from arcjepa.data import families as F  # noqa: E402
from arcjepa.data import hf_loader as H  # noqa: E402
from arcjepa.data import tensorize as T  # noqa: E402

HAVE_DATA = H.has_local_root()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="local HF mirror not available")


# --------------------------------------------------------------------------------------------------------------
# Fixtures (module scoped so the JSONL files are read once)
# --------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tasks():
    if not HAVE_DATA:
        pytest.skip("local HF mirror not available")
    return H.load_tasks()


@pytest.fixture(scope="module")
def training_ids():
    if not HAVE_DATA:
        pytest.skip("local HF mirror not available")
    return H.training_task_ids()


@pytest.fixture(scope="module")
def hf_split():
    if not HAVE_DATA:
        pytest.skip("local HF mirror not available")
    return H.task_splits()


@pytest.fixture(scope="module")
def families(tasks, training_ids):
    t0 = time.time()
    fam = {tid: F.family_of(tasks[tid]) for tid in training_ids}
    assert time.time() - t0 < 30.0
    return fam


# --------------------------------------------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------------------------------------------


@needs_data
def test_root_resolution(monkeypatch, tmp_path):
    root = H.resolve_root()
    assert os.path.isfile(os.path.join(root, "tasks", "train.jsonl"))
    # an explicit bogus root falls through to env / default probing
    assert H.resolve_root(str(tmp_path)) == root
    monkeypatch.setenv(H.ENV_ROOT, root)
    assert H.resolve_root() == root
    assert H.candidate_roots()[0] == root and H.KAGGLE_ROOT in H.candidate_roots()
    assert isinstance(H.DataRootNotFound("x"), FileNotFoundError)
    assert issubclass(H.EvalPublicGuardError, AssertionError)


@needs_data
def test_task_counts_match_card(tasks, training_ids, hf_split):
    assert len(tasks) == H.CARD_COUNTS["tasks"] == 1120
    assert len(training_ids) == 1000
    assert sum(1 for s in hf_split.values() if s == H.EVAL_PUBLIC) == 120
    assert set(training_ids).isdisjoint(t for t, s in hf_split.items() if s == H.EVAL_PUBLIC)
    for tid in ("00576224", training_ids[0], training_ids[-1]):
        t = tasks[tid]
        assert isinstance(t, Task) and t.train and t.test
        assert all(validate_grid(p.input) and validate_grid(p.output) for p in t.train)
        assert all(validate_grid(p.input) and validate_grid(p.output) for p in t.test)  # mirror keeps outputs
    hf = H.read_hf_splits()
    assert len(hf["train"]) == 800 and len(hf["val"]) == 100 and len(hf["test"]) == 100


@needs_data
def test_episode_counts_match_card():
    n = 0
    for split in ("train", "val", "test"):
        eps = H.load_episodes(None, "episodes", split)
        assert eps and all(isinstance(e, Episode) and e.split == split for e in eps)
        n += len(eps)
    ev = H.load_episodes(None, "episodes", H.EVAL_PUBLIC, allow_eval_public=True)
    assert len(ev) == 693
    n += len(ev)
    assert n == H.CARD_COUNTS["episodes"] == 6077
    ep = ev[0]
    assert ep.target_output is not None and validate_grid(ep.test_input)
    assert ep.meta["kind"] in ("canonical", "loo")


@needs_data
def test_eval_public_guard(hf_split):
    with pytest.raises(AssertionError):
        H.load_episodes(None, "episodes", H.EVAL_PUBLIC)
    with pytest.raises(H.EvalPublicGuardError):
        H.load_counterfactuals(None, H.EVAL_PUBLIC)
    with pytest.raises(ValueError):
        H.load_episodes(None, "counterfactual", "train")
    tr = H.load_training_tasks()
    assert len(tr) == 1000 and not any(hf_split[t] == H.EVAL_PUBLIC for t in tr)


@needs_data
def test_other_configs_load():
    sdg = H.load_episodes(None, "sdg_hard", "train")
    assert len(sdg) == H.CARD_COUNTS["sdg_hard"] and sdg[0].source == "sdg-verified"
    rp = H.load_rule_programs(None, "val")
    assert len(rp) == 90 and "source" in next(iter(rp.values()))
    cf = H.load_counterfactuals(None, "val")
    assert len(cf) == 532 and cf[0]["negatives"]


def test_episodes_from_task():
    g = [[1, 0], [0, 2]]
    t = Task("abc", [Pair(g, g), Pair(g, g)], [Pair(g, g), Pair(g, [])])
    eps = H.episodes_from_task(t)
    assert len(eps) == 2 and eps[0].target_output == g and eps[1].target_output is None
    assert eps[1].episode_id == "abc_canonical_1" and len(eps[0].context) == 2


# --------------------------------------------------------------------------------------------------------------
# Re-split
# --------------------------------------------------------------------------------------------------------------


@needs_data
def test_resplit_sizes_disjoint_balanced_deterministic(training_ids, families):
    sp = H.resplit_700_150_150(training_ids, families)
    assert [len(sp[k]) for k in H.RESPLIT_NAMES] == [700, 150, 150]
    sets = [set(sp[k]) for k in H.RESPLIT_NAMES]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert sets[0] | sets[1] | sets[2] == set(training_ids)
    # deterministic and order independent
    shuffled = list(training_ids)
    random.Random(1).shuffle(shuffled)
    assert H.resplit_700_150_150(shuffled, families) == sp
    assert H.resplit_700_150_150(training_ids, families, seed=1) != sp
    # family balance: every family's val / holdout count is within one of its proportional share
    counts = {}
    for tid in training_ids:
        counts[families[tid]] = counts.get(families[tid], 0) + 1
    for name in ("val", "holdout"):
        per_fam = {}
        for tid in sp[name]:
            per_fam[families[tid]] = per_fam.get(families[tid], 0) + 1
        for fam, n in counts.items():
            assert abs(per_fam.get(fam, 0) - 0.15 * n) < 1.0 + 1e-9, (name, fam)


def test_resplit_small_and_capacity_edge_cases():
    ids = [f"t{i:03d}" for i in range(20)]
    fams = {t: ("a" if i < 15 else "b" if i < 19 else "c") for i, t in enumerate(ids)}
    sp = H.resplit_700_150_150(ids, fams)
    assert sum(len(v) for v in sp.values()) == 20 and len(sp["val"]) == 3 and len(sp["holdout"]) == 3
    assert len(set(sp["train"]) | set(sp["val"]) | set(sp["holdout"])) == 20
    with pytest.raises(ValueError):
        H.resplit_700_150_150(ids, fams, sizes={"train": 10, "val": 5, "holdout": 4})


@needs_data
def test_split_files_written(training_ids, families, hf_split, tmp_path):
    root = H.resolve_root()
    sp = H.ensure_resplit(root, package_dir=str(tmp_path), force=True, families=families)
    p_pkg = tmp_path / H.SPLITS_FILENAME
    p_root = Path(root) / H.SPLITS_FILENAME
    assert p_pkg.is_file() and p_root.is_file()
    doc = json.loads(p_pkg.read_text(encoding="utf-8"))
    assert doc["seed"] == H.RESPLIT_SEED and doc["sizes"] == {"train": 700, "val": 150, "holdout": 150}
    assert H.read_splits(str(p_pkg)) == sp == H.read_splits(str(p_root))
    assert H.EVAL_PUBLIC not in doc and not any(hf_split[t] == H.EVAL_PUBLIC for t in sp["holdout"])
    # the committed package copy exists and agrees with the recomputation
    committed = H.PACKAGE_DATA_DIR / H.SPLITS_FILENAME
    assert committed.is_file()
    assert H.read_splits(str(committed)) == sp
    # cached read-back path
    assert H.ensure_resplit(root, package_dir=str(tmp_path)) == sp
    # a stale root copy is healed from the package copy (the source of record)
    stale = dict(doc)
    stale["train"], stale["val"] = list(sp["train"][:-1]) + [sp["val"][0]], [sp["train"][-1]] + list(sp["val"][1:])
    H.write_splits(stale, str(p_root))
    assert H.read_splits(str(p_root)) != sp
    assert H.ensure_resplit(root, package_dir=str(tmp_path)) == sp
    assert H.read_splits(str(p_root)) == sp


# --------------------------------------------------------------------------------------------------------------
# Families
# --------------------------------------------------------------------------------------------------------------


def _task(pairs):
    return Task("x", [Pair(a, b) for a, b in pairs], [])


def test_family_heuristics_on_synthetic_tasks():
    rng = np.random.default_rng(0)
    g1 = rng.integers(0, 4, size=(5, 7)).tolist()
    g2 = rng.integers(0, 4, size=(6, 4)).tolist()
    rot = lambda g: np.rot90(np.asarray(g), 1).tolist()
    assert F.family_of(_task([(g1, rot(g1)), (g2, rot(g2))])) == "geometry"
    tile = lambda g: np.tile(np.asarray(g), (2, 2)).tolist()
    assert F.family_of(_task([(g1, tile(g1)), (g2, tile(g2))])) == "pattern"
    # symmetry completion: right half mirrors the left half
    def sym(g):
        a = np.asarray(g)
        return np.concatenate([a, a[:, ::-1]], axis=1).tolist()
    def broken(g):
        a = np.asarray(sym(g))
        a[0, -1] = 9 if a[0, -1] != 9 else 8
        a[1, -2] = 9 if a[1, -2] != 9 else 8
        return a.tolist()
    assert F.family_of(_task([(broken(g1), sym(g1)), (broken(g2), sym(g2))])) == "symmetry"
    # crop: output is a sub-grid of the input
    assert F.family_of(_task([(g1, np.asarray(g1)[1:4, 2:5].tolist()), (g2, np.asarray(g2)[0:3, 1:3].tolist())])) == "context"
    # counting: one output cell per object (objects = isolated single cells)
    i1 = [[0] * 7 for _ in range(7)]
    i1[0][0] = 3; i1[3][3] = 3; i1[6][6] = 3
    i2 = [[0] * 7 for _ in range(7)]
    i2[0][0] = 3; i2[6][6] = 3
    assert F.family_of(_task([(i1, [[3, 3, 3]]), (i2, [[3, 3]])])) == "counting"
    # object: global recolouring of the objects
    a = [[0, 1, 1, 0], [0, 0, 0, 0], [2, 0, 0, 2]]
    b = [[0, 3, 3, 0], [0, 0, 0, 0], [4, 0, 0, 4]]
    assert F.family_of(_task([(a, b), (a, b)])) == "object"
    # relation: a straight line drawn between two objects (background elsewhere untouched)
    r_in = [[5, 0, 0, 0, 0, 5], [0, 0, 0, 0, 0, 0], [0, 0, 5, 0, 0, 0]]
    r_out = [[5, 1, 1, 1, 1, 5], [0, 0, 0, 0, 0, 0], [0, 0, 5, 0, 0, 0]]
    assert F.family_of(_task([(r_in, r_out), (r_in, r_out)])) == "relation"
    # a recolouring of the whole background is a global colour map -> object
    assert F.family_of(_task([([[5, 0, 0, 5]], [[5, 1, 1, 5]]), ([[5, 0, 5]], [[5, 1, 5]])])) == "object"
    # grows without tiling -> composition
    assert F.family_of(_task([(g1, rng.integers(0, 4, size=(9, 9)).tolist()), (g2, rng.integers(0, 4, size=(9, 9)).tolist())])) == "composition"
    assert F.family_of(Task("e", [], [])) == "composition"


@needs_data
def test_families_cover_real_tasks(tasks, families):
    assert set(families.values()) <= set(F.FAMILIES)
    hist = {f: 0 for f in F.FAMILIES}
    for f in families.values():
        hist[f] += 1
    assert sum(hist.values()) == 1000
    assert sum(1 for v in hist.values() if v > 0) >= 6  # not degenerate
    assert max(hist.values()) < 600  # no single family swallows the pool
    # determinism
    tid = sorted(families)[17]
    assert F.family_of(tasks[tid]) == families[tid]


def test_components_and_helpers():
    a = np.array([[1, 1, 0, 0], [0, 0, 0, 2], [3, 0, 2, 2]])
    comps = F.components(a)
    assert sorted(len(c) for c in comps) == [1, 2, 3]
    comps4 = F.components(a, diagonal=False)
    assert sorted(len(c) for c in comps4) == [1, 2, 3]
    assert F.components(np.zeros((3, 3), dtype=int)) == []
    assert F.dihedral_match(a, a[::-1, :]) == "flip_v"
    assert F.dihedral_match(a, a) is None
    assert F._is_periodic(np.array([[1, 2, 1, 2], [1, 2, 1, 2]]))
    assert not F._is_periodic(np.array([[1, 2, 3], [4, 5, 6]]))


# --------------------------------------------------------------------------------------------------------------
# Tensorisation
# --------------------------------------------------------------------------------------------------------------


def test_grid_tensor_round_trip():
    rng = np.random.default_rng(3)
    for _ in range(50):
        h, w = int(rng.integers(1, 31)), int(rng.integers(1, 31))
        g = rng.integers(0, 10, size=(h, w)).tolist()
        t = T.grid_to_tensor(g)
        assert t.shape == (30, 30) and t.dtype == torch.long
        assert (t == PAD_ID).sum().item() == 900 - h * w
        assert T.tensor_to_grid(t) == g
        m = T.grid_mask(g)
        assert m.dtype == torch.bool and m.sum().item() == h * w and T.mask_to_shape(m) == (h, w)
    assert T.tensor_to_grid(T.grid_to_tensor(None)) == []
    assert T.grid_mask([]).sum().item() == 0
    with pytest.raises(ValueError):
        T.grid_to_tensor([[0] * 31])


class _FakeObj:
    def __init__(self, cells):
        self.cells = frozenset(cells)


def _fake_parser(grid, hypothesis=None, max_objects=64):
    a = np.asarray(grid)
    comps = F.components(a)[:max_objects]
    objs = [_FakeObj(c) for c in comps]
    n = len(objs)
    feats = np.full((n, 32), 0.5, dtype=np.float32)
    rels = np.ones((n, n, 24), dtype=np.float32)
    return objs, feats, rels


def _episode(k_ctx=3, with_target=True):
    rng = np.random.default_rng(k_ctx)
    ctx = []
    for _ in range(k_ctx):
        g = rng.integers(0, 3, size=(6, 8)).tolist()
        ctx.append(Pair(g, [[c % 10 for c in row] for row in g]))
    test_in = rng.integers(0, 3, size=(7, 5)).tolist()
    return Episode("e", "t", "train", ctx, test_in, test_in if with_target else None)


def test_encode_episode_shapes_zero_parser():
    ep = _episode(3)
    d = T.encode_episode(ep, parser=T.zero_parser, max_ctx=10)
    exp = {"ctx_in": (10, 30, 30), "ctx_out": (10, 30, 30), "ctx_mask": (10,), "test_in": (30, 30),
           "target": (30, 30), "obj_feats": (11, 64, 32), "obj_crops": (11, 64, 30, 30), "obj_mask": (11, 64),
           "rel_feats": (11, 64, 64, 24), "n_ctx": ()}
    assert {k: tuple(v.shape) for k, v in d.items()} == exp
    assert d["ctx_mask"].tolist() == [True] * 3 + [False] * 7 and d["n_ctx"].item() == 3
    assert not d["obj_mask"].any() and (d["obj_crops"] == PAD_ID).all()
    assert T.tensor_to_grid(d["ctx_in"][0]) == ep.context[0].input
    assert T.tensor_to_grid(d["ctx_in"][5]) == [] and T.tensor_to_grid(d["target"]) == ep.test_input
    d2 = T.encode_episode(_episode(3, with_target=False), parser=T.zero_parser)
    assert (d2["target"] == PAD_ID).all()
    # more demos than max_ctx are truncated
    d3 = T.encode_episode(_episode(12), parser=T.zero_parser, max_ctx=10)
    assert d3["n_ctx"].item() == 10 and d3["ctx_mask"].all()


def test_encode_episode_with_parser_objects():
    ep = _episode(2)
    d = T.encode_episode(ep, parser=_fake_parser, max_ctx=4)
    assert d["obj_feats"].shape == (5, 64, 32) and d["obj_mask"].shape == (5, 64)
    for slot, grid in [(0, ep.context[0].input), (1, ep.context[1].input), (4, ep.test_input)]:
        n = min(len(F.components(np.asarray(grid))), 64)
        assert d["obj_mask"][slot].sum().item() == n
        assert torch.allclose(d["obj_feats"][slot, :n], torch.full((n, 32), 0.5))
        assert d["rel_feats"][slot, :n, :n].sum().item() == n * n * 24
        # crop of the first object reproduces its colours from the grid
        if n:
            crop = d["obj_crops"][slot, 0]
            assert crop.dtype == torch.int8 and (crop != PAD_ID).sum().item() == len(F.components(np.asarray(grid))[0])
    assert not d["obj_mask"][2].any() and not d["obj_mask"][3].any()


def test_collate_and_dataset():
    eps = [_episode(1), _episode(3), _episode(2)]
    ds = T.EpisodeDataset(eps, parser=T.zero_parser, max_ctx=6)
    assert len(ds) == 3 and ds.episode_ids == ["e", "e", "e"]
    batch = T.collate([ds[i] for i in range(3)])
    assert batch["ctx_in"].shape == (3, 6, 30, 30) and batch["obj_feats"].shape == (3, 7, 64, 32)
    assert batch["n_ctx"].tolist() == [1, 3, 2]
    # heterogeneous max_ctx is padded
    mixed = [T.encode_episode(eps[0], T.zero_parser, max_ctx=2), T.encode_episode(eps[1], T.zero_parser, max_ctx=5)]
    b2 = T.collate(mixed)
    assert b2["ctx_in"].shape == (2, 5, 30, 30) and (b2["ctx_in"][0, 2:] == PAD_ID).all()
    assert b2["obj_mask"].shape == (2, 6, 64) and not b2["ctx_mask"][0, 1:].any()
    loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=T.EpisodeDataset.collate_fn)
    b3 = next(iter(loader))
    assert b3["test_in"].shape == (2, 30, 30)
    with pytest.raises(ValueError):
        T.collate([])


def test_resolve_parser_fallback():
    fn = T.resolve_parser(None)
    assert callable(fn)
    objs, f, r = T.zero_parser([[1]])
    assert objs == [] and f.shape == (0, 32) and r.shape == (0, 0, 24)


@needs_data
def test_real_episode_dataset():
    eps = H.load_episodes(None, "episodes", "val")[:4]
    ds = T.EpisodeDataset(eps, parser=T.zero_parser)
    b = T.collate([ds[i] for i in range(len(ds))])
    assert b["ctx_in"].shape == (4, 10, 30, 30)
    for i, ep in enumerate(eps):
        assert T.tensor_to_grid(b["test_in"][i]) == ep.test_input
        assert T.tensor_to_grid(b["target"][i]) == ep.target_output
        assert b["n_ctx"][i].item() == min(len(ep.context), 10)
