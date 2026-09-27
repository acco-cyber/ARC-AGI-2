"""Tests for the frozen v2 split (Train-670 / Val-150 / Hard-180) and the Hard-180 harness. CPU, < 60 s."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arcjepa.data import hf_loader as H  # noqa: E402
from arcjepa.data import splits as S  # noqa: E402
from arcjepa.eval import hard180 as HB  # noqa: E402

#: the pinned identity of the Hard-180 ids (sha256 of "\n".join(sorted(ids))). NEVER update this to make a test
#: pass: a different digest means the locked development harness changed.
PINNED_HARD180_SHA256 = "109f247cbc0ccf729e1a66d0747bcb2fde5e68c3715a90f76a271e94d447f9db"
SPLIT_PATH = ROOT / "data" / "splits_670_150_180.json"
OLD_PATH = ROOT / "data" / "splits_700_150_150.json"

HAVE_DATA = H.has_local_root()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="local HF mirror not available")


@pytest.fixture(scope="module")
def doc():
    return S.load_split()


@pytest.fixture(scope="module")
def old():
    return json.loads(OLD_PATH.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ split
def test_hard180_sha256_is_pinned(doc):
    assert S.HARD180_SHA256 == PINNED_HARD180_SHA256
    assert doc["hard180_sha256"] == PINNED_HARD180_SHA256
    assert S.hard180_sha256(doc["hard180"]) == PINNED_HARD180_SHA256
    assert S.hard180_sha256(list(reversed(doc["hard180"]))) == PINNED_HARD180_SHA256  # order-free


def test_sizes_disjoint_and_union(doc, old):
    train, val, hard = set(doc["train"]), set(doc["val"]), set(doc["hard180"])
    assert (len(doc["train"]), len(doc["val"]), len(doc["hard180"])) == (670, 150, 180)
    assert (len(train), len(val), len(hard)) == (670, 150, 180)
    assert not (train & val) and not (train & hard) and not (val & hard)
    union = set(old["train"]) | set(old["val"]) | set(old["holdout"])
    assert len(union) == 1000 and train | val | hard == union
    # construction: clean 150 = v1 holdout, 30 from v1 train, val unchanged, train = v1 train minus the 30
    assert set(doc["hard180_clean150"]) == set(old["holdout"]) and len(doc["hard180_clean150"]) == 150
    moved = set(doc["hard180_from_old_train"])
    assert len(moved) == 30 and moved <= set(old["train"])
    assert set(doc["hard180_clean150"]) | moved == hard
    assert val == set(old["val"]) and train == set(old["train"]) - moved
    assert set(doc["hard180_scores"]) == hard and set(doc["hard180_features"]) == hard
    # the 30 moved ids are the 30 highest-scoring v1 train ids (cutoff strictly between rank 30 and 31)
    cut = doc["hardness_cutoff"]
    assert min(doc["hard180_scores"][t] for t in moved) == cut["rank30_score"] > cut["rank31_score"]
    assert doc["seed"] == S.HARD180_SEED == 20260927 and "method" in doc and "eval_public" not in doc


@needs_data
def test_union_is_the_1000_training_ids_and_no_public_eval_id(doc):
    rows = H.load_task_rows(splits=("train", "val", "test"))  # never opens eval_public.jsonl
    training = {t for t, r in rows.items() if r.get("source_split") == "training"}
    assert len(training) == 1000
    assert set(doc["train"]) | set(doc["val"]) | set(doc["hard180"]) == training
    eval_ids = set(H.read_hf_splits()["eval_public"])  # the id list only
    assert len(eval_ids) == 120
    for name in ("train", "val", "hard180"):
        assert not set(doc[name]) & eval_ids, name


@needs_data
def test_rebuild_is_deterministic(doc):
    again = S.build_split()
    for k in ("train", "val", "hard180", "hard180_clean150", "hard180_from_old_train", "hard180_sha256",
              "hard180_scores", "hard180_features"):
        assert again[k] == doc[k], k


def test_tampered_split_is_rejected(doc, tmp_path):
    bad = dict(doc)
    swap_in = doc["train"][0]
    bad["hard180"] = sorted(doc["hard180"][1:] + [swap_in])
    bad["train"] = sorted(doc["train"][1:] + [doc["hard180"][0]])
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(S.SplitIntegrityError):
        S.load_split(p)
    with pytest.raises(AssertionError):  # SplitIntegrityError is an AssertionError
        H.read_splits(str(p))


def test_write_split_refuses_overwrite(tmp_path, doc):
    p = tmp_path / "s.json"
    S.write_split(doc, p)
    with pytest.raises(FileExistsError):
        S.write_split(doc, p)
    assert S.load_split(p)["hard180"] == doc["hard180"]
    assert b"\r\n" not in p.read_bytes()


def test_split_ids_aliases(doc):
    assert S.split_ids("train670") == sorted(doc["train"])
    assert S.split_ids("h180") == sorted(doc["hard180"])
    assert S.split_ids("clean150") == sorted(doc["hard180_clean150"])
    with pytest.raises(KeyError):
        S.split_ids("eval_public")


# ------------------------------------------------------------------------------------------------ loader wiring
def test_load_resplit_defaults_to_train670(doc):
    sp = H.load_resplit()
    assert len(sp["train"]) == 670 and sorted(sp["train"]) == sorted(doc["train"])
    assert sorted(sp["holdout"]) == sorted(doc["hard180"]) == sorted(sp["hard180"])
    assert sorted(sp["val"]) == sorted(doc["val"])
    old = H.load_resplit(split_file="data/splits_700_150_150.json")
    assert len(old["train"]) == 700 and len(old["holdout"]) == 150
    with pytest.raises(H.SplitFileMissing):
        H.load_resplit(split_file="data/no_such_split.json")


def test_configs_use_the_v2_split_file():
    from arcjepa.training import common as C

    for name in ("base", "debug", "l4x4", "kaggle"):
        cfg = C.load_config(ROOT / "configs" / f"{name}.yaml")
        sf = C.cfg_get(cfg, "data.split_file")
        assert sf and Path(H.resolve_split_file(sf)).resolve() == SPLIT_PATH.resolve(), name


# ------------------------------------------------------------------------------------------------ scoring helpers
def test_cell_error_and_failure_categories():
    t = [[1, 2], [3, 4]]
    assert HB.cell_error(t, t) == 0.0
    assert HB.cell_error([[1, 2], [3, 0]], t) == 0.25
    assert HB.cell_error([[1, 2, 3]], t) == 1.0 and HB.cell_error(None, t) == 1.0
    fc = HB.failure_category
    assert fc([True, True], [0.0, 0.0], [True, True], False) == "solved"
    assert fc([True, False], [0.0, 0.05], [True, True], True) == "exact_fit_wrong_on_test"
    assert fc([False], [1.0], [False], False) == "wrong_shape"
    assert fc([True, False], [0.0, 0.05], [True, True], False) == "near_miss_90"
    assert fc([False], [0.3], [True], False) == "no_candidate_close"
    assert fc([False], [1.0], [False], False, has_attempts=False) == "no_candidate_close"
    assert set(HB.FAILURE_CATEGORIES) == {"solved", "wrong_shape", "exact_fit_wrong_on_test", "near_miss_90",
                                          "no_candidate_close"}


def test_task_record_fields_from_synthetic_diagnostics():
    from arcjepa.core.types import Pair, Task

    task = Task("x", [Pair([[1]], [[2]])], [Pair([[1, 1]], [[2, 2]]), Pair([[3]], [[4]])])
    attempts = [([[2, 2]], [[0, 0]]), ([[1, 1]], [[4, 0]])]
    rec = HB.task_record(task, attempts, {"n_exact": 0, "beam_expansions": 7, "astar_nodes": 3,
                                          "evo_generations": 2, "n_candidates": 5, "difficulty": 0.4},
                         runtime=1.25, split="hard180", family="object", evo_pop=10)
    for k in HB.PER_TASK_FIELDS:
        assert k in rec, k
    assert rec["task_pass"] is False and rec["output_pass"] == 1 and rec["output_total"] == 2
    assert rec["best_cell_error"] == 0.0 and rec["cell_error_per_output"] == [0.0, 1.0]
    assert rec["failure_category"] == "wrong_shape"
    assert rec["search_nodes"] == 7 + 3 + 2 * 10 and rec["candidate_count"] == 5
    assert rec["repair_rounds"] is None and "repair_rounds" in rec["missing_counters"]


def test_regression_check():
    cur = [{"task_id": t, "task_pass": p} for t, p in [("a", True), ("b", False), ("c", False), ("d", True)]]
    base = [{"task_id": t, "task_pass": p} for t, p in [("a", True), ("b", True), ("c", True), ("d", False)]]
    r = HB.regression_check(cur, base, ["a", "b", "c", "d"])
    assert r["delta"] == -1 and r["accepted"] and r["lost"] == ["b", "c"] and r["gained"] == ["d"]
    r2 = HB.regression_check(cur, base, ["a", "b", "c", "d"], tolerance=0)
    assert not r2["accepted"]


# ------------------------------------------------------------------------------------------------ harness
@needs_data
def test_harness_three_tasks_writes_every_field(tmp_path, doc):
    out = tmp_path / "reports"
    s = HB.run("hard180", None, budget=3.0, workers=1, max_tasks=3, out=str(out), tag="t")
    summary_path, rows_path = HB.output_paths(str(out), "hard180", "t")
    assert summary_path.name == "t_h180.json" and rows_path.name == "t_h180_per_task.jsonl"
    assert summary_path.is_file() and rows_path.is_file()
    rows = HB.read_rows(rows_path)
    ids = sorted(doc["hard180"])[:3]
    assert [r["task_id"] for r in rows] == ids
    for r in rows:
        for k in HB.PER_TASK_FIELDS:
            assert k in r, k
        assert r["split"] == "hard180" and r["subset"] in ("clean150", "from_old_train")
        assert isinstance(r["task_pass"], bool) and 0 <= r["output_pass"] <= r["output_total"]
        assert 0.0 <= r["best_cell_error"] <= 1.0 and r["failure_category"] in HB.FAILURE_CATEGORIES
        assert r["family"] and r["hardness"] is not None and r["runtime"] < 3.0 * 1.1 + 2.0
        assert r["beam_expansions"] is not None and r["candidate_count"] is not None
        assert len(r["attempts"]) == r["output_total"]
    disk = json.loads(summary_path.read_text(encoding="utf-8"))
    for k in ("headline", "task_pass@2", "output_pass@2", "exact_program", "near_90_cell", "runtime_s",
              "failure_categories", "per_family", "subsets", "search", "notes", "hard180_sha256"):
        assert k in disk, k
    assert disk["n_tasks"] == 3 and disk["complete"] and disk["hard180_sha256"] == PINNED_HARD180_SHA256
    assert set(disk["subsets"]) == {"hard180_clean150", "hard180_from_old_train"}
    assert sum(disk["failure_categories"].values()) == 3
    assert s["headline"] == disk["headline"]

    # resumable: a second invocation solves nothing new and keeps the rows
    s2 = HB.run("hard180", None, budget=3.0, workers=1, max_tasks=3, out=str(out), tag="t")
    assert len(HB.read_rows(rows_path)) == 3 and s2["headline"] == s["headline"]
    # another configuration may not append to the same file
    with pytest.raises(RuntimeError):
        HB.run("hard180", None, budget=4.0, workers=1, max_tasks=4, out=str(out), tag="t")
    # regression gate against itself: accepted, delta 0; CLI summarise-only exits 0
    s3 = HB.run("hard180", None, budget=3.0, workers=1, max_tasks=3, out=str(out), tag="t",
                baseline=str(summary_path))
    assert s3["regression"]["accepted"] and s3["regression"]["delta"] == 0
    rc = HB.main(["--budget", "3", "--max-tasks", "3", "--out", str(out), "--tag", "t", "--summarize-only"])
    assert rc == 0
