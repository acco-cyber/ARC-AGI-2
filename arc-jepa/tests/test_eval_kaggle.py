"""Tests for arcjepa/eval, kaggle/ (notebook builders, submission validator) and the Kaggle submission runner."""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import py_compile
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from arcjepa.core.types import Pair, Task
from arcjepa.eval import (SPEC_KEYS, analyze_errors, cell_accuracy, competition_score, error_record, evaluate,
                          near_miss, normalize_attempts, read_diagnostics, replay_solver, retrieval_at_k,
                          score_task, summarize_search)
from arcjepa.utils.kaggle_submit_runner import (RunnerConfig, fair_share_seconds, fallback_attempts,
                                                order_shortest_first, run_submission)

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = ROOT / "kaggle"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(f"_t_{name}", KAGGLE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VS = _load("validate_submission")
BT = _load("build_train_nb")
BI = _load("build_infer_nb")

A = [[1, 2], [3, 4]]
B = [[4, 3], [2, 1]]
C = [[0]]


def flip(g: List[List[int]]) -> List[List[int]]:
    return [list(reversed(r)) for r in g]


# ============================================================================================ metric

def test_score_task_hand_cases() -> None:
    assert score_task([(A, B)], [A]) == 1.0            # attempt_1 hit
    assert score_task([(B, A)], [A]) == 1.0            # attempt_2 hit
    assert score_task([(B, C)], [A]) == 0.0
    assert score_task([(A, C), (C, C)], [A, B]) == 0.5  # one of two test outputs
    assert score_task([(A, C)], [A, B]) == 0.5          # missing attempts for test 2 count as wrong
    assert score_task([{"attempt_1": C, "attempt_2": B}], [B]) == 1.0
    assert score_task([], [A]) == 0.0
    assert score_task([(A, A)], []) == 0.0


def test_competition_score_mean_over_tasks() -> None:
    sols = {"t1": [A], "t2": [A, B], "t3": [B]}
    sub = {"t1": [{"attempt_1": C, "attempt_2": A}],
           "t2": [{"attempt_1": A, "attempt_2": C}, {"attempt_1": C, "attempt_2": C}]}  # t3 missing -> 0
    assert competition_score(sub, sols) == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert VS.score_submission(sub, sols) == pytest.approx(competition_score(sub, sols))
    assert competition_score({}, {}) == 0.0
    # a near miss is not a hit
    assert competition_score({"t3": [{"attempt_1": [[4, 3], [2, 2]], "attempt_2": C}]}, {"t3": [B]}) == 0.0


def test_normalize_attempts_formats() -> None:
    assert normalize_attempts([(A, B)]) == [(A, B)]
    assert normalize_attempts([[A, B]]) == [(A, B)]
    assert normalize_attempts([{"attempt_1": A}]) == [(A, None)]
    assert normalize_attempts([[[1, 2], [3, 4]]]) == [(None, None)]  # a bare 2-row grid is not an attempt pair
    assert normalize_attempts(None) == []


def _task(tid: str, train: List[Pair], tests: List[Pair]) -> Task:
    return Task(tid, train, tests)


def test_evaluate_with_dummy_solver(tmp_path: Path) -> None:
    tr = [Pair(A, flip(A)), Pair(B, flip(B))]
    tasks = {
        "a_right2": _task("a_right2", tr, [Pair(B, flip(B))]),
        "b_half": _task("b_half", tr, [Pair(A, flip(A)), Pair(B, flip(B))]),
        "c_raises": _task("c_raises", tr, [Pair(A, flip(A))]),
        "d_unknown": _task("d_unknown", tr, [Pair(A, [])]),
    }

    def solver(t: Task) -> Any:
        if t.task_id == "c_raises":
            raise ValueError("boom")
        if t.task_id == "a_right2":
            return [(C, flip(B))], {"beam_expansions": 12, "candidate_rank": 1, "bucket": 0}
        if t.task_id == "b_half":
            return [(flip(A), C), ([[3, 4], [1, 1]], C)]
        return [(C, C)]

    path = tmp_path / "diag.json"
    res = evaluate(tasks, solver, diagnostics_path=str(path))
    assert res["score"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert res["n_tasks"] == 3 and res["n_unscored"] == 1 and res["n_solver_errors"] == 1
    assert res["n_test_outputs"] == 4 and res["n_correct_outputs"] == 2
    assert sum(v["n"] for v in res["per_family"].values()) == 3
    for rec in res["diagnostics"]:
        assert all(k in rec for k in SPEC_KEYS)
    by_id = {r["task_id"]: r for r in res["diagnostics"]}
    assert by_id["a_right2"]["correct"] == [True] and by_id["a_right2"]["beam_expansions"] == 12
    assert by_id["b_half"]["correct"] == [True, False]
    assert "error" in by_id["c_raises"]
    ea = res["error_analysis"]
    assert ea["n_outputs"] == 4 and ea["n_exact"] == 2 and ea["n_shape_mismatch"] == 1
    assert ea["n_shape_match_wrong"] == 1  # b_half test 2: 3 of 4 cells right
    assert res["search_stats"]["n_tasks"] == 4
    assert len(read_diagnostics(str(path))) == 4
    # max_tasks and replay
    assert evaluate(tasks, solver, max_tasks=1)["n_tasks"] == 1
    replay = evaluate(tasks, replay_solver(res["submission"]))
    assert replay["score"] == pytest.approx(res["score"])


def test_error_analysis() -> None:
    assert cell_accuracy(A, A) == 1.0
    assert cell_accuracy([[1, 2], [3, 0]], A) == 0.75
    assert cell_accuracy([[1, 2]], A) == 0.0
    assert cell_accuracy(None, A) == 0.0
    nm = near_miss([[[1]], [[1, 2], [3, 0]]], A)
    assert nm == {"exact": False, "shape_match": True, "best_cell_acc": 0.75, "wrong_cells": 1, "best_attempt": 2}
    assert near_miss([C, A], A)["exact"]
    rows = [error_record("x", 0, [A, C], A), error_record("y", 0, [[[1, 2], [3, 0]], C], A),
            error_record("z", 0, [C, C], A)]
    s = analyze_errors(rows)
    assert (s["n_exact"], s["n_shape_mismatch"], s["n_shape_match_wrong"]) == (1, 1, 1)
    assert sum(s["cell_acc_histogram"].values()) == 1 and s["near_miss_90"] == 0


def test_search_stats() -> None:
    r = retrieval_at_k([1, 3, None, 10], ks=(1, 4, 8, 16))
    assert r == {"retrieval@1": 0.25, "retrieval@4": 0.5, "retrieval@8": 0.5, "retrieval@16": 0.75}
    diags = [{"score": 1.0, "beam_expansions": 10, "inference_ms": 5.0, "bucket": 0, "candidate_rank": 1},
             {"score": 0.0, "beam_expansions": 1000, "astar_nodes": 50, "inference_ms": 50.0, "bucket": 3},
             {"correct": [True, False], "beam_expansions": 3, "bucket": 0, "rule_retrieval_r8": 1.0}]
    s = summarize_search(diags)
    assert s["n_tasks"] == 3 and s["beam_expansions"]["max"] == 1000
    assert s["accuracy_by_bucket"]["0"]["accuracy"] == pytest.approx(0.75)
    assert sum(b["n"] for b in s["accuracy_vs_nodes"]) == 3
    assert s["program_rank"]["retrieval@1"] == pytest.approx(1 / 3)
    assert s["rule_retrieval_r8"] == 1.0


# ============================================================================================ validator

CHALLENGES = {
    "t1": {"train": [{"input": A, "output": flip(A)}, {"input": B, "output": flip(B)}], "test": [{"input": B}]},
    "t2": {"train": [{"input": A, "output": [[5, 5, 5]]}, {"input": B, "output": [[5, 5, 5]]}],
           "test": [{"input": A}, {"input": [[7]]}]},
}


def _good() -> Dict[str, Any]:
    return {"t1": [{"attempt_1": B, "attempt_2": A}],
            "t2": [{"attempt_1": [[5, 5, 5]], "attempt_2": C}, {"attempt_1": C, "attempt_2": C}]}


def test_validator_accepts_valid() -> None:
    assert VS.validate_submission(_good(), CHALLENGES) == []
    assert VS.validate_submission(_good(), task_ids=["t1", "t2"]) == []


def _mutate(kind: str) -> Any:
    s = _good()
    if kind == "not_dict":
        return [s]
    if kind == "missing_task":
        del s["t2"]
    elif kind == "extra_task":
        s["t9"] = s["t1"]
    elif kind == "one_attempt":
        del s["t1"][0]["attempt_2"]
    elif kind == "extra_key":
        s["t1"][0]["attempt_3"] = A
    elif kind == "value_10":
        s["t1"][0]["attempt_1"] = [[10]]
    elif kind == "negative":
        s["t1"][0]["attempt_1"] = [[-1]]
    elif kind == "float":
        s["t1"][0]["attempt_1"] = [[1.0]]
    elif kind == "bool":
        s["t1"][0]["attempt_1"] = [[True]]
    elif kind == "string":
        s["t1"][0]["attempt_1"] = [["1"]]
    elif kind == "ragged":
        s["t1"][0]["attempt_1"] = [[1, 2], [3]]
    elif kind == "empty_grid":
        s["t1"][0]["attempt_1"] = []
    elif kind == "empty_row":
        s["t1"][0]["attempt_1"] = [[]]
    elif kind == "too_tall":
        s["t1"][0]["attempt_1"] = [[0]] * 31
    elif kind == "too_wide":
        s["t1"][0]["attempt_1"] = [[0] * 31]
    elif kind == "wrong_count":
        s["t2"] = s["t2"][:1]
    elif kind == "entry_not_dict":
        s["t1"] = [[B, A]]
    elif kind == "value_not_list":
        s["t1"] = {"attempt_1": B, "attempt_2": A}
    elif kind == "empty_list":
        s["t1"] = []
    return s


@pytest.mark.parametrize("kind", ["not_dict", "missing_task", "extra_task", "one_attempt", "extra_key", "value_10",
                                  "negative", "float", "bool", "string", "ragged", "empty_grid", "empty_row",
                                  "too_tall", "too_wide", "wrong_count", "entry_not_dict", "value_not_list",
                                  "empty_list"])
def test_validator_rejects_malformed(kind: str) -> None:
    assert VS.validate_submission(_mutate(kind), CHALLENGES), kind


def test_validate_file_and_cli(tmp_path: Path) -> None:
    ch = tmp_path / "ch.json"
    ch.write_text(json.dumps(CHALLENGES), encoding="utf-8")
    good, bad, broken = tmp_path / "good.json", tmp_path / "bad.json", tmp_path / "broken.json"
    good.write_text(json.dumps(_good()), encoding="utf-8")
    bad.write_text(json.dumps(_mutate("value_10")), encoding="utf-8")
    broken.write_text("{not json", encoding="utf-8")
    assert VS.validate_file(str(good), str(ch)) == []
    assert VS.validate_file(str(bad), str(ch))
    assert VS.validate_file(str(broken))
    assert VS.validate_file(str(tmp_path / "missing.json"))
    assert VS.main([str(good), "--challenges", str(ch)]) == 0
    assert VS.main([str(bad), "--challenges", str(ch)]) == 1


def test_fallbacks_and_repair() -> None:
    fb = VS.fallback_submission(CHALLENGES)
    assert VS.validate_submission(fb, CHALLENGES) == []
    # t1 is not an identity task: no attempt is the test input; both are 2x2 fills (the predicted shape) with the
    # two most common demo output colours
    t1 = fb["t1"][0]
    assert B not in (t1["attempt_1"], t1["attempt_2"]) and t1["attempt_1"] != t1["attempt_2"]
    assert all(len({v for r in g for v in r}) == 1 and (len(g), len(g[0])) == (2, 2) for g in t1.values())
    assert fb["t2"][0]["attempt_1"] == [[5, 5, 5]] and fb["t2"][1]["attempt_1"] == [[5, 5, 5]]  # constant output
    assert fb["t2"][1]["attempt_2"] == [[7]]                 # identity only as the last resort
    assert {t: fallback_attempts(d) for t, d in CHALLENGES.items()} == fb  # runner and inline agree
    ident_task = {"train": [{"input": A, "output": A}, {"input": B, "output": B}], "test": [{"input": [[7]]}]}
    assert VS.fallback_attempts(ident_task)[0]["attempt_1"] == [[7]] == fallback_attempts(ident_task)[0]["attempt_1"]
    rep = VS.repair_submission(_mutate("value_10"), CHALLENGES)
    assert VS.validate_submission(rep, CHALLENGES) == []
    assert rep["t1"][0]["attempt_1"] == fb["t1"][0]["attempt_1"] and rep["t1"][0]["attempt_2"] == A
    assert VS.validate_submission(VS.repair_submission(None, CHALLENGES), CHALLENGES) == []


def test_fallbacks_agree_with_the_solver_and_avoid_the_identity() -> None:
    """runner / inline / arcjepa.search fallbacks give the same two grids on real tasks, and never the identity
    for a task whose demos are not all identity mappings."""
    from arcjepa.core.types import task_from_json
    from arcjepa.search import fallback_grids

    path = Path(os.environ.get("ARCJEPA_DATA", r"E:\Claude code\arc2\dataset\hf_arc2_episodes")) / "tasks" / "train.jsonl"
    if not path.is_file():
        pytest.skip(f"real task file missing: {path}")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if len(rows) >= 60:
                break
            rows.append(json.loads(line))
    n_identity_tasks = 0
    for row in rows:
        ch = {"train": row["train"], "test": [{"input": p["input"]} for p in row["test"]]}
        fa, fv = fallback_attempts(ch), VS.fallback_attempts(ch)
        assert fa == fv, row["task_id"]
        task = task_from_json(row["task_id"], row)
        pairs = [p for p in task.train]
        ident = all(p.input == p.output for p in pairs)
        n_identity_tasks += ident
        for tp, att in zip(task.test, fa):
            fbs = fallback_grids(tp.input, pairs)
            first_two = [fbs[0], next((g for g in fbs[1:] if g != fbs[0]), fbs[0])]
            assert first_two == [att["attempt_1"], att["attempt_2"]], row["task_id"]
            if not ident:
                assert tp.input not in first_two, row["task_id"]
    assert n_identity_tasks < len(rows)


def _pkg(d: Path, *, debug: bool = False, created: int = 1_790_000_000, fmt: str = "arcjepa-package-v1",
         weights: bytes = b"\x08\x00\x00\x00\x00\x00\x00\x00{}      ", config: Any = None) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    cfg = config if config is not None else json.dumps({
        "format": fmt, "created_unix": created, "n_parameters": 557041 if debug else 39_530_000,
        "training": {"config_path": "configs/debug.yaml" if debug else "configs/kaggle.yaml"}})
    (d / "config.json").write_text(cfg, encoding="utf-8")
    (d / "model.safetensors").write_bytes(weights)
    return d


def test_find_package_kaggle_mount_layouts(tmp_path: Path) -> None:
    from arcjepa.utils.kaggle_submit_runner import find_package, package_info

    def fresh(name: str) -> Path:
        root = tmp_path / name / "input"
        root.mkdir(parents=True)
        return root

    def code_dataset(root: Path) -> None:  # the code dataset with its local runs/ debug packages (decoys)
        (root / "arc-jepa-code" / "arcjepa").mkdir(parents=True)
        (root / "arc-jepa-code" / "arcjepa" / "__init__.py").write_text("", encoding="utf-8")
        _pkg(root / "arc-jepa-code" / "runs" / "e2e" / "pkg", debug=True)
        _pkg(root / "arc-jepa-code" / "runs" / "fake" / "arc-jepa-train" / "arc_jepa_pkg", debug=True)

    layouts = {
        "new": "notebooks/poby7722/arc-jepa-train/arc_jepa_pkg",
        "old": "arc-jepa-train/arc_jepa_pkg",
        "nested_output": "notebooks/poby7722/arc-jepa-train/output/arc_jepa_pkg",
        "versioned": "notebooks/poby7722/arc-jepa-train/versions/3/arc_jepa_pkg",
    }
    for name, rel in layouts.items():
        root = fresh(name)
        code_dataset(root)
        want = _pkg(root / rel)
        assert find_package([], search_roots=[str(root)]) == str(want), name
    # decoys only: nothing is picked (a code tree is never searched)
    root = fresh("decoy_only")
    code_dataset(root)
    assert find_package([], search_roots=[str(root)]) is None
    # a smoke-trained kernel output is still found (the notebook's dev gate flags it), after any real package
    root = fresh("debug_vs_real")
    smoke = _pkg(root / "arc-jepa-train-smoke" / "arc_jepa_pkg", debug=True)
    assert find_package([], search_roots=[str(root)]) == str(smoke) and package_info(str(smoke))["debug"]
    real = _pkg(root / "arc-jepa-train" / "arc_jepa_pkg")
    assert find_package([], search_roots=[str(root)]) == str(real)
    # newest real package first; broken ones are skipped
    root = fresh("versions")
    _pkg(root / "a" / "arc_jepa_pkg", created=100)
    newest = _pkg(root / "b" / "arc_jepa_pkg", created=200)
    _pkg(root / "c" / "arc_jepa_pkg", created=300, config="{not json")
    _pkg(root / "d" / "arc_jepa_pkg", created=400, weights=b"")
    _pkg(root / "e" / "arc_jepa_pkg", created=500, fmt="something-else")
    assert find_package([], search_roots=[str(root)]) == str(newest)
    # generic (not arc_jepa_pkg-named) folders: real ones up to max_depth, debug ones only with allow_debug
    root = fresh("generic")
    dbg = _pkg(root / "some-output" / "pkg", debug=True)
    assert find_package([], search_roots=[str(root)]) is None
    assert find_package([], search_roots=[str(root)], allow_debug=True) == str(dbg)
    gen = _pkg(root / "other-output" / "model_pkg")
    assert find_package([], search_roots=[str(root)]) == str(gen)
    # explicit candidates win, debug or not; invalid explicit candidates fall through to the walk
    assert find_package([str(dbg)], search_roots=[str(root)]) == str(dbg)
    assert find_package([str(tmp_path / "missing"), str(root)], search_roots=[str(root)]) == str(gen)
    assert find_package([], search_roots=[str(tmp_path / "nowhere")]) is None


# ============================================================================================ notebooks

def _statement_first_lines(src: str) -> List[str]:
    """First source line of every statement (continuation lines of a multi-line expression excluded)."""
    lines = src.splitlines()
    return [lines[n.lineno - 1] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.stmt)]


def _check_notebook(path: Path, tmp_path: Path) -> Dict[str, Any]:
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(str(path), as_version=4)
    nbformat.validate(nb)
    ids = [c["id"] for c in nb.cells]
    assert len(ids) == len(set(ids)) and all(ids)
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        f = tmp_path / f"{path.stem}-{c['id']}.py"
        f.write_text(c.source, encoding="utf-8")
        py_compile.compile(str(f), cfile=str(f) + "c", doraise=True)  # a `%magic` / `!cmd` line cannot compile
        # no IPython magic / shell escape at the start of any statement (a line starting with "%" inside a
        # multi-line expression, e.g. `"..." % (a, b)`, is ordinary Python)
        assert not any(line.lstrip().startswith(("!", "%")) for line in _statement_first_lines(c.source)), c["id"]
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_sources(nb: Dict[str, Any]) -> List[str]:
    return ["".join(c["source"]) for c in nb["cells"]]


def test_notebooks_build_validate_and_compile(tmp_path: Path) -> None:
    out = tmp_path / "kaggle"
    tr, inf = BT.build(out), BI.build(out)
    for paths in (tr, inf):
        _check_notebook(Path(paths["notebook"]), tmp_path)
        assert Path(paths["kernel_notebook"]).read_bytes() == Path(paths["notebook"]).read_bytes()
    mt = json.loads(Path(tr["metadata"]).read_text(encoding="utf-8"))
    mi = json.loads(Path(inf["metadata"]).read_text(encoding="utf-8"))
    for m in (mt, mi):
        assert m["docker_image"] == DOCKER and m["machine_shape"] == "NvidiaL4"
        assert m["enable_internet"] is False and m["enable_gpu"] is True
        assert m["dataset_sources"] == ["poby7722/arc-jepa-code", "poby7722/arc-agi-2-jepa-episodes"]
    # the inference kernel needs the competition data; the training kernel reads no ARC competition file, so it
    # may drop the mount (review 09-26)
    assert mi["competition_sources"] == ["arc-prize-2026-arc-agi-2"]
    assert mt["competition_sources"] in ([], ["arc-prize-2026-arc-agi-2"])
    assert mt["id"] == "poby7722/arc-jepa-train" and mt["code_file"] == "arc-jepa-train.ipynb"
    assert mt["kernel_sources"] == []
    assert mi["id"] == "poby7722/arc-jepa-infer" and mi["code_file"] == "arc-jepa-infer.ipynb"
    assert mi["kernel_sources"] == ["poby7722/arc-jepa-train"]
    # the committed notebooks are up to date with the builders
    for rel in ("arc-jepa-train.ipynb", "train/kernel-metadata.json", "train/arc-jepa-train.ipynb",
                "arc-jepa-infer.ipynb", "infer/kernel-metadata.json", "infer/arc-jepa-infer.ipynb"):
        assert (KAGGLE / rel).is_file(), rel
        assert (KAGGLE / rel).read_bytes() == (out / rel).read_bytes(), f"{rel} is stale: rerun the builders"
    train_srcs = _cell_sources(json.loads((out / "arc-jepa-train.ipynb").read_text(encoding="utf-8")))
    assert any("arcjepa.synthetic.dataset" in s for s in train_srcs)
    # smoke stays the default build; --full writes kaggle/train_full with FULL defaults and identical metadata
    assert any('os.environ.get("ARCJEPA_SMOKE", "1")' in s for s in train_srcs)
    full = BT.build(out, full=True)
    fsrcs = _cell_sources(_check_notebook(Path(full["kernel_notebook"]), tmp_path))
    assert any('os.environ.get("ARCJEPA_SMOKE", "0")' in s for s in fsrcs)
    assert any('os.environ.get("ARCJEPA_TRAIN_HOURS", "7.5")' in s for s in fsrcs)
    assert any("WALL_HOURS" in s for s in fsrcs)
    assert not re.findall(r"__[A-Z][A-Z0-9_]*__", "".join(fsrcs)), "unsubstituted build placeholder"
    assert json.loads(Path(full["metadata"]).read_text(encoding="utf-8")) == mt
    assert any('"--device", "auto"' in s and "memory.max_synthetic=4000" in s for s in fsrcs)
    for rel in ("train_full/kernel-metadata.json", "train_full/arc-jepa-train.ipynb"):
        assert (KAGGLE / rel).read_bytes() == (out / rel).read_bytes(), f"{rel} is stale: rerun build_train_nb --full"
    inb = json.loads((out / "arc-jepa-infer.ipynb").read_text(encoding="utf-8"))
    isrc = json.dumps(inb)
    for needle in ("KAGGLE_IS_COMPETITION_RERUN", "arc-agi_test_challenges.json", "fallback_submission",
                   "validate_submission", "run_submission", "arc-agi_evaluation_challenges.json",
                   "notebooks/poby7722/arc-jepa-train/arc_jepa_pkg", "arc-jepa-train/arc_jepa_pkg"):
        assert needle in isrc, needle
    # the dev gate is the last cell and never fires in a competition rerun
    gate = inb["cells"][-1]
    assert gate["id"] == "infer-09-dev-gate"
    assert "if not IS_RERUN:" in "".join(gate["source"]) and "raise RuntimeError" in "".join(gate["source"])
    # the code copy after the fallback submission is wrapped (a copy error must not fail a rerun)
    code = next(s for c, s in zip(inb["cells"], _cell_sources(inb)) if c["id"] == "infer-04-code")
    assert "copytree" in code and "except Exception as exc" in code


# ============================================================================================ runner

def _flip_task(g: List[List[int]]) -> Dict[str, Any]:
    return {"train": [{"input": A, "output": flip(A)}, {"input": [[1, 2, 3], [4, 5, 6]], "output": [[3, 2, 1], [6, 5, 4]]},
                      {"input": [[7, 0, 0]], "output": [[0, 0, 7]]}],
            "test": [{"input": g}]}


RUN_CH = {
    "big": _flip_task([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2]]),
    "small": _flip_task([[3, 4]]),
    "two": {"train": [{"input": A, "output": A}, {"input": B, "output": B}], "test": [{"input": A}, {"input": [[9]]}]},
}
RUN_SOL = {"big": [flip(RUN_CH["big"]["test"][0]["input"])], "small": [[[4, 3]]], "two": [A, [[9]]]}


def test_runner_order_and_fair_share() -> None:
    assert order_shortest_first(RUN_CH) == ["two", "small", "big"]  # by total cells (15 < 18 < 29)
    assert fair_share_seconds(100.0, 10, 1, lo=2, hi=50) == 10.0
    assert fair_share_seconds(100.0, 10, 4, lo=2, hi=50) == 40.0
    assert fair_share_seconds(100.0, 1, 4, lo=2, hi=50) == 50.0
    assert fair_share_seconds(1.0, 100, 1, lo=2, hi=50) == 1.0   # never beyond the time left
    assert fair_share_seconds(0.0, 3) == 0.0 and fair_share_seconds(5.0, 0) == 0.0


def test_run_submission_in_process(tmp_path: Path) -> None:
    out, diag = tmp_path / "submission.json", tmp_path / "diag.json"
    cfg = RunnerConfig(total_seconds=9.0, reserve_seconds=0.5, min_task_seconds=0.5, max_task_seconds=3.0,
                       rewrite_every_s=0.0, workers=0)
    t0 = time.time()
    summary = run_submission(RUN_CH, str(out), pkg_dir=None, cfg=cfg, diagnostics_path=str(diag))
    assert time.time() - t0 < 15.0
    sub = json.loads(out.read_text(encoding="utf-8"))
    assert VS.validate_submission(sub, RUN_CH) == []
    assert summary["n_tasks"] == 3 and summary["model_loaded"] is False
    assert VS.score_submission(sub, RUN_SOL) == pytest.approx(1.0)  # flips + identity are all in reach
    assert len(json.loads(diag.read_text(encoding="utf-8"))["tasks"]) == summary["n_solved"] + summary["n_errors"]


def test_run_submission_deadline_keeps_fallbacks(tmp_path: Path) -> None:
    out = tmp_path / "submission.json"
    cfg = RunnerConfig(total_seconds=1.0, reserve_seconds=1.0, workers=0)  # no time at all
    summary = run_submission(RUN_CH, str(out), cfg=cfg, initial={"small": [{"attempt_1": [[4, 3]], "attempt_2": [[99]]}]})
    sub = json.loads(out.read_text(encoding="utf-8"))
    assert VS.validate_submission(sub, RUN_CH) == [] and summary["n_timeout"] == 3
    assert sub["small"][0]["attempt_1"] == [[4, 3]]          # valid initial attempt kept
    assert sub["small"][0]["attempt_2"] == [[0, 0]] or VS.grid_error(sub["small"][0]["attempt_2"]) is None


def test_run_submission_process_pool(tmp_path: Path) -> None:
    out = tmp_path / "submission.json"
    # per-task budgets are capped at 2 s; the global deadline only has to cover spawning the pool, which takes
    # 10-20 s when every core is busy (the run returns as soon as the three tasks are done)
    cfg = RunnerConfig(total_seconds=90.0, reserve_seconds=0.5, min_task_seconds=0.5, max_task_seconds=2.0,
                       workers=2, devices=["cpu"], threads_per_worker=1, rewrite_every_s=1.0, poll_s=0.2)
    summary = run_submission(RUN_CH, str(out), cfg=cfg)
    sub = json.loads(out.read_text(encoding="utf-8"))
    assert VS.validate_submission(sub, RUN_CH) == []
    assert summary["n_solved"] == 3, summary
    assert len({d.get("pid") for d in summary["diagnostics"].values()} - {os.getpid()}) >= 1
    assert summary["pool_restarts"] == 0 and summary["quarantined"] == []
    assert summary["model_loaded"] is False and summary["model_loaded_fraction"] == 0.0  # no package: symbolic


class _KillOnUnpickle:
    """Unpickling this object terminates the process that unpickles it: a stand-in for a worker that is
    OOM-killed / segfaults on one task (the parent only pickles it)."""

    def __reduce__(self) -> Any:
        return (os._exit, (3,))


def test_run_submission_requeues_after_worker_crash(tmp_path: Path) -> None:
    ch: Dict[str, Any] = {f"f{i}": _flip_task([[i, (i + 1) % 10, (i + 2) % 10]]) for i in range(5)}
    poison = _flip_task([[1, 2], [3, 4]])
    poison["poison"] = _KillOnUnpickle()  # kills whichever worker receives this task, every time
    ch["p"] = poison
    out = tmp_path / "submission.json"
    cfg = RunnerConfig(total_seconds=150.0, reserve_seconds=0.5, min_task_seconds=0.5, max_task_seconds=1.0,
                       workers=2, devices=["cpu"], threads_per_worker=1, rewrite_every_s=1.0, poll_s=0.1)
    t0 = time.time()
    summary = run_submission(ch, str(out), cfg=cfg)
    assert time.time() - t0 < 150.0
    sub = json.loads(out.read_text(encoding="utf-8"))
    assert VS.validate_submission(sub, ch) == []
    # the crashing task is quarantined after its second crash (it ran alone as the only suspect); every task that
    # was in flight with it was re-queued into a fresh pool and solved
    assert summary["quarantined"] == ["p"] and summary["pool_restarts"] == 2, summary
    assert "quarantined" in summary["diagnostics"]["p"]["error"]
    assert summary["n_solved"] == 5 and summary["n_timeout"] == 0 and not summary.get("unstarted")
    assert sub["p"] == fallback_attempts(poison)
    sols = {f"f{i}": [flip(ch[f"f{i}"]["test"][0]["input"])] for i in range(5)}
    assert VS.score_submission(sub, sols) == pytest.approx(1.0)


# ============================================================================================ notebook end to end

def _notebook_script(nb_path: Path, dest: Path) -> Path:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    dest.write_text(code, encoding="utf-8")
    return dest


def _run_infer(tmp_path: Path, mode: str, extra_env: Dict[str, str]) -> subprocess.CompletedProcess:
    comp = tmp_path / "comp"
    comp.mkdir(exist_ok=True)
    (comp / "arc-agi_test_challenges.json").write_text(json.dumps(RUN_CH), encoding="utf-8")
    (comp / "arc-agi_evaluation_challenges.json").write_text(json.dumps(RUN_CH), encoding="utf-8")
    (comp / "arc-agi_evaluation_solutions.json").write_text(json.dumps(RUN_SOL), encoding="utf-8")
    work = tmp_path / f"work_{mode}"
    script = _notebook_script(KAGGLE / "arc-jepa-infer.ipynb", tmp_path / f"infer_{mode}.py")
    env = {k: v for k, v in os.environ.items() if k not in ("KAGGLE_IS_COMPETITION_RERUN", "ARCJEPA_PKG")}
    # one torch thread, as the Kaggle pool workers pin theirs: with every core busy, torch's default pool of one
    # thread per core stalls the CPU model stage of the debug package past the 2 s task budget
    env.update({"ARCJEPA_WORK": str(work), "ARCJEPA_COMP_DIR": str(comp), "ARCJEPA_CODE": str(ROOT),
                "ARCJEPA_WORKERS": "0", "ARCJEPA_MAX_TASK_SECONDS": "2", "ARCJEPA_THREADS": "1"})
    env.update(extra_env)
    return subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=120,
                          cwd=str(tmp_path))


#: Global / dev budgets of the notebook end-to-end runs. Per-task budgets are capped at 2 s
#: (ARCJEPA_MAX_TASK_SECONDS) and the runs return as soon as the 3 tasks are done, so this is only a safety deadline
#: that must also cover the notebook start (code copy + imports: 10-20 s when every core is busy).
E2E_HOURS = "0.03"


def test_infer_notebook_rerun_mode_end_to_end(tmp_path: Path) -> None:
    r = _run_infer(tmp_path, "rerun", {"KAGGLE_IS_COMPETITION_RERUN": "1", "ARCJEPA_GLOBAL_HOURS": E2E_HOURS})
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    sub = json.loads((tmp_path / "work_rerun" / "submission.json").read_text(encoding="utf-8"))
    assert VS.validate_submission(sub, RUN_CH) == []
    assert "valid = True" in r.stdout
    diag = json.loads((tmp_path / "work_rerun" / "rerun_diagnostics.json").read_text(encoding="utf-8"))
    assert diag["summary"]["n_solved"] == 3
    assert VS.score_submission(sub, RUN_SOL) == pytest.approx(1.0)


def test_infer_notebook_dev_mode_end_to_end(tmp_path: Path) -> None:
    extra = {"ARCJEPA_DEV_HOURS": E2E_HOURS}
    pkg = ROOT / "runs" / "debug" / "package"
    if (pkg / "config.json").is_file():
        extra["ARCJEPA_PKG"] = str(pkg)  # exercise the neural path with the debug package when it exists
        extra["ARCJEPA_ALLOW_DEBUG_PKG"] = "1"  # the gate would (rightly) reject a debug package on Kaggle
    else:
        extra["ARCJEPA_ALLOW_SYMBOLIC"] = "1"
    r = _run_infer(tmp_path, "dev", extra)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "COMPETITION METRIC" in r.stdout and "valid = True" in r.stdout
    if "ARCJEPA_PKG" in extra:
        assert "dev gate passed" in r.stdout, r.stdout[-3000:]
    sub = json.loads((tmp_path / "work_dev" / "submission.json").read_text(encoding="utf-8"))
    assert VS.validate_submission(sub, RUN_CH) == []
    diag = json.loads((tmp_path / "work_dev" / "dev_eval_diagnostics.json").read_text(encoding="utf-8"))
    assert len(diag["tasks"]) == 3 and diag["summary"]["score"] == pytest.approx(1.0)


def test_infer_notebook_dev_gate_fails_a_broken_commit(tmp_path: Path) -> None:
    """Dev/commit mode with the code dataset missing must fail loudly (not save a fallback-only version)."""
    r = _run_infer(tmp_path, "dev", {"ARCJEPA_DEV_HOURS": E2E_HOURS, "ARCJEPA_CODE": str(tmp_path / "no_code")})
    assert r.returncode != 0, r.stdout[-3000:]
    assert "DEV GATE FAILED" in r.stderr and "code dataset" in r.stderr, r.stderr[-3000:]
    sub = json.loads((tmp_path / "work_dev" / "submission.json").read_text(encoding="utf-8"))
    assert VS.validate_submission(sub, RUN_CH) == []  # the fallback file was still written first
    # a missing package fails the gate too
    r = _run_infer(tmp_path, "dev", {"ARCJEPA_DEV_HOURS": E2E_HOURS, "ARCJEPA_PKG": str(tmp_path / "no_pkg")})
    assert r.returncode != 0 and "no trained package" in r.stderr, r.stdout[-2000:] + r.stderr[-2000:]


def test_infer_notebook_rerun_never_fails_without_code(tmp_path: Path) -> None:
    """Competition rerun with the code dataset missing: no gate, the fallback submission stands and is valid."""
    r = _run_infer(tmp_path, "rerun", {"KAGGLE_IS_COMPETITION_RERUN": "1", "ARCJEPA_GLOBAL_HOURS": E2E_HOURS,
                                       "ARCJEPA_CODE": str(tmp_path / "no_code")})
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "the fallback submission stands" in r.stdout and "valid = True" in r.stdout
    sub = json.loads((tmp_path / "work_rerun" / "submission.json").read_text(encoding="utf-8"))
    assert sub == VS.fallback_submission(RUN_CH)
