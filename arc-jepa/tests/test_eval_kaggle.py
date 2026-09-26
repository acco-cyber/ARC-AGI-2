"""Tests for arcjepa/eval, kaggle/ (notebook builders, submission validator) and the Kaggle submission runner."""
from __future__ import annotations

import importlib.util
import json
import os
import py_compile
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
    assert fb["t1"][0]["attempt_1"] == B                     # identity
    assert fb["t2"][1]["attempt_2"] == [[5, 5, 5]]           # most common demo output shape + colour
    assert {t: fallback_attempts(d) for t, d in CHALLENGES.items()} == fb  # runner and inline agree
    rep = VS.repair_submission(_mutate("value_10"), CHALLENGES)
    assert VS.validate_submission(rep, CHALLENGES) == []
    assert rep["t1"][0]["attempt_1"] == B and rep["t1"][0]["attempt_2"] == A
    assert VS.validate_submission(VS.repair_submission(None, CHALLENGES), CHALLENGES) == []


# ============================================================================================ notebooks

def _check_notebook(path: Path, tmp_path: Path) -> Dict[str, Any]:
    nbformat = pytest.importorskip("nbformat")
    nb = nbformat.read(str(path), as_version=4)
    nbformat.validate(nb)
    ids = [c["id"] for c in nb.cells]
    assert len(ids) == len(set(ids)) and all(ids)
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        assert not any(line.lstrip().startswith(("!", "%")) for line in c.source.splitlines())
        f = tmp_path / f"{path.stem}-{c['id']}.py"
        f.write_text(c.source, encoding="utf-8")
        py_compile.compile(str(f), cfile=str(f) + "c", doraise=True)
    return json.loads(path.read_text(encoding="utf-8"))


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
        assert m["competition_sources"] == ["arc-prize-2026-arc-agi-2"]
        assert m["dataset_sources"] == ["poby7722/arc-jepa-code", "poby7722/arc-agi-2-jepa-episodes"]
    assert mt["id"] == "poby7722/arc-jepa-train" and mt["code_file"] == "arc-jepa-train.ipynb"
    assert mt["kernel_sources"] == []
    assert mi["id"] == "poby7722/arc-jepa-infer" and mi["code_file"] == "arc-jepa-infer.ipynb"
    assert mi["kernel_sources"] == ["poby7722/arc-jepa-train"]
    # the committed notebooks are up to date with the builders
    for rel in ("arc-jepa-train.ipynb", "train/kernel-metadata.json", "train/arc-jepa-train.ipynb",
                "arc-jepa-infer.ipynb", "infer/kernel-metadata.json", "infer/arc-jepa-infer.ipynb"):
        assert (KAGGLE / rel).is_file(), rel
        assert (KAGGLE / rel).read_bytes() == (out / rel).read_bytes(), f"{rel} is stale: rerun the builders"
    src = "".join(json.loads((out / "arc-jepa-train.ipynb").read_text(encoding="utf-8"))["cells"][5]["source"])
    assert "arcjepa.synthetic.dataset" in src
    isrc = json.dumps(json.loads((out / "arc-jepa-infer.ipynb").read_text(encoding="utf-8")))
    for needle in ("KAGGLE_IS_COMPETITION_RERUN", "arc-agi_test_challenges.json", "fallback_submission",
                   "validate_submission", "run_submission", "arc-agi_evaluation_challenges.json"):
        assert needle in isrc, needle


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
    cfg = RunnerConfig(total_seconds=25.0, reserve_seconds=0.5, min_task_seconds=0.5, max_task_seconds=2.0,
                       workers=2, devices=["cpu"], threads_per_worker=1, rewrite_every_s=1.0, poll_s=0.2)
    summary = run_submission(RUN_CH, str(out), cfg=cfg)
    sub = json.loads(out.read_text(encoding="utf-8"))
    assert VS.validate_submission(sub, RUN_CH) == []
    assert summary["n_solved"] == 3, summary
    assert len({d.get("pid") for d in summary["diagnostics"].values()} - {os.getpid()}) >= 1


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
    env.update({"ARCJEPA_WORK": str(work), "ARCJEPA_COMP_DIR": str(comp), "ARCJEPA_CODE": str(ROOT),
                "ARCJEPA_WORKERS": "0", "ARCJEPA_MAX_TASK_SECONDS": "2"})
    env.update(extra_env)
    return subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=120,
                          cwd=str(tmp_path))


def test_infer_notebook_rerun_mode_end_to_end(tmp_path: Path) -> None:
    r = _run_infer(tmp_path, "rerun", {"KAGGLE_IS_COMPETITION_RERUN": "1", "ARCJEPA_GLOBAL_HOURS": "0.0025"})
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    sub = json.loads((tmp_path / "work_rerun" / "submission.json").read_text(encoding="utf-8"))
    assert VS.validate_submission(sub, RUN_CH) == []
    assert "valid = True" in r.stdout
    diag = json.loads((tmp_path / "work_rerun" / "rerun_diagnostics.json").read_text(encoding="utf-8"))
    assert diag["summary"]["n_solved"] == 3
    assert VS.score_submission(sub, RUN_SOL) == pytest.approx(1.0)


def test_infer_notebook_dev_mode_end_to_end(tmp_path: Path) -> None:
    extra = {"ARCJEPA_DEV_HOURS": "0.0025"}
    pkg = ROOT / "runs" / "debug" / "package"
    if (pkg / "config.json").is_file():
        extra["ARCJEPA_PKG"] = str(pkg)  # exercise the neural path with the debug package when it exists
    r = _run_infer(tmp_path, "dev", extra)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "COMPETITION METRIC" in r.stdout and "valid = True" in r.stdout
    sub = json.loads((tmp_path / "work_dev" / "submission.json").read_text(encoding="utf-8"))
    assert VS.validate_submission(sub, RUN_CH) == []
    diag = json.loads((tmp_path / "work_dev" / "dev_eval_diagnostics.json").read_text(encoding="utf-8"))
    assert len(diag["tasks"]) == 3 and diag["summary"]["score"] == pytest.approx(1.0)
