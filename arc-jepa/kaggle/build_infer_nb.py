"""Build the Kaggle inference notebook ``kaggle/arc-jepa-infer.ipynb`` + ``kaggle/infer/kernel-metadata.json``.

The notebook (kernel ``poby7722/arc-jepa-infer``; sources: code + episodes datasets, the competition, and the
training kernel ``poby7722/arc-jepa-train`` whose output holds ``arc_jepa_pkg``):

* writes fallback attempts for every test input of ``arc-agi_test_challenges.json`` to
  ``/kaggle/working/submission.json`` FIRST (identity grid + most-common demo output shape fill; pure stdlib, so
  this works even when the code dataset is missing);
* **rerun mode** (``KAGGLE_IS_COMPETITION_RERUN`` set): solves the test challenges with
  ``arcjepa.utils.kaggle_submit_runner.run_submission`` under a global 11 h budget (fair-share seconds per
  task, shortest-first, ``submission.json`` rewritten every 60 s); ``model=None`` symbolic mode when the
  package is missing;
* **dev mode** (otherwise): solves the 120 public evaluation tasks (competition files, else the episodes
  dataset's ``tasks/eval_public.jsonl``), scores them with ``arcjepa.eval.evaluate`` and prints the competition
  metric; the test challenges' submission reuses the dev attempts for overlapping ids;
* validates the final file with the logic of ``kaggle/validate_submission.py`` (inlined) and repairs any
  malformed entry from the fallbacks;
* **dev gate** (dev / commit runs only, never in a competition rerun): the last cell raises when the code dataset
  is missing or does not import, no trained package is found, the package is a debug-config (smoke) package, or
  no solver process could load it, so a broken version fails its commit and cannot be submitted. In rerun mode
  the notebook never fails on these (the fallback / symbolic submission stands).

Usage: ``python kaggle/build_infer_nb.py [--out-dir kaggle]``.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

KAGGLE_DIR = Path(__file__).resolve().parent


def _load_train_builder() -> Any:
    spec = importlib.util.spec_from_file_location("arcjepa_build_train_nb", KAGGLE_DIR / "build_train_nb.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_tb = _load_train_builder()
code_cell, markdown_cell, notebook = _tb.code_cell, _tb.markdown_cell, _tb.notebook
kernel_metadata, write_json = _tb.kernel_metadata, _tb.write_json
OWNER = _tb.OWNER
TRAIN_KERNEL = f"{OWNER}/{_tb.TRAIN_SLUG}"
INFER_SLUG = "arc-jepa-infer"
INFER_TITLE = "ARC JEPA Infer"
INFER_NOTEBOOK = f"{INFER_SLUG}.ipynb"
INLINE_BEGIN = "# ---- BEGIN INLINE"
INLINE_END = "# ---- END INLINE ----"


def inline_validator_source(path: Optional[Path] = None) -> str:
    """The stdlib block of ``validate_submission.py`` between the INLINE markers (embedded in the notebook)."""
    text = (path or KAGGLE_DIR / "validate_submission.py").read_text(encoding="utf-8")
    start = text.index(INLINE_BEGIN)
    start = text.index("\n", start) + 1
    end = text.index(INLINE_END)
    return "# ---- submission validation + fallbacks (inlined from kaggle/validate_submission.py)\n" + text[start:end]


INFER_HEADER = """
# ARC-JEPA: inference (neural-guided program search -> submission.json)

Loads the offline package `arc_jepa_pkg` exported by `poby7722/arc-jepa-train` (or runs the pure symbolic
search when it is missing), solves every task with `arcjepa.search.solver.solve_task` and writes
`submission.json` (two attempts per test input). Fallback attempts are written before any search and the file is
rewritten every 60 s, so a valid submission always exists.

* Competition rerun (`KAGGLE_IS_COMPETITION_RERUN`): `arc-agi_test_challenges.json`, global budget
  `ARCJEPA_GLOBAL_HOURS` (default 11 h), fair-share seconds per task, shortest task first.
* Interactive / commit run: dev mode on the 120 public evaluation tasks (`ARCJEPA_DEV_HOURS`, default 1 h,
  `ARCJEPA_DEV_MAX_TASKS` to cap), scored with the competition metric. The last cell is a **gate**: it raises
  (the commit fails and cannot be submitted) when the code dataset is missing, the trained package is not found
  or is a debug/smoke package, or no solver process could load it. A competition rerun never fails on these.

Other switches: `ARCJEPA_WORKERS` (default `auto`), `ARCJEPA_THREADS` (torch threads per solver process, default
auto = CPUs / workers), `ARCJEPA_PKG` (explicit package folder),
`ARCJEPA_MAX_TASK_SECONDS` (default `auto` = max(1800, 2 x budget x workers / tasks)); local testing only:
`ARCJEPA_ALLOW_SYMBOLIC=1`, `ARCJEPA_ALLOW_DEBUG_PKG=1` (the gate reports instead of raising).
"""

INFER_SETUP_SRC = r'''
import glob
import json
import os
import shutil
import sys
import time
import traceback

T0 = time.time()
IS_RERUN = bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN"))
GLOBAL_BUDGET_S = float(os.environ.get("ARCJEPA_GLOBAL_HOURS", "11")) * 3600.0
DEV_HOURS = float(os.environ.get("ARCJEPA_DEV_HOURS", "1.0"))
DEV_MAX_TASKS = int(os.environ.get("ARCJEPA_DEV_MAX_TASKS", "0") or 0)
MAX_TASK_ENV = os.environ.get("ARCJEPA_MAX_TASK_SECONDS", "auto")
ALLOW_SYMBOLIC = os.environ.get("ARCJEPA_ALLOW_SYMBOLIC", "0") == "1"    # local tests only (Kaggle sets no env)
ALLOW_DEBUG_PKG = os.environ.get("ARCJEPA_ALLOW_DEBUG_PKG", "0") == "1"  # local tests only
WORKERS_ENV = os.environ.get("ARCJEPA_WORKERS", "auto")
THREADS = int(os.environ.get("ARCJEPA_THREADS", "0") or 0) or None  # torch threads per solver process (auto)
WORK = os.environ.get("ARCJEPA_WORK", "/kaggle/working")
SUBMISSION = os.path.join(WORK, "submission.json")
COMP_DIRS = [d for d in (os.environ.get("ARCJEPA_COMP_DIR", ""),
                         "/kaggle/input/competitions/arc-prize-2026-arc-agi-2",
                         "/kaggle/input/arc-prize-2026-arc-agi-2") if d]
RESERVE_S = min(600.0, 0.05 * GLOBAL_BUDGET_S)  # kept for the final write + validation
os.makedirs(WORK, exist_ok=True)
print("mode:", "COMPETITION RERUN" if IS_RERUN else "dev (public evaluation)")
'''

INFER_LOAD_SRC = r'''
# ---- challenge files; fallback submission.json is written FIRST


def find_comp_file(name):
    for d in COMP_DIRS:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    hits = sorted(glob.glob(os.path.join("/kaggle/input", "*", name)) + glob.glob(os.path.join("/kaggle/input", "*", "*", name)))
    return hits[0] if hits else None


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


TEST_PATH = find_comp_file("arc-agi_test_challenges.json")
test_challenges = load_json(TEST_PATH) if TEST_PATH else {}
if test_challenges:
    write_json_atomic(fallback_submission(test_challenges), SUBMISSION)
    print("fallback submission written for %d test tasks (%d test inputs)" % (
        len(test_challenges), sum(len(t.get("test", [])) for t in test_challenges.values())))
else:
    print("WARNING: arc-agi_test_challenges.json not found under", COMP_DIRS)
'''

#: locate the code dataset, copy it to /kaggle/working/arc-jepa (falling back to the read-only mount when the copy
#: fails: this cell runs after the fallback submission.json exists, so it must never throw) and import it
INFER_CODE_SRC = r'''
# ---- locate the ARC-JEPA code dataset, copy it to /kaggle/working/arc-jepa, put it on sys.path
CODE_CANDIDATES = [os.environ.get("ARCJEPA_CODE", ""), "/kaggle/input/datasets/poby7722/arc-jepa-code",
                   "/kaggle/input/arc-jepa-code"]
CODE_DST = os.path.join(WORK, "arc-jepa")


def find_code_root(cands):
    """First folder holding arcjepa/__init__.py: the candidates (up to two levels deep), then /kaggle/input."""
    for base in [c for c in cands if c and os.path.isdir(c)]:
        for pat in ("", "*", "*/*"):
            for d in (sorted(glob.glob(os.path.join(base, pat))) if pat else [base]):
                if os.path.isfile(os.path.join(d, "arcjepa", "__init__.py")):
                    return d
    hits = sorted(glob.glob("/kaggle/input/**/arcjepa/__init__.py", recursive=True))
    return os.path.dirname(os.path.dirname(hits[0])) if hits else None


CODE_SRC = None
CODE_PATH = None
CODE_OK = False
try:
    CODE_SRC = find_code_root(CODE_CANDIDATES)
    if CODE_SRC:
        CODE_PATH = CODE_DST
        if os.path.abspath(CODE_SRC) != os.path.abspath(CODE_DST):
            try:
                shutil.copytree(CODE_SRC, CODE_DST, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "runs", ".git"))
            except Exception as exc:  # importing from the read-only mount works too
                print("copying the code failed (%r); importing it from %s" % (exc, CODE_SRC))
                CODE_PATH = CODE_SRC
        if CODE_PATH not in sys.path:
            sys.path.insert(0, CODE_PATH)
        os.environ["PYTHONPATH"] = CODE_PATH + os.pathsep + os.environ.get("PYTHONPATH", "")  # spawn workers
        import arcjepa  # noqa: F401

        CODE_OK = True
except Exception:
    traceback.print_exc()
print("code source:", CODE_SRC, "-> imported from", CODE_PATH, "| import ok:", CODE_OK)

# ---- the episodes dataset (task mirror; dev mode reads tasks/eval_public.jsonl when the competition files lack it)
EPISODES_ROOT = None
for base in [os.environ.get("ARCJEPA_DATA", ""), "/kaggle/input/datasets/poby7722/arc-agi-2-jepa-episodes",
             "/kaggle/input/arc-agi-2-jepa-episodes"]:
    if not base or not os.path.isdir(base):
        continue
    hits = sorted(glob.glob(os.path.join(base, "tasks", "train.jsonl")) + glob.glob(os.path.join(base, "*", "tasks", "train.jsonl")))
    if hits:
        EPISODES_ROOT = os.path.dirname(os.path.dirname(hits[0]))
        break
if EPISODES_ROOT:
    os.environ["ARCJEPA_DATA"] = EPISODES_ROOT
print("episodes dataset:", EPISODES_ROOT)
'''

INFER_PKG_SRC = r'''
# ---- model package (kernel output of poby7722/arc-jepa-train, mounted under /kaggle/input); symbolic mode when
# missing. Explicit mounts first, then a walk for arc_jepa_pkg folders (config.json + model.safetensors).
PKG_DIR = None
PKG_INFO = {}
RUNNER_OK = False
if CODE_OK:
    try:
        from arcjepa.utils.kaggle_submit_runner import (RunnerConfig, find_package, load_model_package,
                                                        package_info, run_submission)

        RUNNER_OK = True
        PKG_DIR = find_package([os.environ.get("ARCJEPA_PKG", ""),
                                "/kaggle/input/notebooks/poby7722/arc-jepa-train/arc_jepa_pkg",
                                "/kaggle/input/arc-jepa-train/arc_jepa_pkg",
                                "/kaggle/input/kernels/poby7722/arc-jepa-train/arc_jepa_pkg"],
                               search_roots=["/kaggle/input"])
        PKG_INFO = package_info(PKG_DIR) if PKG_DIR else {}
    except Exception:
        traceback.print_exc()
try:
    import torch

    N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
except Exception:
    N_GPU = 0
N_CPU = os.cpu_count() or 1
if WORKERS_ENV == "auto":
    N_WORKERS = max(1, min(16, N_CPU - 1))
else:
    N_WORKERS = max(0, int(WORKERS_ENV))
if N_WORKERS > 0 and "__file__" in globals():
    # executed as a plain script (local simulation only): spawn workers would re-run this unguarded script.
    # Kaggle runs the notebook in a Jupyter kernel (no __file__), where the worker pool is used.
    print("script mode: solving in-process instead of %d spawn workers" % N_WORKERS)
    N_WORKERS = 0
DEVICES = ["cuda:%d" % (i % N_GPU) for i in range(max(1, N_WORKERS))] if (N_GPU and PKG_DIR) else None


def task_cap_seconds(budget_s, n_tasks):
    """Per-task budget cap: ARCJEPA_MAX_TASK_SECONDS, or auto = max(1800, 2 x budget x workers / tasks) so the
    cap never leaves the global budget idle when there are many workers."""
    if MAX_TASK_ENV != "auto":
        return float(MAX_TASK_ENV)
    return max(1800.0, 2.0 * budget_s * max(1, N_WORKERS) / max(1, n_tasks))


print("code:", CODE_OK, "| runner:", RUNNER_OK, "| package:", PKG_DIR or "none (symbolic mode)",
      "| GPUs", N_GPU, "| CPUs", N_CPU, "| workers", N_WORKERS)
if PKG_DIR:
    _c = PKG_INFO.get("created_unix")
    print("package created", time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(_c)) if _c else "?",
          "| parameters", PKG_INFO.get("n_parameters"), "| training config", PKG_INFO.get("config_path"))
    if PKG_INFO.get("debug"):
        print("WARNING: this is a DEBUG (smoke) package: the training kernel did not run the full config")
'''

INFER_RERUN_SRC = r'''
# ---- competition rerun: solve the hidden test tasks within the global budget
RERUN_SUMMARY = None
if IS_RERUN and RUNNER_OK and test_challenges:
    try:
        cfg = RunnerConfig(total_seconds=GLOBAL_BUDGET_S, reserve_seconds=RESERVE_S, min_task_seconds=2.0,
                           max_task_seconds=task_cap_seconds(GLOBAL_BUDGET_S, len(test_challenges)),
                           rewrite_every_s=60.0, workers=N_WORKERS, devices=DEVICES, threads_per_worker=THREADS)
        RERUN_SUMMARY = run_submission(test_challenges, SUBMISSION, pkg_dir=PKG_DIR, cfg=cfg,
                                       initial=load_json(SUBMISSION), start_time=T0,
                                       diagnostics_path=os.path.join(WORK, "rerun_diagnostics.json"))
        print({k: v for k, v in RERUN_SUMMARY.items() if k not in ("diagnostics", "unstarted")})
        if PKG_DIR and not RERUN_SUMMARY.get("model_loaded"):
            print("WARNING: package %s was found but no worker loaded it: symbolic search only" % PKG_DIR)
    except Exception:
        traceback.print_exc()
elif IS_RERUN:
    print("rerun without a working solver: the fallback submission stands")
'''

INFER_DEV_SRC = r'''
# ---- dev mode: the 120 public evaluation tasks, scored with the competition metric
DEV_RESULT = None
DEV_SUMMARY = None
if not IS_RERUN:
    eval_ch, eval_sol = {}, {}
    ch_path = find_comp_file("arc-agi_evaluation_challenges.json")
    sol_path = find_comp_file("arc-agi_evaluation_solutions.json")
    if ch_path and sol_path:
        eval_ch, eval_sol = load_json(ch_path), load_json(sol_path)
    elif EPISODES_ROOT and os.path.isfile(os.path.join(EPISODES_ROOT, "tasks", "eval_public.jsonl")):
        with open(os.path.join(EPISODES_ROOT, "tasks", "eval_public.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    eval_ch[row["task_id"]] = {"train": row["train"], "test": [{"input": p["input"]} for p in row["test"]]}
                    eval_sol[row["task_id"]] = [p["output"] for p in row["test"]]
    ids = sorted(eval_ch)[:DEV_MAX_TASKS] if DEV_MAX_TASKS else sorted(eval_ch)
    eval_ch = {t: eval_ch[t] for t in ids}
    eval_sol = {t: eval_sol[t] for t in ids if t in eval_sol}
    print("dev tasks:", len(eval_ch))
    dev_sub_path = os.path.join(WORK, "dev_eval_submission.json")
    if eval_ch and RUNNER_OK:
        try:
            cfg = RunnerConfig(total_seconds=DEV_HOURS * 3600.0, reserve_seconds=min(30.0, 0.05 * DEV_HOURS * 3600.0),
                               min_task_seconds=2.0,
                               max_task_seconds=task_cap_seconds(DEV_HOURS * 3600.0, len(eval_ch)),
                               rewrite_every_s=60.0, workers=N_WORKERS, devices=DEVICES, threads_per_worker=THREADS)
            dev_summary = run_submission(eval_ch, dev_sub_path, pkg_dir=PKG_DIR, cfg=cfg, start_time=time.time(),
                                         diagnostics_path=os.path.join(WORK, "dev_run_diagnostics.json"))
            DEV_SUMMARY = {k: v for k, v in dev_summary.items() if k != "diagnostics"}
            print("dev run: model loaded by %s of the workers | pool restarts %d | quarantined %s" % (
                dev_summary.get("model_loaded_fraction"), dev_summary.get("pool_restarts", 0),
                dev_summary.get("quarantined")))
            dev_sub = load_json(dev_sub_path)
            from arcjepa.core.types import task_from_json
            from arcjepa.eval.evaluate import competition_score, evaluate, replay_solver

            tasks = {}
            for t, d in eval_ch.items():
                tests = [{"input": tp["input"], "output": o} for tp, o in zip(d["test"], eval_sol.get(t, []))]
                tasks[t] = task_from_json(t, {"train": d["train"], "test": tests})
            DEV_RESULT = evaluate(tasks, replay_solver(dev_sub, dev_summary["diagnostics"]),
                                  diagnostics_path=os.path.join(WORK, "dev_eval_diagnostics.json"))
            print("COMPETITION METRIC (public eval, %d tasks): %.4f  [cross-check %.4f]" % (
                DEV_RESULT["n_tasks"], DEV_RESULT["score"], competition_score(dev_sub, eval_sol)))
            print("outputs solved: %d / %d | solver errors %d | exact programs found on %d tasks" % (
                DEV_RESULT["n_correct_outputs"], DEV_RESULT["n_test_outputs"], dev_summary["n_errors"],
                dev_summary["n_exact_found"]))
            print("per family:", json.dumps(DEV_RESULT["per_family"]))
            print("near misses:", json.dumps({k: v for k, v in DEV_RESULT["error_analysis"].items()
                                              if k.startswith("near_miss_") and k != "near_miss_task_ids"}))
            if test_challenges:  # reuse the dev attempts for overlapping ids
                write_json_atomic(repair_submission(dev_sub, test_challenges), SUBMISSION)
        except Exception:
            traceback.print_exc()
    elif eval_ch:
        fb = fallback_submission(eval_ch)
        print("no solver available; fallback-only metric: %.4f" % score_submission(fb, eval_sol))
'''

INFER_VALIDATE_SRC = r'''
# ---- final validation of submission.json (repairs malformed entries from the fallbacks)
if test_challenges:
    final = load_json(SUBMISSION) if os.path.isfile(SUBMISSION) else {}
    errs = validate_submission(final, test_challenges)
    if errs:
        print("repairing %d problems, e.g." % len(errs), errs[:5])
        write_json_atomic(repair_submission(final, test_challenges), SUBMISSION)
        errs = validate_submission(load_json(SUBMISSION), test_challenges)
    print("submission.json: %d tasks, valid = %s, %.1f min elapsed" % (
        len(test_challenges), not errs, (time.time() - T0) / 60.0))
    if errs:
        raise RuntimeError("submission.json is invalid: %s" % errs[:5])
else:
    print("no test challenges found; nothing to validate")
'''

INFER_GATE_SRC = r'''
# ---- dev / commit gate: a broken setup fails this run loudly, so the version cannot be submitted.
# Never active in a competition rerun (there the fallback / symbolic submission above stands).
GATE_PROBLEMS = []
if not IS_RERUN:
    if not CODE_OK:
        GATE_PROBLEMS.append("the arcjepa code dataset was not found or does not import (looked in %s)"
                             % [c for c in CODE_CANDIDATES if c])
    elif not RUNNER_OK:
        GATE_PROBLEMS.append("arcjepa.utils.kaggle_submit_runner does not import")
    if RUNNER_OK and not PKG_DIR:
        GATE_PROBLEMS.append("no trained package (arc_jepa_pkg: config.json + model.safetensors) under /kaggle/input;"
                             " is the poby7722/arc-jepa-train output attached and complete?")
    if PKG_DIR and PKG_INFO.get("debug") and not ALLOW_DEBUG_PKG:
        GATE_PROBLEMS.append("the package %s was trained with the debug config (%s parameters): the training kernel"
                             " ran in smoke mode" % (PKG_DIR, PKG_INFO.get("n_parameters")))
    if PKG_DIR and RUNNER_OK:
        frac = DEV_SUMMARY.get("model_loaded_fraction") if DEV_SUMMARY else None
        loaded = None if frac is None else frac > 0
        if loaded is None:  # no dev run / no task finished: try one load here
            try:
                loaded = load_model_package(PKG_DIR, "cpu") is not None
            except Exception:
                traceback.print_exc()
                loaded = False
        if not loaded:
            GATE_PROBLEMS.append("no solver process could load the package %s (see the warnings above)" % PKG_DIR)
    if GATE_PROBLEMS:
        GATE_MSG = "DEV GATE FAILED (this version must not be submitted):\n - " + "\n - ".join(GATE_PROBLEMS)
        if ALLOW_SYMBOLIC:
            print(GATE_MSG + "\n(ARCJEPA_ALLOW_SYMBOLIC=1: reported, not raised)")
        else:
            raise RuntimeError(GATE_MSG)
    else:
        print("dev gate passed: code OK, package %s (%s parameters) loaded" % (PKG_DIR, PKG_INFO.get("n_parameters")))
'''


def build_infer_notebook() -> Dict[str, Any]:
    """The inference notebook as nbformat JSON."""
    return notebook([
        markdown_cell("infer-00-header", INFER_HEADER),
        code_cell("infer-01-setup", INFER_SETUP_SRC),
        code_cell("infer-02-validator", inline_validator_source()),
        code_cell("infer-03-fallback", INFER_LOAD_SRC),
        code_cell("infer-04-code", INFER_CODE_SRC),
        code_cell("infer-05-package", INFER_PKG_SRC),
        code_cell("infer-06-rerun", INFER_RERUN_SRC),
        code_cell("infer-07-dev", INFER_DEV_SRC),
        code_cell("infer-08-validate", INFER_VALIDATE_SRC),
        code_cell("infer-09-dev-gate", INFER_GATE_SRC),
    ])


def build(out_dir: Optional[Path] = None) -> Dict[str, str]:
    """Write ``<out_dir>/arc-jepa-infer.ipynb``, ``<out_dir>/infer/kernel-metadata.json`` and the notebook copy
    ``<out_dir>/infer/arc-jepa-infer.ipynb``; returns the written paths."""
    out = Path(out_dir) if out_dir else KAGGLE_DIR
    nb = build_infer_notebook()
    meta = kernel_metadata(INFER_SLUG, INFER_TITLE, INFER_NOTEBOOK, kernel_sources=[TRAIN_KERNEL])
    return {
        "notebook": write_json(nb, out / INFER_NOTEBOOK),
        "kernel_notebook": write_json(nb, out / "infer" / INFER_NOTEBOOK),
        "metadata": write_json(meta, out / "infer" / "kernel-metadata.json"),
    }


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Build the ARC-JEPA Kaggle inference notebook.")
    ap.add_argument("--out-dir", default=None, help="output folder (default: this kaggle/ folder)")
    a = ap.parse_args(argv)
    return build(Path(a.out_dir) if a.out_dir else None)


if __name__ == "__main__":
    for k, v in main().items():
        print(f"{k}: {v}")
