"""Build the Kaggle training notebook ``kaggle/arc-jepa-train.ipynb`` + ``kaggle/train/kernel-metadata.json``.

The notebook (kernel ``poby7722/arc-jepa-train``, NvidiaL4 x 4, no internet, pinned image):

1. copies the ARC-JEPA code from the code dataset (``/kaggle/input/datasets/poby7722/arc-jepa-code``, also
   probing ``/kaggle/input/arc-jepa-code``) to ``/kaggle/working/arc-jepa`` and puts it on ``sys.path``;
2. generates the synthetic phase-1 tasks in-session (``python -m arcjepa.synthetic.dataset``, 4 CPU workers,
   target 100k within 30 min: a 2k calibration run measures throughput and the target is reduced when the
   rate is too low);
3. runs ``arcjepa.training.train_all`` time-boxed by ``ARCJEPA_TRAIN_HOURS`` (default 9.5) with ``torchrun``
   over all GPUs, falling back to a single process (which resumes the torchrun checkpoints);
4. makes sure the package is exported to ``/kaggle/working/arc_jepa_pkg`` and prints a summary.

``ARCJEPA_SMOKE=1`` (the default build, for the first push) switches to ``configs/debug.yaml``, 200 synthetic tasks
and a few minutes of single-process training.

Usage: ``python kaggle/build_train_nb.py [--out-dir kaggle]``. The notebook is also copied next to its metadata
(``kaggle/train/arc-jepa-train.ipynb``) so ``kaggle/train`` is a pushable kernel folder.

``python kaggle/build_train_nb.py --full`` writes the FULL-mode kernel folder ``kaggle/train_full/`` (same kernel id
``poby7722/arc-jepa-train`` and metadata) whose notebook defaults to ``ARCJEPA_SMOKE=0`` and
``ARCJEPA_TRAIN_HOURS=7.5`` (Kaggle cannot set environment variables): v1 model, torchrun on 4 x L4, every step
capped at ``ARCJEPA_WALL_HOURS`` (default train hours + 1 = 8.5 h).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

KAGGLE_DIR = Path(__file__).resolve().parent
OWNER = "poby7722"
DOCKER_IMAGE = ("gcr.io/kaggle-private-byod/python@sha256:"
                "320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868")
MACHINE_SHAPE = "NvidiaL4"
COMPETITION = "arc-prize-2026-arc-agi-2"
CODE_DATASET = f"{OWNER}/arc-jepa-code"
EPISODES_DATASET = f"{OWNER}/arc-agi-2-jepa-episodes"
DATASET_SOURCES = [CODE_DATASET, EPISODES_DATASET]
TRAIN_SLUG = "arc-jepa-train"
TRAIN_TITLE = "ARC JEPA Train"
TRAIN_NOTEBOOK = f"{TRAIN_SLUG}.ipynb"


# ============================================================================================ notebook helpers

def _lines(src: str) -> List[str]:
    src = src.strip("\n") + "\n"
    lines = src.splitlines(keepends=True)
    lines[-1] = lines[-1].rstrip("\n")
    return lines


def code_cell(cell_id: str, source: str) -> Dict[str, Any]:
    """nbformat 4.5 code cell with a fixed id."""
    return {"cell_type": "code", "execution_count": None, "id": cell_id, "metadata": {}, "outputs": [],
            "source": _lines(source)}


def markdown_cell(cell_id: str, source: str) -> Dict[str, Any]:
    """nbformat 4.5 markdown cell with a fixed id."""
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": _lines(source)}


def notebook(cells: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Plain nbformat 4.5 notebook JSON (Python 3 kernel)."""
    return {
        "cells": list(cells),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def kernel_metadata(slug: str, title: str, code_file: str, *, kernel_sources: Sequence[str] = ()) -> Dict[str, Any]:
    """Kaggle ``kernel-metadata.json`` for an ARC-JEPA kernel (GPU L4, no internet, pinned image)."""
    return {
        "id": f"{OWNER}/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": list(DATASET_SOURCES),
        "kernel_sources": list(kernel_sources),
        "competition_sources": [COMPETITION],
        "model_sources": [],
        "docker_image": DOCKER_IMAGE,
        "machine_shape": MACHINE_SHAPE,
    }


def write_json(obj: Any, path: Path) -> str:
    """Write ``obj`` as indented UTF-8 JSON (LF newlines, trailing newline); returns the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    return str(path)


# ============================================================================================ shared cells

#: locate the code dataset, copy it to /kaggle/working/arc-jepa and import it (shared by both notebooks)
CODE_SETUP_SRC = r'''
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


CODE_SRC = find_code_root(CODE_CANDIDATES)
CODE_OK = False
if CODE_SRC:
    if os.path.abspath(CODE_SRC) != os.path.abspath(CODE_DST):
        shutil.copytree(CODE_SRC, CODE_DST, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "runs", ".git"))
    if CODE_DST not in sys.path:
        sys.path.insert(0, CODE_DST)
    os.environ["PYTHONPATH"] = CODE_DST + os.pathsep + os.environ.get("PYTHONPATH", "")
    try:
        import arcjepa  # noqa: F401
        CODE_OK = True
    except Exception as exc:  # the caller decides how to degrade
        print("arcjepa import failed:", repr(exc))
print("code source:", CODE_SRC, "-> copied to", CODE_DST, "| import ok:", CODE_OK)

# ---- the episodes dataset (task mirror), probed like arcjepa.data.hf_loader does
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

#: run a subprocess with streamed output and a hard wall-clock limit (kills only its own process group)
RUN_CMD_SRC = r'''
import signal
import subprocess
import threading


def run_cmd(cmd, timeout_s, log_path=None, cwd=None):
    """Run cmd, stream its output (and append it to log_path), kill its own process group after timeout_s.
    Returns the exit code (None when it was stopped by the timeout)."""
    print("$", " ".join(str(c) for c in cmd), "| timeout %.0fs" % timeout_s, flush=True)
    popen_kw = {"start_new_session": True} if os.name == "posix" else {}
    proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, cwd=cwd or CODE_DST, env=os.environ.copy(), **popen_kw)
    log_fh = open(log_path, "a", encoding="utf-8") if log_path else None

    def pump():
        for line in proc.stdout:
            sys.stdout.write(line)
            if log_fh:
                log_fh.write(line)
        sys.stdout.flush()

    th = threading.Thread(target=pump, daemon=True)
    th.start()
    try:
        rc = proc.wait(timeout=max(1.0, timeout_s))
    except subprocess.TimeoutExpired:
        print("timeout: stopping", cmd[:4], flush=True)
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGTERM)  # our own child's process group, never by name
            else:
                proc.terminate()
            proc.wait(timeout=60)
        except Exception:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        rc = None
    th.join(timeout=10)
    if log_fh:
        log_fh.close()
    print("exit code:", rc, flush=True)
    return rc
'''


# ============================================================================================ train notebook

TRAIN_HEADER = """
# ARC-JEPA: training (synthetic generation, stages A-D, package export)

Object-centric transformation JEPA with latent-guided program search for ARC-AGI-2. This kernel generates the
synthetic phase-1 tasks in-session, runs `arcjepa.training.train_all` (stages A->B->C->D, time-boxed) on the four
L4 GPUs with `torchrun`, and exports the offline package to `/kaggle/working/arc_jepa_pkg`, which the inference
kernel `poby7722/arc-jepa-infer` mounts via `kernel_sources`.

Build mode: **__MODE__**. Environment switches (Kaggle cannot set them, so the build bakes the defaults):
`ARCJEPA_SMOKE` (default `__SMOKE_DEFAULT__`; `1` = debug config, 200 tasks, a few minutes),
`ARCJEPA_TRAIN_HOURS` (default `__TRAIN_HOURS_DEFAULT__`, measured from the notebook start),
`ARCJEPA_WALL_HOURS` (default train hours + 1: hard cap for every step of the notebook),
`ARCJEPA_SYNTH_TASKS` (default `100000`), `ARCJEPA_SYNTH_WORKERS` (default `4`),
`ARCJEPA_SYNTH_MAX_MINUTES` (default `30`).
"""

TRAIN_SETUP_SRC = r'''
import glob
import json
import os
import shutil
import sys
import time

T0 = time.time()
SMOKE = os.environ.get("ARCJEPA_SMOKE", "__SMOKE_DEFAULT__") == "1"
TRAIN_HOURS = float(os.environ.get("ARCJEPA_TRAIN_HOURS", "__TRAIN_HOURS_DEFAULT__"))
WALL_HOURS = float(os.environ.get("ARCJEPA_WALL_HOURS", str(TRAIN_HOURS + 1.0)))  # hard cap for the whole notebook
SMOKE_HOURS = float(os.environ.get("ARCJEPA_SMOKE_HOURS", "0.05"))
SYNTH_TARGET = int(os.environ.get("ARCJEPA_SYNTH_TASKS", "100000"))
SYNTH_WORKERS = int(os.environ.get("ARCJEPA_SYNTH_WORKERS", "4"))
SYNTH_MAX_S = float(os.environ.get("ARCJEPA_SYNTH_MAX_MINUTES", "30")) * 60.0
SYNTH_SEED = int(os.environ.get("ARCJEPA_SYNTH_SEED", "1"))
KEEP_SYNTH = os.environ.get("ARCJEPA_KEEP_SYNTH", "0") == "1"
WORK = os.environ.get("ARCJEPA_WORK", "/kaggle/working")
RUN_DIR = os.path.join(WORK, "run")
PKG_DIR = os.path.join(WORK, "arc_jepa_pkg")
SYNTH_PATH = os.path.join(WORK, "synthetic_phase1.jsonl")
LOG_PATH = os.path.join(WORK, "train_stdout.log")
os.makedirs(WORK, exist_ok=True)


def wall_left():
    """Seconds left before the notebook's hard wall-clock cap (WALL_HOURS after T0)."""
    return WALL_HOURS * 3600.0 - (time.time() - T0)


print("SMOKE" if SMOKE else "FULL", "run | train hours", TRAIN_HOURS, "| wall cap hours", WALL_HOURS,
      "| synthetic target", SYNTH_TARGET, "x", SYNTH_WORKERS, "workers")
'''

TRAIN_ENV_SRC = r'''
import torch

N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
print("python", sys.version.split()[0], "| torch", torch.__version__, "| GPUs", N_GPU,
      [torch.cuda.get_device_name(i) for i in range(N_GPU)], "| CPUs", os.cpu_count())
if not CODE_OK:
    raise RuntimeError("the ARC-JEPA code dataset (poby7722/arc-jepa-code) is not mounted or not importable")
'''

TRAIN_SYNTH_SRC = r'''
# ---- synthetic phase-1 tasks, generated in-session on CPU workers within SYNTH_MAX_S


def count_lines(path):
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


def generate(n, path, seed, workers, timeout_s):
    """python -m arcjepa.synthetic.dataset into path (via a .part file); True on success."""
    part = path + ".part"
    rc = run_cmd([sys.executable, "-m", "arcjepa.synthetic.dataset", "--n", n, "--out", part, "--seed", seed,
                  "--workers", workers], timeout_s, LOG_PATH)
    if rc == 0 and os.path.isfile(part):
        os.replace(part, path)
        return True
    if os.path.isfile(part):
        os.remove(part)
    return False


t_syn = time.time()
if os.path.isfile(SYNTH_PATH):
    print("reusing", SYNTH_PATH)
elif SMOKE:
    if not generate(200, SYNTH_PATH, SYNTH_SEED, SYNTH_WORKERS, 600):
        raise RuntimeError("synthetic smoke generation failed")
else:
    calib_path = os.path.join(WORK, "synthetic_calib.jsonl")
    n_cal = min(2000, SYNTH_TARGET)
    t_cal = time.time()
    if not generate(n_cal, calib_path, SYNTH_SEED + 1000, SYNTH_WORKERS, SYNTH_MAX_S / 3):
        raise RuntimeError("synthetic calibration run failed")
    rate = n_cal / max(1e-3, time.time() - t_cal)
    left = SYNTH_MAX_S - (time.time() - t_syn) - 30.0
    n_main = int(min(SYNTH_TARGET, 0.85 * rate * left)) // 1000 * 1000
    print("synthetic throughput %.0f tasks/s -> generating %d tasks (target %d)" % (rate, n_main, SYNTH_TARGET))
    ok = n_main > n_cal and generate(n_main, SYNTH_PATH, SYNTH_SEED, SYNTH_WORKERS, left)
    if ok:
        os.remove(calib_path)
    else:
        print("falling back to the %d calibration tasks" % n_cal)
        os.replace(calib_path, SYNTH_PATH)
N_SYNTH = count_lines(SYNTH_PATH)
print("synthetic tasks:", N_SYNTH, "in %.0fs" % (time.time() - t_syn))
'''

TRAIN_RUN_SRC = r'''
# ---- train_all: torchrun over all GPUs, single-process fallback (resumes from the same --out)
CONFIG = os.path.join(CODE_DST, "configs", "debug.yaml" if SMOKE else "kaggle.yaml")
MARGIN_H = 0.02 if SMOKE else 0.25  # export check + summary


def package_ok():
    """A loadable package: config.json + weights (the export writes these before building the memory)."""
    return os.path.isfile(os.path.join(PKG_DIR, "config.json")) and any(
        os.path.isfile(os.path.join(PKG_DIR, w)) for w in ("model.safetensors", "model.pt"))


def package_complete():
    """A loadable package whose memory was written too (config.json "export_complete")."""
    if not package_ok():
        return False
    try:
        with open(os.path.join(PKG_DIR, "config.json"), encoding="utf-8") as fh:
            return bool(json.load(fh).get("export_complete", True))
    except (OSError, ValueError):
        return False


def stages_done():
    """Stages recorded as finished in the run's metrics (stage_end records)."""
    done = set()
    try:
        with open(os.path.join(RUN_DIR, "metrics.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("kind") == "stage_end" and r.get("stage"):
                    done.add(r["stage"])
    except OSError:
        pass
    return done


def hours_left():
    h = TRAIN_HOURS - (time.time() - T0) / 3600.0 - MARGIN_H
    return max(0.01, min(h, SMOKE_HOURS) if SMOKE else h)


common = ["--config", CONFIG, "--out", RUN_DIR, "--export-dir", PKG_DIR, "--set", "synthetic.path=" + SYNTH_PATH,
          "--set", "synthetic.n_tasks=%d" % N_SYNTH]
if EPISODES_ROOT:
    common += ["--set", "data.root=" + EPISODES_ROOT]
t_train = time.time()
rc = None
if not SMOKE and N_GPU > 1:
    h = hours_left()
    # train_all stops its stages at h*(1 - reserve_frac) and exports in the reserve; the kill is a backstop
    # that also leaves wall time for the fallback export below
    limit = min(h * 3600 + 1800, wall_left() - 1800)
    rc = run_cmd([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=%d" % N_GPU,
                  "-m", "arcjepa.training.train_all", "--hours", "%.4f" % h] + common, limit, LOG_PATH)
if not package_complete():
    h = hours_left()
    done = stages_done()
    if not SMOKE and set("ABCD") <= done:
        print("all stages finished (exit %s) but the package is incomplete: the export cell re-exports" % rc)
    elif not SMOKE and wall_left() < 2100:
        print("torchrun exit %s; %.0fs of wall time left: skipping the fallback training" % (rc, wall_left()))
    else:
        if not SMOKE and N_GPU > 1:
            msg = "torchrun did not produce a package (exit %s, stages done %s); single-process fallback for %.2fh"
            print(msg % (rc, sorted(done), h))
        limit = min(h * 3600 + 1200, wall_left() - 1500) if not SMOKE else h * 3600 + 1800
        rc = run_cmd([sys.executable, "-m", "arcjepa.training.train_all", "--hours", "%.4f" % h] + common,
                     limit, LOG_PATH)
print("training finished in %.1f min, package ok: %s, complete: %s" % (
    (time.time() - t_train) / 60, package_ok(), package_complete()))
'''

TRAIN_EXPORT_SRC = r'''
# ---- make sure a complete offline package exists: re-export from the last checkpoint otherwise, on CUDA when
# present and with a reduced memory (the full 20k-task memory cannot be rebuilt inside this cell's time box;
# on CPU the v1 model needs ~9 s per memory task, so the CPU path keeps only a token synthetic memory)
ckpt = os.path.join(RUN_DIR, "checkpoints", "last.pt")
if not package_complete() and os.path.isfile(ckpt):
    export_cmd = [sys.executable, "-m", "arcjepa.training.export", "--checkpoint", ckpt, "--out", PKG_DIR,
                  "--config", CONFIG, "--device", "auto", "--set", "synthetic.path=" + SYNTH_PATH,
                  "--set", "synthetic.n_tasks=%d" % N_SYNTH]
    if EPISODES_ROOT:
        export_cmd += ["--set", "data.root=" + EPISODES_ROOT]
    if not SMOKE:
        if N_GPU > 0:
            export_cmd += ["--set", "memory.max_synthetic=4000", "--set", "memory.pseudo_label_seconds=0"]
        else:
            export_cmd += ["--set", "memory.max_synthetic=64", "--set", "memory.include_real=false",
                           "--set", "memory.pseudo_label_seconds=0"]
    limit = 3600 if SMOKE else max(60.0, min(2400.0, wall_left() - 300))
    run_cmd(export_cmd, limit, LOG_PATH)
print("package ok: %s, complete: %s" % (package_ok(), package_complete()))
if package_ok():
    from arcjepa.model.arcjepa import ARCJEPA

    m = ARCJEPA.load_package(PKG_DIR, device="cpu")
    print("package loads: %.2fM parameters, memory %s" % (
        sum(p.numel() for p in m.parameters()) / 1e6, None if m.memory is None else len(m.memory)))
    del m
for f in sorted(glob.glob(os.path.join(PKG_DIR, "*"))):
    print("  %-24s %10d bytes" % (os.path.basename(f), os.path.getsize(f)))
'''

TRAIN_SUMMARY_SRC = r'''
# ---- summary
report = {"smoke": SMOKE, "package_ok": package_ok(), "package_complete": package_complete(),
          "package_dir": PKG_DIR, "synthetic_tasks": N_SYNTH, "minutes": round((time.time() - T0) / 60, 1),
          "train_hours": TRAIN_HOURS, "wall_hours": WALL_HOURS, "config": CONFIG}
summary_path = os.path.join(RUN_DIR, "train_summary.json")
if os.path.isfile(summary_path):
    with open(summary_path, encoding="utf-8") as fh:
        report["train_summary"] = json.load(fh)
metrics_path = os.path.join(RUN_DIR, "metrics.jsonl")
if os.path.isfile(metrics_path):
    rows = []
    with open(metrics_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    report["n_metric_rows"] = len(rows)
    report["last_train"] = next((r for r in reversed(rows) if r.get("kind") == "train"), None)
    report["last_eval"] = next((r for r in reversed(rows) if r.get("kind") == "eval"), None)
with open(os.path.join(WORK, "train_report.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=1, default=str)
print(json.dumps(report, indent=1, default=str)[:6000])
if not KEEP_SYNTH and os.path.isfile(SYNTH_PATH):
    os.remove(SYNTH_PATH)  # keep the kernel output small: the package is what the inference kernel needs
if not package_ok():
    raise RuntimeError("no ARC-JEPA package was exported")
'''


#: baked-in defaults per build mode (Kaggle kernels cannot set environment variables). FULL: the v1 model on
#: 4 x L4 with torchrun; TRAIN_HOURS 7.5 from the notebook start (synthetic <= 0.5 h, stages + export inside
#: train_all's --hours), wall cap 8.5 h for every step including the fallbacks (12 h Kaggle limit, quota headroom).
BUILD_MODES: Dict[str, Dict[str, str]] = {
    "smoke": {"__MODE__": "smoke (debug config)", "__SMOKE_DEFAULT__": "1", "__TRAIN_HOURS_DEFAULT__": "9.5"},
    "full": {"__MODE__": "FULL (v1 model, configs/kaggle.yaml)", "__SMOKE_DEFAULT__": "0",
             "__TRAIN_HOURS_DEFAULT__": "7.5"},
}
FULL_KERNEL_DIR = "train_full"


def _bake(src: str, mode: str) -> str:
    for key, val in BUILD_MODES[mode].items():
        src = src.replace(key, val)
    return src


def build_train_notebook(full: bool = False) -> Dict[str, Any]:
    """The training notebook as nbformat JSON (``full``: FULL-mode defaults baked in, else smoke)."""
    mode = "full" if full else "smoke"
    return notebook([
        markdown_cell("train-00-header", _bake(TRAIN_HEADER, mode)),
        code_cell("train-01-setup", _bake(TRAIN_SETUP_SRC, mode)),
        code_cell("train-02-code", CODE_SETUP_SRC),
        code_cell("train-03-run-cmd", RUN_CMD_SRC),
        code_cell("train-04-env", TRAIN_ENV_SRC),
        code_cell("train-05-synthetic", TRAIN_SYNTH_SRC),
        code_cell("train-06-train", TRAIN_RUN_SRC),
        code_cell("train-07-export", TRAIN_EXPORT_SRC),
        code_cell("train-08-summary", TRAIN_SUMMARY_SRC),
    ])


def build(out_dir: Optional[Path] = None, full: bool = False) -> Dict[str, str]:
    """Default (smoke): write ``<out_dir>/arc-jepa-train.ipynb``, ``<out_dir>/train/kernel-metadata.json`` and the
    notebook copy ``<out_dir>/train/arc-jepa-train.ipynb``. ``full``: write the FULL-mode kernel folder
    ``<out_dir>/train_full/`` (same kernel id and metadata, ARCJEPA_SMOKE default "0", ARCJEPA_TRAIN_HOURS default
    "7.5"). Returns the written paths."""
    out = Path(out_dir) if out_dir else KAGGLE_DIR
    meta = kernel_metadata(TRAIN_SLUG, TRAIN_TITLE, TRAIN_NOTEBOOK)
    if full:
        nb = build_train_notebook(full=True)
        return {
            "kernel_notebook": write_json(nb, out / FULL_KERNEL_DIR / TRAIN_NOTEBOOK),
            "metadata": write_json(meta, out / FULL_KERNEL_DIR / "kernel-metadata.json"),
        }
    nb = build_train_notebook()
    return {
        "notebook": write_json(nb, out / TRAIN_NOTEBOOK),
        "kernel_notebook": write_json(nb, out / "train" / TRAIN_NOTEBOOK),
        "metadata": write_json(meta, out / "train" / "kernel-metadata.json"),
    }


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description="Build the ARC-JEPA Kaggle training notebook.")
    ap.add_argument("--out-dir", default=None, help="output folder (default: this kaggle/ folder)")
    ap.add_argument("--full", action="store_true",
                    help="write the FULL-mode kernel folder <out-dir>/train_full (ARCJEPA_SMOKE=0, 7.5 train hours)")
    a = ap.parse_args(argv)
    return build(Path(a.out_dir) if a.out_dir else None, full=a.full)


if __name__ == "__main__":
    for k, v in main().items():
        print(f"{k}: {v}")
