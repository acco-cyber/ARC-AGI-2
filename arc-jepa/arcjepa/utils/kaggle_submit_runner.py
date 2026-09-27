"""Budgeted, crash-safe submission runner used by the Kaggle inference notebook (INTERFACES §8).

``run_submission(challenges, out_path, ...)`` solves every task of a challenges dict
(``{task_id: {"train": [...], "test": [{"input": g}, ...]}}``) with :func:`arcjepa.search.solver.solve_task`:

* fallback attempts for every test input (:func:`fallback_attempts`: constant demo output, the predicted output
  shape filled with the two most common demo output colours; the identity grid only as a last resort) are written
  to ``out_path`` FIRST, before any search;
* tasks run shortest-first (fewest total cells), each with a fair share of the time left:
  ``clip(seconds_left * slots / tasks_left, min_task_seconds, max_task_seconds)``, never beyond the global
  deadline ``start_time + total_seconds - reserve_seconds``;
* ``out_path`` is rewritten atomically every ``rewrite_every_s`` seconds and at the end; only valid grids ever
  replace the fallbacks;
* ``workers > 0`` solves tasks in ``spawn`` worker processes (each loads the package once, on its own device).
  **Worker crashes** (OOM kill, segfault, CUDA abort) break the whole ``ProcessPoolExecutor``: every task that
  was in flight (and a task whose submit hit the broken pool) is re-queued into a fresh pool, up to
  ``max_pool_restarts`` rebuilds under the same global deadline. Re-queued tasks are *suspects* and run at most
  one at a time; a task in flight during ``max_task_breaks`` crashes is quarantined (keeps its fallback). Only
  when rebuilding keeps failing are the remaining non-suspect tasks solved in-process (each capped by a thread
  timeout); suspects never run in the notebook process;
* ``workers == 0`` solves in-process (thread-capped). When the package is missing or fails to load, the solver
  runs with ``model=None`` (pure symbolic search); ``summary["model_loaded"]`` / ``["model_loaded_fraction"]``
  report what the workers actually loaded.

The per-process ``_WORKER`` dict holds the model loaded by the pool initializer (process-local state of the
worker processes only).
"""
from __future__ import annotations

import concurrent.futures as cf
import gc
import json
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from arcjepa.core.types import Grid, task_from_json, validate_grid

__all__ = ["RunnerConfig", "task_cost", "order_shortest_first", "fair_share_seconds", "predicted_output_shape",
           "fallback_attempts", "fallback_submission", "package_info", "find_package", "load_model_package",
           "build_solve_config", "solve_one", "write_json_atomic", "run_submission"]

log = logging.getLogger(__name__)

ATTEMPT_KEYS = ("attempt_1", "attempt_2")
PACKAGE_CONFIG = "config.json"
PACKAGE_FORMAT = "arcjepa-package-v1"
PACKAGE_DIRNAME = "arc_jepa_pkg"
WEIGHT_FILES = ("model.safetensors", "model.pt")

#: process-local state of pool workers (model, config); empty in the parent process
_WORKER: Dict[str, Any] = {}


@dataclass
class RunnerConfig:
    """Budget and parallelism knobs of :func:`run_submission`."""

    total_seconds: float = 11 * 3600.0
    reserve_seconds: float = 120.0
    min_task_seconds: float = 2.0
    max_task_seconds: float = 1800.0
    rewrite_every_s: float = 60.0
    workers: int = 0
    devices: Optional[List[str]] = None
    threads_per_worker: Optional[int] = None
    solve_overrides: Dict[str, Any] = field(default_factory=dict)
    poll_s: float = 2.0
    #: fresh pools built after worker crashes before giving up on the pool
    max_pool_restarts: int = 6
    #: a task in flight during this many pool crashes is quarantined (keeps its fallback)
    max_task_breaks: int = 2


# ============================================================================================ ordering / budget

def task_cost(task: Mapping[str, Any]) -> int:
    """Size proxy used for shortest-first ordering: total cells over all demo and test grids."""
    n = 0
    for p in list(task.get("train", [])) + list(task.get("test", [])):
        if not isinstance(p, Mapping):
            continue
        for k in ("input", "output"):
            g = p.get(k)
            if isinstance(g, list) and g and isinstance(g[0], list):
                n += len(g) * len(g[0])
    return n


def order_shortest_first(challenges: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Task ids sorted by (:func:`task_cost`, id)."""
    return sorted(challenges, key=lambda t: (task_cost(challenges[t]), t))


def fair_share_seconds(seconds_left: float, tasks_left: int, slots: int = 1, lo: float = 2.0,
                       hi: float = 1800.0) -> float:
    """Fair per-task budget: ``seconds_left * slots / tasks_left`` clipped to ``[lo, hi]`` and to the time left."""
    if tasks_left <= 0 or seconds_left <= 0:
        return 0.0
    share = seconds_left * max(1, slots) / float(tasks_left)
    return max(0.0, min(max(lo, min(hi, share)), seconds_left))


# ============================================================================================ fallbacks / io

def _shape(g: Grid) -> Tuple[int, int]:
    return (len(g), len(g[0]))


def predicted_output_shape(demo: Sequence[Tuple[Grid, Grid]], test_input: Any) -> Optional[Tuple[int, int]]:
    """Output shape implied by valid demo ``(input, output)`` grids: same as input, a consistent integer scaling or
    a constant shape (same rule as :func:`arcjepa.search.diversity.predict_output_shape`), else None."""
    if not demo or not validate_grid(test_input):
        return None
    ih, iw = _shape(test_input)
    shapes = [(_shape(i), _shape(o)) for i, o in demo]
    if all(a == b for a, b in shapes):
        return (ih, iw)
    fr = {(o[0] / i[0], o[1] / i[1]) for i, o in shapes}
    if len(fr) == 1:
        a, b = next(iter(fr))
        h, w = ih * a, iw * b
        if abs(h - round(h)) < 1e-9 and abs(w - round(w)) < 1e-9 and 1 <= round(h) <= 30 and 1 <= round(w) <= 30:
            return (int(round(h)), int(round(w)))
    outs = {o for _, o in shapes}
    return next(iter(outs)) if len(outs) == 1 else None


def fallback_attempts(task: Mapping[str, Any]) -> List[Dict[str, Grid]]:
    """Always-valid attempts for every test input, without any search: the first two distinct grids of
    [constant demo output (all demo outputs equal); the identity for identity tasks (every demo output equals its
    input); the predicted output shape (else the most common demo output shape) filled with the most common and
    the second most common demo output colour; the identity; ``[[0]]``].

    The identity (0 of 1,076 test outputs of the public training tasks) is a last resort. Same rule as
    :func:`arcjepa.search.diversity.fallback_grids` and the inlined copy in ``kaggle/validate_submission.py``.
    """
    demo = [(p["input"], p["output"]) for p in task.get("train", []) or []
            if isinstance(p, Mapping) and validate_grid(p.get("input")) and validate_grid(p.get("output"))]
    outs = [o for _, o in demo]
    ident_ok = bool(demo) and all(i == o for i, o in demo)
    colours = [c for c, _ in Counter(v for o in outs for row in o for v in row).most_common(2)]
    common = Counter(_shape(o) for o in outs).most_common(1)[0][0] if outs else None
    res: List[Dict[str, Grid]] = []
    for tp in task.get("test", []) or [{}]:
        g = tp.get("input") if isinstance(tp, Mapping) else None
        valid_in = validate_grid(g)
        opts: List[Grid] = []
        if outs and all(o == outs[0] for o in outs):
            opts.append(outs[0])
        if valid_in and ident_ok:
            opts.append(g)
        if outs:
            shape = (predicted_output_shape(demo, g) if valid_in else None) or common
            opts.extend([[int(c)] * shape[1] for _ in range(shape[0])] for c in colours)
        if valid_in:
            opts.append(g)
        opts.append([[0]])
        a1 = opts[0]
        a2 = next((o for o in opts[1:] if o != a1), a1)
        res.append({"attempt_1": [list(r) for r in a1], "attempt_2": [list(r) for r in a2]})
    return res


def fallback_submission(challenges: Mapping[str, Mapping[str, Any]]) -> Dict[str, List[Dict[str, Grid]]]:
    """:func:`fallback_attempts` for every task."""
    return {str(t): fallback_attempts(challenges[t]) for t in challenges}


def write_json_atomic(obj: Any, path: str) -> None:
    """JSON-dump ``obj`` to ``path`` through a temporary file and an atomic rename."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    os.replace(tmp, path)


def _merge(sub: Dict[str, List[Dict[str, Grid]]], tid: str, attempts: Sequence[Mapping[str, Any]]) -> int:
    """Replace fallbacks of ``tid`` by the valid grids of ``attempts``; returns the number of grids taken."""
    cur = sub.get(tid, [])
    n = 0
    for i, a in enumerate(attempts):
        if i >= len(cur) or not isinstance(a, Mapping):
            continue
        for k in ATTEMPT_KEYS:
            if validate_grid(a.get(k)):
                cur[i][k] = a[k]
                n += 1
    return n


# ============================================================================================ model

def package_info(pkg_dir: str) -> Dict[str, Any]:
    """What ``pkg_dir`` holds: ``ok`` (config.json parses to a dict of format ``arcjepa-package-v1`` and a
    non-empty weights file exists), ``reason`` when not ok, ``debug`` (trained with ``configs/debug.yaml``),
    ``named`` (folder called ``arc_jepa_pkg``), ``created_unix``, ``n_parameters``, ``config_path``."""
    info: Dict[str, Any] = {"path": pkg_dir, "ok": False, "reason": None, "debug": False,
                            "named": os.path.basename(os.path.normpath(pkg_dir)) == PACKAGE_DIRNAME,
                            "created_unix": None, "n_parameters": None, "config_path": None, "weights": None}
    cfg_path = os.path.join(pkg_dir, PACKAGE_CONFIG)
    if not os.path.isfile(cfg_path):
        info["reason"] = "no config.json"
        return info
    weights = [w for w in WEIGHT_FILES if os.path.isfile(os.path.join(pkg_dir, w))
               and os.path.getsize(os.path.join(pkg_dir, w)) > 0]
    if not weights:
        info["reason"] = "no non-empty model.safetensors / model.pt"
        return info
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        info["reason"] = f"config.json unreadable ({exc.__class__.__name__})"
        return info
    if not isinstance(meta, dict):
        info["reason"] = "config.json is not an object"
        return info
    fmt = meta.get("format")
    if fmt is not None and fmt != PACKAGE_FORMAT:
        info["reason"] = f"format {fmt!r} is not {PACKAGE_FORMAT}"
        return info
    training = meta.get("training") if isinstance(meta.get("training"), dict) else {}
    cpath = str(training.get("config_path") or "")
    info.update({"ok": True, "weights": weights[0], "created_unix": meta.get("created_unix"),
                 "n_parameters": meta.get("n_parameters"), "config_path": cpath or None,
                 "debug": os.path.basename(cpath.replace("\\", "/")) == "debug.yaml"})
    return info


def find_package(candidates: Sequence[str] = (), search_roots: Sequence[str] = ("/kaggle/input", "/kaggle/working"),
                 max_depth: int = 4, named_max_depth: int = 7, allow_debug: bool = False) -> Optional[str]:
    """Folder of the exported ARC-JEPA package to load, or None.

    1. The explicit ``candidates`` in order (any valid package, debug ones included: naming a folder is explicit).
    2. A walk of ``search_roots``: folders named ``arc_jepa_pkg`` up to ``named_max_depth`` levels deep (the
       training kernel's output, mounted e.g. at ``/kaggle/input/notebooks/<owner>/arc-jepa-train/arc_jepa_pkg``,
       ``/kaggle/input/arc-jepa-train/arc_jepa_pkg`` or one level deeper), and any other package folder up to
       ``max_depth`` levels that was not trained with the debug config (unless ``allow_debug``). Code trees
       (folders holding ``arcjepa/__init__.py``, i.e. the code dataset with its local ``runs/``) are never
       searched. Preference: non-debug, then ``arc_jepa_pkg``-named, then the newest ``created_unix``.

    Only folders whose :func:`package_info` is ok qualify.
    """
    for c in candidates:
        if c and os.path.isdir(c):
            info = package_info(c)
            if info["ok"]:
                return c
            log.warning("package candidate %s skipped: %s", c, info["reason"])
    found: List[Dict[str, Any]] = []
    for root in search_roots:
        if not root or not os.path.isdir(root):
            continue
        base = os.path.normpath(root).count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root):
            depth = os.path.normpath(dirpath).count(os.sep) - base
            if os.path.isfile(os.path.join(dirpath, "arcjepa", "__init__.py")):
                dirnames[:] = []  # a code tree never holds the trained package (debug decoys live in its runs/)
                continue
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "__pycache__")
            if depth >= named_max_depth:
                dirnames[:] = []
            if PACKAGE_CONFIG not in filenames:
                continue
            named = os.path.basename(os.path.normpath(dirpath)) == PACKAGE_DIRNAME
            if not named and depth > max_depth:
                continue
            info = package_info(dirpath)
            if info["ok"] and (named or allow_debug or not info["debug"]):
                found.append(info)
    found.sort(key=lambda i: (bool(i["debug"]), not i["named"], -float(i["created_unix"] or 0), i["path"]))
    return found[0]["path"] if found else None


def load_model_package(pkg_dir: Optional[str], device: Optional[str] = None) -> Any:
    """``ARCJEPA.load_package(pkg_dir)`` on ``device`` (default cuda:0 when available, else cpu), or ``None``
    when ``pkg_dir`` is missing or the load fails (symbolic mode)."""
    if not pkg_dir:
        return None
    try:
        import torch

        from arcjepa.model.arcjepa import ARCJEPA

        dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        return ARCJEPA.load_package(pkg_dir, device=dev)
    except Exception as exc:  # noqa: BLE001 - symbolic mode instead of failing the submission
        log.warning("package %s could not be loaded (%s); symbolic mode", pkg_dir, exc)
        return None


def build_solve_config(model: Any, per_task_seconds: float, overrides: Optional[Mapping[str, Any]] = None) -> Any:
    """``SolveConfig`` from the package's search/tta/repair settings (spec defaults without a package), with the
    given per-task budget, the package memory attached and ``overrides`` applied."""
    from arcjepa.search.solver import SolveConfig

    meta = getattr(model, "package_meta", None) or {}
    kw = dict(overrides or {})
    kw["per_task_seconds"] = float(per_task_seconds)
    cfg = SolveConfig.from_spec_yaml(meta, **kw)
    cfg.memory = getattr(model, "memory", None) if model is not None else None
    return cfg


def solve_one(task_id: str, task: Mapping[str, Any], seconds: float, model: Any = None,
              overrides: Optional[Mapping[str, Any]] = None) -> Tuple[str, List[Dict[str, Grid]], Dict[str, Any]]:
    """Solve one challenges task within ``seconds``: ``(task_id, attempts dicts, diagnostics)``. Never raises
    (errors return the fallback attempts with ``diagnostics["error"]``)."""
    t0 = time.perf_counter()
    try:
        from arcjepa.eval.diagnostics import jsonable
        from arcjepa.search.solver import solve_task

        t = task_from_json(task_id, dict(task))
        cfg = build_solve_config(model, seconds, overrides)
        attempts, diag = solve_task(t, model, cfg)
        out = [{"attempt_1": a1, "attempt_2": a2} for a1, a2 in attempts]
        diag = jsonable(diag)
    except Exception as exc:  # noqa: BLE001
        log.exception("solve_one failed on %s", task_id)
        out, diag = fallback_attempts(task), {"task_id": task_id, "error": repr(exc)}
    diag["wall_s"] = round(time.perf_counter() - t0, 3)
    diag["budget_s"] = round(float(seconds), 3)
    diag["pid"] = os.getpid()
    diag["model"] = model is not None
    return task_id, out, diag


def _solve_capped(task_id: str, task: Mapping[str, Any], seconds: float, model: Any,
                  overrides: Optional[Mapping[str, Any]], cap_s: float
                  ) -> Optional[Tuple[str, List[Dict[str, Grid]], Dict[str, Any]]]:
    """:func:`solve_one` in a daemon thread joined for at most ``cap_s`` seconds (None when it overran; the solver
    stops on its own deadline, so the thread ends soon after)."""
    box: Dict[str, Any] = {}

    def target() -> None:
        box["res"] = solve_one(task_id, task, seconds, model, overrides)

    th = threading.Thread(target=target, name=f"arcjepa-solve-{task_id}", daemon=True)
    th.start()
    th.join(max(0.1, cap_s))
    return box.get("res")


# ============================================================================================ workers

def _init_worker(pkg_dir: Optional[str], device_queue: Any, threads: Optional[int],
                 overrides: Mapping[str, Any]) -> None:
    """Pool initializer: pin threads, take a device from ``device_queue``, load the package once and freeze the
    long-lived heap (short gen-2 GC pauses during the time-boxed searches)."""
    dev = None
    try:
        dev = device_queue.get(timeout=30) if device_queue is not None else None
    except Exception:  # noqa: BLE001
        dev = None
    if threads and (pkg_dir or "torch" in sys.modules):  # symbolic mode never needs torch: faster pool (re)starts
        try:
            import torch

            torch.set_num_threads(int(threads))
        except Exception:  # noqa: BLE001
            pass
    _WORKER["model"] = load_model_package(pkg_dir, dev)
    _WORKER["overrides"] = dict(overrides or {})
    _WORKER["device"] = dev
    try:  # import the solver stack now, so the first task's wall time is solving, not importing (~10 s under load)
        import arcjepa.eval.diagnostics  # noqa: F401
        import arcjepa.search.solver  # noqa: F401
    except Exception:  # noqa: BLE001 - solve_one reports the import error per task
        pass
    try:
        gc.collect()
        gc.freeze()
    except Exception:  # noqa: BLE001
        pass


def _worker_solve(task_id: str, task: Mapping[str, Any], seconds: float
                  ) -> Tuple[str, List[Dict[str, Grid]], Dict[str, Any]]:
    """Pool task: :func:`solve_one` with the worker's model."""
    res = solve_one(task_id, task, seconds, _WORKER.get("model"), _WORKER.get("overrides"))
    res[2]["device"] = _WORKER.get("device")
    return res


def _default_devices(n: int) -> List[str]:
    try:
        import torch

        g = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001
        g = 0
    return [f"cuda:{i % g}" for i in range(n)] if g else ["cpu"] * n


def _terminate_pool(ex: cf.ProcessPoolExecutor) -> None:
    """Stop our own pool's worker processes (by handle, never by name)."""
    procs = list(getattr(ex, "_processes", {}).values()) if getattr(ex, "_processes", None) else []
    try:
        ex.shutdown(wait=False, cancel_futures=True)
    except Exception:  # noqa: BLE001
        pass
    for p in procs:
        try:
            if p.is_alive():
                p.terminate()
        except Exception:  # noqa: BLE001
            pass


def _next_task(pending: Sequence[str], suspects: Set[str], inflight: Sequence[str]) -> Optional[str]:
    """First pending task that may start now: suspects (in flight at an earlier pool crash) one at a time."""
    busy_suspect = any(t in suspects for t in inflight)
    for tid in pending:
        if tid in suspects and busy_suspect:
            continue
        return tid
    return None


def _model_loaded_fraction(diags: Mapping[str, Mapping[str, Any]]) -> Optional[float]:
    """Fraction of worker processes (by pid, over completed tasks) that ran with the model loaded."""
    by_pid: Dict[Any, bool] = {}
    for d in diags.values():
        if "pid" in d and "model" in d:
            by_pid[d["pid"]] = by_pid.get(d["pid"], False) or bool(d["model"])
    return (sum(by_pid.values()) / float(len(by_pid))) if by_pid else None


# ============================================================================================ driver

def run_submission(challenges: Mapping[str, Mapping[str, Any]], out_path: str, *, pkg_dir: Optional[str] = None,
                   cfg: Optional[RunnerConfig] = None, initial: Optional[Mapping[str, Any]] = None,
                   start_time: Optional[float] = None, diagnostics_path: Optional[str] = None) -> Dict[str, Any]:
    """Solve ``challenges`` into ``out_path`` under the global budget (see module docstring).

    ``initial`` (e.g. a submission already written by the notebook) seeds the attempts; ``start_time``
    (``time.time()`` of the notebook start) anchors the global deadline. Returns a summary with
    ``n_tasks, n_solved, n_timeout, n_errors, n_exact_found, seconds, model_loaded, model_loaded_fraction,
    pool_restarts, quarantined, diagnostics``.
    """
    cfg = cfg or RunnerConfig()
    t_start = time.time()
    start = float(start_time) if start_time is not None else t_start
    deadline = start + float(cfg.total_seconds) - float(cfg.reserve_seconds)
    sub = fallback_submission(challenges)
    if initial:
        for tid, att in initial.items():
            if tid in sub and isinstance(att, list):
                _merge(sub, tid, att)
    write_json_atomic(sub, out_path)
    last_write = time.time()
    order = order_shortest_first(challenges)
    diags: Dict[str, Dict[str, Any]] = {}
    summary: Dict[str, Any] = {"n_tasks": len(order), "workers": int(cfg.workers), "pkg_dir": pkg_dir,
                               "pool_restarts": 0, "quarantined": [], "in_process_fallback": False}

    def maybe_write(force: bool = False) -> None:
        nonlocal last_write
        if force or time.time() - last_write >= cfg.rewrite_every_s:
            write_json_atomic(sub, out_path)
            last_write = time.time()

    def take(res: Tuple[str, List[Dict[str, Grid]], Dict[str, Any]]) -> None:
        tid, att, d = res
        _merge(sub, tid, att)
        diags[tid] = d

    def solve_in_process(tids: List[str], model: Any) -> None:
        """Thread-capped in-process solving of ``tids`` (removed from ``pending`` as they start)."""
        try:  # import the solver stack before the first task's time cap starts (~10 s under heavy load)
            import arcjepa.eval.diagnostics  # noqa: F401
            import arcjepa.search.solver  # noqa: F401
        except Exception:  # noqa: BLE001 - solve_one reports the import error per task
            pass
        for tid in list(tids):
            left = deadline - time.time()
            if left < 0.5:
                break
            pending.remove(tid)
            sec = fair_share_seconds(left, len(pending) + 1, 1, cfg.min_task_seconds, cfg.max_task_seconds)
            res = _solve_capped(tid, challenges[tid], sec, model, cfg.solve_overrides,
                                min(deadline - time.time(), 2.0 * sec + 30.0))
            if res is None:
                diags[tid] = {"task_id": tid, "error": "in-process solve overran its time cap"}
            else:
                take(res)
            maybe_write()

    pending = list(order)
    if cfg.workers <= 0:
        if cfg.threads_per_worker:  # explicit only: pins this (the notebook's) process, as the pool workers are
            try:
                import torch

                torch.set_num_threads(int(cfg.threads_per_worker))
            except Exception:  # noqa: BLE001
                pass
        model = load_model_package(pkg_dir, (cfg.devices or [None])[0])
        summary["model_loaded"] = model is not None
        summary["model_loaded_fraction"] = 1.0 if model is not None else 0.0
        solve_in_process(list(pending), model)
    else:
        n = int(cfg.workers)
        devices = list(cfg.devices or _default_devices(n))
        ctx = mp.get_context("spawn")
        threads = cfg.threads_per_worker or max(1, (os.cpu_count() or 1) // n)
        breaks: Counter = Counter()
        suspects: Set[str] = set()
        restarts = 0
        while pending and deadline - time.time() >= 0.5:
            if restarts > int(cfg.max_pool_restarts):
                summary["in_process_fallback"] = True
                break
            q = ctx.Queue()
            for i in range(n):
                q.put(devices[i % len(devices)])
            ex = cf.ProcessPoolExecutor(max_workers=n, mp_context=ctx, initializer=_init_worker,
                                        initargs=(pkg_dir, q, threads, dict(cfg.solve_overrides)))
            inflight: Dict[cf.Future, str] = {}
            lost: List[str] = []
            broken = False
            try:
                while (pending or inflight) and not broken:
                    left = deadline - time.time()
                    if left < 0.5:
                        break
                    while len(inflight) < n and left >= 0.5:
                        tid = _next_task(pending, suspects, list(inflight.values()))
                        if tid is None:
                            break
                        pending.remove(tid)
                        sec = fair_share_seconds(left, len(pending) + 1 + len(inflight), n, cfg.min_task_seconds,
                                                 cfg.max_task_seconds)
                        try:
                            fut = ex.submit(_worker_solve, tid, dict(challenges[tid]), sec)
                        except (cf.process.BrokenProcessPool, RuntimeError) as exc:
                            log.error("submit of %s hit a broken pool (%s)", tid, exc)
                            pending.insert(0, tid)  # never started: not a suspect
                            broken = True
                            break
                        inflight[fut] = tid
                    if broken or not inflight:
                        break
                    done, _ = cf.wait(list(inflight), timeout=max(0.05, min(cfg.poll_s, deadline - time.time())),
                                      return_when=cf.FIRST_COMPLETED)
                    for f in done:
                        tid = inflight.pop(f)
                        try:
                            take(f.result())
                            suspects.discard(tid)
                        except cf.process.BrokenProcessPool as exc:
                            log.error("worker pool broke while solving %s: %s", tid, exc)
                            broken = True
                            lost.append(tid)
                        except Exception as exc:  # noqa: BLE001 - e.g. an unpicklable result: keep the fallback
                            log.warning("worker failed on %s: %s", tid, exc)
                            diags[tid] = {"task_id": tid, "error": repr(exc)}
                    maybe_write()
            finally:
                if broken:
                    lost.extend(inflight.values())  # every future of a broken pool is dead
                else:
                    for tid in inflight.values():
                        diags.setdefault(tid, {"task_id": tid, "error": "global deadline reached while solving"})
                inflight = {}
                _terminate_pool(ex)
            if not broken:
                break
            restarts += 1
            requeue: List[str] = []
            for tid in lost:
                breaks[tid] += 1
                if breaks[tid] >= int(cfg.max_task_breaks):
                    suspects.discard(tid)
                    summary["quarantined"].append(tid)
                    diags[tid] = {"task_id": tid, "error": f"quarantined: in flight during {breaks[tid]} worker "
                                                          "pool crashes; the fallback attempts stand"}
                else:
                    suspects.add(tid)
                    requeue.append(tid)
            pending = requeue + pending
            log.error("worker pool crash %d: %d task(s) re-queued, %d quarantined so far", restarts, len(requeue),
                      len(summary["quarantined"]))
        summary["pool_restarts"] = restarts
        if summary["in_process_fallback"] and pending and deadline - time.time() >= 0.5:
            log.error("the worker pool keeps failing: solving %d non-suspect task(s) in-process",
                      sum(1 for t in pending if t not in suspects))
            model = load_model_package(pkg_dir, devices[0] if devices else None)
            solve_in_process([t for t in pending if t not in suspects], model)
        frac = _model_loaded_fraction(diags)
        summary["model_loaded_fraction"] = frac
        summary["model_loaded"] = bool(frac)
        summary["suspects_unsolved"] = sorted(t for t in suspects if t not in diags)
    maybe_write(force=True)
    summary.update({
        "n_solved": sum(1 for d in diags.values() if "error" not in d),
        "n_errors": sum(1 for d in diags.values() if "error" in d),
        "n_timeout": len(order) - len(diags),
        "n_exact_found": sum(1 for d in diags.values() if (d.get("n_exact") or 0) > 0),
        "seconds": round(time.time() - t_start, 2),
        "diagnostics": diags,
    })
    if pending:
        summary["unstarted"] = list(pending)
    if diagnostics_path:
        write_json_atomic({"summary": {k: v for k, v in summary.items() if k != "diagnostics"},
                           "tasks": list(diags.values())}, diagnostics_path)
    log.info("run_submission: %d tasks, %d solved, %d errors, %d not reached, %d pool restarts in %.1fs",
             len(order), summary["n_solved"], summary["n_errors"], summary["n_timeout"], summary["pool_restarts"],
             summary["seconds"])
    return summary
