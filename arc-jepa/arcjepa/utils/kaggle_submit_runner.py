"""Budgeted, crash-safe submission runner used by the Kaggle inference notebook (INTERFACES §8).

``run_submission(challenges, out_path, ...)`` solves every task of a challenges dict
(``{task_id: {"train": [...], "test": [{"input": g}, ...]}}``) with :func:`arcjepa.search.solver.solve_task`:

* fallback attempts for every test input (identity grid + most-common demo output shape filled with the most
  common demo output colour) are written to ``out_path`` FIRST, before any search;
* tasks run shortest-first (fewest total cells), each with a fair share of the time left:
  ``clip(seconds_left * slots / tasks_left, min_task_seconds, max_task_seconds)``, never beyond the global
  deadline ``start_time + total_seconds - reserve_seconds``;
* ``out_path`` is rewritten atomically every ``rewrite_every_s`` seconds and at the end; only valid grids ever
  replace the fallbacks;
* ``workers > 0`` solves tasks in ``spawn`` worker processes (each loads the package once, on its own device);
  ``workers == 0`` solves in-process. When the package is missing or fails to load, the solver runs with
  ``model=None`` (pure symbolic search).

The per-process ``_WORKER`` dict holds the model loaded by the pool initializer (process-local state of the
worker processes only).
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import multiprocessing as mp
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from arcjepa.core.types import Grid, task_from_json, validate_grid

__all__ = ["RunnerConfig", "task_cost", "order_shortest_first", "fair_share_seconds", "fallback_attempts",
           "fallback_submission", "find_package", "load_model_package", "build_solve_config", "solve_one",
           "write_json_atomic", "run_submission"]

log = logging.getLogger(__name__)

ATTEMPT_KEYS = ("attempt_1", "attempt_2")
PACKAGE_CONFIG = "config.json"

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

def fallback_attempts(task: Mapping[str, Any]) -> List[Dict[str, Grid]]:
    """attempt_1 = identity (the test input), attempt_2 = the most common demo output shape filled with the most
    common demo output colour; one dict per test input, always valid grids."""
    outs = [p.get("output") for p in task.get("train", []) if isinstance(p, Mapping)]
    outs = [o for o in outs if validate_grid(o)]
    if outs:
        shape = Counter((len(o), len(o[0])) for o in outs).most_common(1)[0][0]
        colour = Counter(v for o in outs for row in o for v in row).most_common(1)[0][0]
    else:
        shape, colour = (1, 1), 0
    res: List[Dict[str, Grid]] = []
    for tp in task.get("test", []) or [{}]:
        g = tp.get("input") if isinstance(tp, Mapping) else None
        ident = [list(r) for r in g] if validate_grid(g) else [[0]]
        res.append({"attempt_1": ident, "attempt_2": [[int(colour)] * shape[1] for _ in range(shape[0])]})
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

def find_package(candidates: Sequence[str] = (), search_roots: Sequence[str] = ("/kaggle/input", "/kaggle/working"),
                 max_depth: int = 4) -> Optional[str]:
    """First directory holding an exported ARC-JEPA package (``config.json`` + weights): the explicit
    ``candidates`` first, then a bounded walk of ``search_roots`` (preferring folders named ``arc_jepa_pkg``)."""
    def ok(d: str) -> bool:
        return (os.path.isfile(os.path.join(d, PACKAGE_CONFIG))
                and any(os.path.isfile(os.path.join(d, w)) for w in ("model.safetensors", "model.pt")))

    for c in candidates:
        if c and ok(c):
            return c
    found: List[str] = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        base = root.rstrip("/\\").count(os.sep)
        for dirpath, dirnames, _ in os.walk(root):
            if dirpath.count(os.sep) - base >= max_depth:
                dirnames[:] = []
            dirnames.sort()
            if ok(dirpath):
                found.append(dirpath)
    found.sort(key=lambda d: (os.path.basename(d) != "arc_jepa_pkg", d))
    return found[0] if found else None


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
    return task_id, out, diag


# ============================================================================================ workers

def _init_worker(pkg_dir: Optional[str], device_queue: Any, threads: Optional[int],
                 overrides: Mapping[str, Any]) -> None:
    """Pool initializer: pin threads, take a device from ``device_queue`` and load the package once."""
    dev = None
    try:
        dev = device_queue.get(timeout=30) if device_queue is not None else None
    except Exception:  # noqa: BLE001
        dev = None
    try:
        import torch

        if threads:
            torch.set_num_threads(int(threads))
    except Exception:  # noqa: BLE001
        pass
    _WORKER["model"] = load_model_package(pkg_dir, dev)
    _WORKER["overrides"] = dict(overrides or {})
    _WORKER["device"] = dev


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
    ex.shutdown(wait=False, cancel_futures=True)
    for p in procs:
        try:
            if p.is_alive():
                p.terminate()
        except Exception:  # noqa: BLE001
            pass


# ============================================================================================ driver

def run_submission(challenges: Mapping[str, Mapping[str, Any]], out_path: str, *, pkg_dir: Optional[str] = None,
                   cfg: Optional[RunnerConfig] = None, initial: Optional[Mapping[str, Any]] = None,
                   start_time: Optional[float] = None, diagnostics_path: Optional[str] = None) -> Dict[str, Any]:
    """Solve ``challenges`` into ``out_path`` under the global budget (see module docstring).

    ``initial`` (e.g. a submission already written by the notebook) seeds the attempts; ``start_time``
    (``time.time()`` of the notebook start) anchors the global deadline. Returns a summary with
    ``n_tasks, n_solved, n_timeout, n_errors, n_exact_found, seconds, model_loaded, diagnostics``.
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
    summary: Dict[str, Any] = {"n_tasks": len(order), "workers": int(cfg.workers), "pkg_dir": pkg_dir}

    def maybe_write(force: bool = False) -> None:
        nonlocal last_write
        if force or time.time() - last_write >= cfg.rewrite_every_s:
            write_json_atomic(sub, out_path)
            last_write = time.time()

    def take(res: Tuple[str, List[Dict[str, Grid]], Dict[str, Any]]) -> None:
        tid, att, d = res
        _merge(sub, tid, att)
        diags[tid] = d

    pending = list(order)
    if cfg.workers <= 0:
        model = load_model_package(pkg_dir, (cfg.devices or [None])[0])
        summary["model_loaded"] = model is not None
        while pending:
            left = deadline - time.time()
            if left < 0.5:
                break
            tid = pending.pop(0)
            sec = fair_share_seconds(left, len(pending) + 1, 1, cfg.min_task_seconds, cfg.max_task_seconds)
            take(solve_one(tid, challenges[tid], sec, model, cfg.solve_overrides))
            maybe_write()
    else:
        n = int(cfg.workers)
        devices = list(cfg.devices or _default_devices(n))
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        for i in range(n):
            q.put(devices[i % len(devices)])
        threads = cfg.threads_per_worker or max(1, (os.cpu_count() or 1) // n)
        ex = cf.ProcessPoolExecutor(max_workers=n, mp_context=ctx, initializer=_init_worker,
                                    initargs=(pkg_dir, q, threads, dict(cfg.solve_overrides)))
        inflight: Dict[cf.Future, str] = {}
        summary["model_loaded"] = bool(pkg_dir)
        try:
            while pending or inflight:
                left = deadline - time.time()
                if left < 0.5:
                    break
                while pending and len(inflight) < n and left >= 0.5:
                    tid = pending.pop(0)
                    sec = fair_share_seconds(left, len(pending) + 1 + len(inflight), n, cfg.min_task_seconds,
                                             cfg.max_task_seconds)
                    inflight[ex.submit(_worker_solve, tid, dict(challenges[tid]), sec)] = tid
                done, _ = cf.wait(list(inflight), timeout=max(0.05, min(cfg.poll_s, deadline - time.time())),
                                  return_when=cf.FIRST_COMPLETED)
                for f in done:
                    tid = inflight.pop(f)
                    try:
                        take(f.result())
                    except Exception as exc:  # noqa: BLE001 - worker crash: keep the fallback
                        log.warning("worker failed on %s: %s", tid, exc)
                        diags[tid] = {"task_id": tid, "error": repr(exc)}
                maybe_write()
        except cf.process.BrokenProcessPool as exc:
            log.error("worker pool broke (%s); solving the rest in-process", exc)
            pending = list(inflight.values()) + pending
            inflight = {}
            model = load_model_package(pkg_dir, None)
            while pending and deadline - time.time() >= 0.5:
                tid = pending.pop(0)
                sec = fair_share_seconds(deadline - time.time(), len(pending) + 1, 1, cfg.min_task_seconds,
                                         cfg.max_task_seconds)
                take(solve_one(tid, challenges[tid], sec, model, cfg.solve_overrides))
                maybe_write()
        finally:
            for tid in inflight.values():
                diags.setdefault(tid, {"task_id": tid, "error": "global deadline reached while solving"})
            _terminate_pool(ex)
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
    log.info("run_submission: %d tasks, %d solved, %d errors, %d not reached in %.1fs", len(order),
             summary["n_solved"], summary["n_errors"], summary["n_timeout"], summary["seconds"])
    return summary
