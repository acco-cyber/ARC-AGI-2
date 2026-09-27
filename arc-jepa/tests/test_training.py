"""Tests for arcjepa.training (CPU, < 60 s): config loading, schedules, samplers, datasets, stage resume,
retrieval, and an end-to-end ``train_all`` run (debug config) that exports a package loadable by
``ARCJEPA.load_package``."""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arcjepa.model.arcjepa import ARCJEPA  # noqa: E402
from arcjepa.training import common as C  # noqa: E402
from arcjepa.training.train_real import EvalLeakError, family_weights, stream_episodes  # noqa: E402

DEBUG_CFG = ROOT / "configs" / "debug.yaml"

# Small overrides that keep the end-to-end run well inside the test budget.
FAST = ["synthetic.n_tasks=60", "synthetic.heldout_min=4", "synthetic.heldout_max=4", "synthetic.max_ctx=3",
        "stages.A.max_steps=2", "stages.B.max_steps=2", "stages.C.max_steps=2", "stages.D.max_steps=2",
        "stages.A.batch_size=2", "stages.B.batch_size=2", "stages.C.batch_size=2", "stages.D.batch_size=2",
        "stages.B.max_ctx=3", "stages.B.max_per_config=4", "stages.B.val_episodes=2", "memory.max_synthetic=6",
        "memory.max_real=2", "memory.pseudo_label_seconds=0.1", "training.log_every=1", "eval.every_steps=0"]


def _has_data() -> bool:
    try:
        from arcjepa.data.hf_loader import has_local_root

        return has_local_root()
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="module")
def synth(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    """A 60-task synthetic store shared by the unit tests."""
    d = tmp_path_factory.mktemp("synth")
    cfg = C.load_config(DEBUG_CFG, FAST)
    store, train, held = C.prepare_synthetic(cfg, d, C.DistInfo())
    return {"cfg": cfg, "store": store, "train": train, "held": held, "dir": d}


# ------------------------------------------------------------------------------------------------ config
def test_configs_load_and_inherit() -> None:
    base = C.load_config(ROOT / "configs" / "base.yaml")
    assert base["training"]["lr"] == pytest.approx(1.5e-4)
    assert base["training"]["betas"] == [0.9, 0.95]
    assert base["training"]["weight_decay"] == pytest.approx(0.05)
    assert base["training"]["grad_clip"] == pytest.approx(1.0)
    assert [base["stages"][s]["lr"] for s in "ABCD"] == pytest.approx([1.5e-4, 3e-5, 1e-4, 1e-4])
    assert "eval_public" not in base["stages"]["B"]["hf_splits"]
    dbg = C.load_config(DEBUG_CFG, ["stages.A.max_steps=3", "eval.retrieval_k=[1, 4]"])
    assert dbg["model"]["preset"] == "tiny" and dbg["stages"]["A"]["max_steps"] == 3
    assert dbg["stages"]["A"]["lr"] == pytest.approx(1.5e-4)  # inherited from base
    assert dbg["eval"]["retrieval_k"] == [1, 4]
    assert dbg["synthetic"]["n_tasks"] == 200
    assert (base["stages"]["A"]["batch_size"], base["stages"]["B"]["batch_size"]) == (128, 32)  # spec per-GPU
    for name in ("l4x4", "kaggle"):
        c = C.load_config(ROOT / "configs" / f"{name}.yaml")
        assert c["training"]["precision"] == "bf16"
        st = c["stages"]
        # measured per-GPU micro-batches that fit a 22 GiB L4 (8 x ~1.3 GB stage-A / 4 x ~2.4 GB stage-B episodes at
        # 30x30 in bf16), with gradient accumulation up to the documented global batches over 4 GPUs
        assert {s: (st[s]["batch_size"], st[s]["accum_steps"]) for s in "ABCD"} == {
            "A": (8, 8), "B": (4, 8), "C": (8, 8), "D": (8, 4)}
        assert {s: st[s]["batch_size"] * st[s]["accum_steps"] * 4 for s in "ABCD"} == {
            "A": 256, "B": 128, "C": 256, "D": 128}
        assert c["training"]["oom_max_split"] == 2  # a CUDA OOM halves the micro-batch once
        fr = c["training"]["time_fractions"]
        assert set(fr) == set("ABCD") and sum(fr.values()) == pytest.approx(1.0) and fr["A"] > 0.5 > fr["B"] > 0
        assert "sdg_hard" not in st["B"]["configs"] and not st["B"]["allow_sdg_hard"]
    kag = C.load_config(ROOT / "configs" / "kaggle.yaml")
    assert kag["export"]["dir"] == "/kaggle/working/arc_jepa_pkg"
    assert kag["memory"]["pseudo_label_total_seconds"] <= 300 and kag["memory"]["max_synthetic"] == 20000
    mc = C.model_config_from(dbg)
    assert mc.name == "tiny" and mc.cell_dim == 64 and mc.rule_dim == 64
    assert C.model_config_from(base).jepa_dim == 512


def test_lr_multiplier_warmup_and_cosine() -> None:
    vals = [C.lr_multiplier(s, 100, 10, 0.01) for s in range(100)]
    assert vals[0] == pytest.approx(0.1) and vals[9] == pytest.approx(1.0)
    assert all(a >= b - 1e-12 for a, b in zip(vals[9:], vals[10:]))  # monotone after warmup
    assert C.lr_multiplier(100, 100, 10, 0.01) == pytest.approx(0.01)
    assert C.lr_multiplier(0, 1, 5, 0.0) == pytest.approx(1.0)


def test_time_shrunk_schedule_keeps_warmup_fraction() -> None:
    # stage A on 4 x L4: ~18.5k epoch-planned steps with a 10 % warmup, but the time box allows ~3k
    total, warmup = C.shrink_schedule(4, 18_550, 1_855, 0.1, sec_per_step=3.5, seconds_left=3.5 * 2_996)
    assert total == 3_000 and warmup == 300
    assert C.lr_multiplier(300, total, warmup, 0.01) == pytest.approx(1.0, abs=1e-3)  # peak lr reached at 10 %
    # a later, faster estimate never grows the schedule or the warmup back
    assert C.shrink_schedule(500, total, warmup, 0.1, 1.0, 1e9) == (3_000, 300)
    # a slower estimate shrinks both again, never below step + 1
    assert C.shrink_schedule(500, total, warmup, 0.1, 10.0, 10.0 * 1_500) == (2_000, 200)
    assert C.shrink_schedule(900, 2_000, 200, 0.1, 10.0, 0.0) == (901, 90)


def test_stage_budget_split() -> None:
    from arcjepa.training.train_all import stage_budget

    fr = {"A": 0.45, "B": 0.25, "C": 0.15, "D": 0.15}
    assert stage_budget("A", "ABCD", 100.0, fr) == pytest.approx(45.0)
    assert stage_budget("C", "CD", 100.0, fr) == pytest.approx(50.0)
    assert stage_budget("D", "D", -5.0, fr) == 0.0


# ------------------------------------------------------------------------------------------------ sampling
def test_epoch_batch_sampler_deterministic_sharded_and_weighted() -> None:
    s0 = C.EpochBatchSampler(50, 4, seed=3, rank=0, world_size=2)
    s1 = C.EpochBatchSampler(50, 4, seed=3, rank=1, world_size=2)
    assert s0.batches(0) == C.EpochBatchSampler(50, 4, seed=3, rank=0, world_size=2).batches(0)
    assert s0.batches(0) != s0.batches(1)
    a = [i for b in s0.batches(0) for i in b]
    b = [i for bb in s1.batches(0) for i in bb]
    assert len(s0.batches(0)) == len(s1.batches(0)) == len(s0)
    assert not set(a) & set(b)
    w = [10.0 if i < 5 else 1.0 for i in range(40)]
    ws = C.EpochBatchSampler(40, 5, seed=1, weights=w, epoch_size=20)
    for ep in range(5):
        idx = ws.indices(ep)
        assert len(idx) == 20 and len(set(idx)) == 20  # without replacement: never a duplicate in an epoch
    hits = sum(sum(1 for i in ws.indices(ep) if i < 5) for ep in range(20)) / 20
    assert hits > 5 * 20 / 40 * 1.5  # the up-weighted items are oversampled
    fw = family_weights(["a", "a", "a", "a", "b"], 1.0)
    assert fw[4] == pytest.approx(4 * fw[0])


# ------------------------------------------------------------------------------------------------ data
def test_synth_dataset_items_and_negatives(synth: Dict[str, Any]) -> None:
    store = synth["store"]
    assert len(store) == 60 and synth["held"] and not set(synth["held"]) & set(synth["train"])
    ds = C.SynthEpisodeDataset(store, synth["train"], None, max_ctx=5, out_objects=True, negatives=8, seed=5)
    it = ds[0]
    assert it["ctx_in"].shape == (5, 30, 30) and it["out_obj_mask"].shape == (6, 64)
    assert it["out_rel_feats"].shape == (6, 64, 64, 24)
    assert bool(it["target"].ne(10).any())  # synthetic episodes always carry a known target
    assert isinstance(it["program"], str) and len(it["negatives"]) == 8
    assert it["program"] not in it["negatives"]
    again = ds[0]
    assert torch.equal(it["ctx_in"], again["ctx_in"]) and it["negatives"] == again["negatives"]
    batch = C.collate_items([ds[i] for i in range(3)])
    assert batch["ctx_in"].shape[0] == 3 and len(batch["programs_neg"]) == 3


@pytest.mark.skipif(not _has_data(), reason="local HF mirror not available")
def test_real_stream_never_reads_eval_public() -> None:
    from arcjepa.data.hf_loader import resolve_root

    root = resolve_root()
    with pytest.raises(EvalLeakError):
        stream_episodes(root, "episodes", "eval_public", lambda t: True, 1)
    eps = stream_episodes(root, "episodes", "train", lambda t: True, 3)
    assert len(eps) == 3 and all(e.target_output for e in eps)


# ------------------------------------------------------------------------------------------------ optimiser / stages
def _ctx(cfg: Dict[str, Any], synth: Dict[str, Any], out: Path) -> C.TrainContext:
    C.set_seed(0)
    model = C.build_model(cfg)
    return C.TrainContext(cfg=cfg, model=model, target=C.build_target(model), device=torch.device("cpu"),
                          dist=C.DistInfo(), out_dir=out, metrics=C.MetricsLogger(out / "metrics.jsonl"),
                          ckpt=C.Checkpointer(out / "ck"), parser=None, store=synth["store"],
                          synth_train=synth["train"], synth_heldout=synth["held"])


def test_optimizer_groups_freeze_core(synth: Dict[str, Any]) -> None:
    model = C.build_model(synth["cfg"])
    opt, trainable = C.build_optimizer(model, synth["cfg"], {"lr": 1e-4, "core_lr_mult": 0.0,
                                                             "train_modules": ["program_encoder", "scorer"]})
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.scorer.parameters())
    assert opt.defaults["betas"] == (0.9, 0.95)
    n = sum(p.numel() for p in trainable)
    assert n == sum(p.numel() for m in (model.program_encoder, model.scorer) for p in m.parameters())
    assert {g["weight_decay"] for g in opt.param_groups} == {0.0, 0.05}


def test_run_stage_logs_and_resumes_mid_stage(synth: Dict[str, Any], tmp_path: Path, monkeypatch) -> None:
    cfg = C.load_config(DEBUG_CFG, FAST + ["training.checkpoint_minutes=0", "stages.D.max_steps=3",
                                           "eval.every_steps=2"])
    ctx = _ctx(cfg, synth, tmp_path)
    saved: List[Dict[str, Any]] = []
    real_save = C.Checkpointer.save

    def spy(self: C.Checkpointer, state: Dict[str, Any], name: str = "last.pt"):
        saved.append({k: (dict(v) if isinstance(v, dict) else v) for k, v in state.items()})
        return real_save(self, state, name)

    monkeypatch.setattr(C.Checkpointer, "save", spy)
    from arcjepa.training.train_program_encoder import run_stage_d

    before = [p.detach().clone() for p in ctx.model.scorer.parameters()]
    summ = run_stage_d(ctx, 120.0)
    assert summ["steps"] == 3 and summ["steps_this_run"] == 3 and ctx.completed == ["D"]
    assert any(not torch.equal(a, b) for a, b in zip(before, ctx.model.scorer.parameters()))
    recs = ctx.metrics.read()
    train_recs = [r for r in recs if r["kind"] == "train"]
    assert train_recs and all(k in train_recs[-1] for k in ("loss_total", "loss_prog", "loss_rank", "collapse_std"))
    assert any("retrieval@1" in r and "retrieval@8" in r for r in recs if r["kind"] == "eval")
    mid = [s for s in saved if s.get("stage") == "D" and s["stage_state"]["step"] == 1]
    assert mid, "a mid-stage checkpoint must have been written"
    # resume from the step-1 checkpoint: exactly the 2 remaining steps run
    ctx2 = _ctx(cfg, synth, tmp_path / "r")
    ctx2.model.load_state_dict(mid[0]["model"])
    ctx2.target.load_state_dict(mid[0]["target"])
    ctx2.resume = mid[0]
    summ2 = run_stage_d(ctx2, 120.0)
    assert summ2["steps"] == 3 and summ2["steps_this_run"] == 2


# ------------------------------------------------------------------------------------------------ end to end
def test_train_all_debug_end_to_end_and_package(tmp_path: Path) -> None:
    from arcjepa.training.train_all import main

    out = tmp_path / "run"
    args = ["--config", str(DEBUG_CFG), "--hours", "0.02", "--out", str(out)]
    for ov in FAST:
        args += ["--set", ov]
    summary = main(args)
    assert summary["completed"] == ["A", "B", "C", "D"] or (not _has_data() and "B" in summary["completed"])
    pkg = out / "package"
    for f in ("config.json", "vocab.json", "program_memory.npz", "programs.json"):
        assert (pkg / f).is_file(), f
    assert (pkg / "model.safetensors").is_file() or (pkg / "model.pt").is_file()
    assert (out / "checkpoints" / "last.pt").is_file()
    recs = [json.loads(l) for l in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    kinds = {r["kind"] for r in recs}
    assert {"start", "train", "eval", "stage_end", "export"} <= kinds
    assert all(r["stage"] in "ABCD" for r in recs if r["kind"] == "train")
    for r in recs:
        if r["kind"] == "train":
            assert r["loss_total"] is not None and r["collapse_std"] is not None

    model = ARCJEPA.load_package(pkg)
    meta = json.loads((pkg / "config.json").read_text(encoding="utf-8"))
    assert meta["model"]["name"] == "tiny" and model.cfg.rule_dim == 64
    mem = model.memory
    assert mem is not None and len(mem) == meta["memory"]["size"] >= 6
    sources = {rec.get("source") for rec in mem.records}
    assert "synthetic" in sources
    programs = json.loads((pkg / "programs.json").read_text(encoding="utf-8"))["records"]
    from arcjepa.data.hf_loader import EVAL_PUBLIC

    # the exported memory holds only synthetic train-split tasks and 700-split training tasks
    assert {r.get("source") for r in programs} <= {"synthetic", "arc"}
    assert all(str(r["task_id"]).startswith("syn") for r in programs if r.get("source") == "synthetic")
    if _has_data():
        from arcjepa.data.hf_loader import load_resplit
        from arcjepa.training.train_real import official_task_splits

        splits = official_task_splits()
        assert all(splits.get(r["task_id"]) != EVAL_PUBLIC for r in programs if r.get("source") == "arc")
        train700 = set(load_resplit()["train"])
        arc_ids = [r["task_id"] for r in programs if r.get("source") == "arc"]
        assert arc_ids and set(arc_ids) <= train700
    # the loaded model reproduces the trained weights exactly
    state = torch.load(out / "checkpoints" / "last.pt", map_location="cpu", weights_only=False)
    ref = ARCJEPA(model.cfg, model.tokenizer)
    ref.load_state_dict(state["model"])
    ref.eval()
    r = torch.randn(3, model.cfg.rule_dim)
    progs = [rec["program"] for rec in mem.records if rec["program"]][:3]
    with torch.no_grad():
        assert torch.allclose(model.score_programs(r[:1], progs), ref.score_programs(r[:1], progs), atol=1e-6)
    hits = mem.query(r[0].numpy(), k=4)
    assert len(hits) == 4

    # a second invocation resumes: every stage is already complete, so it only re-exports
    summary2 = main(args)
    assert summary2["stages"] == {} and summary2["completed"] == summary["completed"]
    assert meta["export_complete"] is True


# ------------------------------------------------------------------------------------------------ 09-26 fixes
def test_stage_b_refuses_sdg_hard_and_configs_exclude_it() -> None:
    from arcjepa.training.train_real import DEFAULT_CONFIGS, stage_b_configs

    assert "sdg_hard" not in DEFAULT_CONFIGS
    for name in ("base", "debug", "l4x4", "kaggle"):
        b = C.load_config(ROOT / "configs" / f"{name}.yaml")["stages"]["B"]
        assert "sdg_hard" not in (b.get("configs") or DEFAULT_CONFIGS), name
        assert not b.get("allow_sdg_hard"), name
        assert stage_b_configs(b)[1] == set()
    with pytest.raises(EvalLeakError):
        stage_b_configs({"configs": ["episodes", "sdg_hard"]})
    with pytest.raises(EvalLeakError):
        stage_b_configs({"configs": ["sdg_hard"], "allow_sdg_hard": False, "sdg_allowlist": ["x"]})
    confs, allow = stage_b_configs({"configs": ["sdg_hard"], "allow_sdg_hard": True, "sdg_allowlist": ["abc"]})
    assert confs == ["sdg_hard"] and allow == {"abc"}


@pytest.mark.skipif(not _has_data(), reason="local HF mirror not available")
def test_stage_b_keeps_only_700_train_ids_and_memory_is_train_only() -> None:
    from arcjepa.data.hf_loader import load_resplit
    from arcjepa.training.train_all import memory_sources
    from arcjepa.training.train_real import load_real_data

    splits = load_resplit()
    train_ids = set(splits["train"])
    cfg = C.load_config(DEBUG_CFG, ["stages.B.max_per_config=40", "stages.B.val_episodes=4"])
    eps, fams, val = load_real_data(cfg)
    assert eps and len(eps) == len(fams)
    assert {e.task_id for e in eps} <= train_ids
    assert {e.task_id for e in val} <= set(splits["val"])
    # opting into sdg_hard with an empty allowlist still admits no sdg row
    cfg2 = C.load_config(DEBUG_CFG, ["stages.B.configs=[sdg_hard]", "stages.B.allow_sdg_hard=true",
                                     "stages.B.max_per_config=40", "stages.B.val_episodes=0"])
    eps2, _, _ = load_real_data(cfg2)
    assert eps2 == []
    # memory: real tasks are 700-split train ids only; blocked = eval + val + holdout
    _, real, _, blocked = memory_sources(C.load_config(DEBUG_CFG, ["memory.max_real=12"]), None, [])
    assert real and set(real) <= train_ids and not set(real) & set(blocked)
    assert set(splits["val"]) <= set(blocked) and set(splits["holdout"]) <= set(blocked)


def test_init_distributed_sets_long_collective_timeout(monkeypatch) -> None:
    import datetime

    import torch.distributed as tdist

    seen: Dict[str, Any] = {}
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(tdist, "is_initialized", lambda: False)
    monkeypatch.setattr(tdist, "init_process_group", lambda **kw: seen.update(kw))
    d = C.init_distributed("cpu")
    assert d.enabled and d.rank == 1 and not d.is_main
    assert seen["backend"] == "gloo" and seen["timeout"] >= datetime.timedelta(hours=1)


def test_train_all_leaves_process_group_before_export(tmp_path: Path, monkeypatch) -> None:
    """Every rank tears the process group down before rank 0 exports (no collective pending in the export)."""
    from arcjepa.training import export as E
    from arcjepa.training import train_all as TA

    calls: List[str] = []
    real_cleanup, real_export = TA.cleanup_distributed, E.export_package
    monkeypatch.setattr(TA, "cleanup_distributed", lambda d: (calls.append("cleanup"), real_cleanup(d)))
    monkeypatch.setattr(E, "export_package", lambda *a, **k: (calls.append("export"), real_export(*a, **k))[1])
    args = ["--config", str(DEBUG_CFG), "--hours", "0.02", "--out", str(tmp_path / "run"), "--stages", "A"]
    for ov in FAST + ["memory.include_real=false"]:
        args += ["--set", ov]
    summary = TA.main(args)
    assert calls[:2] == ["cleanup", "export"], calls
    assert summary["package"]["memory_size"] > 0


def test_export_writes_loadable_package_before_memory(tmp_path: Path, monkeypatch) -> None:
    from arcjepa.training import export as E

    cfg = C.load_config(DEBUG_CFG, FAST)
    model = C.build_model(cfg)

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("killed while building the memory")

    monkeypatch.setattr(E, "build_program_memory", boom)
    with pytest.raises(RuntimeError):
        E.export_package(model, cfg, tmp_path / "pkg")
    meta = json.loads((tmp_path / "pkg" / "config.json").read_text(encoding="utf-8"))
    assert meta["export_complete"] is False
    loaded = ARCJEPA.load_package(tmp_path / "pkg")  # loadable without memory
    assert loaded.memory is None and loaded.cfg.rule_dim == model.cfg.rule_dim
    assert E.resolve_export_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert E.resolve_export_device("cpu").type == "cpu"


def test_pseudo_label_total_budget_caps_the_search(monkeypatch) -> None:
    from arcjepa.core.types import Pair, Task
    from arcjepa.training import export as E

    seen: List[float] = []

    def fake(task: Any, seconds: float) -> Any:  # a search that uses its whole budget
        import time

        seen.append(seconds)
        time.sleep(seconds)
        return None

    monkeypatch.setattr(E, "pseudo_label", fake)
    cfg = C.load_config(DEBUG_CFG, FAST)
    model = C.build_model(cfg)
    g = [[1, 0], [0, 1]]
    tasks = {f"t{i}": Task(f"t{i}", [Pair(g, g), Pair(g, g)], [Pair(g, g)]) for i in range(6)}
    mem = E.build_program_memory(model, real_tasks=tasks, pseudo_label_seconds=0.5, pseudo_label_total_seconds=0.9)
    assert len(mem) == 6 and seen and all(s <= 0.5 for s in seen)
    assert sum(seen) <= 0.9 + 1e-6


def test_bf16_autocast_forward_backward_tiny() -> None:
    """The CUDA bf16 path (never run on CPU by the trainer) emulated with CPU autocast: no dtype errors."""
    from arcjepa.model.losses import jepa_losses

    from arcjepa.core.types import Episode, Pair

    cfg = C.load_config(DEBUG_CFG, FAST)
    C.set_seed(0)
    model = C.build_model(cfg)
    target = C.build_target(model)
    g = [[(r * 3 + c) % 10 for c in range(6)] for r in range(5)]
    eps = [Episode(episode_id=f"e{i}", task_id=f"e{i}", split="synthetic",
                   context=[Pair(g, [row[::-1] for row in g]) for _ in range(3)], test_input=g,
                   target_output=[row[::-1] for row in g], source="synthetic") for i in range(2)]
    batch = C.collate_items([C.encode_item(e, C.resolve_parser(None), 3, out_objects=True) for e in eps])
    progs = ["(REFLECT_V INPUT)", "(REFLECT_H INPUT)"]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = jepa_losses(model, target, batch, programs_pos=progs, programs_neg=[[progs[1]], [progs[0]]],
                          weights=C.loss_weights(cfg, {}))
    assert torch.isfinite(out["total"].float())
    out["total"].float().backward()
    assert any(p.grad is not None for p in model.encoder.parameters())
    rel = model.encoder.relations
    n = model.cfg.max_objects
    with torch.autocast("cpu", dtype=torch.bfloat16):
        full, pooled = rel(torch.randn(2, n, model.cfg.object_dim), torch.rand(2, n, n, model.cfg.rel_feat_dim),
                           torch.arange(n).expand(2, n) < 5, return_tokens=True)
    assert full.shape[:3] == (2, n, n) and torch.isfinite(full.float()).all()


def test_run_stage_halves_micro_batch_on_cuda_oom(synth: Dict[str, Any], tmp_path: Path, monkeypatch) -> None:
    from arcjepa.training.train_program_encoder import run_stage_d

    cfg = C.load_config(DEBUG_CFG, FAST + ["stages.D.max_steps=2", "eval.every_steps=0"])
    ctx = _ctx(cfg, synth, tmp_path)
    real = C.jepa_losses
    sizes: List[int] = []

    def flaky(model: Any, target: Any, batch: Dict[str, Any], **kw: Any) -> Any:
        n = int(batch["ctx_in"].shape[0])
        sizes.append(n)
        if n > 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real(model, target, batch, **kw)

    monkeypatch.setattr(C, "jepa_losses", flaky)
    summ = run_stage_d(ctx, 120.0)
    assert summ["steps"] == 2 and summ["oom_split"] == 2
    assert sizes[0] == 2 and set(sizes[1:]) == {1}
    assert any(r["kind"] == "oom" for r in ctx.metrics.read())
    # an OOM that persists at the halved micro-batch is re-raised (halving happens once by default)
    monkeypatch.setattr(C, "jepa_losses", lambda *a, **k: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("x")))
    with pytest.raises(torch.cuda.OutOfMemoryError):
        run_stage_d(_ctx(cfg, synth, tmp_path / "b"), 120.0)
    assert C.split_bounds(5, 2) == [(0, 2), (2, 5)] and C.split_bounds(1, 4) == [(0, 1)]
