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
    for name in ("l4x4", "kaggle"):
        c = C.load_config(ROOT / "configs" / f"{name}.yaml")
        assert c["training"]["precision"] == "bf16" and c["stages"]["A"]["batch_size"] == 64
    assert C.load_config(ROOT / "configs" / "kaggle.yaml")["export"]["dir"] == "/kaggle/working/arc_jepa_pkg"
    mc = C.model_config_from(dbg)
    assert mc.name == "tiny" and mc.cell_dim == 64 and mc.rule_dim == 64
    assert C.model_config_from(base).jepa_dim == 512


def test_lr_multiplier_warmup_and_cosine() -> None:
    vals = [C.lr_multiplier(s, 100, 10, 0.01) for s in range(100)]
    assert vals[0] == pytest.approx(0.1) and vals[9] == pytest.approx(1.0)
    assert all(a >= b - 1e-12 for a, b in zip(vals[9:], vals[10:]))  # monotone after warmup
    assert C.lr_multiplier(100, 100, 10, 0.01) == pytest.approx(0.01)
    assert C.lr_multiplier(0, 1, 5, 0.0) == pytest.approx(1.0)


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

    if _has_data():
        from arcjepa.training.train_real import official_task_splits

        splits = official_task_splits()
        assert all(splits.get(r["task_id"]) != EVAL_PUBLIC for r in programs if r.get("source") == "arc")
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
