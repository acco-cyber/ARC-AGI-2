"""Tests for ``arcjepa.model`` (tiny config, CPU, < 60 s)."""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Dict

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The tiny model is launch-bound (thousands of ~1 KB kernels); a large thread pool only adds spin-up overhead
# per kernel, so the tests run with a small pool (this only affects the test process).
torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))

from arcjepa.core.types import PAD_ID  # noqa: E402
from arcjepa.model import (ARCJEPA, CellEncoder, EMATargetEncoder, GridJEPAEncoder, LossWeights, ModelConfig,  # noqa: E402
                           ObjectEncoder, ProgramEncoder, ProgramTokenizer, RelationEncoder, RuleLatent, Scorer,
                           SimpleNode, TransformationMemory, TransformationPredictor, jepa_losses, make_grid_batch,
                           neighbor_features, parse_sexpr)
from arcjepa.model.program_encoder import SPEC_PRIMITIVE_NAMES, STRUCTURAL_OPS  # noqa: E402

CFG = ModelConfig.tiny()
S, N = CFG.max_side, CFG.max_objects


# --------------------------------------------------------------------------- synthetic data helpers
def rand_grid(g: torch.Generator, h: int, w: int) -> torch.Tensor:
    grid = torch.full((S, S), PAD_ID, dtype=torch.long)
    grid[:h, :w] = torch.randint(0, 10, (h, w), generator=g)
    return grid


def rand_objects(g: torch.Generator, n_valid: int, rows: int) -> Dict[str, torch.Tensor]:
    crops = torch.full((rows, N, S, S), PAD_ID, dtype=torch.long)
    feats = torch.zeros(rows, N, CFG.obj_feat_dim)
    mask = torch.zeros(rows, N, dtype=torch.bool)
    rel = torch.zeros(rows, N, N, CFG.rel_feat_dim)
    for r in range(rows):
        for i in range(n_valid):
            h, w = int(torch.randint(1, 4, (1,), generator=g)), int(torch.randint(1, 4, (1,), generator=g))
            crops[r, i, :h, :w] = torch.randint(1, 10, (h, w), generator=g)
            feats[r, i] = torch.rand(CFG.obj_feat_dim, generator=g)
            mask[r, i] = True
        rel[r, :n_valid, :n_valid] = torch.rand(n_valid, n_valid, CFG.rel_feat_dim, generator=g)
    return {"obj_crops": crops, "obj_feats": feats, "obj_mask": mask, "rel_feats": rel}


def grid_batch(b: int = 2, n_obj: int = 3, seed: int = 0, h: int = 5, w: int = 6) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    grid = torch.stack([rand_grid(g, h, w) for _ in range(b)])
    objs = rand_objects(g, n_obj, b)
    return make_grid_batch(grid, None, objs["obj_crops"], objs["obj_feats"], objs["obj_mask"], objs["rel_feats"])


def episode_batch(b: int = 2, k: int = 3, seed: int = 0, with_out_objects: bool = True) -> Dict[str, torch.Tensor]:
    """B episodes with K context pairs; episode 1 has its last context slot padded and an unknown target."""
    g = torch.Generator().manual_seed(seed)
    ctx_in = torch.stack([torch.stack([rand_grid(g, 4 + i, 5) for i in range(k)]) for _ in range(b)])
    ctx_out = torch.stack([torch.stack([rand_grid(g, 4 + i, 5) for i in range(k)]) for _ in range(b)])
    ctx_mask = torch.ones(b, k, dtype=torch.bool)
    if b > 1 and k > 1:
        ctx_mask[1, -1] = False
        ctx_in[1, -1] = PAD_ID
        ctx_out[1, -1] = PAD_ID
    test_in = torch.stack([rand_grid(g, 6, 4) for _ in range(b)])
    target = torch.stack([rand_grid(g, 6, 4) for _ in range(b)])
    if b > 1:
        target[1] = PAD_ID
    objs = rand_objects(g, 3, b * (k + 1))
    batch = {"ctx_in": ctx_in, "ctx_out": ctx_out, "ctx_mask": ctx_mask, "test_in": test_in, "target": target}
    for key, v in objs.items():
        batch[key] = v.view(b, k + 1, *v.shape[1:])
    if with_out_objects:
        objs_out = rand_objects(g, 2, b * (k + 1))
        for key, v in objs_out.items():
            batch["out_" + key] = v.view(b, k + 1, *v.shape[1:])
    return batch


# --------------------------------------------------------------------------- config
def test_config_presets_and_roundtrip(tmp_path: Path) -> None:
    tiny, v1 = ModelConfig.tiny(), ModelConfig.v1()
    assert tiny.cell_dim == 64 and tiny.cell_layers == 2 and tiny.cell_embed_dim == 32
    assert v1.cell_dim == 256 and v1.cell_layers == 12 and v1.heads == 8 and v1.ffn_dim == 1024
    assert v1.cell_embed_dim == 128 and v1.object_layers == 4 and v1.relation_layers == 4 and v1.edge_dim == 64
    assert v1.global_dim == 384 and v1.jepa_dim == 512 and v1.rule_dim == 256 and v1.program_dim == 256
    assert v1.predictor_dim == 512 and v1.predictor_layers == 6 and v1.program_layers == 6
    assert v1.rule_hidden == (1024, 512) and v1.shape_dim == 128 and v1.program_sym_dim == 128
    assert ModelConfig.v1_wide().predictor_ffn_dim == 2048 and v1.predictor_ffn_dim == 1024
    assert ModelConfig.from_dict(v1.to_dict()) == v1
    path = tmp_path / "config.json"
    tiny.save_json(path)
    assert ModelConfig.load_json(path) == tiny
    with pytest.raises(ValueError):
        ModelConfig(cell_dim=66, heads=8)


# --------------------------------------------------------------------------- encoders
def test_cell_encoder_shapes_and_neighbor_features() -> None:
    torch.manual_seed(0)
    enc = CellEncoder(CFG)
    batch = grid_batch(b=2)
    mask = batch["grid"] != PAD_ID
    feats = neighbor_features(batch["grid"], mask)
    assert feats.shape == (2, S, S, 16)
    assert torch.isfinite(feats).all() and float(feats.min()) >= 0.0 and float(feats.max()) <= 1.0 + 1e-6
    assert float(feats[~mask].abs().sum()) == 0.0
    tokens, pooled = enc(batch["grid"], mask)
    assert tokens.shape == (2, S * S, CFG.cell_dim) and pooled.shape == (2, CFG.cell_dim)
    assert torch.isfinite(tokens).all() and torch.isfinite(pooled).all()
    # padded cells carry zero tokens; an all-PAD grid still yields a finite pooled vector
    assert float(tokens.detach().view(2, S, S, -1)[~mask].abs().sum()) == 0.0
    empty = torch.full((1, S, S), PAD_ID, dtype=torch.long)
    _, p2 = enc(empty)
    assert torch.isfinite(p2).all()


def test_object_encoder_shapes() -> None:
    torch.manual_seed(0)
    enc = ObjectEncoder(CFG)
    batch = grid_batch(b=3, n_obj=4)
    batch["obj_mask"][2] = False  # a grid without any parsed object
    toks, pooled = enc(batch["obj_crops"], batch["obj_feats"], batch["obj_mask"])
    assert toks.shape == (3, N, CFG.object_dim) and pooled.shape == (3, CFG.object_dim)
    assert torch.isfinite(toks).all() and torch.isfinite(pooled).all()
    toks_d = toks.detach()
    assert float(toks_d[~batch["obj_mask"]].abs().sum()) == 0.0
    assert float(toks_d[batch["obj_mask"]].abs().sum()) > 0.0


def test_relation_encoder_shapes_and_subsampling() -> None:
    torch.manual_seed(0)
    enc = RelationEncoder(CFG)
    b = 2
    g = torch.Generator().manual_seed(1)
    obj_tokens = torch.randn(b, N, CFG.object_dim, generator=g)
    rel_feats = torch.rand(b, N, N, CFG.rel_feat_dim, generator=g)
    mask = torch.zeros(b, N, dtype=torch.bool)
    mask[0, :16] = True  # 16 * 15 = 240 ordered pairs > max_pairs (128) -> subsampled
    mask[1, :3] = True
    rel_tokens, pooled = enc(obj_tokens, rel_feats, mask)
    assert rel_tokens.shape == (b, N, N, CFG.relation_dim) and pooled.shape == (b, CFG.relation_dim)
    assert torch.isfinite(rel_tokens).all() and torch.isfinite(pooled).all()
    pair_valid = mask[:, :, None] & mask[:, None, :] & ~torch.eye(N, dtype=torch.bool)
    rel_d = rel_tokens.detach()
    assert float(rel_d[~pair_valid].abs().sum()) == 0.0
    assert float(rel_d[pair_valid].abs().sum()) > 0.0
    none_tokens, pooled2 = enc(obj_tokens, rel_feats, mask, return_tokens=False)
    assert none_tokens is None and torch.allclose(pooled, pooled2, atol=1e-5)


def test_jepa_encoder_shapes_and_determinism() -> None:
    torch.manual_seed(0)
    enc = GridJEPAEncoder(CFG).eval()
    batch = grid_batch(b=2)
    out = enc(batch)
    assert out["z"].shape == (2, CFG.jepa_dim) and out["z_global"].shape == (2, CFG.global_dim)
    assert out["obj_tokens"].shape == (2, N, CFG.object_dim) and out["rel_pooled"].shape == (2, CFG.relation_dim)
    assert out["cell_pooled"].shape == (2, CFG.cell_dim)
    out2 = enc(batch)
    assert torch.allclose(out["z"], out2["z"], atol=1e-6)
    # grid only (no objects) is accepted
    out3 = enc({"grid": batch["grid"]})
    assert out3["z"].shape == (2, CFG.jepa_dim) and torch.isfinite(out3["z"]).all()


# --------------------------------------------------------------------------- EMA target
def test_ema_target_encoder_update() -> None:
    torch.manual_seed(0)
    online = GridJEPAEncoder(CFG)
    target = EMATargetEncoder(online, 0.996, 0.9995)
    assert math.isclose(target.tau(0, 100), 0.996) and math.isclose(target.tau(100, 100), 0.9995)
    assert 0.996 < target.tau(50, 100) < 0.9995
    before = [p.detach().clone() for p in target.encoder.parameters()]
    for p in target.encoder.parameters():
        assert not p.requires_grad
    with torch.no_grad():
        for p in online.parameters():
            p.add_(1.0)
    tau = target.update(0, 100)
    assert math.isclose(tau, 0.996)
    after = list(target.encoder.parameters())
    diffs = [(a - b0).abs().max().item() for a, b0 in zip(after, before)]
    assert max(diffs) > 0.0
    assert all(math.isclose(d, 0.004, rel_tol=1e-3) for d in diffs)  # (1 - tau) * 1.0
    target.train()
    assert not target.encoder.training
    out = target(grid_batch(b=1))
    assert not out["z"].requires_grad and out["z"].shape == (1, CFG.jepa_dim)


# --------------------------------------------------------------------------- predictor / rule latent
def test_predictor_shapes() -> None:
    torch.manual_seed(0)
    pred = TransformationPredictor(CFG)
    b = 2
    z_x = torch.randn(b, CFG.jepa_dim)
    r = torch.randn(b, CFG.rule_dim)
    obj = torch.randn(b, N, CFG.object_dim)
    mask = torch.zeros(b, N, dtype=torch.bool)
    mask[:, :3] = True
    out = pred(z_x, r, obj, mask)
    assert out["z_hat"].shape == (b, CFG.jepa_dim) and out["obj_hat"].shape == (b, N, CFG.object_dim)
    assert out["rel_hat"].shape == (b, CFG.relation_dim)
    out2 = pred(z_x, r)  # no objects at all
    assert torch.isfinite(out2["z_hat"]).all() and out2["obj_hat"].shape == (b, N, CFG.object_dim)


def test_rule_latent_shapes_and_masking() -> None:
    torch.manual_seed(0)
    rl = RuleLatent(CFG).eval()
    b, k = 2, 3
    z_x = torch.randn(b, k, CFG.jepa_dim)
    z_y = torch.randn(b, k, CFG.jepa_dim)
    r = rl(z_x, z_y, torch.ones(b, k, dtype=torch.bool))
    assert r.shape == (b, CFG.rule_dim) and torch.isfinite(r).all()
    mask = torch.tensor([[True, True, False], [True, True, True]])
    r_masked = rl(z_x, z_y, mask)
    r_two = rl(z_x[:, :2], z_y[:, :2], torch.ones(b, 2, dtype=torch.bool))
    assert torch.allclose(r_masked[0], r_two[0], atol=1e-5)  # padded demo has no influence
    r_none = rl(z_x, z_y, torch.zeros(b, k, dtype=torch.bool))
    assert torch.isfinite(r_none).all()


# --------------------------------------------------------------------------- program tokenizer / encoder / scorer
def same_tree(a, b) -> bool:
    """Structural equality across Node implementations (dsl Node vs SimpleNode)."""
    if hasattr(a, "op") != hasattr(b, "op"):
        return False
    if not hasattr(a, "op"):
        return a == b and type(a) is type(b)
    return a.op == b.op and len(a.args) == len(b.args) and all(same_tree(x, y) for x, y in zip(a.args, b.args))


PROGRAMS = [
    SimpleNode("RECOLOR", (SimpleNode("SELECT_LARGEST", (SimpleNode("GET_COMPONENTS4", (SimpleNode("INPUT"),)),)), 3)),
    SimpleNode("RENDER", (SimpleNode("FILTER", (SimpleNode("GET_COMPONENTS4", (SimpleNode("INPUT"),)), "SAME_COLOR",
                          SimpleNode("SELECT_LARGEST", (SimpleNode("GET_COMPONENTS4", (SimpleNode("INPUT"),)),)))),
                          SimpleNode("INPUT"))),
    SimpleNode("APPLY_TO_EACH", (SimpleNode("GET_COMPONENTS8", (SimpleNode("INPUT"),)),
                                 SimpleNode("RECOLOR", (SimpleNode("OBJ"), 2)))),
    SimpleNode("ROTATE90", (SimpleNode("INPUT"),)),
    SimpleNode("RENDER", (SimpleNode("MOVE", (SimpleNode("SELECT_SMALLEST", (SimpleNode("GET_COMPONENTS8",
               (SimpleNode("INPUT"),)),)), (1, -2))), SimpleNode("INPUT"))),
    SimpleNode("IF", (SimpleNode("GET_SYMMETRY", (SimpleNode("INPUT"),)), SimpleNode("REFLECT_H", (SimpleNode("INPUT"),)),
                      SimpleNode("INPUT"))),
    SimpleNode("ALIGN", (SimpleNode("SELECT_ALL", (SimpleNode("INPUT"),)), "center", True)),
    SimpleNode("REPEAT_N", (SimpleNode("INPUT"), 7)),
    SimpleNode("INPUT"),
]


def test_tokenizer_round_trip_and_vocab() -> None:
    tok = ProgramTokenizer()
    for name in SPEC_PRIMITIVE_NAMES + STRUCTURAL_OPS:
        assert name in tok.vocab
    assert len(SPEC_PRIMITIVE_NAMES) == 72
    for prog in PROGRAMS:
        ids = tok.encode(prog)
        assert ids[0] == tok.BOS and ids[-1] == tok.EOS and (len(ids) - 2) % 3 == 0
        back = tok.decode(ids)
        assert type(back) is tok.node_cls
        assert back.to_str() == prog.to_str(), (back.to_str(), prog.to_str())
        assert same_tree(back, prog)
        # decode is robust to padding and works from a string form too
        assert tok.decode(ids + [tok.PAD] * 4).to_str() == prog.to_str()
        assert tok.decode(tok.encode(prog.to_str())).to_str() == prog.to_str()
        assert tok.UNK not in ids  # every op and literal of the test programs is in the vocabulary
    assert parse_sexpr("(RECOLOR (SELECT_LARGEST (GET_COMPONENTS4 INPUT)) 3)") == PROGRAMS[0]
    assert parse_sexpr("(MOVE OBJ (1 -2))") == SimpleNode("MOVE", (SimpleNode("OBJ"), (1, -2)))
    tokens, mask = tok.pad_batch([tok.encode(p) for p in PROGRAMS])
    assert tokens.shape == mask.shape and tokens.shape[0] == len(PROGRAMS)
    assert (tokens[~mask] == tok.PAD).all() and mask[:, 0].all()
    assert tok.kind_ids_tensor().shape == (tok.vocab_size,)


def test_tokenizer_round_trip_on_dsl_programs() -> None:
    """Exact decode(encode(p)) == p on random typed programs of the real DSL (skipped before the DSL lands)."""
    grammar = pytest.importorskip("arcjepa.dsl.grammar")
    import random

    tok = ProgramTokenizer()
    rng = random.Random(0)
    for _ in range(200):
        prog = grammar.random_program(rng, rng.randint(1, 6))
        ids = tok.encode(prog)
        assert tok.UNK not in ids and tok.unk_lit_id not in ids, prog.to_str()
        assert tok.decode(ids) == prog, prog.to_str()
        assert tok.decode(tok.encode(prog.to_str())) == prog


def test_tokenizer_save_load(tmp_path: Path) -> None:
    tok = ProgramTokenizer()
    tok.save(tmp_path / "vocab.json")
    tok2 = ProgramTokenizer.load(tmp_path / "vocab.json")
    assert tok2.itos == tok.itos and tok2.encode(PROGRAMS[0]) == tok.encode(PROGRAMS[0])


def test_program_encoder_and_scorer_shapes() -> None:
    torch.manual_seed(0)
    tok = ProgramTokenizer()
    enc = ProgramEncoder(CFG, tok)
    tokens, mask = tok.pad_batch([tok.encode(p) for p in PROGRAMS])
    z_p = enc(tokens, mask)
    assert z_p.shape == (len(PROGRAMS), CFG.program_dim) and torch.isfinite(z_p).all()
    scorer = Scorer(CFG)
    r = torch.randn(len(PROGRAMS), CFG.rule_dim)
    s = scorer(r, z_p)
    assert s.shape == (len(PROGRAMS),)
    s_many = scorer(r[:2], z_p[:6].view(2, 3, -1))
    assert s_many.shape == (2, 3)
    # one token per AST node: extra padding does not change z_p; an all-PAD row (missing candidate) is finite
    enc.eval()
    with torch.no_grad():
        z_ref = enc(tokens, mask)
        tokens_long, mask_long = tok.pad_batch([tok.encode(p) for p in PROGRAMS] + [[tok.PAD]], max_len=None)
        tokens_long = torch.cat([tokens_long, torch.full((tokens_long.shape[0], 7), tok.PAD)], dim=1)
        mask_long = torch.cat([mask_long, torch.zeros(mask_long.shape[0], 7, dtype=torch.bool)], dim=1)
        mask_long[-1] = False
        z_long = enc(tokens_long, mask_long)
    assert torch.allclose(z_ref, z_long[:-1], atol=1e-5) and torch.isfinite(z_long[-1]).all()
    with pytest.raises(ValueError):
        enc(tokens[:, 1:], mask[:, 1:])  # streams must start with <bos>


# --------------------------------------------------------------------------- memory
def test_memory_add_query_save_load(tmp_path: Path) -> None:
    mem = TransformationMemory(dim=8)
    rng = torch.Generator().manual_seed(0)
    lat = torch.randn(20, 8, generator=rng)
    for i in range(20):
        mem.add(lat[i], f"(PROG_{i} INPUT)", complexity=i % 5 + 1, family=("geometry" if i % 2 else "object"))
    assert len(mem) == 20
    hits = mem.query(lat[7], k=5)
    assert len(hits) == 5 and hits[0]["index"] == 7 and hits[0]["program"] == "(PROG_7 INPUT)"
    assert math.isclose(hits[0]["score"], 1.0, abs_tol=1e-5)
    assert all(hits[i]["score"] >= hits[i + 1]["score"] for i in range(4))
    fam = mem.query(lat[7], k=5, family="geometry")
    assert all(h["family"] == "geometry" for h in fam)
    mem.save(tmp_path / "program_memory.npz")
    assert (tmp_path / "program_memory.npz").exists() and (tmp_path / "program_memory.json").exists()
    loaded = TransformationMemory.load(tmp_path / "program_memory.npz")
    assert len(loaded) == 20 and loaded.query(lat[3], k=1)[0]["index"] == 3
    inst = TransformationMemory()
    inst.load(tmp_path / "program_memory")
    assert len(inst) == 20 and inst.dim == 8
    assert TransformationMemory().query(lat[0]) == []
    mem.save(tmp_path / "memory.v1")  # dotted stem: files are memory.v1.npz / memory.v1.json
    assert (tmp_path / "memory.v1.npz").exists() and (tmp_path / "memory.v1.json").exists()
    assert len(TransformationMemory.load(tmp_path / "memory.v1.json")) == 20


# --------------------------------------------------------------------------- composite model
def test_arcjepa_api() -> None:
    torch.manual_seed(0)
    model = ARCJEPA(CFG).eval()
    batch = episode_batch(b=2, k=3)
    with torch.no_grad():
        z = model.encode_grid(grid_batch(b=2))
        assert z.shape == (2, CFG.jepa_dim)
        r = model.rule_from_episode(batch)
        assert r.shape == (2, CFG.rule_dim) and torch.isfinite(r).all()
        z_t = model.encode_grid(batch)  # episode batch -> test input
        assert z_t.shape == (2, CFG.jepa_dim)
        z_hat = model.predict(z_t, r)
        assert z_hat.shape == (2, CFG.jepa_dim)
        s = model.score_programs(r, model.tokenize(PROGRAMS[:2]))
        assert s.shape == (2,)
        s_all = model.score_programs(r[0], PROGRAMS)  # one rule latent vs many candidate programs
        assert s_all.shape == (len(PROGRAMS),)
        tok3, m3 = model.as_token_batch([PROGRAMS[:3], PROGRAMS[3:5]], nested=True)
        assert tok3.shape[:2] == (2, 3) and not m3[1, 2].any()
        s_nested = model.score_programs(r, (tok3, m3))
        assert s_nested.shape == (2, 3)
        out = model(batch)
        assert out["z_hat"].shape == (2, CFG.jepa_dim) and out["r_task"].shape == (2, CFG.rule_dim)
    # without output objects the context outputs are encoded object-free
    with torch.no_grad():
        r2 = model.rule_from_episode(episode_batch(b=2, k=2, with_out_objects=False))
    assert torch.isfinite(r2).all()


def test_losses_finite_and_decrease() -> None:
    torch.manual_seed(0)
    model = ARCJEPA(CFG)
    target = EMATargetEncoder(model.encoder, CFG.ema_start, CFG.ema_end)
    batch = episode_batch(b=2, k=3)
    pos = PROGRAMS[:2]
    neg = [PROGRAMS[2:5], PROGRAMS[4:7]]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, foreach=True)
    steps = 30
    totals = []
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad()
        losses = jepa_losses(model, target, batch, programs_pos=pos, programs_neg=neg, weights=LossWeights())
        for key in ("L_g", "L_o", "L_r", "L_prog", "L_rank", "L_var", "total"):
            assert torch.isfinite(losses[key]).all(), key
        losses["total"].backward()
        opt.step()
        target.update(step, steps)
        totals.append(float(losses["total"].detach()))
    elapsed = time.time() - t0
    print(f"\n30-step loss curve ({elapsed:.1f}s): first={totals[0]:.4f} last={totals[-1]:.4f}")
    assert totals[-1] < totals[0]
    assert sum(totals[-5:]) / 5 < sum(totals[:5]) / 5
    assert float(losses["L_o"].detach()) > 0.0 and float(losses["n_pred"]) == 6.0  # 5 valid ctx pairs + 1 target


def test_losses_without_programs_or_target() -> None:
    torch.manual_seed(0)
    model = ARCJEPA(CFG)
    target = EMATargetEncoder(model.encoder)
    batch = episode_batch(b=2, k=2, with_out_objects=False)
    batch.pop("target")
    losses = jepa_losses(model, target, batch)
    assert float(losses["L_prog"]) == 0.0 and float(losses["L_rank"]) == 0.0
    assert torch.isfinite(losses["total"]) and float(losses["L_o"]) == 0.0  # no output objects -> no object loss
    losses["total"].backward()


def test_v1_parameter_count() -> None:
    """Full-config parameter count (built on the meta device: no memory, no init cost)."""
    counts = {}
    for name, cfg in (("v1", ModelConfig.v1()), ("v1_wide", ModelConfig.v1_wide())):
        with torch.device("meta"):
            model = ARCJEPA(cfg)
        counts[name] = model.num_parameters()
        if name == "v1":
            parts = {k: sum(p.numel() for p in getattr(model, k).parameters())
                     for k in ("encoder", "predictor", "rule_latent", "program_encoder", "scorer")}
    print(f"\nARCJEPA v1 parameters: {counts['v1'] / 1e6:.2f} M ("
          + ", ".join(f"{k} {v / 1e6:.2f} M" for k, v in parts.items())
          + f"); v1_wide (predictor FFN 2048): {counts['v1_wide'] / 1e6:.2f} M")
    assert 10_000_000 <= counts["v1"] <= 40_000_000
    assert counts["v1_wide"] > counts["v1"]
