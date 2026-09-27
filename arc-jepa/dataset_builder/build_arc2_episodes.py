#!/usr/bin/env python3
"""Build the task-level ARC-AGI-2 reasoning dataset (masked-demonstration episodes).

Sources
  * official ARC-AGI-2 tasks (github.com/arcprize/ARC-AGI-2, Apache-2.0): 1000 training + 120 evaluation
  * google/ARC-GEN generator programs (Apache-2.0) + locally generated fresh pairs (60 per covered task)
  * our own 228 sandbox-verified SDG puzzles (30 pairs each)

Outputs (one directory per config, one JSONL per split; see README.md for schemas)
  tasks/            raw tasks with the task-level split label and per-task statistics
  episodes/         masked-demonstration episodes (canonical + leave-one-out), unaugmented
  episodes_aug/     K deterministic dihedral+colour augmentations per episode (train/val/test only)
  arcgen_fresh/     fresh-pair episodes from ARC-GEN generators, split inherited from the task split
  sdg_hard/         episodes from the 228 hard SDG puzzles
  counterfactual/   negatives (wrong outputs) per episode for contrastive / JEPA InfoNCE training
  rule_programs/    ARC-GEN generator source per covered task (the "latent rule" as executable code)
  splits.json, stats.json, analysis.md

Split rule: TASK-level. 1000 training tasks -> 800 train / 100 val / 100 test (seeded shuffle);
120 public evaluation tasks -> eval_public (benchmark only, never train). No episode of a task
appears in more than one split; ARC-GEN/SDG episodes inherit the split of their task.
"""
import argparse, glob, gzip, hashlib, json, os, random, re, statistics, sys, time
from collections import Counter, defaultdict

DIHEDRAL = ["identity", "rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose", "anti_transpose"]


def dihedral(grid, name):
    g = [list(r) for r in grid]
    if name == "identity":
        return g
    if name == "rot90":      # clockwise
        return [list(r) for r in zip(*g[::-1])]
    if name == "rot180":
        return [r[::-1] for r in g[::-1]]
    if name == "rot270":
        return [list(r) for r in zip(*g)][::-1]
    if name == "flip_h":     # left-right
        return [r[::-1] for r in g]
    if name == "flip_v":     # up-down
        return [list(r) for r in g[::-1]]
    if name == "transpose":
        return [list(r) for r in zip(*g)]
    if name == "anti_transpose":
        return [list(r) for r in zip(*g[::-1])][::-1]
    raise ValueError(name)


def recolor(grid, perm):
    return [[perm[c] for c in row] for row in grid]


def shape(g):
    return [len(g), len(g[0]) if g else 0]


def colors(g):
    return sorted({c for row in g for c in row})


def grid_key(g):
    return hashlib.sha1(json.dumps(g, separators=(",", ":")).encode()).hexdigest()[:16]


def is_symmetric(g):
    for name in DIHEDRAL[1:]:
        try:
            if dihedral(g, name) == g:
                return True
        except Exception:
            pass
    return False


def shape_relation(pairs):
    rel = set()
    outs = {tuple(shape(p["output"])) for p in pairs}
    for p in pairs:
        si, so = shape(p["input"]), shape(p["output"])
        if si == so:
            rel.add("same")
        elif so[0] * so[1] > si[0] * si[1]:
            rel.add("grows")
        else:
            rel.add("shrinks")
    if len(rel) == 1:
        r = rel.pop()
        if r != "same" and len(outs) == 1:
            return "output_constant_shape"
        return r
    return "mixed"


def load_arc_tasks(root):
    tasks = {}
    for split_dir in ("training", "evaluation"):
        for fp in sorted(glob.glob(os.path.join(root, split_dir, "*.json"))):
            tid = os.path.splitext(os.path.basename(fp))[0]
            d = json.load(open(fp, encoding="utf-8"))
            tasks[tid] = {"task_id": tid, "source_split": split_dir, "train": d["train"], "test": d["test"]}
    return tasks


def task_stats(t):
    pairs = t["train"] + t["test"]
    ins = [shape(p["input"]) for p in pairs]
    outs = [shape(p["output"]) for p in pairs]
    return {
        "n_train": len(t["train"]),
        "n_test": len(t["test"]),
        "input_shapes": ins,
        "output_shapes": outs,
        "max_side": max(max(s) for s in ins + outs),
        "max_area": max(s[0] * s[1] for s in ins + outs),
        "n_colors_in": len(set(c for p in pairs for c in colors(p["input"]))),
        "n_colors_out": len(set(c for p in pairs for c in colors(p["output"]))),
        "shape_relation": shape_relation(pairs),
        "any_symmetric_input": any(is_symmetric(p["input"]) for p in pairs),
        "output_shape_consistent": len({tuple(s) for s in outs}) == 1,
    }


def write_jsonl(path, rows, gz=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    opener = gzip.open if gz else open
    with opener(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc-root", required=True, help=".../ARC-AGI-2-main/data")
    ap.add_argument("--arcgen-dir", default=r"E:\Claude code\arc2\phaseC\arcgen_puzzles")
    ap.add_argument("--arcgen-repo", default=r"E:\Claude code\arc2\assets\ARC-GEN\tasks")
    ap.add_argument("--sdg-dir", default=r"E:\Claude code\arc2\phaseC\sdg\puzzles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--aug", type=int, default=8, help="augmentations per real episode")
    ap.add_argument("--arcgen-episodes", type=int, default=12)
    ap.add_argument("--sdg-episodes", type=int, default=8)
    ap.add_argument("--n-val", type=int, default=100)
    ap.add_argument("--n-test", type=int, default=100)
    a = ap.parse_args()
    t0 = time.time()
    rng = random.Random(a.seed)

    # ---------- 1. tasks + split ----------
    tasks = load_arc_tasks(a.arc_root)
    train_ids = sorted(t for t, v in tasks.items() if v["source_split"] == "training")
    eval_ids = sorted(t for t, v in tasks.items() if v["source_split"] == "evaluation")
    assert len(train_ids) == 1000 and len(eval_ids) == 120, (len(train_ids), len(eval_ids))
    shuffled = train_ids[:]
    rng.shuffle(shuffled)
    split_of = {}
    for i, tid in enumerate(shuffled):
        split_of[tid] = "val" if i < a.n_val else ("test" if i < a.n_val + a.n_test else "train")
    for tid in eval_ids:
        split_of[tid] = "eval_public"
    splits = defaultdict(list)
    for tid, s in split_of.items():
        splits[s].append(tid)
    splits = {k: sorted(v) for k, v in splits.items()}

    arcgen_programs = {}
    for fp in glob.glob(os.path.join(a.arcgen_repo, "task_*.py")):
        tid = os.path.basename(fp)[5:-3]
        arcgen_programs[tid] = open(fp, encoding="utf-8").read()
    arcgen_puzzles = {}
    for fp in glob.glob(os.path.join(a.arcgen_dir, "*.json")):
        tid = os.path.splitext(os.path.basename(fp))[0]
        d = json.load(open(fp, encoding="utf-8"))
        pairs = d if isinstance(d, list) else d.get("pairs", [])
        pairs = [p for p in pairs if isinstance(p, dict) and "input" in p and "output" in p]
        if len(pairs) >= 4:
            arcgen_puzzles[tid] = pairs

    task_rows = defaultdict(list)
    tstats = {}
    for tid, t in tasks.items():
        st = task_stats(t)
        st["arcgen_generator"] = tid in arcgen_programs
        st["arcgen_fresh_pairs"] = len(arcgen_puzzles.get(tid, []))
        tstats[tid] = st
        row = {"task_id": tid, "source_split": t["source_split"], "split": split_of[tid],
               "train": t["train"], "test": t["test"], **st}
        task_rows[split_of[tid]].append(row)

    # ---------- 2. episodes (canonical + leave-one-out) ----------
    ep_rows = defaultdict(list)
    ep_index = {}  # episode_id -> row (for aug / counterfactual)
    for tid, t in tasks.items():
        s = split_of[tid]
        pairs = t["train"] + t["test"]
        n_tr = len(t["train"])
        for j, tp in enumerate(t["test"]):
            eid = f"{tid}_canon_{j}"
            row = {"episode_id": eid, "task_id": tid, "split": s, "kind": "canonical",
                   "target_index": n_tr + j, "context": t["train"], "test_input": tp["input"],
                   "target_output": tp["output"], "n_context": n_tr, "target_shape": shape(tp["output"])}
            ep_rows[s].append(row); ep_index[eid] = row
        for i, p in enumerate(pairs):
            ctx = [q for k, q in enumerate(pairs) if k != i]
            eid = f"{tid}_loo_{i}"
            row = {"episode_id": eid, "task_id": tid, "split": s, "kind": "loo",
                   "target_index": i, "context": ctx, "test_input": p["input"],
                   "target_output": p["output"], "n_context": len(ctx), "target_shape": shape(p["output"])}
            ep_rows[s].append(row); ep_index[eid] = row

    # ---------- 3. augmentations ----------
    aug_rows = defaultdict(list)
    for s in ("train", "val", "test"):
        for row in ep_rows[s]:
            r2 = random.Random(f"{a.seed}:{row['episode_id']}")
            seen = set()
            for k in range(a.aug):
                for _ in range(20):
                    dname = r2.choice(DIHEDRAL)
                    perm = list(range(10))
                    tail = perm[1:]
                    r2.shuffle(tail)
                    perm = [0] + tail
                    key = (dname, tuple(perm))
                    if key not in seen:
                        seen.add(key); break
                f = lambda g: recolor(dihedral(g, dname), perm)
                aug_rows[s].append({
                    "episode_id": f"{row['episode_id']}_aug{k}", "base_episode_id": row["episode_id"],
                    "task_id": row["task_id"], "split": s, "kind": row["kind"], "aug_id": k,
                    "dihedral": dname, "color_perm": perm,
                    "context": [{"input": f(p["input"]), "output": f(p["output"])} for p in row["context"]],
                    "test_input": f(row["test_input"]), "target_output": f(row["target_output"]),
                    "n_context": row["n_context"], "target_shape": shape(f(row["target_output"]))})

    # ---------- 4. ARC-GEN fresh episodes ----------
    ag_rows = defaultdict(list)
    for tid, pairs in sorted(arcgen_puzzles.items()):
        if tid not in split_of or split_of[tid] == "eval_public":
            continue
        s = split_of[tid]
        r2 = random.Random(f"{a.seed}:arcgen:{tid}")
        for e in range(a.arcgen_episodes):
            k = r2.choice([3, 4, 5])
            idx = r2.sample(range(len(pairs)), min(k + 1, len(pairs)))
            tgt, ctx = idx[0], idx[1:]
            ag_rows[s].append({
                "episode_id": f"{tid}_arcgen_{e}", "task_id": tid, "split": s, "source": "google/ARC-GEN",
                "context": [pairs[i] for i in ctx], "test_input": pairs[tgt]["input"],
                "target_output": pairs[tgt]["output"], "n_context": len(ctx),
                "target_shape": shape(pairs[tgt]["output"])})

    # ---------- 5. SDG hard episodes ----------
    sdg_rows = []
    sdg_n = 0
    for fp in sorted(glob.glob(os.path.join(a.sdg_dir, "*.json"))):
        if re.search(r"\.(meta|inputs)\.json$", fp):
            continue
        pid = os.path.splitext(os.path.basename(fp))[0]
        d = json.load(open(fp, encoding="utf-8"))
        pairs = d if isinstance(d, list) else d.get("pairs", [])
        pairs = [p for p in pairs if isinstance(p, dict) and "input" in p and "output" in p]
        if len(pairs) < 4:
            continue
        sdg_n += 1
        r2 = random.Random(f"{a.seed}:sdg:{pid}")
        for e in range(a.sdg_episodes):
            k = r2.choice([3, 4, 5])
            idx = r2.sample(range(len(pairs)), min(k + 1, len(pairs)))
            tgt, ctx = idx[0], idx[1:]
            sdg_rows.append({
                "episode_id": f"{pid}_sdg_{e}", "task_id": pid, "split": "train", "source": "sdg-verified",
                "context": [pairs[i] for i in ctx], "test_input": pairs[tgt]["input"],
                "target_output": pairs[tgt]["output"], "n_context": len(ctx),
                "target_shape": shape(pairs[tgt]["output"])})

    # ---------- 6. counterfactual negatives ----------
    by_shape = defaultdict(list)
    for s in ("train", "val", "test", "eval_public"):
        for row in ep_rows[s]:
            by_shape[(s, tuple(row["target_shape"]))].append(row)
    cf_rows = defaultdict(list)
    for s in ("train", "val", "test", "eval_public"):
        for row in ep_rows[s]:
            r2 = random.Random(f"{a.seed}:cf:{row['episode_id']}")
            truth = row["target_output"]
            negs = []
            # (a) wrong orientation of the true output
            for _ in range(10):
                dname = r2.choice(DIHEDRAL[1:])
                try:
                    g = dihedral(truth, dname)
                except Exception:
                    continue
                if g != truth:
                    negs.append({"kind": "dihedral_of_truth", "detail": dname, "grid": g}); break
            # (b) wrong colours of the true output
            for _ in range(10):
                perm = list(range(10)); tail = perm[1:]; r2.shuffle(tail); perm = [0] + tail
                g = recolor(truth, perm)
                if g != truth:
                    negs.append({"kind": "color_perm_of_truth", "detail": perm, "grid": g}); break
            # (c) near-miss: corrupt 5-15% of cells
            h, w = shape(truth)
            n_cells = h * w
            n_corrupt = max(1, int(round(n_cells * r2.uniform(0.05, 0.15))))
            palette = colors(truth) if len(colors(truth)) > 1 else list(range(10))
            g = [list(r) for r in truth]
            for (yy, xx) in r2.sample([(y, x) for y in range(h) for x in range(w)], n_corrupt):
                choices = [c for c in palette if c != g[yy][xx]] or [(g[yy][xx] + 1) % 10]
                g[yy][xx] = r2.choice(choices)
            negs.append({"kind": "cell_corrupt", "detail": n_corrupt, "grid": g})
            # (d) a plausible-looking output from another task with the same shape
            cands = [o for o in by_shape[(s, tuple(row["target_shape"]))] if o["task_id"] != row["task_id"]]
            if cands:
                o = r2.choice(cands)
                if o["target_output"] != truth:
                    negs.append({"kind": "other_task_same_shape", "detail": o["episode_id"], "grid": o["target_output"]})
            cf_rows[s].append({"episode_id": row["episode_id"], "task_id": row["task_id"], "split": s,
                               "target_shape": row["target_shape"], "negatives": negs})

    # ---------- 7. rule programs ----------
    rp_rows = defaultdict(list)
    for tid, src in sorted(arcgen_programs.items()):
        if tid in split_of and split_of[tid] != "eval_public":
            rp_rows[split_of[tid]].append({"task_id": tid, "split": split_of[tid], "language": "python",
                                           "source": "google/ARC-GEN", "license": "Apache-2.0",
                                           "generator_source": src})

    # ---------- 8. write ----------
    out = a.out
    os.makedirs(out, exist_ok=True)
    counts = {}
    for s, rows in task_rows.items():
        counts[f"tasks/{s}"] = write_jsonl(os.path.join(out, "tasks", f"{s}.jsonl"), rows)
    for s, rows in ep_rows.items():
        counts[f"episodes/{s}"] = write_jsonl(os.path.join(out, "episodes", f"{s}.jsonl"), rows)
    for s, rows in aug_rows.items():
        counts[f"episodes_aug/{s}"] = write_jsonl(os.path.join(out, "episodes_aug", f"{s}.jsonl"), rows)
    for s, rows in ag_rows.items():
        counts[f"arcgen_fresh/{s}"] = write_jsonl(os.path.join(out, "arcgen_fresh", f"{s}.jsonl"), rows)
    counts["sdg_hard/train"] = write_jsonl(os.path.join(out, "sdg_hard", "train.jsonl"), sdg_rows)
    for s, rows in cf_rows.items():
        counts[f"counterfactual/{s}"] = write_jsonl(os.path.join(out, "counterfactual", f"{s}.jsonl"), rows)
    for s, rows in rp_rows.items():
        counts[f"rule_programs/{s}"] = write_jsonl(os.path.join(out, "rule_programs", f"{s}.jsonl"), rows)
    json.dump({"seed": a.seed, "rule": "task-level; 1000 training tasks -> 800/100/100 seeded shuffle; "
                                       "120 public evaluation tasks -> eval_public (benchmark only)",
               **splits}, open(os.path.join(out, "splits.json"), "w"), indent=1)

    # ---------- 9. analysis ----------
    def summarize(ids):
        st = [tstats[t] for t in ids]
        areas = [x["max_area"] for x in st]
        return {
            "n_tasks": len(ids),
            "n_train_pairs_hist": dict(sorted(Counter(x["n_train"] for x in st).items())),
            "n_test_hist": dict(sorted(Counter(x["n_test"] for x in st).items())),
            "multi_test_frac": round(sum(x["n_test"] > 1 for x in st) / len(st), 4),
            "max_side_median": statistics.median(x["max_side"] for x in st),
            "max_area_median": statistics.median(areas),
            "max_area_mean": round(statistics.mean(areas), 1),
            "frac_max_side_30": round(sum(x["max_side"] == 30 for x in st) / len(st), 4),
            "shape_relation_hist": dict(Counter(x["shape_relation"] for x in st)),
            "n_colors_in_median": statistics.median(x["n_colors_in"] for x in st),
            "n_colors_out_median": statistics.median(x["n_colors_out"] for x in st),
            "symmetric_input_frac": round(sum(x["any_symmetric_input"] for x in st) / len(st), 4),
            "output_shape_consistent_frac": round(sum(x["output_shape_consistent"] for x in st) / len(st), 4),
            "arcgen_generator_frac": round(sum(x["arcgen_generator"] for x in st) / len(st), 4),
            "arcgen_fresh_pairs_total": sum(x["arcgen_fresh_pairs"] for x in st),
        }
    stats = {"built": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": a.seed, "counts": counts,
             "training_1000": summarize(train_ids), "evaluation_120": summarize(eval_ids),
             "split_train_800": summarize(splits["train"]), "split_val_100": summarize(splits["val"]),
             "split_test_100": summarize(splits["test"]),
             "sdg_puzzles": sdg_n, "arcgen_programs": len(arcgen_programs),
             "arcgen_puzzles_with_fresh_pairs": len(arcgen_puzzles),
             "elapsed_s": round(time.time() - t0, 1)}
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1)

    L = []
    L.append("# ARC-AGI-2 task analysis (computed from the official task files, %s)\n" % stats["built"])
    L.append("| statistic | training (1000) | public evaluation (120) |\n|---|---|---|")
    keys = ["n_train_pairs_hist", "n_test_hist", "multi_test_frac", "max_side_median", "max_area_median",
            "max_area_mean", "frac_max_side_30", "shape_relation_hist", "n_colors_in_median",
            "n_colors_out_median", "symmetric_input_frac", "output_shape_consistent_frac",
            "arcgen_generator_frac", "arcgen_fresh_pairs_total"]
    for k in keys:
        L.append(f"| {k} | {stats['training_1000'][k]} | {stats['evaluation_120'][k]} |")
    L.append("\n## Row counts per config/split\n")
    L.append("| file | rows |\n|---|---|")
    for k, v in counts.items():
        L.append(f"| {k} | {v} |")
    L.append("\n## Split sizes\n")
    for s in ("train", "val", "test", "eval_public"):
        L.append(f"- {s}: {len(splits[s])} tasks")
    open(os.path.join(out, "analysis.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
