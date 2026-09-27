#!/usr/bin/env python3
"""Publish one ARC-JEPA model version to Hugging Face, GitHub (release + asset) and Kaggle Models.

  set HF_TOKEN, GH_TOKEN (and a working kaggle CLI), then
  python publish_version.py --version v1 --pkg <arc_jepa_pkg dir> --metrics <metrics.json> --notes "..."

Each target is independent: a failure on one is reported and the others still run.
"""
import argparse, datetime, json, os, shutil, subprocess, sys, tempfile, urllib.request, urllib.error, zipfile

HF_REPO = "koushikz1/arc-jepa"
GH_REPO = "acco-cyber/ARC-AGI-2"
KG_OWNER, KG_MODEL, KG_FRAMEWORK, KG_INSTANCE = "poby7722", "arc-jepa", "PyTorch", "default"
PKG_FILES = ["model.safetensors", "model.pt", "config.json", "vocab.json", "program_memory.npz", "programs.json"]


def model_card(version, metrics, notes, history):
    rows = "\n".join(f"| {h['version']} | {h.get('date','')} | {h.get('public_eval','n/a')} | {h.get('val150','n/a')} | "
                     f"{h.get('params','n/a')} | {h.get('notes','')} |" for h in history)
    return f"""---
license: apache-2.0
tags: [arc-agi, arc-agi-2, jepa, program-synthesis, abstract-reasoning]
datasets: [koushikz1/arc-agi-2-jepa-episodes]
---

# ARC-JEPA {version}

Object-centric transformation JEPA with latent-guided program search for ARC-AGI-2
(code: https://github.com/{GH_REPO}/tree/main/arc-jepa). The package is the input of the solver: a hierarchical
transformation-JEPA (cell/object/relation encoders, EMA target, predictor, rule latent, program encoder, scorer)
plus a transformation memory, used as a prior for a typed-DSL program search with exact demonstration
verification.

## Measured results (every number is measured, none is projected)

| version | date | public eval (120 tasks, competition metric) | val 150 (training-distribution split) | params | notes |
|---|---|---|---|---|---|
{rows}

"public eval" is the official 120-task ARC-AGI-2 evaluation set, never used for training or tuning. "val 150" is
the held-out 150-task split of the 1,000 training tasks (easier distribution). The Kaggle leaderboard score, when
available, is reported in the GitHub release notes.

## This version
{notes}

## Files
`model.safetensors` (weights), `config.json` (model + search config), `vocab.json` (program tokenizer),
`program_memory.npz` + `programs.json` (transformation memory: synthetic + training tasks only), `metrics.json`.

## Load
```python
from arcjepa.model.arcjepa import ARCJEPA
model = ARCJEPA.load_package("path/to/this/repo", device="cuda")
```
"""


def pkg_zip(pkg, out_zip):
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in os.listdir(pkg):
            p = os.path.join(pkg, f)
            if os.path.isfile(p):
                z.write(p, arcname=f"arc_jepa_pkg/{f}")
    return out_zip


def stage(pkg, metrics, card, dst):
    os.makedirs(dst, exist_ok=True)
    for f in PKG_FILES:
        if os.path.isfile(os.path.join(pkg, f)):
            shutil.copy2(os.path.join(pkg, f), os.path.join(dst, f))
    json.dump(metrics, open(os.path.join(dst, "metrics.json"), "w"), indent=1)
    open(os.path.join(dst, "README.md"), "w", encoding="utf-8").write(card)
    return dst


def push_hf(version, dst, message):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(HF_REPO, repo_type="model", private=True, exist_ok=True)
    info = api.upload_folder(folder_path=dst, repo_id=HF_REPO, repo_type="model", commit_message=message)
    try:
        api.create_tag(HF_REPO, tag=version, repo_type="model", tag_message=message)
    except Exception as e:  # tag exists -> move it
        api.delete_tag(HF_REPO, tag=version, repo_type="model")
        api.create_tag(HF_REPO, tag=version, repo_type="model", tag_message=message)
    return f"https://huggingface.co/{HF_REPO}/tree/{version} ({info.oid[:8] if hasattr(info, 'oid') else info})"


def gh(method, url, data=None, raw=None, ctype="application/json"):
    body = raw if raw is not None else (json.dumps(data).encode() if data is not None else None)
    r = urllib.request.Request(url, data=body, method=method, headers={
        "Authorization": f"token {os.environ['GH_TOKEN']}", "Accept": "application/vnd.github+json",
        "User-Agent": "arc-jepa-publisher", "Content-Type": ctype})
    with urllib.request.urlopen(r, timeout=300) as resp:
        return json.loads(resp.read().decode() or "{}")


def push_gh(version, zip_path, notes):
    tag = f"arc-jepa-{version}"
    try:
        rel = gh("GET", f"https://api.github.com/repos/{GH_REPO}/releases/tags/{tag}")
        gh("PATCH", f"https://api.github.com/repos/{GH_REPO}/releases/{rel['id']}", {"body": notes})
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        rel = gh("POST", f"https://api.github.com/repos/{GH_REPO}/releases",
                 {"tag_name": tag, "target_commitish": "main", "name": f"ARC-JEPA {version}", "body": notes,
                  "draft": False, "prerelease": True})
    name = os.path.basename(zip_path)
    for a in rel.get("assets", []):
        if a["name"] == name:
            gh("DELETE", f"https://api.github.com/repos/{GH_REPO}/releases/assets/{a['id']}")
    up = rel["upload_url"].split("{")[0] + f"?name={name}"
    gh("POST", up, raw=open(zip_path, "rb").read(), ctype="application/zip")
    return rel["html_url"]


def push_kaggle(version, dst, notes):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    mdir = tempfile.mkdtemp()
    json.dump({"ownerSlug": KG_OWNER, "title": "ARC-JEPA", "slug": KG_MODEL, "subtitle":
               "Transformation JEPA + latent-guided typed-DSL program search for ARC-AGI-2", "isPrivate": True,
               "description": open(os.path.join(dst, "README.md"), encoding="utf-8").read().split("---", 2)[-1],
               "publishTime": "", "provenanceSources": ""}, open(os.path.join(mdir, "model-metadata.json"), "w"))
    exists = subprocess.run(["kaggle", "models", "get", f"{KG_OWNER}/{KG_MODEL}"], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    if exists.returncode != 0:
        r = subprocess.run(["kaggle", "models", "create", "-p", mdir], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        if r.returncode != 0:
            raise RuntimeError("models create: " + (r.stdout + r.stderr)[-400:])
    inst = f"{KG_OWNER}/{KG_MODEL}/{KG_FRAMEWORK}/{KG_INSTANCE}"
    got = subprocess.run(["kaggle", "models", "instances", "get", inst], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    if got.returncode != 0:
        json.dump({"ownerSlug": KG_OWNER, "modelSlug": KG_MODEL, "instanceSlug": KG_INSTANCE, "framework": KG_FRAMEWORK,
                   "overview": "ARC-JEPA offline package (weights, config, vocab, transformation memory).",
                   "usage": "# Model Usage\nLoad with `arcjepa.model.arcjepa.ARCJEPA.load_package(dir)`; code at "
                            f"https://github.com/{GH_REPO}/tree/main/arc-jepa\n\n# Changelog\n{version}: {notes[:500]}",
                   "licenseName": "Apache 2.0", "fineTunable": False, "trainingData": [], "modelInstanceType": "Unspecified",
                   "baseModelInstanceId": 0, "externalBaseModelUrl": ""},
                  open(os.path.join(dst, "model-instance-metadata.json"), "w"))
        r = subprocess.run(["kaggle", "models", "instances", "create", "-p", dst], capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        os.remove(os.path.join(dst, "model-instance-metadata.json"))
    else:
        r = subprocess.run(["kaggle", "models", "instances", "versions", "create", inst, "-p", dst, "-n", f"{version}: {notes[:200]}"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    if r.returncode != 0:
        raise RuntimeError("kaggle models: " + (r.stdout + r.stderr)[-400:])
    return f"https://www.kaggle.com/models/{KG_OWNER}/{KG_MODEL}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True); ap.add_argument("--pkg", required=True)
    ap.add_argument("--metrics", required=True); ap.add_argument("--notes", required=True)
    ap.add_argument("--history", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_versions.json"))
    ap.add_argument("--targets", default="hf,gh,kaggle")
    a = ap.parse_args()
    metrics = json.load(open(a.metrics))
    hist = json.load(open(a.history)) if os.path.isfile(a.history) else []
    entry = {"version": a.version, "date": datetime.date.today().isoformat(), "public_eval": metrics.get("public_eval"),
             "val150": metrics.get("val150"), "params": metrics.get("params"), "notes": a.notes[:160]}
    hist = [h for h in hist if h["version"] != a.version] + [entry]
    json.dump(hist, open(a.history, "w"), indent=1)
    card = model_card(a.version, metrics, a.notes, hist)
    work = tempfile.mkdtemp(prefix=f"arcjepa_{a.version}_")
    dst = stage(a.pkg, metrics, card, os.path.join(work, "pkg"))
    zip_path = pkg_zip(dst, os.path.join(work, f"arc_jepa_pkg_{a.version}.zip"))
    msg = f"ARC-JEPA {a.version}: {a.notes[:120]}"
    out = {}
    for t in a.targets.split(","):
        try:
            if t == "hf":
                out[t] = push_hf(a.version, dst, msg)
            elif t == "gh":
                out[t] = push_gh(a.version, zip_path, card.split("---", 2)[-1])
            elif t == "kaggle":
                out[t] = push_kaggle(a.version, dst, a.notes)
        except Exception as e:
            out[t] = f"FAILED: {e!r}"[:500]
        print(t, "->", out[t], flush=True)
    return out


if __name__ == "__main__":
    main()
