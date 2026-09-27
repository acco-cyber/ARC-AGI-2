#!/usr/bin/env python3
"""Publish a trained ARC-JEPA package to Hugging Face, Kaggle Models and GitHub (release asset).

Usage (PowerShell):
  $env:HF_TOKEN="..."; $env:GH_TOKEN="..."
  python publish_model.py --pkg <arc_jepa_pkg dir> --card <MODEL_CARD.md> --version-notes "..." [--hf] [--kaggle] [--github]

* HF:      model repo koushikz1/arc-jepa (private), uploads the package files + README.md (model card).
* Kaggle:  model poby7722/arc-jepa, framework PyTorch, instance "default", new version per call.
* GitHub:  release "arc-jepa-<tag>" on acco-cyber/ARC-AGI-2 with the package zipped as an asset
           (weights stay out of the git tree; the model card + config/vocab go into arc-jepa/model/).
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time, urllib.request, urllib.error, zipfile

HF_REPO = "koushikz1/arc-jepa"
KAGGLE_MODEL = "poby7722/arc-jepa"
GH_REPO = "acco-cyber/ARC-AGI-2"
PKG_FILES = ["model.safetensors", "model.pt", "config.json", "vocab.json", "program_memory.npz", "programs.json"]


def pkg_files(pkg):
    fs = [os.path.join(pkg, f) for f in PKG_FILES if os.path.isfile(os.path.join(pkg, f))]
    if not any(f.endswith(("model.safetensors", "model.pt")) for f in fs) or not os.path.isfile(os.path.join(pkg, "config.json")):
        sys.exit(f"not a complete ARC-JEPA package: {pkg}")
    return fs


def publish_hf(pkg, card, notes):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(HF_REPO, repo_type="model", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        for f in pkg_files(pkg):
            shutil.copy2(f, d)
        shutil.copy2(card, os.path.join(d, "README.md"))
        info = api.upload_folder(folder_path=d, repo_id=HF_REPO, repo_type="model", commit_message=notes)
    print("HF:", info)


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-2000:], r.stderr[-2000:], flush=True)
    return r.returncode, r.stdout + r.stderr


def publish_kaggle(pkg, card, notes):
    owner, slug = KAGGLE_MODEL.split("/")
    with tempfile.TemporaryDirectory() as d:
        mdir = os.path.join(d, "model"); os.makedirs(mdir)
        meta = {"ownerSlug": owner, "title": "ARC-JEPA", "slug": slug, "subtitle":
                "Object-centric transformation JEPA with latent-guided program search for ARC-AGI-2",
                "isPrivate": True, "description": open(card, encoding="utf-8").read(),
                "publishTime": "", "provenanceSources": ""}
        json.dump(meta, open(os.path.join(mdir, "model-metadata.json"), "w"), indent=1)
        rc, out = run(["kaggle", "models", "get", KAGGLE_MODEL])
        if rc != 0:
            rc, out = run(["kaggle", "models", "create", "-p", mdir])
        idir = os.path.join(d, "instance"); os.makedirs(idir)
        for f in pkg_files(pkg):
            shutil.copy2(f, idir)
        imeta = {"ownerSlug": owner, "modelSlug": slug, "instanceSlug": "default", "framework": "PyTorch",
                 "overview": "ARC-JEPA offline package (weights, config, vocab, transformation memory).",
                 "usage": "from arcjepa.model.arcjepa import ARCJEPA; m = ARCJEPA.load_package(path)",
                 "licenseName": "Apache 2.0", "fineTunable": False, "trainingData": [],
                 "modelInstanceType": "Unspecified", "baseModelInstanceId": 0, "externalBaseModelUrl": ""}
        json.dump(imeta, open(os.path.join(idir, "model-instance-metadata.json"), "w"), indent=1)
        rc, out = run(["kaggle", "models", "instances", "get", f"{KAGGLE_MODEL}/PyTorch/default"])
        if rc != 0:
            rc, out = run(["kaggle", "models", "instances", "create", "-p", idir])
        else:
            os.remove(os.path.join(idir, "model-instance-metadata.json"))
            rc, out = run(["kaggle", "models", "instances", "versions", "create", f"{KAGGLE_MODEL}/PyTorch/default",
                           "-p", idir, "-n", notes])
        if rc != 0:
            sys.exit("Kaggle model publish failed")


def gh(method, url, data=None, raw=None, ctype="application/json"):
    body = raw if raw is not None else (json.dumps(data).encode() if data is not None else None)
    r = urllib.request.Request(url, data=body, method=method, headers={
        "Authorization": f"token {os.environ['GH_TOKEN']}", "Accept": "application/vnd.github+json",
        "User-Agent": "arc-jepa-publisher", "Content-Type": ctype})
    with urllib.request.urlopen(r, timeout=300) as resp:
        return json.loads(resp.read().decode() or "{}")


def publish_github(pkg, card, notes, tag):
    with tempfile.TemporaryDirectory() as d:
        zpath = os.path.join(d, f"arc_jepa_pkg_{tag}.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in pkg_files(pkg):
                z.write(f, os.path.basename(f))
            z.write(card, "MODEL_CARD.md")
        try:
            rel = gh("POST", f"https://api.github.com/repos/{GH_REPO}/releases",
                     {"tag_name": f"arc-jepa-{tag}", "name": f"ARC-JEPA model {tag}", "body": notes, "draft": False,
                      "prerelease": False, "target_commitish": "main"})
        except urllib.error.HTTPError as e:
            sys.exit(f"GitHub release create failed: {e.code} {e.read().decode()[:300]}")
        up = rel["upload_url"].split("{")[0] + f"?name={os.path.basename(zpath)}"
        asset = gh("POST", up, raw=open(zpath, "rb").read(), ctype="application/zip")
        print("GitHub release:", rel["html_url"], "asset:", asset.get("browser_download_url"),
              f"({os.path.getsize(zpath) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True); ap.add_argument("--card", required=True)
    ap.add_argument("--version-notes", required=True); ap.add_argument("--tag", default=time.strftime("v%Y%m%d-%H%M"))
    ap.add_argument("--hf", action="store_true"); ap.add_argument("--kaggle", action="store_true")
    ap.add_argument("--github", action="store_true")
    a = ap.parse_args()
    pkg_files(a.pkg)
    if a.hf:
        publish_hf(a.pkg, a.card, a.version_notes)
    if a.kaggle:
        publish_kaggle(a.pkg, a.card, a.version_notes)
    if a.github:
        publish_github(a.pkg, a.card, a.version_notes, a.tag)


if __name__ == "__main__":
    main()
