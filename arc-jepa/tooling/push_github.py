#!/usr/bin/env python3
"""Push a local folder into a GitHub repo path with the Git Data API (no git binary needed).

Usage:  set GH_TOKEN, then
  python push_github.py --root "E:/Claude code/arc2/ARC-AGI-2/arc-jepa" --repo acco-cyber/ARC-AGI-2 \
                        --prefix arc-jepa --branch main --message "ARC-JEPA: ..."
Creates blobs for every file under --root (skipping ignored patterns), builds a tree on top of the current
branch tree, commits with the current head as parent, and fast-forwards the branch ref.
"""
import argparse, base64, fnmatch, json, os, sys, time, urllib.request, urllib.error

API = "https://api.github.com"
IGNORE_DIRS = {"__pycache__", ".pytest_cache", "runs", ".git", "out", "card", "out_v1"}
IGNORE_GLOBS = ["*.pyc", "*.safetensors", "*.pt", "*.npz", "*.log", "kaggle.json", ".env", "*.zip", "synth*.jsonl"]
MAX_FILE_MB = 20


def req(method, url, token, data=None):
    body = json.dumps(data).encode() if data is not None else None
    r = urllib.request.Request(url, data=body, method=method, headers={
        "Authorization": f"token {token}", "Accept": "application/vnd.github+json",
        "User-Agent": "arc-jepa-pusher", "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(r, timeout=60) as resp:
                return json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if e.code in (403, 429, 502, 503) and attempt < 4:
                time.sleep(2 + 3 * attempt); continue
            raise RuntimeError(f"{method} {url} -> {e.code}: {msg[:300]}")


def iter_files(root):
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in IGNORE_DIRS]
        for fn in fns:
            if any(fnmatch.fnmatch(fn, g) for g in IGNORE_GLOBS):
                continue
            p = os.path.join(dp, fn)
            if os.path.getsize(p) > MAX_FILE_MB * 1024 * 1024:
                print("skip (too large):", p); continue
            yield p, os.path.relpath(p, root).replace("\\", "/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--repo", required=True)
    ap.add_argument("--prefix", default=""); ap.add_argument("--branch", default="main")
    ap.add_argument("--message", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    token = os.environ.get("GH_TOKEN")
    if not token:
        sys.exit("GH_TOKEN not set")
    files = list(iter_files(a.root))
    print(f"{len(files)} files to push under '{a.prefix}'")
    if a.dry_run:
        for _, rel in files[:200]:
            print("  ", rel)
        return
    ref = req("GET", f"{API}/repos/{a.repo}/git/ref/heads/{a.branch}", token)
    head = ref["object"]["sha"]
    commit = req("GET", f"{API}/repos/{a.repo}/git/commits/{head}", token)
    base_tree = commit["tree"]["sha"]
    tree = []
    for i, (p, rel) in enumerate(files):
        raw = open(p, "rb").read()
        try:
            content, enc = raw.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            content, enc = base64.b64encode(raw).decode(), "base64"
        blob = req("POST", f"{API}/repos/{a.repo}/git/blobs", token, {"content": content, "encoding": enc})
        path = f"{a.prefix}/{rel}" if a.prefix else rel
        tree.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        if (i + 1) % 25 == 0:
            print(f"  blobs {i + 1}/{len(files)}")
    new_tree = req("POST", f"{API}/repos/{a.repo}/git/trees", token, {"base_tree": base_tree, "tree": tree})
    new_commit = req("POST", f"{API}/repos/{a.repo}/git/commits", token,
                     {"message": a.message, "tree": new_tree["sha"], "parents": [head]})
    req("PATCH", f"{API}/repos/{a.repo}/git/refs/heads/{a.branch}", token, {"sha": new_commit["sha"], "force": False})
    print("pushed commit", new_commit["sha"], "->", f"https://github.com/{a.repo}/tree/{a.branch}/{a.prefix}")


if __name__ == "__main__":
    main()
