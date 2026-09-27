"""Derive a quota-fit copy of the FULL training kernel: new slug, shorter time box, shorter synthetic step.

python make_quota_train_kernel.py --hours 1.5 --synth-minutes 15 --slug arc-jepa-train-v2
Reads arc-jepa/kaggle/train_full/{arc-jepa-train.ipynb,kernel-metadata.json}, writes arc-jepa/kaggle/train_quota/.
"""
import argparse, json, os, re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arc-jepa", "kaggle")
ap = argparse.ArgumentParser()
ap.add_argument("--hours", default="1.5"); ap.add_argument("--synth-minutes", default="15")
ap.add_argument("--slug", default="arc-jepa-train-v2"); ap.add_argument("--title", default="ARC JEPA Train v2")
a = ap.parse_args()
src, dst = os.path.join(ROOT, "train_full"), os.path.join(ROOT, "train_quota")
os.makedirs(dst, exist_ok=True)
nb = json.load(open(os.path.join(src, "arc-jepa-train.ipynb"), encoding="utf-8"))
hits = {"hours": 0, "synth": 0, "smoke": 0}
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    s2, n = re.subn(r'os\.environ\.get\("ARCJEPA_TRAIN_HOURS", "[0-9.]+"\)', f'os.environ.get("ARCJEPA_TRAIN_HOURS", "{a.hours}")', s)
    hits["hours"] += n
    s2, n = re.subn(r'os\.environ\.get\("ARCJEPA_SYNTH_MAX_MINUTES", "[0-9.]+"\)', f'os.environ.get("ARCJEPA_SYNTH_MAX_MINUTES", "{a.synth_minutes}")', s2)
    hits["synth"] += n
    hits["smoke"] += len(re.findall(r'os\.environ\.get\("ARCJEPA_SMOKE", "0"\)', s2))
    if s2 != s:
        lines = s2.split("\n")
        c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
assert hits["hours"] >= 1 and hits["synth"] >= 1 and hits["smoke"] >= 1, hits
with open(os.path.join(dst, "arc-jepa-train.ipynb"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
meta = json.load(open(os.path.join(src, "kernel-metadata.json"), encoding="utf-8"))
meta["id"], meta["title"] = f"poby7722/{a.slug}", a.title
with open(os.path.join(dst, "kernel-metadata.json"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write(json.dumps(meta, indent=1) + "\n")
print("wrote", dst, hits, meta["id"])
