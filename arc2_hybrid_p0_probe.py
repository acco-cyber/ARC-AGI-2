# P0 CPU probe on the NVARC image: can an isolated CPython 3.12 + vLLM 0.19.0 (offline wheelhouse)
# live next to NVARC's Python 3.11 stack? No GPU is used. Every check is isolated and reported.
import glob, json, os, shutil, subprocess, sys, tarfile, time

T0 = time.time()
R = {}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def sh(cmd, timeout=600, env=None):
    t = time.time()
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout, env=env)
        return p.returncode, (p.stdout + p.stderr), round(time.time() - t, 1)
    except Exception as e:
        return -1, repr(e), round(time.time() - t, 1)


def check(name, fn):
    t = time.time()
    try:
        R[name] = {"ok": True, "out": fn()}
    except BaseException as e:
        R[name] = {"ok": False, "out": repr(e)[:1500]}
    R[name]["s"] = round(time.time() - t, 1)
    log(f"CHECK {name}: {'OK' if R[name]['ok'] else 'FAIL'} ({R[name]['s']}s) {str(R[name]['out'])[:600]}")


check("kernel_python", lambda: sys.version)
check("glibc", lambda: sh("ldd --version | head -1")[1].strip())
check("os", lambda: sh("cat /etc/os-release | head -2")[1].strip())
check("system_pythons", lambda: sorted(glob.glob("/usr/bin/python3*") + glob.glob("/usr/local/bin/python3*")))
check("disk", lambda: sh("df -h /tmp /kaggle/working | tail -2")[1].strip())
check("inputs", lambda: sh("ls /kaggle/input /kaggle/input/* 2>/dev/null | head -40")[1])

PY = "/tmp/py312/python/bin/python3.12"


def get_python():
    tgz = glob.glob("/kaggle/input/**/cpython312.tar.gz", recursive=True)
    if tgz:
        shutil.rmtree("/tmp/py312", ignore_errors=True)
        os.makedirs("/tmp/py312")
        with tarfile.open(tgz[0]) as tf:
            tf.extractall("/tmp/py312")
        return f"extracted {tgz[0]}"
    exe = glob.glob("/kaggle/input/**/python/bin/python3.12", recursive=True)
    if exe:  # Kaggle auto-extracted the archive: copy (input mount has no exec bit)
        src = os.path.dirname(os.path.dirname(os.path.dirname(exe[0])))
        shutil.copytree(src, "/tmp/py312", dirs_exist_ok=True)
        return f"copied extracted tree {src}"
    raise FileNotFoundError("no cpython312 in inputs")


check("standalone_python", get_python)
check("py312_runs", lambda: sh([PY, "-c", "import sys,ssl,sqlite3,ctypes; print(sys.version)"])[1].strip())
check("py312_pip", lambda: sh([PY, "-m", "pip", "--version"])[1].strip())

WH = None
whl = glob.glob("/kaggle/input/**/vllm-0.19.0*.whl", recursive=True)
if whl:
    WH = os.path.dirname(whl[0])
R["wheelhouse"] = {"ok": bool(WH), "out": WH}
log("wheelhouse:", WH)

CLEAN = {"PATH": "/tmp/py312/python/bin:/usr/local/cuda/bin:/usr/bin:/bin", "HOME": "/tmp",
         "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0",
         "TMPDIR": "/tmp", "PIP_NO_CACHE_DIR": "1"}


def install():
    rc, out, s = sh([PY, "-m", "pip", "install", "--no-index", "--no-warn-conflicts", "--disable-pip-version-check",
                     "--find-links", WH, "vllm==0.19.0"], timeout=2400, env=CLEAN)
    if rc != 0:
        raise RuntimeError(f"rc={rc} after {s}s: {out[-2500:]}")
    return f"installed in {s}s; site-packages {sh('du -sh /tmp/py312/python/lib/python3.12/site-packages')[1].strip()}"


check("pip_install_vllm", install)
check("imports", lambda: sh([PY, "-c", "import torch, vllm, transformers, triton; print('torch', torch.__version__, torch.version.cuda, '| vllm', vllm.__version__, '| transformers', transformers.__version__, '| triton', triton.__version__)"], timeout=900, env=CLEAN)[1].strip()[-1200:])


def help_flags():
    rc, out, s = sh([PY, "-m", "vllm.entrypoints.openai.api_server", "--help=all"], timeout=900, env=CLEAN)
    if "--max-model-len" not in out:
        rc, out, s = sh([PY, "-m", "vllm.entrypoints.openai.api_server", "--help"], timeout=900, env=CLEAN)
    flags = ["--language-model-only", "--reasoning-parser", "--enable-prefix-caching", "--kv-cache-dtype",
             "--speculative-config", "--reasoning-config", "--max-num-seqs", "--enable-chunked-prefill",
             "--default-chat-template-kwargs", "--attention-backend", "--async-scheduling", "--generation-config"]
    return {"rc": rc, "s": s, "present": {f: (f in out) for f in flags}, "tail": out[-600:] if rc else ""}


check("server_help", help_flags)

MODEL = None
cfgs = [c for c in glob.glob("/kaggle/input/**/config.json", recursive=True)
        if glob.glob(os.path.join(os.path.dirname(c), "*.safetensors"))]
for c in cfgs:
    try:
        if "Qwen3_5" in json.dumps(json.load(open(c)).get("architectures", [])):
            MODEL = os.path.dirname(c)
    except Exception:
        pass
R["model_dir"] = {"ok": bool(MODEL), "out": MODEL}
log("model:", MODEL)

PROBE = r'''
import json, sys
from vllm.transformers_utils.config import get_config
m = sys.argv[1]
c = get_config(m, trust_remote_code=False)
tc = getattr(c, "text_config", c)
out = {"cls": type(c).__name__, "model_type": getattr(c, "model_type", None),
       "layers": getattr(tc, "num_hidden_layers", None), "quant": str(getattr(tc, "quantization_config", getattr(c, "quantization_config", None)))[:200]}
from vllm.model_executor.models.registry import ModelRegistry
out["arch_supported"] = [a for a in (c.architectures or []) if a in ModelRegistry.get_supported_archs()]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(m)
msgs = [{"role": "user", "content": "hi"}]
for eff in ("xhigh", "low"):
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, reasoning_effort=eff, enable_thinking=True)
    out[f"template_{eff}_tail"] = s[-90:]
s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
out["template_nothink_tail"] = s[-60:]
from vllm.reasoning import ReasoningParserManager
out["qwen3_reasoning_parser"] = "qwen3" in str(ReasoningParserManager.reasoning_parsers.keys() if hasattr(ReasoningParserManager, "reasoning_parsers") else ReasoningParserManager.__dict__)
ev = json.load(open(sys.argv[2]))
lens = sorted(len(tok(json.dumps(t)).input_ids) for t in ev.values())
out["eval_task_tokens_p50_p90_max"] = [lens[len(lens) // 2], lens[int(len(lens) * 0.9)], lens[-1]]
print("PROBE_JSON " + json.dumps(out))
'''


def model_probe():
    if not MODEL:
        raise FileNotFoundError("no Qwen3.8 model dir")
    open("/tmp/model_probe.py", "w").write(PROBE)
    ev = glob.glob("/kaggle/input/**/arc-agi_evaluation_challenges.json", recursive=True)[0]
    rc, out, s = sh([PY, "/tmp/model_probe.py", MODEL, ev], timeout=900, env=CLEAN)
    i = out.find("PROBE_JSON ")
    if i < 0:
        raise RuntimeError(out[-2500:])
    return json.loads(out[i + 11:].splitlines()[0])


check("model_config_tokenizer", model_probe)
check("kernel_nvarc_imports_untouched", lambda: sh([sys.executable, "-c", "import torch, transformers; print(torch.__version__, transformers.__version__)"], timeout=300)[1].strip()[-300:])
check("model_read_speed", lambda: sh(f"dd if=$(ls {MODEL}/*.safetensors | head -1) of=/dev/null bs=16M 2>&1 | tail -1", timeout=600)[1].strip() if MODEL else "no model")

json.dump(R, open("/kaggle/working/p0_report.json", "w"), indent=1, default=str)
print("=" * 90)
for k, v in R.items():
    print(f"{k:<32} {'OK  ' if v.get('ok') else 'FAIL'} {str(v.get('out'))[:200]}")
print("=" * 90)
open("/kaggle/working/submission.json", "w").write("{}")
log("done")
