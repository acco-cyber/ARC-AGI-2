"""Flash-Next W4A16 launcher shared by Kaggle submissions and local evaluation.

Without arguments it is the Kaggle notebook: it restores the pinned runtime from the attachments,
loads the target and draft, starts SGLang, runs the shared ``kaggle_runner.py`` with the
``codex_lite`` policy and validates the atomic submission. Local runs execute the same source
files and serving/runner commands; only the task JSON, the installed runtime and compiler cache
(``--deployment-root``/``--cache``) and the model weight paths differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import tarfile
from contextlib import contextmanager
from datetime import datetime, timezone
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

COMPETITION = "arc-prize-2026-arc-agi-2"
SERVED_MODEL = "qwen38-flash-next-w4a16"
PORT = 30000
# Kaggle stops the notebook at 12 h; stopping the runner and server takes about a minute.
KERNEL_BUDGET_SECONDS = 11 * 3600 + 55 * 60
# Since v21 second attempts after unverified firsts carry their work; since v22 every request also carries a -1
# logit bias on "Hmm" and "Alternatively" (codex_lite.CODEX_LITE_LOGIT_BIASES); since v27 each request's output is
# capped at 12,288 tokens instead of 24,576 (codex_lite.CODEX_LITE_REQUEST_CAPS); since v28 the task message also shows
# every grid as an image (codex_lite.task_images).
ATTEMPT_MODE = "codex_lite_cap12k_images"
# Attempt modes whose task message also shows the grids as images (codex_lite.task_images): the server loads the
# target's vision tower; every other mode serves the language model only.
IMAGE_ATTEMPT_MODES = ("codex_lite_cap12k_images",)
TEMPERATURE = 0.7
REASONING_EFFORT = "medium"
TOKEN_CAP = 65_536
CONTEXT_LENGTH = 139_264
# Ten running attempts since v28 (8 through v27). Decode context parallelism 2 stores each token on one rank of its
# pair (flash_next_qsa_dcp.py), about 1.8x the tokens in the same memory, but at 12 running free device memory fell
# to 0.05 GiB while serving (cache kernel version 11; builds that ran 12 h kept 0.31-0.36 GiB in their save runs).
# Ten drops the graphs for 11-12 (~0.02 GB each) and 8 mamba slots (0.11 GB), which the KV cap (L4_KV_TOKENS) keeps
# free: v28's save run kept 0.38 GiB (codex_lite_kaggle_20261028_v28c12f/). Admission grows only while KV use is at
# most 60%; a running attempt averages ~37.5k tokens (p90 peak 92k).
MAX_RUNNING_REQUESTS = 10
# 0.96 through v20, which left 0.54 GB per L4 after graph capture. v21 needs 0.60 GB for the buffers allocated
# after the KV pool is sized, which the draft's freed vocabulary tables covered through v20 (see the reclaim
# below). At 0.95 its cache kernel kept 0.09 GB after capture, and a cold Triton autotune's 256 MB benchmark buffer
# failed; 0.93 restores v20's margin (user, 2026-10-24). 0.92 since v28, whose DCP 2 and graphs 1-10 kept 0.64 GB
# after capture and 0.38 GiB at the lowest while serving.
KAGGLE_MEMORY_FRACTION = "0.92"
# Lazy extra buffer, 4 slots a request (v28; v30 again; eager at 5 through v27 and in v29), 13.9 MB per slot per L4:
# the second ping-pong slot exists only around a tracking point. v29 returned to the eager buffer over pub_pwgraph
# seed 2234's degraded held server, but that was the QSA DCP decode kernel's int32 row offsets (fixed in v30,
# l4_hclocal_20261029/findings.md); fresh servers scored lazy and eager alike (s2234 76.81 and 77.36).
MAMBA_SLOTS_PER_REQUEST = 4
MAMBA_RADIX_CACHE_STRATEGY = "extra_buffer_lazy"
# Kaggle's FP8 KV pool in rows per rank (663,552 tokens across each DCP pair), passed as --max-total-tokens: SGLang
# caps the rows before the DCP doubling. v28-v29 had 343,360, what 0.92 gave 12 running with 48 mamba slots (cache
# kernel version 11), so the memory 10 running frees stays free instead of growing the pool. v30's local HC up
# projection costs 0.25 GB a rank and the lazy buffer returns 0.14 GB, so 11,584 rows (~10 KB a row a rank) fewer
# keep v29's free memory (Modal hclocal_01 at 305,600 rows had 0.12 GiB more than v29's 0.42 GiB lowest free).
# v21's 8-running profile without DCP had 484,032.
L4_KV_TOKENS = 331_776
# Local runs on Hopper GPUs (phi1's H200s) admit up to 160 running attempts under the same adaptive
# admission (user, 2026-10-21): KV peaked at 10% at 128 running there, so H200 evaluations no longer
# mimic the L4 profile; Kaggle's L4 throughput and score are estimated from generated tokens. Other
# local GPUs (vast L4s) keep Kaggle's profile.
H200_MAX_RUNNING_REQUESTS = 160
MODEL_STAGE_WORKERS = 8
# Ordinary saves (and the cache kernel) run 16 public tasks, first attempts only, for at most SMOKE_SECONDS:
# enough concurrent attempts to hold the production profile at 10 running. Since v17 they are the 16 tasks
# whose attempts used the fewest tokens in phi1 runs of v16's code (at most 5.1k-23.2k generated tokens in
# each of 6 attempts), so a first attempt reliably finishes and post-verifies within the window. The first 16
# sorted public tasks (through v16) mostly need 20-60k tokens; v17's first save run finished none.
SMOKE_KEYS = ("1818057f", "1ae2feb7", "2ba387bc", "38007db0", "45a5af55", "58490d8a", "58f5dbd5", "5961cc34",
              "7b5033c1", "7ed72f31", "8f215267", "981571dc", "9aaea919", "a395ee82", "bf45cf4b", "db695cfb")
# 10 minutes from v13 (v12's fastest save-run attempt finished in 233 s); 20 since v22, so that the save run
# measures L4 decode throughput over more than v21's 470 s and 6 finished attempts. The cache kernel runs the same
# window and must finish all 16 attempts (user, 2026-10-28: the cache has to cover everything the save run compiles):
# every run's first real prefill stalls ~115 s, and some kernels compile one variant per running count, the low
# counts only while the last attempts finish (shorter windows missed variants: version 9 at 6 minutes, 10 at 2).
SMOKE_SECONDS = 20 * 60
# The startup heartbeat (cgroup memory and CPU counters) goes to resources.jsonl every 30 s and to the live log
# every 5 minutes.
HEARTBEAT_LOG_SECONDS = 5 * 60
# Files under the cache root that are host-specific or transient (the libcuda link, Triton's tool copies, compiler
# temporaries): never exported, and not compiles.
CACHE_EXCLUDED = ("cuda_link", "triton_tools", "tmp")
# Compile caches under HOME, outside the exported cache root, which a run must not add to (TILELANG_CACHE_DIR keeps
# TileLang's inside the root; through version 10 every run recompiled its kernels).
HOME_CACHES = (".tilelang", ".deep_gemm", ".triton", ".nv", ".cache", ".cupy")
# Before loading weights, a sandboxed command must import what the task prompt promises.
SANDBOX_PREFLIGHT = "python -c 'import numpy, scipy; print(numpy.__version__, scipy.__version__)'"
SANDBOX_PREFLIGHT_SECONDS = 300
SEED_BASE = 1234
# Kaggle's L4 memory budget; local H200 evaluations use 2048 (--chunked-prefill-size). 1024 (v31 cache kernel version 14,
# breakable): prefill -5% per new token against v30, but lowest free 0.37 -> 0.17 GiB on rank 0, and 0.12 GiB within
# 84 minutes of a vast.ai L4 full-path run (inducer/results/codex_lite_kaggle_20261031_v31/findings.md).
PREFILL_CHUNK = 512
# Kaggle runs since v28: the TP group's bf16 decode all-reduces of at most 320 KiB per rank go through pinned
# /dev/shm memory instead of NCCL's SHM transport (flash_next_host_ar.py; neutral on Kaggle's L4s, decode p25 71.2
# against 71.1 ms, codex_lite_kaggle_20261028_v28/). Local runs keep NCCL, which is faster on NVLink hosts, unless
# --host-all-reduce on.
HOST_ALL_REDUCE = True
# Decode context parallelism since v28: at TP4 the target's 2 KV heads are each stored on two ranks; DCP 2
# splits each head's sequence between those two ranks. Stock QSA ignores DCP (every rank kept every token, and the
# cache kernel crashed once slots passed the physical rows), so flash_next_qsa_dcp.py shards the target's KV and
# merges the ranks' sparse attention; the replicated indexer cache keeps the gain near 1.8x rather than 2x.
DCP_SIZE = 2
# Kaggle runs since v29: the target's vocab-parallel logits all-gather sends each rank's top 32 logits and ids instead
# of its whole shard (flash_next_serving.py, SGLANG_FLASH_NEXT_TOPK_LOGITS). Exact for top-k 20 with the four -1
# filler biases; -3.6% decode step at 10 running on L4 (inducer/results/l4_decode_profile_20261028_v1/).
TOPK_LOGITS = 32
# Kaggle runs since v29: prefill chunks replay a CUDA graph (one bucket at the chunk size) with GDN, QSA attention, the
# QSA indexer and PLE run eagerly between captured segments (flash_next_serving.py, SGLANG_FLASH_NEXT_PREFILL_GRAPH).
# v29-v30 captured it through tc_piecewise (eager compiler): on L4 a 512-token batch dropped 585 -> 516 ms for 0.15 GB
# per GPU (inducer/results/prefill_graph_phi1_20261029/). Since v31 SGLang's breakable backend captures it, with the
# MTP draft's prefill kept eager: vast.ai 4xL4 at 512-token chunks, -6.5 to -6.8% time against tc_piecewise on a fixed
# prefill-heavy workload (cold prompts -10%, cached continuations -0.5 to -2%) at the same sampled peak VRAM
# (codex/v30-inference-speed). phi1 public 120 with it (at 1024): 81.11 (v31_rehearsal_phi1_20261031_v1/).
PREFILL_GRAPH = True
PREFILL_GRAPH_BACKEND = "breakable"
# With the host all-reduce, a decode MoE layer's down projection, top-k sum, gated shared-expert add and all-reduce run
# as one kernel that overlaps the all-reduce with the GEMM, bit-exact against the unfused path (flash_next_moe_down_ar.py,
# SGLANG_FLASH_NEXT_MOE_DOWN_AR; Kaggle 4xL4 microbenchmark 271 -> 221 us a layer at 40 tokens). Modal 4xL4 paired A/B
# with the shared expert serialized: decode step 70.1 -> 68.5 / 77.9 -> 76.3 ms at 8/10 running; with it beside the
# routed experts again, both kernels against v30 before them: -6.4 / -4.9% at 8/10 running
# (inducer/results/moe_down_ar_20261030/). Local runs need --host-all-reduce on.
MOE_DOWN_AR = True
# Decode HC branch mixes (106 a step) through ForgeKernel's CUDA kernel instead of the Triton one
# (flash_next_hc_local_mix.cuh, SGLANG_FLASH_NEXT_HC_MIX_CUDA; bit-identical to Triton). Modal 4xL4 paired A/B with
# MOE_DOWN_AR on: HC mix 3.9 -> 2.1 ms a step; decode step 68.5 -> 66.9 / 76.5 -> 73.9 ms at 8/10 running, both kernels
# against v30 before them -4.6 / -5.1% (inducer/results/moe_down_ar_20261030/).
HC_MIX_CUDA = True
# Since v31 the QSA indexer's decode scoring in speculative verify and draft-extend graphs runs ForgeKernel's CUDA kernel,
# one CTA per request's 4 verify rows, instead of TileLang's per-row kernel over the whole context limit
# (flash_next_qsa_index_decode.cuh, SGLANG_FLASH_NEXT_QSA_DECODE). The TileLang kernel costs 2.46 ms per decode step
# at 10 running on L4 (13 calls); cold-L2 L4 82.8 against 326 us a call at 32k mean context, 40 rows.
QSA_DECODE = True
DECODE_LOG_INTERVAL = 40  # Decode steps per SGLang decode log line (SGLang's default), passed explicitly.
TARGET_MODEL = (
    "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1"
)
# Since v21: 480 of 512 routed experts (mass-ranked REAP selection) in every target layer and in the MTP draft.
# Since v29 the target is INT8 dense: GPTQ W8A16 (group 128) in_proj_qkvz, attention q/k/v, shared-expert gate/up
# and linear-attention out_proj on layers 4-43 (scripts/vastai/pruning/quantize_int8_dense.py DENSE_2026_10_28;
# v21 to v28 converted only in_proj_qkvz, target/1). It frees 0.23 GiB per L4 at the same speed; public 120 on phi1
# 79.55 over 4 seeds. The target attachment holds only what changed (the 14 shards and the configs); every other
# file comes from TARGET_MODEL. Our files are instances of one private Kaggle Model, which Kaggle mounts over NFS
# (Datasets are GCSFUSE, v17). The target ref is also the compile-cache identity of these weights.
REAP_TARGET_DATASET = "andy2709/arc-flash-next-480/transformers/target-int8dense/1"
REAP_TARGET_MODEL = "andy2709/arc-flash-next-480/transformers/target-int8dense/1"
# The 480-expert draft with dense tensors re-fit to this target (2026-10-23, multi-step objective): replay accept
# length 2.807 -> 2.992 on 1,939 requests. Its attachment adds the export record, which the attachment search
# requires so that the un-refit draft can never be picked up instead.
REAP_DRAFT_MODEL = "andy2709/arc-flash-next-480/transformers/draft-refit/1"
REAP_DRAFT_EXPORT = "mtp_finetune_export.json"
REAP_MANIFEST = "reap_v2_manifest.json"
REAP_EXPERTS = 480
# "warmup" (the cache kernel's copy) runs the save-run workload without the pinned cache and
# exports the compile cache that submissions restore.
LAUNCH_MODE = "submission"
COMPILE_CACHE_KERNEL = "andy2709/arc-flash-next-reap-v2-cache"
COMPILE_CACHE_ARCHIVE = "flash_next_compile_cache.tar.gz"
# Cache kernel version 15 (v31: profile [10, 40, 1-10], breakable 512 prefill graph, QSA decode kernel): 16/16, no
# errors, 0 compiles outside the cache root, 0.37 GiB lowest free (codex_lite_kaggle_20261031_v31b/). The manifest
# records the prefill graph, so restore refuses version 14 (146cc968..., [breakable, 1024]) and version 13 (91f0a14a...).
COMPILE_CACHE_SHA256 = "6e26d71c8a8386870cda2483b1d5f516c5008ae9b25ca1e00c5e61fe726908bb"
SGLANG_VERSION = "0.5.20"
WORK = Path("/kaggle/working")
SCRATCH = Path("/kaggle/temp/arc_flash_next_submission")
CACHE_MANIFEST = "flash_next_compile_cache_manifest.json"


def find_directory_with(*required: str) -> Path:
    for root, directories, files in os.walk("/kaggle/input"):
        names = set(directories) | set(files)
        if all(name in names for name in required):
            return Path(root)
    raise FileNotFoundError(f"could not locate input directory containing {required}")


def competition_file(name: str) -> Path:
    candidates = (
        Path(f"/kaggle/input/competitions/{COMPETITION}/{name}"),
        Path(f"/kaggle/input/{COMPETITION}/{name}"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return find_directory_with(name) / name


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


class StartupRecorder:
    def __init__(self, path):
        self.path = Path(path)
        self.started = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, stage, **fields):
        row = {
            "stage": stage,
            "elapsed_seconds": time.monotonic() - self.started,
            "at": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(
            f"[startup] {row['elapsed_seconds']:.2f}s {stage} " + json.dumps(fields),
            flush=True,
        )
        return row

    @contextmanager
    def timed(self, stage):
        started = time.monotonic()
        self.event(stage + "_begin")
        try:
            yield
        except BaseException:
            self.event(stage + "_failed", duration_seconds=time.monotonic() - started)
            raise
        else:
            self.event(stage + "_end", duration_seconds=time.monotonic() - started)


def mount_diagnostics(*paths):
    """Mounts under /kaggle and the mount holding each path, from /proc/self/mountinfo.

    Attachments moved from NFS to GCSFUSE around 2026-10-20, and weight loading became many
    times slower; the save-run log shows which storage each session got. Never raises.
    """
    try:
        mounts = []
        with open("/proc/self/mountinfo") as stream:
            for line in stream:
                fields = line.split()
                tail = fields[fields.index("-") + 1:]
                mounts.append({"mount_point": fields[4].replace("\\040", " "), "fstype": tail[0],
                               "source": tail[1] if len(tail) > 1 else ""})

        def holding(path):
            resolved = os.path.realpath(path)
            return max((mount for mount in mounts if resolved == mount["mount_point"]
                        or resolved.startswith(mount["mount_point"].rstrip("/") + "/")),
                       key=lambda mount: len(mount["mount_point"]), default=None)

        return {"kaggle_mounts": [mount for mount in mounts if mount["mount_point"].startswith("/kaggle")],
                "paths": {str(path): holding(path) for path in paths}}
    except Exception as exc:  # A diagnostic must never fail the notebook.
        return {"error": repr(exc)}


def relocate_triton_groups(cache):
    """Triton group indexes store absolute paths, unlike the kernel binaries."""
    groups = children = 0
    for path in cache.rglob("__grp__*.json"):
        data = json.loads(path.read_text())
        paths = data.get("child_paths")
        if not isinstance(paths, dict) or not paths:
            raise ValueError("invalid Triton cache group: " + str(path))
        relocated = {}
        for name, original in paths.items():
            if Path(name).name != name or Path(original).name != name:
                raise ValueError("unexpected Triton cache member: " + name)
            target = path.parent / name
            if not target.is_file():
                raise FileNotFoundError(target)
            relocated[name] = str(target.resolve())
            children += 1
        data["child_paths"] = relocated
        write_json(path, data)
        groups += 1
    return {"groups": groups, "child_paths": children}


def restore_bundle(source, destination, manifest, recorder):
    destination.mkdir(parents=True, exist_ok=False)
    if (
        shutil.disk_usage(destination).free
        < sum(x["uncompressed_bytes"] for x in manifest["archives"].values())
        + 4 * 2**30
    ):
        raise RuntimeError("insufficient scratch space for runtime")
    for kind, record in manifest["archives"].items():
        path = source / record["archive"]
        # No separate SHA-256 pass: it re-read the runtime from /kaggle/input, which is slow on
        # GCSFUSE. Reading the gzip stream to its end checks its CRC-32 and size, and the pinned
        # manifest, SGLang version and installed.lock still identify the runtime.
        with recorder.timed(kind + "_archive_restore"):
            with tarfile.open(path, "r:gz") as archive:
                archive.extractall(destination, filter="data")
                while archive.fileobj.read(16 * 2**20):  # tarfile stops before the gzip trailer.
                    pass
    if file_sha256(destination / "installed.lock") != manifest["dependencies_sha256"]:
        raise ValueError("dependency lock checksum mismatch")


def check_reap_files(directory: Path) -> dict:
    files = json.loads((directory / REAP_MANIFEST).read_text())["files"]
    for name, record in files.items():
        if (directory / name).stat().st_size != record["bytes"]:
            raise ValueError(f"REAP v2 file size differs from its manifest: {name}")
    return files


def build_reap_target(attachment: Path, reap: Path, destination: Path, recorder) -> Path:
    """Link the REAP v2 shards, index and config over the attachment's other files.

    The 95 GiB PLE table and the other unchanged shards stay on the original attachment.
    Every tensor the REAP index places in one of those files must be in its header.
    """
    from flash_next_loading import safetensors_header  # Shipped beside kaggle_runner.py, on sys.path here.

    files = check_reap_files(reap)
    destination.mkdir(parents=True, exist_ok=False)
    for path in attachment.iterdir():
        if path.is_file() and path.name not in files:
            (destination / path.name).symlink_to(path)
    for name in files:
        (destination / name).symlink_to(reap / name)
    weight_map = json.loads((destination / "model.safetensors.index.json").read_text())["weight_map"]
    linked: dict[str, set[str]] = {}
    for tensor, shard in weight_map.items():
        if shard not in files:
            linked.setdefault(shard, set()).add(tensor)
    for shard, tensors in sorted(linked.items()):
        if not (destination / shard).is_file():
            raise FileNotFoundError(f"model attachment lacks REAP v2 shard {shard}")
        missing = tensors - set(safetensors_header(destination / shard)[1])
        if missing:
            raise ValueError(f"attachment shard {shard} lacks {len(missing)} REAP v2 tensors")
    experts = json.loads((destination / "config.json").read_text())["text_config"]["num_experts"]
    if experts != REAP_EXPERTS:
        raise ValueError(f"REAP v2 config has {experts} routed experts, not {REAP_EXPERTS}")
    recorder.event("reap_target_view", destination=str(destination), reap_files=len(files),
                   linked_shards=len(linked), tensors=len(weight_map))
    return destination


def prefill_profile() -> list:
    """Kaggle's prefill graph [backend, bucket]: its captured kernels differ by both, so a cache records them."""
    return [PREFILL_GRAPH_BACKEND if PREFILL_GRAPH else "off", PREFILL_CHUNK]


def restore_compile_cache(runtime: Path, manifest: dict, recorder) -> None:
    """Add the REAP v2 cache kernel's compiled kernels to the restored bundle cache."""
    if COMPILE_CACHE_SHA256 is None:
        recorder.event("compile_cache_not_pinned")
        return
    # Only the cache kernel's output also holds its warmup report.
    source = find_directory_with(CACHE_MANIFEST, COMPILE_CACHE_ARCHIVE, "warmup_report.json")
    pins = json.loads((source / CACHE_MANIFEST).read_text())
    expected = {"sglang_version": SGLANG_VERSION, "sha256": COMPILE_CACHE_SHA256,
                "runtime_manifest_sha256": manifest["runtime_manifest_sha256"],
                "reap_target_dataset": REAP_TARGET_DATASET, "prefill_graph": prefill_profile()}
    # A cache built for a covering profile (more running, a superset of the graph sizes) is accepted: it holds most
    # of a smaller profile's kernels, and the smaller profile's own variants (mamba slots, graph set) compile at start.
    requests, _, graph_sizes = request_profile(None)
    cached = pins.get("request_profile") or [0, 0, []]
    covered = (cached[0] >= requests and cached[1] == MAMBA_SLOTS_PER_REQUEST * cached[0]
               and set(graph_sizes) <= set(cached[2]))
    if {key: pins.get(key) for key in expected} != expected or not covered:
        raise ValueError("compile cache was built for a different runtime, profile or checkpoint")
    with recorder.timed("compile_cache_verify"):
        if file_sha256(source / COMPILE_CACHE_ARCHIVE) != COMPILE_CACHE_SHA256:
            raise ValueError("compile cache archive checksum mismatch")
    with recorder.timed("compile_cache_restore"):
        with tarfile.open(source / COMPILE_CACHE_ARCHIVE, "r:gz") as archive:
            members = archive.getmembers()
            if any(member.name != "cache" and not member.name.startswith("cache/") for member in members):
                raise ValueError("compile cache member outside cache/")
            # Links are host-specific: prepare_cuda_linker relinks libcuda at every start.
            members = [member for member in members if member.isfile() or member.isdir()]
            archive.extractall(runtime, members=members, filter="data")
    recorder.event("compile_cache_restored", files=len(members),
                   triton_relocation=relocate_triton_groups(runtime / "cache"))


def sandbox_preflight_probe(workspace: Path, command: str = SANDBOX_PREFLIGHT, *,
                            first_yield_ms: int = 10_000, limit_seconds: int = SANDBOX_PREFLIGHT_SECONDS) -> str:
    """Source that runs ``command`` in a codex_lite sandbox and prints its final observation as JSON.

    v11's first save failed when the import outlasted a single 10 s yield on a slow Kaggle VM
    (0.6 s in cache kernel v1); like a model, the probe polls the session until it exits.
    """
    return (
        "import json, pathlib, re, time, codex_lite\n"
        f"workspace = codex_lite.Workspace(pathlib.Path({str(workspace)!r}), {{'train': [], 'test': []}})\n"
        "started = time.monotonic()\n"
        f"latest = workspace.exec_command({{'cmd': {command!r}, 'yield_time_ms': {first_yield_ms}}})\n"
        "output = latest\n"
        "while (running := re.search(r'Process running with session ID (\\d+)', latest)) and "
        f"time.monotonic() - started < {limit_seconds}:\n"
        f"    left_ms = ({limit_seconds} - (time.monotonic() - started)) * 1000\n"
        "    latest = workspace.write_stdin({'session_id': int(running[1]), "
        "'yield_time_ms': max(250, min(30000, left_ms))})\n"
        "    output += '\\n' + latest\n"
        "print(json.dumps({'output': output, 'seconds': round(time.monotonic() - started, 2), "
        "'uid': workspace.uid, 'network_isolated': codex_lite.NETWORK_ISOLATED}))\n"
    )


def preflight_passed(output: str) -> bool:
    """The last observation in a polled preflight is its final status."""
    statuses = [line for line in output.splitlines() if line.startswith("Process ")]
    return bool(statuses) and statuses[-1] == "Process exited with code 0"


def server_log_metrics(log_path: Path) -> dict:
    """KV pool size and decode throughput from rank 0's SGLang log, in one pass.

    End-to-end tokens/s counts DECODE_LOG_INTERVAL steps of running x accept-length tokens per
    logged line; the medians are SGLang's own per-line aggregate decode rates.
    """
    kv_tokens, rows = None, []
    for line in Path(log_path).read_text(errors="replace").splitlines():
        if "TP0" not in line:
            continue
        if kv_tokens is None and "KV Cache is allocated" in line and "#tokens: " in line:
            kv_tokens = int(line.split("#tokens: ")[1].split(",")[0])
        elif "Decode batch" in line and "#running-req: " in line:
            rows.append((datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S").timestamp(),
                         int(line.split("#running-req: ")[1].split(",")[0]),
                         float(line.split("accept len: ")[1].split(",")[0]),
                         float(line.rsplit("gen throughput (token/s): ", 1)[1].split(",")[0])))
    rates: dict[int, list[float]] = {}
    for _, running, _, rate in rows:
        rates.setdefault(running, []).append(rate)
    decode = {}
    if len(rows) >= 2 and rows[-1][0] > rows[0][0]:
        span = rows[-1][0] - rows[0][0]
        decode = {"tokens_per_second": round(sum(DECODE_LOG_INTERVAL * n * accept
                                                 for _, n, accept, _ in rows[1:]) / span, 1),
                  "mean_running": round(sum(row[1] for row in rows) / len(rows), 2),
                  "mean_accept_len": round(sum(row[2] for row in rows) / len(rows), 3),
                  "window_seconds": span}
    return {"kv_pool_tokens": kv_tokens, "decode": decode,
            "median_decode_tokens_per_second_by_running": {
                str(n): sorted(v)[len(v) // 2] for n, v in sorted(rates.items())}}


def run_report(server_log_path: Path, attempts_path: Path, **fields) -> dict:
    """Throughput and health of a save or cache-kernel run; diagnostics never fail the run."""
    report = {"server_ready_seconds": fields["server_ready_seconds"], "budget_reached": fields["budget_reached"],
              "request_profile": fields["profile"]}
    try:
        report.update(server_log_metrics(server_log_path))
        attempts = []
        if attempts_path.exists():
            with attempts_path.open() as stream:
                for line in stream:
                    try:
                        attempts.append(json.loads(line))
                    except json.JSONDecodeError:  # A record torn by stopping the runner.
                        pass
        report.update(attempts_completed=len(attempts), generated_tokens=sum(
            (a.get("stats") or {}).get("completion_tokens_used", 0) for a in attempts))
    except Exception as exc:  # A malformed log line must not fail a save that served correctly.
        report["error"] = repr(exc)[:500]
    return report


def stage_model(source, destination, recorder, *, workers=MODEL_STAGE_WORKERS):
    """Copy once to disk before TP ranks load, as in the successful FP8 benchmark."""
    if not 1 <= workers <= 32:
        raise ValueError("model staging workers must be between 1 and 32")
    if destination.exists():
        raise FileExistsError(destination)
    entries = sorted(source.iterdir())
    files = [path for path in entries if path.is_file()]
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("checkpoint index has no weight_map")
    names = weight_map.values()
    if any(not isinstance(name, str) or not name or Path(name).name != name for name in names):
        raise ValueError("checkpoint index must reference root-level shard files")
    shards = set(names)
    missing = shards - {path.name for path in files}
    if missing:
        raise FileNotFoundError("missing checkpoint shards: " + ", ".join(sorted(missing)))
    size = sum(path.stat().st_size for path in files)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < size + 4 * 2**30:
        raise RuntimeError("insufficient local scratch space for checkpoint staging")
    recorder.event(
        "model_stage_layout", destination=str(destination), workers=workers,
        files=len(files), indexed_shards=len(shards), bytes=size,
        ignored_directories=[path.name for path in entries if path.is_dir()],
    )
    with recorder.timed("model_stage"):
        destination.mkdir()
        started = last_log = time.monotonic()
        copied = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            transfers = {pool.submit(shutil.copyfile, path, destination / path.name): path
                         for path in files}
            for index, transfer in enumerate(as_completed(transfers), 1):
                path = transfers[transfer]
                transfer.result()
                copied += path.stat().st_size
                if time.monotonic() - last_log >= 30 or index == len(files):
                    recorder.event(
                        "model_stage_progress", destination=str(destination),
                        files=index, total_files=len(files), bytes=copied,
                        total_bytes=size,
                        bytes_per_second=copied / max(time.monotonic() - started, 1e-9),
                    )
                    last_log = time.monotonic()
        for path in files:
            if path.stat().st_size != (destination / path.name).stat().st_size:
                raise ValueError(f"incomplete checkpoint copy: {path.name}")
    recorder.event("model_staged", destination=str(destination), files=len(files), bytes=size)
    return destination


def prepare_triton_tools(sglang_site: Path, cache_root: Path) -> dict[str, str]:
    """Restore executable Triton tools stripped by Kaggle's read-only mount."""
    source_root = sglang_site / "triton" / "backends" / "nvidia" / "bin"
    target_root = cache_root / "triton_tools"
    target_root.mkdir(parents=True, exist_ok=True)
    variables = {
        "cuobjdump": "TRITON_CUOBJDUMP_PATH",
        "nvdisasm": "TRITON_NVDISASM_PATH",
        "ptxas": "TRITON_PTXAS_PATH",
        "ptxas-blackwell": "TRITON_PTXAS_BLACKWELL_PATH",
    }
    environment = {}
    for name, variable in variables.items():
        source = source_root / name
        target = target_root / name
        if not target.is_file() or target.stat().st_size != source.stat().st_size:
            shutil.copy2(source, target)
        target.chmod(0o755)
        environment[variable] = str(target)
    return environment


def prepare_cuda_linker(cache_root: Path) -> Path:
    """Expose the real driver library under the unversioned linker name."""
    candidates = (
        Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
        Path("/lib/x86_64-linux-gnu/libcuda.so.1"),
        Path("/usr/local/nvidia/lib64/libcuda.so.1"),
        Path("/usr/lib64/libcuda.so.1"),
    )
    driver = next((path.resolve() for path in candidates if path.is_file()), None)
    if driver is None:
        result = subprocess.run(
            ["ldconfig", "-p"], capture_output=True, text=True, check=True
        )
        for line in result.stdout.splitlines():
            if "libcuda.so.1" in line and "=>" in line:
                candidate = Path(line.rsplit("=>", 1)[1].strip())
                if candidate.is_file():
                    driver = candidate.resolve()
                    break
    if driver is None:
        raise FileNotFoundError("could not resolve the NVIDIA driver libcuda.so.1")

    link_root = cache_root / "cuda_link"
    link_root.mkdir(parents=True, exist_ok=True)
    link = link_root / "libcuda.so"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(driver)
    return link_root


def prepare_cuda_runtime_linker(cuda_home: Path) -> None:
    """Expose CUDA 13's versioned wheel library where SGLang's JIT links it."""
    cudart = cuda_home / "lib/libcudart.so.13"
    if not cudart.is_file():
        raise FileNotFoundError(cudart)
    lib64 = cuda_home / "lib64"
    lib64.mkdir(exist_ok=True)
    for name in ("libcudart.so", "libcudart.so.13"):
        link = lib64 / name
        if not link.exists():
            link.symlink_to(Path("..") / "lib" / cudart.name)


def terminate(process: subprocess.Popen, grace_seconds: float = 30.0) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def tail_lines(path: Path, count: int = 200) -> str:
    with path.open(errors="replace") as stream:
        return "".join(deque(stream, maxlen=count))


def cgroup_counters() -> dict[str, str]:
    """This process's cgroup v2 memory and CPU counters (the container root on Kaggle)."""
    relative = "/"
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                relative = line[3:]
    except OSError:
        pass
    root = Path("/sys/fs/cgroup") / relative.lstrip("/")
    counters = {}
    for name in ("memory.current", "memory.peak", "memory.max", "memory.events", "memory.pressure",
                 "cpu.max", "cpu.stat", "cpu.pressure"):
        path = root / name
        if path.is_file():
            counters[name] = path.read_text().strip()
    return counters


def cache_files(cache_root: Path) -> set[str]:
    """Files under the cache root that an export would hold, relative to it."""
    return {str(path.relative_to(cache_root)) for path in cache_root.rglob("*")
            if path.is_file() and not path.is_symlink() and path.relative_to(cache_root).parts[0] not in CACHE_EXCLUDED}


def home_cache_files() -> set[str]:
    """Files in the compile caches under HOME (HOME_CACHES), relative to HOME."""
    home = Path.home()
    return {str(path.relative_to(home)) for name in HOME_CACHES for path in (home / name).rglob("*")
            if path.is_file() and not path.is_symlink()}


def compiled_files(new: set[str]) -> dict:
    """New cache files (what compiled) by top directory, and the Triton kernels among them."""
    names = sorted(new)
    return {"files": len(names), "by_dir": dict(Counter(Path(name).parts[0] for name in names)),
            "triton_kernels": sorted({Path(name).stem for name in names if name.endswith(".cubin")}),
            "first_files": names[:40]}


def wait_for_server(process: subprocess.Popen, log_path: Path, scratch: Path, port: int, seconds: float,
                    resources_path: Path) -> None:
    """Poll until SGLang is healthy, for at most ``seconds`` (the rest of the notebook budget)."""
    deadline = time.time() + seconds
    last_status = 0
    last_logged = 0
    log_offset = 0
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"SGLang exited during startup with {process.returncode}:\n"
                + tail_lines(log_path)
            )
        if time.time() - last_status >= 30:
            with log_path.open(errors="replace") as stream:
                stream.seek(log_offset)
                for line in stream:
                    if any(marker in line for marker in (
                        "[weight-loading]", "Load weight", "Engine startup", "Traceback", "Error",
                    )):
                        print(line.rstrip(), flush=True)
                log_offset = stream.tell()
            staged_bytes = 0
            for path in (scratch / "weights/gpu_window").glob("*"):
                try:
                    staged_bytes += path.stat().st_size
                except FileNotFoundError:
                    pass  # The loader releases completed shards concurrently.
            resource = {
                "stage": "server_startup_heartbeat",
                "at": datetime.now(timezone.utc).isoformat(),
                "gpu_stage_file_bytes": staged_bytes,
                "filesystem_free_bytes": shutil.disk_usage(scratch).free,
                **cgroup_counters(),
            }
            with resources_path.open("a") as stream:
                stream.write(json.dumps(resource) + "\n")
            if time.time() - last_logged >= HEARTBEAT_LOG_SECONDS:
                print("[startup] " + json.dumps(resource), flush=True)
                last_logged = time.time()
            last_status = time.time()
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=5
            ) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError("SGLang did not become healthy within the notebook budget")


def validate_submission(path, challenges):
    data = json.loads(path.read_text())
    if set(data) != set(challenges):
        raise ValueError("submission task IDs differ")
    for key, task in challenges.items():
        if len(data[key]) != len(task["test"]):
            raise ValueError("submission output count differs")
        for pair in data[key]:
            if set(pair) != {"attempt_1", "attempt_2"}:
                raise ValueError("submission must contain both attempts")
            for grid in pair.values():
                if not isinstance(grid, list) or not 1 <= len(grid) <= 30:
                    raise ValueError("invalid grid height")
                width = len(grid[0]) if isinstance(grid[0], list) else 0
                if not 1 <= width <= 30 or not all(
                    isinstance(row, list)
                    and len(row) == width
                    and all(type(x) is int and 0 <= x <= 9 for x in row)
                    for row in grid
                ):
                    raise ValueError("invalid grid")


def score_submission(submission: dict, solutions: dict) -> dict:
    """Kaggle's metric: each task scores the fraction of its outputs matched by either attempt."""
    per_task = {}
    for key, truths in solutions.items():
        pairs = submission.get(key, [])
        hits = [i < len(pairs) and truth in (pairs[i].get("attempt_1"), pairs[i].get("attempt_2"))
                for i, truth in enumerate(truths)]
        per_task[key] = sum(hits) / len(truths)
    return {"score": 100 * sum(per_task.values()) / len(per_task), "tasks": len(per_task),
            "solved_outputs": sum(round(v * len(solutions[k])) for k, v in per_task.items()),
            "per_task": per_task}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--challenges", type=Path,
                        help="task JSON (default: this Kaggle run's competition file)")
    parser.add_argument("--solutions", type=Path,
                        help="public solutions: post-verify every attempt and score the submission")
    parser.add_argument("--deployment-root", type=Path,
                        help="installed H200 deployment (.venv-sglang, runtime/compat, models, cache) "
                             "used instead of the Kaggle attachments")
    parser.add_argument("--model-path", type=Path, help="target weights (default: attachment or <root>/models/target)")
    parser.add_argument("--draft-path", type=Path, help="MTP draft weights (default: attachment or <root>/models/draft)")
    parser.add_argument("--cache", type=Path,
                        help="compiler cache (default: the restored Kaggle cache or <root>/cache/replicas/launcher)")
    parser.add_argument("--source", type=Path, help="built inducer runtime dataset (default: the Kaggle attachment)")
    parser.add_argument("--work", type=Path, help="output directory (default: /kaggle/working)")
    parser.add_argument("--gpus", default="0,1,2,3", help="CUDA devices for the four TP ranks")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--kv-tokens", type=int,
                        help=f"cap the KV pool, e.g. {L4_KV_TOKENS} to reproduce Kaggle's L4 pool on larger GPUs")
    parser.add_argument("--memory-fraction",
                        help=f"static GPU memory fraction (local default 0.85; Kaggle {KAGGLE_MEMORY_FRACTION})")
    parser.add_argument("--max-running-requests", type=int,
                        help=f"server batch and episode admission ceiling (default {MAX_RUNNING_REQUESTS}; 160 "
                             f"for local runs on Hopper GPUs); sizes decode graphs and {MAMBA_SLOTS_PER_REQUEST} "
                             "Mamba slots per request")
    parser.add_argument("--chunked-prefill-size", type=int,
                        help=f"prefill chunk and per-batch prefill token limit (default {PREFILL_CHUNK}, Kaggle's)")
    parser.add_argument("--seed-base", type=int, help=f"first attempt's seed (default {SEED_BASE}); the second adds 1")
    parser.add_argument("--temperature", type=float, help=f"sampling temperature (default {TEMPERATURE})")
    parser.add_argument("--top-p", type=float, help="nucleus sampling (default codex_lite's 0.95; top-k stays 20)")
    parser.add_argument("--logit-bias", help='JSON object of token id to additive logit bias, e.g. \'{"77264": -1}\'')
    parser.add_argument("--host-all-reduce", choices=("on", "off"),
                        help="local runs: decode all-reduces through pinned host memory (default off; Kaggle runs "
                             f"use HOST_ALL_REDUCE, {'on' if HOST_ALL_REDUCE else 'off'})")
    parser.add_argument("--dcp-size", type=int, choices=(1, 2),
                        help=f"local runs: decode context parallel size (default {DCP_SIZE}; with the target's 2 KV "
                             "heads at TP4 a larger size frees nothing more)")
    parser.add_argument("--topk-logits", type=int,
                        help=f"local runs: each rank's logits sent in the all-gather (default {TOPK_LOGITS}; 0 sends the "
                             "whole shard)")
    parser.add_argument("--moe-down-ar", choices=("on", "off"),
                        help=f"local runs: fused MoE down projection + host all-reduce (needs the host all-reduce; "
                             f"default {'on' if MOE_DOWN_AR else 'off'})")
    parser.add_argument("--hc-mix-cuda", choices=("on", "off"),
                        help=f"local runs: CUDA HC branch mix for 1-64 tokens (default {'on' if HC_MIX_CUDA else 'off'})")
    parser.add_argument("--prefill-graph", choices=("on", "off"),
                        help=f"local runs: prefill CUDA graph (default {'on' if PREFILL_GRAPH else 'off'})")
    parser.add_argument("--prefill-graph-backend", choices=("breakable", "tc_piecewise"),
                        help=f"local runs: prefill CUDA graph backend (default {PREFILL_GRAPH_BACKEND})")
    parser.add_argument("--qsa-decode", choices=("on", "off"),
                        help=f"local runs: CUDA QSA indexer decode scoring (default {'on' if QSA_DECODE else 'off'})")
    parser.add_argument("--serve-only", action="store_true",
                        help="local runs: start SGLang, write <work>/server_ready.json and serve until SIGTERM; "
                             "evaluations then run through scripts/inducer/serve_eval.py")
    args = parser.parse_args(argv)
    if args.serve_only and args.deployment_root is None:
        parser.error("--serve-only is for local runs and requires --deployment-root")
    local_only = ("draft_path", "model_path", "memory_fraction", "kv_tokens", "max_running_requests",
                  "chunked_prefill_size", "seed_base", "temperature", "top_p", "logit_bias", "host_all_reduce",
                  "dcp_size", "topk_logits", "prefill_graph", "prefill_graph_backend", "moe_down_ar", "hc_mix_cuda",
                  "qsa_decode")
    overrides = ["--" + name.replace("_", "-") for name in local_only if getattr(args, name) is not None]
    if args.deployment_root is None and overrides:
        parser.error(f"{', '.join(overrides)}: overrides are for local runs and require --deployment-root")
    if args.chunked_prefill_size is not None and args.chunked_prefill_size < 1:
        parser.error("--chunked-prefill-size must be positive")
    if args.max_running_requests is not None and not 1 <= args.max_running_requests <= H200_MAX_RUNNING_REQUESTS:
        parser.error(f"--max-running-requests must be between 1 and {H200_MAX_RUNNING_REQUESTS}")
    return args


def runner_command(python: str, runtime_root: Path, work: Path, port: int, model_path: Path, seed_base: int,
                   requests: int, challenges_path: Path, submission_path: Path, progress_path: Path,
                   overrides: tuple = ()) -> list[str]:
    """The kaggle_runner invocation (also used by scripts/inducer/serve_eval.py against a held server)."""
    command = [
        python,
        str(runtime_root / "kaggle_runner.py"),
        "--teacher-token-cap",
        str(TOKEN_CAP),
        "--trace-out",
        str(work / "attempts.jsonl"),
        "--base-url",
        f"http://127.0.0.1:{port}/v1",
        "--model",
        SERVED_MODEL,
        "--seed-base",
        str(seed_base),
        "--attempt-mode",
        ATTEMPT_MODE,
        "--tokenizer-path",
        str(model_path),
        "--context-window",
        str(CONTEXT_LENGTH),
        "--request-timeout",
        "14400",
        "--time-budget",
        "14400",
        "--concurrency",
        str(requests),
        "--adaptive-concurrency",
        "--adaptive-min-inflight",
        "1",
        "--adaptive-initial-inflight",
        str(min(4, requests)),
        "--adaptive-max-inflight",
        str(requests),
        # Coverage first; second attempts start where the first did not verify.
        "--secondary-unverified-first",
        "--adaptive-poll-interval",
        "5",
        "--challenges",
        str(challenges_path),
        "--submission",
        str(submission_path),
        "--progress",
        str(progress_path),
    ]
    # Local sampling experiments; Kaggle runs keep the runner's (codex_lite's) defaults.
    for flag, value in overrides:
        if value is not None:
            command.extend([flag, str(value)])
    return command


def request_profile(max_running: int | None, default: int = MAX_RUNNING_REQUESTS) -> tuple[int, int, list[str]]:
    """Server request ceiling, Mamba slots and decode graph batch sizes (Kaggle: 10, 40, 1-10)."""
    requests = max_running or default
    # A batch replays the smallest captured graph that holds it, at that graph's cost (with graphs 1, 2, 4, 8, 16, 9-15
    # running all paid the bs-16 step on L4, ~100 ms). Up to 16 every size has its graph; each costs ~0.02 GB of
    # target-verify memory on L4 (0.41 GB for 1-12 against 0.26 GB for 1, 2, 4, 8). Above 16 a ladder of 8 to 64, then
    # 16, bounds the padding near 10%.
    ladder = (*range(1, 17), *range(24, 65, 8), *range(80, requests + 1, 16))
    sizes = sorted({n for n in ladder if n <= requests} | {requests})
    return requests, MAMBA_SLOTS_PER_REQUEST * requests, [str(n) for n in sizes]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or [])
    kernel_started = time.monotonic()
    local = args.deployment_root is not None
    rerun = os.getenv("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    work = args.work or WORK
    scratch = work / "scratch" if local else SCRATCH
    warmup = LAUNCH_MODE == "warmup"
    if warmup and (rerun or local):
        raise RuntimeError("the warmup copy is a Kaggle utility kernel, not a submission")
    # Ordinary Kaggle saves run the smoke; a task JSON selects every task in it.
    smoke = not rerun and args.challenges is None and not warmup
    challenges_path = args.challenges or competition_file(
        "arc-agi_test_challenges.json"
        if rerun
        else "arc-agi_evaluation_challenges.json"
    )
    challenges = json.loads(challenges_path.read_text())

    runtime_root = args.source or find_directory_with(
        "kaggle_runner.py",
        "loop_driver.py",
        "adaptive_concurrency.py",
    )
    manifest = json.loads((runtime_root / "launch_manifest.json").read_text())
    for name, digest in manifest["source_sha256"].items():
        if file_sha256(runtime_root / name) != digest:
            raise ValueError(f"runtime source checksum mismatch: {name}")
    if local:
        root = args.deployment_root.resolve()
        bundle_manifest_path = root / "artifacts/runtime_manifest.json"
    else:
        bundle_root = find_directory_with(
            "flash_next_runtime.tar.gz", "runtime_manifest.json"
        )
        bundle_manifest_path = bundle_root / "runtime_manifest.json"
    if file_sha256(bundle_manifest_path) != manifest["runtime_manifest_sha256"]:
        raise ValueError("SGLang bundle manifest checksum mismatch")
    bundle = json.loads(bundle_manifest_path.read_text())
    if bundle.get("sglang_version") != SGLANG_VERSION:
        raise ValueError("SGLang version differs from the validated L4 runtime")
    work.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    # Persist the complete schema before the expensive setup, just as in FP8.
    write_json(
        work / "submission.json",
        {
            key: [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in task["test"]]
            for key, task in challenges.items()
        },
    )
    write_json(work / "launch_manifest.json", manifest)
    recorder = StartupRecorder(work / "startup.jsonl")
    recorder.event("filesystems", **mount_diagnostics(challenges_path, work, scratch))
    if local:
        # The same pinned distributions, installed once with H200 compiler caches.
        sglang_site = root / ".venv-sglang/lib/python3.12/site-packages"
        python = str(root / ".venv-sglang/bin/python")
        compat = root / "runtime/compat"
        if file_sha256(root / "runtime/installed.lock") != bundle["dependencies_sha256"]:
            raise ValueError("dependency lock checksum mismatch")
        # Absolute: compilers run from their own build directories with TMPDIR under this cache.
        cache_root = (args.cache or root / "cache/replicas/launcher").resolve()
    else:
        runtime = scratch / "runtime"
        restore_bundle(bundle_root, runtime, bundle, recorder)
        if not warmup:
            restore_compile_cache(runtime, manifest, recorder)
        sglang_site = runtime / bundle["site"]
        python = str(runtime / bundle["python"])
        compat = runtime / "compat"
        cache_root = args.cache or runtime / "cache"
    sys.path.insert(0, str(runtime_root))
    from draft_view_adapter import prepare_draft_view
    from flash_next_serving import patch_sglang as patch_serving

    # The one Flash-Next patch set for SGLang 0.5.20; Kaggle's loader hooks below build on it.
    recorder.event("sglang_patches", files=patch_serving(sglang_site / "sglang"))

    loading_environment = {}
    if local:
        model_path = args.model_path or root / "models/target"
        draft_path = args.draft_path or root / "models/draft"
    else:
        from flash_next_loading import PLAN_ENV, patch_sglang, prepare_plan

        # PLE and the other unchanged files remain on the attachment. The loader stages
        # only two GPU shards at a time; the separate small draft is copied in full.
        model_path = build_reap_target(
            Path("/kaggle/input/models") / TARGET_MODEL,
            find_directory_with(REAP_MANIFEST, "phi1_reap_export.json"),
            scratch / "weights/reap_v2_target", recorder,
        )
        loading_plan = work / "weight_loading_plan.json"
        prepare_plan(model_path, scratch / "weights/gpu_window", loading_plan)
        patch_sglang(sglang_site)
        reap_draft = find_directory_with(REAP_MANIFEST, "mtp-routed-experts-int4.safetensors", REAP_DRAFT_EXPORT)
        check_reap_files(reap_draft)
        draft_path = stage_model(reap_draft, scratch / "weights/draft", recorder)
        loading_environment[PLAN_ENV] = str(loading_plan)
    draft_view = prepare_draft_view(model_path, draft_path, scratch)

    submission_path = work / "submission.json"
    progress_path = work / "inducer_progress.json"

    for name in ("cuda", "inductor", "triton", "tilelang", "sglang", "torch_extensions", "tmp"):
        (cache_root / name).mkdir(parents=True, exist_ok=True)
    triton_tool_environment = prepare_triton_tools(sglang_site, cache_root)
    cuda_link_directory = prepare_cuda_linker(cache_root)
    if local:
        capability = subprocess.run(
            ["nvidia-smi", "--id", args.gpus.split(",")[0], "--query-gpu=compute_cap",
             "--format=csv,noheader"], capture_output=True, text=True, check=True,
        ).stdout.strip()
        hardware_environment = {
            "TORCH_CUDA_ARCH_LIST": capability,
            "FLASHINFER_CUDA_ARCH_LIST": capability + ("a" if capability.startswith("9") else ""),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        }
        # Other owners' small processes may share these GPUs; --kv-tokens sizes the pool.
        memory_fraction = args.memory_fraction or "0.85"
        default_requests = H200_MAX_RUNNING_REQUESTS if capability.startswith("9") else MAX_RUNNING_REQUESTS
        kv_tokens = args.kv_tokens
        # Local runs keep NCCL unless --host-all-reduce on.
        host_all_reduce = args.host_all_reduce == "on"
    else:
        hardware_environment = {"TORCH_CUDA_ARCH_LIST": "8.9", "FLASHINFER_CUDA_ARCH_LIST": "8.9"}
        memory_fraction = KAGGLE_MEMORY_FRACTION
        default_requests = MAX_RUNNING_REQUESTS
        kv_tokens = L4_KV_TOKENS
        host_all_reduce = HOST_ALL_REDUCE
    dcp_size = args.dcp_size or DCP_SIZE
    topk_logits = TOPK_LOGITS if args.topk_logits is None else args.topk_logits
    prefill_graph = PREFILL_GRAPH if args.prefill_graph is None else args.prefill_graph == "on"
    moe_down_ar = host_all_reduce and (MOE_DOWN_AR if args.moe_down_ar is None else args.moe_down_ar == "on")
    hc_mix_cuda = HC_MIX_CUDA if args.hc_mix_cuda is None else args.hc_mix_cuda == "on"
    prefill_graph_backend = args.prefill_graph_backend or PREFILL_GRAPH_BACKEND
    qsa_decode = QSA_DECODE if args.qsa_decode is None else args.qsa_decode == "on"

    python_path = os.pathsep.join(
        [
            str(runtime_root),
            str(sglang_site),
            str(sglang_site / "nvidia_cutlass_dsl/dsl_packages"),
        ]
    )
    cuda_home = sglang_site / "nvidia/cu13"
    prepare_cuda_runtime_linker(cuda_home)
    environment = {
        **os.environ,
        **loading_environment,
        "PYTHONPATH": python_path,
        "CUDA_HOME": str(cuda_home),
        "CUDACXX": str(cuda_home / "bin/nvcc"),
        "PATH": os.pathsep.join(
            (
                str(cuda_home / "bin"),
                str(Path(python).parent),
                str(sglang_site / "ninja/data/bin"),
                os.environ.get("PATH", ""),
            )
        ),
        "LD_LIBRARY_PATH": os.pathsep.join(
            (
                str(compat),
                str(cuda_home / "lib64"),
                str(cuda_home / "lib"),
                os.environ.get("LD_LIBRARY_PATH", ""),
            )
        ),
        "XDG_CACHE_HOME": str(cache_root),
        "CUDA_CACHE_PATH": str(cache_root / "cuda"),
        "FLASHINFER_WORKSPACE_BASE": str(cache_root / "sglang"),
        "SGLANG_CACHE_DIR": str(cache_root / "sglang"),
        "SGLANG_JIT_CACHE_DIR": str(cache_root / "sglang" / "jit"),
        "SGLANG_JIT_CACHE_DEBUG": "1",
        "TORCH_EXTENSIONS_DIR": str(cache_root / "torch_extensions"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "inductor"),
        "TRITON_CACHE_DIR": str(cache_root / "triton"),
        "TILELANG_CACHE_DIR": str(cache_root / "tilelang"),
        "TILELANG_TMP_DIR": str(cache_root / "tmp" / "tilelang"),
        "TMPDIR": str(cache_root / "tmp"),
        "CUDA_VISIBLE_DEVICES": args.gpus,
        **hardware_environment,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # HC mixes all-gather a lowrank slice and a hidden slice instead of all-reducing [T, 4 x hidden].
        "SGLANG_FLASH_NEXT_HC_TP": "1",
        # v30: every rank keeps the whole HC up projection in INT8 (quantized on the CPU at load), so the 97 hidden
        # all-gathers a step go (Modal 4x L4: decode step -4/-5% at 8/10 running, prefill -18% a batch).
        "SGLANG_FLASH_NEXT_HC_LOCAL_UP": "1",
        # pynccl runs only inside CUDA graphs replayed on one stream (eager collectives use the
        # torch.distributed communicator), so graph-mixing support only adds 17-60 us per captured collective.
        "NCCL_GRAPH_MIXING_SUPPORT": "0",
        # Memory reclaim (flash_next_serving.py): the MTP draft never allocates its own full-vocab embedding and
        # lm_head (the target's replace them after the KV pool is sized), and the target's embed_tokens shard is
        # served from pinned host memory like the PLE tables. On L4 only the host embedding is a net saving: through
        # v20 the buffers allocated after pool sizing reused the draft's freed tables, which now go to the pool.
        "SGLANG_FLASH_NEXT_DRAFT_SHARED_VOCAB": "1",
        "SGLANG_FLASH_NEXT_HOST_EMBED": "1",
        **({"SGLANG_FLASH_NEXT_HOST_AR": "1"} if host_all_reduce else {}),
        **({"SGLANG_FLASH_NEXT_MOE_DOWN_AR": "1"} if moe_down_ar else {}),
        **({"SGLANG_FLASH_NEXT_HC_MIX_CUDA": "1"} if hc_mix_cuda else {}),
        **({"SGLANG_FLASH_NEXT_QSA_DECODE": "1"} if qsa_decode else {}),
        # Stock QSA has no DCP path: the target's sparse attention shards its KV through flash_next_qsa_dcp.py.
        **({"SGLANG_FLASH_NEXT_QSA_DCP": "1"} if dcp_size > 1 else {}),
        **({"SGLANG_FLASH_NEXT_TOPK_LOGITS": str(topk_logits)} if topk_logits else {}),
        **({"SGLANG_FLASH_NEXT_PREFILL_GRAPH": "1"} if prefill_graph else {}),
        "MAX_JOBS": "16",
        "LIBRARY_PATH": os.pathsep.join(
            filter(
                None,
                (
                    str(cuda_link_directory),
                    os.environ.get("LIBRARY_PATH", ""),
                ),
            )
        ),
        **triton_tool_environment,
    }
    environment.pop("CUDA_LAUNCH_BLOCKING", None)
    # Before loading weights: a sandboxed command, under the runner's interpreter and the uid an
    # attempt would get, must import the numpy and scipy the task prompt promises.
    probe = subprocess.run(
        [python, "-c", sandbox_preflight_probe(scratch / "codex_lite_preflight")],
        env={**environment, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True, check=False,
        timeout=SANDBOX_PREFLIGHT_SECONDS + 120,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        raise RuntimeError(f"codex_lite sandbox preflight exited with {probe.returncode}:\n"
                           + probe.stderr[-4000:])
    preflight = json.loads(probe.stdout.strip().splitlines()[-1])
    write_json(work / "codex_lite_preflight.json", preflight)
    recorder.event("sandbox_preflight", seconds=preflight.get("seconds"))
    if not preflight_passed(preflight["output"]):
        raise RuntimeError("codex_lite sandbox preflight failed:\n" + preflight["output"])

    requests, mamba_slots, graph_sizes = request_profile(args.max_running_requests, default_requests)
    # The draft hot-token map and the balanced expert placement for the patched server.
    from flash_next_serving import prepare_serving

    hot_token_map, expert_location = prepare_serving(python, environment, work, runtime_root, model_path)
    server_command = [
        python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--load-format",
        "safetensors",
        "--model-loader-extra-config",
        '{"enable_multithread_load":false}',
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        "4",
        "--expert-parallel-size",
        "4",
        *(["--decode-context-parallel-size", str(dcp_size)] if dcp_size > 1 else []),
        "--dtype",
        "bfloat16",
        "--quantization",
        "auto-round",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--mem-fraction-static",
        memory_fraction,
        *(["--max-total-tokens", str(kv_tokens)] if kv_tokens else []),
        "--context-length",
        str(CONTEXT_LENGTH),
        "--page-size",
        "64",
        "--max-running-requests",
        str(requests),
        "--chunked-prefill-size",
        str(args.chunked_prefill_size or PREFILL_CHUNK),
        "--max-prefill-tokens",
        str(args.chunked_prefill_size or PREFILL_CHUNK),
        "--cuda-graph-max-bs-decode",
        str(requests),
        "--cuda-graph-bs-decode",
        *graph_sizes,
        # One prefill bucket at the chunk size: a full ladder costs 0.8-1.1 GB per GPU and raised teacher-forced NLL
        # on phi1 (+0.017 nats over 35 tasks), [512, 1024] did not (prefill_graph_phi1_20261029/).
        # The tc compiler flag is what tc_piecewise needs; breakable ignores it (as in its vast.ai qualification).
        *(["--cuda-graph-backend-prefill", prefill_graph_backend, "--cuda-graph-tc-compiler", "eager",
           "--cuda-graph-bs-prefill", str(args.chunked_prefill_size or PREFILL_CHUNK)]
          if prefill_graph else ["--disable-prefill-cuda-graph"]),
        "--mamba-ssm-dtype",
        "bfloat16",
        "--max-mamba-cache-size",
        str(mamba_slots),
        "--mamba-radix-cache-strategy",
        MAMBA_RADIX_CACHE_STRATEGY,
        "--mamba-track-interval",
        "64",
        "--mamba-backend",
        "triton",
        "--linear-attn-decode-backend",
        "triton",
        "--linear-attn-prefill-backend",
        "triton",
        "--linear-attn-verify-backend",
        "triton",
        "--enable-linear-replayssm-spec",
        "--attention-backend",
        "flashinfer",
        "--moe-runner-backend",
        "auto",
        "--ple-offload-embedding",
        "--trust-remote-code",
        "--json-model-override-args",
        "{}" if ATTEMPT_MODE in IMAGE_ATTEMPT_MODES else '{"language_model_only":true}',
        "--mm-feature-transport",
        "cpu",
        "--image-processor-backend",
        "pil",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--default-chat-template-kwargs",
        '{"preserve_thinking":true}',
        "--chat-template",
        str(model_path / "chat_template.jinja"),
        "--watchdog-timeout",
        "1800",
        "--schedule-policy",
        "lpm",
        "--warmups",
        "structured_output",
        "--enable-cache-report",
        "--enable-metrics",
        "--enable-request-time-stats-logging",
        "--decode-log-interval",
        str(DECODE_LOG_INTERVAL),
        "--load-snapshot-publish-interval",
        "5",
        "--disable-flashinfer-autotune",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--speculative-token-map",
        str(hot_token_map),
        "--init-expert-location",
        str(expert_location),
        "--speculative-draft-model-path",
        str(draft_view),
        "--speculative-draft-model-quantization",
        "compressed-tensors",
        "--speculative-moe-runner-backend",
        "auto",
        "--speculative-draft-kv-cache-dtype",
        "fp8_e4m3",
        "--speculative-accept-threshold-single",
        "1.0",
        "--speculative-accept-threshold-acc",
        "1.0",
    ]
    write_json(work / "server_command.json", server_command)
    server_log_path = work / "sglang_server.log"
    # Save runs and the cache kernel report what compiled during startup and after the server was healthy.
    startup_files = startup_home = ready_files = None
    if smoke or warmup:
        try:  # Diagnostics never fail the run: a failed snapshot stays None and the report has compile_count_error.
            startup_files, startup_home = cache_files(cache_root), home_cache_files()
        except OSError:
            pass
    with server_log_path.open("w") as server_log:
        server = subprocess.Popen(
            server_command,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )

    try:
        print(f"[launcher] rerun={rerun} local={local} challenges={challenges_path}", flush=True)
        print(f"[launcher] model={model_path} draft={draft_path} cache={cache_root}", flush=True)
        print(
            f"[launcher] runtime={runtime_root} sglang_site={sglang_site}", flush=True
        )
        print(
            f"[launcher] solver={ATTEMPT_MODE} temperature={TEMPERATURE} "
            f"reasoning_effort={REASONING_EFFORT} token_cap={TOKEN_CAP} "
            f"server=w4a16_tp4_ep4_nextn_3_1_4 replayssm=True requests={requests} "
            f"max_mamba_cache_size={mamba_slots} kv_tokens={kv_tokens or 'auto'} "
            f"topk_logits={topk_logits or 'off'} "
            f"prefill_graph={prefill_graph_backend if prefill_graph else 'off'} "
            f"prefill_chunk={args.chunked_prefill_size or PREFILL_CHUNK} "
            f"moe_down_ar={'on' if moe_down_ar else 'off'} hc_mix_cuda={'on' if hc_mix_cuda else 'off'} "
            f"qsa_decode={'on' if qsa_decode else 'off'}",
            flush=True,
        )
        # No fixed startup limit (45 minutes through v16): on Kaggle's GCSFUSE dataset mounts loading can
        # take hours, and a late start still leaves inference time. A crashed server ends the wait at once.
        wait_for_server(server, server_log_path, scratch, args.port,
                        KERNEL_BUDGET_SECONDS - (time.monotonic() - kernel_started), work / "resources.jsonl")
        server_ready_seconds = time.monotonic() - kernel_started
        if startup_files is not None:
            try:
                ready_files = cache_files(cache_root)
            except OSError:
                pass
        print("[launcher] SGLang healthy", flush=True)

        if args.serve_only:
            # Local evaluations reuse this server (scripts/inducer/serve_eval.py) until a SIGTERM stops it.
            write_json(work / "server_ready.json", {
                "base_url": f"http://127.0.0.1:{args.port}/v1", "port": args.port, "server_pid": server.pid,
                "model": SERVED_MODEL, "tokenizer_path": str(model_path), "python": python,
                "requests": requests, "context_window": CONTEXT_LENGTH, "runtime_root": str(runtime_root),
                "server_ready_seconds": round(server_ready_seconds, 1),
                "ready_at": datetime.now(timezone.utc).isoformat()})
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
            print(f"[launcher] serving on port {args.port} until SIGTERM (--serve-only)", flush=True)
            while server.poll() is None:
                time.sleep(5)
            raise RuntimeError(f"SGLang exited with {server.returncode}:\n" + tail_lines(server_log_path))
        runner_argv = runner_command(
            python, runtime_root, work, args.port, model_path, SEED_BASE if args.seed_base is None else args.seed_base,
            requests, challenges_path, submission_path, progress_path,
            (("--temperature", args.temperature), ("--top-p", args.top_p), ("--logit-bias", args.logit_bias)))
        if smoke or warmup:
            if not set(SMOKE_KEYS) <= set(challenges):
                raise ValueError("missing required smoke puzzle")
            runner_argv.extend(["--keys", ",".join(SMOKE_KEYS), "--primary-only"])
            print(f"[launcher] save run: {len(SMOKE_KEYS)} public puzzles, first attempts, "
                  f"window={SMOKE_SECONDS}s", flush=True)

        write_json(work / "eval_command.json", runner_argv)
        remaining = KERNEL_BUDGET_SECONDS - (time.monotonic() - kernel_started)
        if remaining <= 0:
            raise TimeoutError("notebook budget exhausted during startup")
        # Attempt workspaces stay off the output directory.
        workspaces = scratch / "codex_lite"
        workspaces.mkdir(parents=True, exist_ok=True)
        runner = subprocess.Popen(
            runner_argv,
            env={**environment, "CUDA_VISIBLE_DEVICES": "", "ARC_CODEX_LITE_ROOT": str(workspaces)},
            start_new_session=True,
        )
        runner_deadline = kernel_started + KERNEL_BUDGET_SECONDS
        if smoke or warmup:
            runner_deadline = min(runner_deadline, time.monotonic() + SMOKE_SECONDS)
        budget_reached = False
        last_sample = None
        while runner.poll() is None:
            if server.poll() is not None:
                terminate(runner)
                raise RuntimeError(
                    f"SGLang exited during evaluation with {server.returncode}:\n"
                    + tail_lines(server_log_path)
                )
            if time.monotonic() >= runner_deadline:
                print(
                    "[launcher] wall budget reached; retaining incremental submission",
                    flush=True,
                )
                terminate(runner)
                budget_reached = True
                break
            if last_sample is None or time.monotonic() - last_sample >= 30:
                # Sandboxes, runner and server share this cgroup, as on the Kaggle VM.
                with (work / "resources.jsonl").open("a") as stream:
                    stream.write(json.dumps({"elapsed_seconds": time.monotonic() - kernel_started,
                                             **cgroup_counters()}) + "\n")
                last_sample = time.monotonic()
            time.sleep(min(5, max(0, runner_deadline - time.monotonic())))
        return_code = 0 if budget_reached else (runner.returncode or 0)
        if return_code != 0:
            raise RuntimeError(f"inducer runner exited with code {return_code}")
    finally:
        terminate(server)

    # Only the cache kernel exports: restore_compile_cache accepts no archive without its warmup report.
    # Before the checks and post-verification, so neither can lose the cache.
    if warmup:
        archive_path = work / COMPILE_CACHE_ARCHIVE
        excluded = {f"cache/{name}" for name in CACHE_EXCLUDED}
        with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
            archive.add(cache_root, arcname="cache",
                        filter=lambda info: None if info.issym() or info.name in excluded else info)
        write_json(
            work / CACHE_MANIFEST,
            {
                "sglang_version": SGLANG_VERSION,
                "runtime_manifest_sha256": manifest["runtime_manifest_sha256"],
                "sha256": file_sha256(archive_path),
                "request_profile": [requests, mamba_slots, graph_sizes],
                "prefill_graph": prefill_profile(),
                "reap_target_dataset": REAP_TARGET_DATASET,
                "launch_mode": LAUNCH_MODE,
            },
        )
        print(f"[launcher] packaged compile cache: {archive_path}", flush=True)
    if smoke or warmup:
        # Save runs stay first-attempts-only; carry second attempts are rehearsed on phi1.
        # The window stops unfinished attempts; every finished one must be a clean first attempt, and a save run must
        # finish one. The cache kernel must finish all 16, or it writes no warmup report and no submission can
        # restore its archive.
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"per_task": {}}
        finished = progress["per_task"]
        if smoke and not finished:
            raise RuntimeError("smoke finished no first attempt within its window")
        if not set(finished) <= set(SMOKE_KEYS) or any(set(phases) != {"primary"} for phases in finished.values()):
            raise RuntimeError("smoke ran outside its public puzzles or past first attempts")
        if any(
            result["error"]
            for phases in finished.values()
            for result in phases.values()
        ):
            raise RuntimeError(
                "smoke contains a failed attempt; inspect inducer_progress.json"
            )
        if warmup and (budget_reached or set(finished) != set(SMOKE_KEYS)):
            raise RuntimeError(f"cache kernel finished {len(finished)} of {len(SMOKE_KEYS)} first attempts within "
                               "its window; its cache would miss kernel variants")
        report = run_report(server_log_path, work / "attempts.jsonl", server_ready_seconds=server_ready_seconds,
                            budget_reached=budget_reached, profile=[requests, mamba_slots, graph_sizes])
        try:  # On a cache built at its own profile a save run compiles nothing: all three counts are zero.
            report["compiled_during_startup"] = compiled_files(ready_files - startup_files)
            report["compiled_after_ready"] = compiled_files(cache_files(cache_root) - ready_files)
            report["compiled_outside_cache_root"] = compiled_files(home_cache_files() - startup_home)
        except Exception as exc:  # Diagnostics never fail the run.
            report["compile_count_error"] = repr(exc)[:500]
        write_json(work / ("warmup_report.json" if warmup else "save_run_report.json"), report)

    # Save runs post-verify their attempts; the cache kernel only needs their compiles.
    solutions_path = args.solutions or (
        competition_file("arc-agi_evaluation_solutions.json") if smoke else None
    )
    if solutions_path is not None:
        subprocess.run(
            [
                python,
                str(runtime_root / "post_verify_traces.py"),
                "--trace-in",
                str(work / "attempts.jsonl"),
                "--solutions",
                str(solutions_path),
                "--ledger-out",
                str(work / "postverify.ledger.jsonl"),
                "--report-out",
                str(work / "postverify.report.json"),
            ],
            env={**environment, "CUDA_VISIBLE_DEVICES": ""},
            check=True,
        )
    validate_submission(submission_path, challenges)
    print(
        f"[launcher] validated submission for {len(challenges)} puzzles at {submission_path}",
        flush=True,
    )
    if args.solutions is not None:
        solutions = json.loads(args.solutions.read_text())
        report = score_submission(json.loads(submission_path.read_text()),
                                  {key: solutions[key] for key in challenges})
        write_json(work / "score.json", report)
        print(f"[launcher] public score {report['score']:.2f} over {report['tasks']} tasks", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
