"""ARC-AGI-2 solver: an open-weight reasoning LLM (llama.cpp, offline) + a 4B test-time-trained
grid model, merged into the two attempts per test output.

Phase LLM: every test output is sent to llama-server as an independent chat request (the model
reads the demonstration pairs and writes the answer grid); every output's first sample is issued
before any second sample, and each request is bounded by the phase deadline.
Phase TTT: the embedded test-time-training solver (its own standalone script, run as a
subprocess with the remaining wall) fits LoRA adapters per task and ranks candidates.
Merge: per test output, the two attempts are the first two DISTINCT grids of the configured
priority order over {LLM candidates by vote, TTT attempt_1, TTT attempt_2}.

Modes: "sequential" (LLM on all GPUs until llm_wall_s, then TTT with what remains) ·
"concurrent" (LLM on llm_gpus and TTT on ttt_gpus at once) · "llm_only" · "ttt_only".
A valid submission.json exists from the first second (test input echoed) and is rewritten as
answers arrive, so a run cut short still scores what it finished. No internet.
"""
import glob
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import base64
import struct
import zlib

T0 = time.time()
CFG = {'mode': 'ttt_then_llm', 'server': 'vllm', 'pip_pkgs': ['vllm==0.19.1', 'transformers==4.57.6'], 'ladder': [{'name': 'fp8_tp4_kvfp8_mtp2', 'model_globs': ['/kaggle/input/models/michaelpoluektov/qwen3-8-27b-fp8/**/config.json', '/kaggle/input/models/dangkhoa2016/qwen-qwen3-8-27b-fp8/**/config.json', '/kaggle/input/models/mikedan7/qwen3-8-27b-fp8-official/**/config.json'], 'args': ['--tensor-parallel-size', '4', '--max-model-len', '131072', '--reasoning-parser', 'qwen3', '--language-model-only', '--gpu-memory-utilization', '0.85', '--max-num-seqs', '32', '--kv-cache-dtype', 'fp8', '--reasoning-config', '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}', '--speculative-config', '{"method": "mtp", "num_speculative_tokens": 2}'], 'load_timeout': 1800}, {'name': 'fp8_tp4_kvfp8_fi', 'model_globs': ['/kaggle/input/models/michaelpoluektov/qwen3-8-27b-fp8/**/config.json', '/kaggle/input/models/dangkhoa2016/qwen-qwen3-8-27b-fp8/**/config.json', '/kaggle/input/models/mikedan7/qwen3-8-27b-fp8-official/**/config.json'], 'args': ['--tensor-parallel-size', '4', '--max-model-len', '131072', '--reasoning-parser', 'qwen3', '--language-model-only', '--gpu-memory-utilization', '0.90', '--max-num-seqs', '32', '--kv-cache-dtype', 'fp8', '--reasoning-config', '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'], 'load_timeout': 1800}, {'name': 'fp8_tp4_fa', 'model_globs': ['/kaggle/input/models/michaelpoluektov/qwen3-8-27b-fp8/**/config.json', '/kaggle/input/models/dangkhoa2016/qwen-qwen3-8-27b-fp8/**/config.json', '/kaggle/input/models/mikedan7/qwen3-8-27b-fp8-official/**/config.json'], 'args': ['--tensor-parallel-size', '4', '--max-model-len', '131072', '--reasoning-parser', 'qwen3', '--language-model-only', '--gpu-memory-utilization', '0.90', '--max-num-seqs', '20', '--attention-backend', 'FLASH_ATTN', '--reasoning-config', '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'], 'load_timeout': 1800}, {'name': 'bf16_tp4_fa', 'model_globs': ['/kaggle/input/models/dangkhoa2016/qwen-qwen3-8-27b/**/config.json'], 'args': ['--tensor-parallel-size', '4', '--max-model-len', '131072', '--reasoning-parser', 'qwen3', '--language-model-only', '--gpu-memory-utilization', '0.92', '--max-num-seqs', '10', '--attention-backend', 'FLASH_ATTN', '--reasoning-config', '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'], 'load_timeout': 1800}], 'samples': 2, 'max_tokens': 98304, 'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0, 'effort': 'low', 'parallel': 32, 'merge': 'llm_first', 'min_request_s': 900, 'llm_min_useful_s': 1800, 'merge_margin_s': 300, 'probe_tasks': 8, 'llm_probe_wall_s': 2400, 'llm_task_mode': 'grid', 'thinking_budget': None, 'commit_thinking_budget': 1024, 'fp8_tables': {'N=3584,K=5120,device_name=NVIDIA_L4,dtype=fp8_w8a8,block_shape=[128,128].json': {'1': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '2': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '4': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '8': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '16': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '24': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '32': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '40': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '48': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '56': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '64': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '80': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3}, '96': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}, '128': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}, '192': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2}, '256': {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 3}, '512': {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 8, 'num_stages': 4}, '1024': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 2}, '2048': {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 3}}, 'N=4096,K=5120,device_name=NVIDIA_L4,dtype=fp8_w8a8,block_shape=[128,128].json': {'1': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '2': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '4': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '8': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '16': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '24': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '32': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '40': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '48': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '56': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '64': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '80': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3}, '96': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}, '128': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2}, '192': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 3}, '256': {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}, '512': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 2}, '1024': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 2}, '2048': {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 3}}, 'N=5120,K=1536,device_name=NVIDIA_L4,dtype=fp8_w8a8,block_shape=[128,128].json': {'1': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3}, '2': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '4': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '8': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '16': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '24': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '32': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '40': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '48': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '56': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '64': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '80': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2}, '96': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2}, '128': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 2}, '192': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3}, '256': {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}, '512': {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 8, 'num_stages': 3}, '1024': {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}, '2048': {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}}, 'N=5120,K=4352,device_name=NVIDIA_L4,dtype=fp8_w8a8,block_shape=[128,128].json': {'1': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '2': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '4': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '8': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '16': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '24': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3}, '32': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 8, 'num_stages': 3}, '40': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '48': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '56': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '64': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '80': {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}, '96': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2}, '128': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 2}, '192': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2}, '256': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}, '512': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 2}, '1024': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 2}, '2048': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 3}}, 'N=8704,K=5120,device_name=NVIDIA_L4,dtype=fp8_w8a8,block_shape=[128,128].json': {'1': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '2': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '4': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '8': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '16': {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '24': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '32': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 3}, '40': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5}, '48': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5}, '56': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5}, '64': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 1, 'num_warps': 4, 'num_stages': 5}, '80': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3}, '96': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 3}, '128': {'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 4, 'num_stages': 2}, '192': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 3}, '256': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 32, 'num_warps': 4, 'num_stages': 3}, '512': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 3}, '1024': {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 3}, '2048': {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8, 'num_warps': 8, 'num_stages': 3}}}, 'sg_samples': [], 'commit_skip_llm': True, 'prompt_variants': [{'sg': True, 'diff': True, 'h2': True}, {'diff': True}, {'sg': True}], 'ttt_wall_s': 10800, 'ttt_grace_s': 7200, 'ttt_env': {'ENS_TRAIN_FRAC': '0.6', 'ENS_MIN_DECODE_BATCHES': '1', 'ENS_DECODE_GRACE_S': '300'}}
TTT_SOURCE = '"""ARC-AGI-2 — test-time training solver for 4x L4.\n\nPer task: fit a small low-rank adapter on that task\'s own demonstrations, decode\ncandidate grids across dihedral x colour-permutation views, re-score each unique\ncandidate under fixed augmentations, and submit the top two.\n\nDesign notes that matter offline:\n  * LoRA is implemented here rather than imported, so the run depends only on\n    torch + transformers. A missing package in a no-internet run does not raise\n    where anyone can see it; it degrades the output and reads as a modelling\n    failure.\n  * One worker per GPU pulling from a shared queue, so a slow task delays only\n    its own worker.\n  * Every worker holds a wall-clock budget and degrades in stages: full TTT ->\n    no TTT -> echo the input. Every task id always receives two well-formed\n    grids, so nothing is ever dropped.\n  * The commit run and the scored re-run are the same file against different\n    data. The shipped placeholder is public training tasks, so solving it in\n    full buys nothing and costs a week\'s GPU quota; `is_public_placeholder`\n    identifies it positively and the run probes instead. It must never fire on\n    real test data, so every uncertain case resolves to the full run.\n"""\nimport json, os, sys, time, gc, math, random, subprocess, hashlib\n\nos.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")\nos.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")\n# Must be set before torch initialises CUDA. TTT allocates and frees a lot of\n# short-lived activation memory, and the resulting fragmentation was worth\n# 0.4-0.75 GiB of reserved-but-unallocated space in a measured OOM.\nos.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")\n# torch 2.10 renamed this; the runtime\'s own OOM message asks for the new name,\n# so set BOTH rather than guessing which one the installed version reads.\nos.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")\n\nimport numpy as np\nimport torch\nimport torch.nn as nn\n\nSTART = time.time()\n# Wall budget for the scored run. Kaggle\'s own cap is 12 h; this sits below it\n# so the run finishes inside the UTC day whose 1/day submission slot it is\n# meant for -- a run that lands after the day boundary forfeits that slot.\n# The competition\'s own 12 h cap less a 10-minute flush buffer -- the same\n# budget the public LB33.89 build uses. Safe to take because the SCORED re-run\n# does not bill the weekly GPU quota (measured), while the COMMIT run is capped\n# to PROBE_WALL_S by the placeholder gate below. Runtimes are obfuscated by up\n# to 10 min, so the buffer is the margin, not the schedule.\nWALL = float(os.environ.get("WALL_S", 12 * 3600 - 600))\nDEADLINE = START + WALL\n\n# 16-token grid vocabulary: id == colour value for 0-9\nNL, USER, ASST, PAD, BOS, EOS = 10, 11, 12, 13, 14, 15\nLEGAL = list(range(10)) + [NL, EOS]\nMAX_SIDE = 30\n\n# Number of augmented demonstration sequences the per-task adapter is fitted on.\n#\n# 96 as of 2026-09-20, and this is the UPWARD half of an axis that had only ever\n# been measured downward at this base. E203 read 48 -> 24 as -4.03 on the server\n# at r64/alpha32; nothing had tested above 48 here. The only upward reads (E158,\n# 128 vs 512) were taken at r256+embed capacity, where the fit was already\n# converged at training_loss 0.007 -- and a kill is conditional on the base\n# strength that made it, which E208 has since moved. E200 also shows the\n# candidate pool widening sharply as adapter capacity falls, so an r64 adapter is\n# the case least likely to be saturated. Priced against it: TTT is 20-28% of the\n# run, so doubling views costs ~25% of the budget = ~-0.85 pts at E217\'s measured\n# 0.034 pts per 1% -- the gain has to clear that before it clears zero.\nTTT_VIEWS = int(os.environ.get("TTT_VIEWS", 48))\n# Drop the loss on the first N demo outputs -- unreachable targets. Default 0\n# (ship-identical to every scored submission so far) until measured on THIS\n# pipeline; see build_ttt for the mechanism and E30 for the offline reading.\nTTT_MASK_FIRST = int(os.environ.get("TTT_MASK_FIRST", 0))\n# 0 = a fresh colour permutation per TTT view (every shipped build). K>0 cycles a\n# pool of K palettes instead: colour-perm diversity at fixed view count and cost.\nTTT_COLOR_PERMS = int(os.environ.get("TTT_COLOR_PERMS", 0))\n# 32, reverting E39\'s kill. E38\'s offline view-count curve (swept post-hoc on a\n# fixed TTT state, so quality is isolated from cost) is logarithmic and puts\n# 16->32 at about +2.3 pass@2. E39 shipped that change and LOST 1.25 -- but at\n# the 9.5h wall, where doubling the dominant phase cost ~52% of the per-task\n# budget and coverage was worth ~0.096 pts per 1%, i.e. about -5 pts. That\n# exchange rate is dead: at the 11.83h wall two successive budget levers\n# returned +0.42 and exactly 0.00, so the budget no longer binds and the cost\n# side of E39\'s arithmetic has gone to ~zero.\n#\n# Two further reasons the price is lower than when E39 ran: the decode loop\n# already breaks on the per-task deadline, so extra views are taken only where\n# slack exists and the change degrades gracefully rather than truncating\n# scoring; and cascade scoring makes a wider pool nearly free to rank\n# (2*|pool| + 6*SCORE_VIEWS forwards, so doubling the pool costs ~+17% of\n# scoring instead of +100%).\n#\n# REVERTED 32 -> 16 (E80). The reasoning above was right that the original kill\n# was priced against a constraint that no longer binds, and wrong that this\n# implied a gain. Measured paired on 89 outputs with BOTH arms given a per-task\n# cap generous enough that neither truncates: pool oracle rises 29.21 -> 34.83\n# and the pool grows 51%, but precision FALLS 2.83% -> 2.23% and pass@2 is\n# EXACTLY equal, 1 up and 1 down. That is dilution, and it means the quality\n# delta was zero all along -- a cost-revert cannot rescue a lever that was never\n# worth anything at full price. 16 is the operating point.\nDECODE_VIEWS = int(os.environ.get("DECODE_VIEWS", 16))\nSAMPLES_PER_VIEW = int(os.environ.get("SAMPLES_PER_VIEW", 1))\n# Raised 4 -> 16 on measured evidence (E56). aug_nll is a MEAN over this\n# many augmentation views, and the ranking at the head is decided by\n# differences of ~0.01 nats/token -- inside the standard error of a\n# 4-sample mean. Re-scoring a FIXED pool (generation identical at every\n# point) gives pass@2 23.26 / 24.42 / 25.00 / 25.00 at K=4 / 8 / 16 / 32:\n# +1.74 by K=16, 3 gained and 0 lost, and flat thereafter. On the server that\n# same change was worth +3.75 (27.64 -> 31.39), 2.15x its offline prediction.\n#\n# Raised again 16 -> 32, and the earlier "16 is the knee" reading was\n# POOL-DEPENDENT rather than wrong: K=32 is exactly flat on the 10-candidate\n# greedy pool (25.00 at K=16 and K=32) but worth +1.17 on the 18-candidate DFS\n# pool (26.74 -> 27.91). A wider pool puts more candidates near the head, where\n# the ranking is decided by ~0.01 nats/token, so it needs a better-resolved\n# estimate. The two levers are complementary, not redundant -- and the cached\n# scorer below makes K=32 on the wider pool cheaper than K=16 was uncached.\n# 16. REVERTED from 24 (E85). The K-curve is a property of the POOL, and on the\n# GREEDY pool this build actually generates it is EXACTLY FLAT -- 25.00 at K=16,\n# 24 and 32. The +0.58 that motivated 24 came entirely from the DFS pool, which\n# is not shipped, and end-to-end on the real pipeline 24 measured -2.33 (0 up /\n# 4 down): no gain to collect, and the extra scoring is not free. Kept as the\n# cautionary note below because the earlier reading looked well-evidenced.\n# WAS: 24, on this reasoning --\n# The cross-seed re-scoring runs swept K on two FIXED pools and they agree: the\n# whole gain sits between 16 and 24, and 24 -> 32 is EXACTLY zero on both.\n#   DFS pool     26.74 (K=16)  27.91 (K=24)  27.91 (K=32)\n#   greedy pool  25.00 (K=16)  25.00 (K=24)  25.00 (K=32)\n#   pooled 344   25.87 (K=16)  26.45 (K=24)  26.45 (K=32)\n# So 24 takes the entire available gain at 75% of 32\'s scoring cost. The earlier\n# "-3.2 pts" arithmetic against K=32 was priced with the per-task exchange rate\n# that E76 later showed is dead, and in any case does not apply at K=24.\nSCORE_VIEWS = int(os.environ.get("SCORE_VIEWS", 16))\n# r256 + embed/lm_head is the reference solution\'s own TTT capacity, adopted\n# 2026-09-03 after measuring that the forkable public port scores 32.64-33.89\n# against this build\'s 27.95 draw mean (E100). The three offline A/Bs that had\n# been read as "recipe matched" were each BELOW this instrument\'s own resolvable\n# floor (E101), and two of the three point estimates leaned the reference\'s way,\n# E25\'s r256+embed bundle by +1.74. Cost measured, not assumed: peak VRAM 15.8\n# GiB against 13.3 at r128, on a 22 GiB card, and the KV term is separately\n# capped by DECODE_KV_GB so the peak cannot run away on a long task.\nLORA_R = int(os.environ.get("LORA_R", 64))\n# rsLoRA scales updates by alpha/sqrt(r), so rank and update magnitude move together\n# unless alpha is set deliberately. 22.6274 = 32/sqrt(2) keeps the update scale at\n# 2.828 (the value r=128 has at alpha=32) while holding r=64\'s parameter count.\nLORA_ALPHA = 32.0\nTTT_LR = float(os.environ.get("TTT_LR", 5e-5))\nMAX_LEN = int(os.environ.get("MAX_LEN", 8192))\n# The reference recipe\'s own per-task cap. Raising this cannot overrun the\n# wall: the worker takes min(TASK_CAP, time_left / tasks_remaining), so the\n# adaptive term is what actually binds and it self-rebalances after every\n# task. At the old 9.5h wall that term was ~570s, so TASK_CAP=600 was the\n# binding constraint and the extra wall would have bought nothing.\nTASK_CAP = float(os.environ.get("TASK_CAP", 1200))\nTEMP = float(os.environ.get("TEMP", 0.8))\n# How many view-decodes share one forward pass. Bounded by KV-cache size,\n# not by compute: a 22 GiB L4 holds ~8 long prompts beside a 4B bf16 model.\n# CAP on views per forward, not a fixed size. Decode is ~52% of a REAL task\n# (measured on public eval; the placeholder says 16% because it is 240 memorised\n# training tasks), and a fixed batch of 8 is wrong in both directions at once:\n# the median task\'s prompt is ~1700 tokens, so 16 views fit easily and the\n# decode was paying two sequential passes where one would do -- while an\n# 8192-token task at batch 8 needs ~9 GiB of KV and drove a 21.8 GiB peak on a\n# 22 GiB card. Sizing the batch by the KV the longest prompt actually needs is\n# therefore FASTER on typical tasks and SAFER on the big ones.\nDECODE_BATCH = int(os.environ.get("DECODE_BATCH", 16))\nDECODE_KV_GB = float(os.environ.get("DECODE_KV_GB", 6.0))\nKV_BYTES_PER_TOK = 147456      # recomputed from the real config at worker start\nDECODE_STATS = {"batches": 0, "views": 0}\n# Shared work queue across the per-GPU subprocesses. Empty = static sharding.\n# Threshold-DFS, enabled PER TASK only when this worker has measured slack.\n# E63 killed DFS as a global setting: it is worth +1.16 pass@2 but costs ~24% of\n# a task, and at 0.096 pts per 1% of budget that is a net loss WHEN THE BUDGET\n# BINDS. It does bind on the mean task (E50 gained +2.36 from more wall; E39 lost\n# 1.25 from more decode) -- but unevenly. Tasks are solved longest-first against\n# an adaptive min(TASK_CAP, time_left/tasks_remaining), so cheap tasks finish\n# under budget and bank time, and any wall left at the end is pure waste.\n# So spend the enumeration exactly where the waste is: project the remaining\n# work from THIS worker\'s own measured mean task cost, and turn DFS on only\n# while the projection leaves headroom. With no slack this is byte-identical to\n# greedy, so it cannot lose; with slack it converts idle wall into candidates.\n# MEASURED OFF. The slack gate itself works -- it fired on 67% of tasks -- but\n# the enumeration it buys converts to exactly ZERO pass@2 (41/172 both arms) for\n# +5% wall, which is the central result again: extra candidates do not help when\n# the scorer cannot pick them out. Kept behind a flag, not deleted, because it\n# becomes live the moment a selector exists that does not read model confidence.\nDFS_ADAPTIVE = int(os.environ.get("DFS_ADAPTIVE", 0))\nDFS_ADAPTIVE_BUDGET = float(os.environ.get("DFS_ADAPTIVE_BUDGET", 2.5))\nDFS_SLACK_FRAC = float(os.environ.get("DFS_SLACK_FRAC", 0.35))\nDFS_MIN_TASKS = int(os.environ.get("DFS_MIN_TASKS", 3))\nADAPT_STATS = {"dfs_on": 0, "dfs_off": 0}\n# SIZE-PROPORTIONAL PER-TASK BUDGET. The worker used to hand every task the same\n# slice, min(TASK_CAP, time_left/tasks_remaining) -- but cost is driven by prompt\n# size and spans ~4x across tasks (p90/p10 = 3.9). Measured on the public eval:\n# the p90 task needs 1.67x the uniform slice while the p10 needs 0.43x, 42% of\n# tasks need more than they get, and the cheap half hands back 41% of its\n# allocation unused. So the budget is MISALLOCATED, not merely scarce -- which is\n# the mechanism behind E50\'s +2.36 for raising the cap, and it is free to fix.\n# Clamped to [0.5x, 2x] the uniform slice so a bad weight cannot starve a task.\nBUDGET_BY_SIZE = int(os.environ.get("BUDGET_BY_SIZE", 1))\n# SENTINEL knob: force every task weight to 1.0, which makes the proportional\n# allocator reduce to the uniform rule it replaces. A run with this set MUST\n# score exactly what the uniform arm scores; if it does not, the allocator is\n# changing something other than the split and no result from it is trustworthy.\nBUDGET_FLAT_WEIGHTS = int(os.environ.get("BUDGET_FLAT_WEIGHTS", 0))\nWEIGHTS = []\nQDIR = os.environ.get("ARC_QDIR", "")\nQUEUE_FILE = "_queue.json"\nNWORKERS = int(os.environ.get("ARC_NWORKERS", 1))\n\n\ndef _decode_batch_for(pids):\n    """How many view-decodes share one forward, from the longest prompt\'s KV."""\n    if not pids:\n        return 1\n    # Reserve for prompt AND the tokens we are about to generate: the decode\n    # budget is now a full grid regardless of prompt length, so sizing on the\n    # prompt alone would under-reserve by up to GRID_MAX_TOK per sequence.\n    longest = max(len(p) for p in pids) + GRID_MAX_TOK\n    n = int(DECODE_KV_GB * (2 ** 30) / max(longest * KV_BYTES_PER_TOK, 1))\n    return max(1, min(DECODE_BATCH, n))\n# Try flash-attention if an attached wheel supplies it. Off by default until a\n# scored run logs ATTN: flash_attention_2, since an unresolved offline package\n# is dropped silently and would otherwise read as "the kernel did not help".\nUSE_FLASH = int(os.environ.get("USE_FLASH", 1))\n# Constrained-beam stage, DISABLED by default. It reliably adds the truth to\n# the candidate pool (+5.0 pts of oracle over 3 full-eval runs, never once\n# removing a truth) but at 0.50% precision against the sampled pool\'s 4.35%,\n# so the ranker cannot find what it adds: across every beam budget tried the\n# pool oracle climbs 20.2 -> 25.2% while pass@2 stays flat at ~19.4%. Kept\n# because the oracle gain is real and becomes cashable the moment a selector\n# exists that does not read the model\'s own confidence. Set BEAM_VIEWS>0 then.\nBEAM_VIEWS = int(os.environ.get("BEAM_VIEWS", 0))\nBEAM_WIDTH = int(os.environ.get("BEAM_WIDTH", 6))\nBEAM_OPT_CAP = int(os.environ.get("BEAM_OPT_CAP", 4))\n# Threshold-DFS decoding. Instead of one greedy rollout per view, enumerate\n# every continuation whose cumulative NLL stays within DFS_BUDGET nats of its\n# own view\'s best live branch. 0.0 collapses to greedy exactly.\n#\n# Why this ships now when the same decoder was measured and held back before:\n# the gain was never in doubt (pool oracle 27.91 -> 35.47 at MAINTAINED\n# precision, 3.01% vs 2.77%, which is the enrichment test a generator change\n# has to pass here) -- the COST was. Decoding each view in its own sequential\n# loop cost 1.44x per task, and per-task budget is worth ~0.096 server points\n# per 1%, so it priced out at -2.94. Views now share one batched decode, the\n# same way greedy already did, so the step count is greedy\'s.\n#\n# The threshold is measured, not the reference\'s. Re-scoring the stored pool at\n# every threshold (free -- cum_nll is on every candidate) gives pass@2\n# 17.44 / 23.26 / 26.16 / 26.74 / 26.74 at X = 0 / 0.8 / 1.609 / 2.5 / inf with\n# SCORE_VIEWS=32, so 2.5 is where the curve flattens: the reference\'s own 1.609\n# leaves 1.17 pts on the table and everything past 2.5 is pure cost.\n# MEASURED AND NOT SHIPPED. On a paired full-eval run against the incumbent\n# (arm0 greedy/SV16 vs arm1 DFS2.5/SV32) DFS is worth +1.16 pass@2 -- but costs\n# +46% of DECODE, and decode is ~50% of a real task. The probe\'s "decode is 16%"\n# came from the placeholder, which is 240 MEMORISED training tasks: there the\n# model is near-certain, the tree has no branches (`cap0`, pool=1) and DFS looks\n# free. On real tasks the row cap binds constantly (`cap2682`) and peak memory\n# reaches 26.6 GiB against a 22 GiB card. At 0.096 pts per 1% of budget the\n# enumeration prices at roughly -2.3 for +1.16. Re-enable only with a cheaper\n# operating point (X=1.609 scores the same +1.16 at near-greedy pool size) AND\n# a row cap that fits the card.\nDFS_BUDGET = float(os.environ.get("DFS_BUDGET", 0.0))\nDFS_WIDTH = int(os.environ.get("DFS_WIDTH", 4))\nDFS_MAX_CANDS = int(os.environ.get("DFS_MAX_CANDS", 4))\n# Global cap on simultaneously-live decode rows, i.e. on KV-cache width. Every\n# view is guaranteed its best row before any view gets a second, so this\n# degrades the enumeration under memory pressure and never drops a view.\nDFS_ROWS = int(os.environ.get("DFS_ROWS", 16))\nDFS_STATS = {"outputs": 0, "seqs": 0, "rows_cap": 0}\n# Scoring-cache telemetry. The cached path is an exact-arithmetic speedup, so\n# a silent fall back to the uncached path costs budget without changing any\n# number -- invisible in the score, visible only here.\nSCORE_CACHE = {"cached": 0, "fallback": 0, "snap": 0, "why": "", "err": -1.0}\n# Ablation switch, off by default: forces the uncached scoring path so a\n# control arm can reproduce the pre-cache build\'s COST exactly. The cache is an\n# exact-arithmetic speedup, so this changes timing and nothing else -- which is\n# precisely why the control needs it.\nNO_SCORE_CACHE = int(os.environ.get("NO_SCORE_CACHE", 0))\n# Tolerance the SCORE-CACHE sentinel holds the cached scorer to, in nats/token.\n#\n# This knob is the whole story behind a cost bug that ran unseen for the entire\n# project. The sentinel compares cached against uncached NLL and, above this\n# threshold, replaces the cached scorer with a stub -- correct, and ~2.4x more\n# expensive on the scoring phase. The threshold was hard-coded at 2e-3 and the\n# discrepancy it measures is bf16 rounding between two computation orders, so it\n# is DATA-DEPENDENT: eval shard 0 reads 7.4e-4 and caches, shard 1 reads 2.9e-3\n# and does not, and the Kaggle commit run reads `SC 0/16` -- uncached on the\n# hidden set, i.e. every scored submission has paid the uncached price while\n# roughly half of every offline measurement paid the cached one.\n#\n# 2e-3 was never calibrated against anything. The quantity that matters is the\n# truth-vs-head ranking margin, ~1.16e-2 nats/token (E93/E118), so 2e-3 is\n# ~6x tighter than the decision it protects. Raising it is only worth doing on\n# measured evidence that the SOLVED SET does not move, which is a counting\n# comparison rather than a score delta -- see the sctol_* arms.\n#\n# CALIBRATED 2026-09-20 to 6e-3, on that evidence (E218). Forcing the cache on\n# where the old threshold refused it left the solved set EXACTLY unchanged on\n# eval shard 2 (11/47, zero swaps, at a measured discrepancy of 4.2e-3) and net\n# unchanged on shard 1 (10/42, one output in, one out, at 2.9e-3), while cutting\n# the scoring phase 2.1-2.3x and total wall 7-10%. 6e-3 sits 1.4x above the\n# largest discrepancy ever observed here and 1.9x below the ranking margin it\n# guards, and a genuinely broken cache errs by at least 5x that -- so the\n# sentinel still fires on the failure it exists for.\nSCORE_CACHE_TOL = float(os.environ.get("SCORE_CACHE_TOL", 6e-3))\n# Per-task RNG seed for test-time training and the augmentation views drawn from\n# it. Default 0 reproduces every previous run bit-for-bit; it exists so a second,\n# DECORRELATED build can be produced from one constant.\n#\n# The point is not a better score -- two seeds score the same within noise, and a\n# seed portfolio was measured and rejected as a way to spend the two attempts on\n# ONE output. This is the other problem: the private rank is the best of the <=2\n# submissions nominated at the deadline, which is a `max` over an uncertain task\n# sample rather than a sum over one output. There, E[max]-E[single] = 0.399*sqrt(d/n)\n# in the disagreement fraction d, so a near-equal but decorrelated second nominee\n# is worth ~1 point where two near-clones are worth ~0.2. The self-check probes\n# and the cost-weight model stay pinned at 0: they must not move between builds.\n# Well-SEPARATED rather than consecutive. Small integers shuffle a 2- or\n# 3-element demo list identically (seeds 1 and 2 duly produced the same\n# sequence digest), so successive draws are drawn from spread-out values.\n# PINNED BACK TO 0 for 2026-09-22: the 33.06 anchor (E199) and every fixed-seed\n# paired read this project has taken are at SEED=0, so a build meant to be\n# compared against them must not also re-roll the seed. 57391482 belongs to v41,\n# whose whole purpose was to vary ONLY this constant.\nSEED = int(os.environ.get("SEED", 229565928))\n# ---- multi-adapter ensemble (default 1 = the single-adapter path, untouched) --\n# K > 1 fits K adapters per task, each on its OWN seed\'s views (adapter 0 is\n# exactly the K=1 adapter), pools every adapter\'s decoded candidates, and ranks\n# the union by aug_nll AVERAGED across the adapters. Why this and not more views\n# on one adapter: the seed changes WHICH tasks get solved far more than any\n# single-adapter knob does, and a larger view set on one adapter made the seed\n# divergence worse, not smaller. Separate adapters are the only lever on that\n# axis. Extra adapters are fitted only out of measured slack inside the task\'s\n# own cap, so a tight task degrades to exactly the K=1 answer.\nTTT_ENSEMBLE = int(os.environ.get("TTT_ENSEMBLE", 2))\nENS_SEED_STRIDE = int(os.environ.get("ENS_SEED_STRIDE", 57391482))\n# Fraction of the task cap held back for joint scoring when deciding whether one\n# more adapter still fits.\nENS_SCORE_RESERVE = float(os.environ.get("ENS_SCORE_RESERVE", 0.2))\n# Short-phase starvation guard (3 h TTT phase of the combo build, ~60-350 s per-task caps): measured in a\n# dress rehearsal on the public eval set, a large task\'s adapter-0 fit could use the whole cap, leaving decode\n# no time, an empty pool and an input-echo answer (85 of 172 outputs). ENS_TRAIN_FRAC < 1 stops adapter-0\n# training at that fraction of the cap; ENS_MIN_DECODE_BATCHES lets adapter 0\'s first decode batches of an\n# output whose pool is still empty run past the cap by up to ENS_DECODE_GRACE_S; when no candidate could be\n# scored at all, candidates are ordered by decode votes instead of pool insertion order. The defaults\n# (1.0, 0) reproduce the previous behaviour exactly.\nENS_TRAIN_FRAC = float(os.environ.get("ENS_TRAIN_FRAC", 1.0))\nENS_MIN_DECODE_BATCHES = int(os.environ.get("ENS_MIN_DECODE_BATCHES", 0))\nENS_DECODE_GRACE_S = float(os.environ.get("ENS_DECODE_GRACE_S", 120))\nSTARVE_STATS = {"train_cut": 0, "grace_batches": 0, "vote_order": 0}\n# Candidates kept for the full-view joint pass after the cheap joint pass.\nENS_KEEP = int(os.environ.get("ENS_KEEP", 8))\n# Offline instrument only: score EVERY pooled candidate under EVERY adapter on its\n# full view set (no cascade) and dump the per-candidate matrix, so any subset of\n# adapters -- including each adapter alone -- can be re-ranked post hoc on the\n# identical pool.\nENS_FULLSCORE = int(os.environ.get("ENS_FULLSCORE", 0))\nENS_DUMP = int(os.environ.get("ENS_DUMP", 0))\nENS_STATS = {"tasks": 0, "adapters": 0, "skipped": 0, "joint": 0}\n# How far past its own per-task cap an ensemble task may run, as a multiple of\n# that cap (0 = K). The cap is sized for ONE adapter, so without this a large task\n# can never afford a second one; the global-slack guard below, not this, is what\n# protects later tasks. Adapter 0 and a single-adapter fallback still see only the\n# original cap, so a task with no slack scores exactly as the K=1 build.\nENS_CAP_MULT = float(os.environ.get("ENS_CAP_MULT", 0)) or float(max(TTT_ENSEMBLE, 1))\n# ---- test-time preference optimisation (default OFF = byte-identical) ------\n# Thirteen selector families have failed here and they share one shape: each\n# tried to READ a better signal out of a fixed model, and the model\'s own\n# likelihood is the thing that is wrong at the head of the ranking (missed\n# truths lose by a median 0.0116 nats/token). This is the one intervention that\n# does not route around the defect -- it changes the model so its likelihood\n# discriminates better, by explicitly optimising the margin between the right\n# answer and the wrong answer that currently outranks it.\n#\n# Pairs come from leave-one-out on the DEMONSTRATIONS, never from the test\n# input: same legitimacy as the CE test-time training above.\n#\n# The published attempt at this (a public pilot on this same chassis) fired ZERO\n# gradient steps on 96/96 tasks, because it required a rollout to match the\n# held-out gold EXACTLY before it would count as `chosen`. That fails for a\n# reason already in this project\'s own results: candidates are ~95% cell-correct\n# and rarely exactly right, so after TTT the model either nails the held-out\n# demo in every view (no negative available) or misses it in all of them (no\n# positive available). The fix is to stop requiring exactness -- take the GOLD\n# itself as `chosen`, which always exists, and the highest-likelihood non-gold\n# rollout as `rejected`, which is the hard negative that actually competes.\nTTDPO = int(os.environ.get("TTDPO", 0))\nTTDPO_BETA = float(os.environ.get("TTDPO_BETA", 0.1))\nTTDPO_LR = float(os.environ.get("TTDPO_LR", 5e-6))\nTTDPO_STEPS = int(os.environ.get("TTDPO_STEPS", 2))\nTTDPO_FOLDS = int(os.environ.get("TTDPO_FOLDS", 3))\nTTDPO_ROLLOUTS = int(os.environ.get("TTDPO_ROLLOUTS", 4))\nTTDPO_S = float(os.environ.get("TTDPO_S", 90.0))\nTTDPO_STATS = {"pairs": 0, "steps": 0, "tasks": 0, "fired": 0,\n               "rollback": 0, "refchk": 0, "nopair": 0, "nostep": 0, "nog": 0, "same": 0}\n# TTT sequences at or below this many tokens skip gradient checkpointing.\n# 0 disables the optimisation (checkpoint everything, the pre-2026-08-26\n# behaviour) and is the control arm. See run_ttt for the sizing evidence.\n# 0 = checkpoint everything. Selective checkpointing is mathematically exact\n# (max |dloss| 1.24e-04) and 1.36x faster where it applies, but TTT is only ~21%\n# of a real task, so it buys ~+0.1 pts -- while RAISING peak memory, and the\n# incumbent already peaks at 21.8 GiB on a 22 GiB card. Not worth the OOM.\nCKPT_MAX_TOK = int(os.environ.get("CKPT_MAX_TOK", 0))\nCKPT_STATS = {"ck": 0, "nock": 0, "oom": 0}\n# Cascade scoring: rank the whole pool on SCORE_CHEAP views, then spend the\n# full SCORE_VIEWS budget only on the top SCORE_KEEP. Sized by exact re-scoring\n# of stored per-view vectors -- see score_pool. 0 disables (score everything).\nSCORE_CHEAP = int(os.environ.get("SCORE_CHEAP", 2))\nSCORE_KEEP = int(os.environ.get("SCORE_KEEP", 6))\nSCORE_CASCADE = {"scored": 0, "pruned": 0}\n# Telemetry, not decoration: the beam is additive and fully wrapped, so if the\n# KV-cache reorder API is missing on the runtime\'s transformers version it\n# contributes nothing and the run still scores -- looking exactly like a lever\n# that does not transfer. These counters are how that is told apart.\nBEAM_STATS = {"outputs": 0, "ran": 0, "novel": 0, "skipped": 0}\n# Counts views whose prompt only fit after dropping demonstrations. Silent\n# truncation is exactly the class of fault this run has been bitten by, so\n# it gets a counter rather than being inferred from a score.\nTRUNC_STATS = {"views": 0}\n# Largest grid this competition allows, in tokens: 30x30 cells + 30 row breaks\n# + EOS. The decode budget must never fall below it, or the true grid is\n# unemittable BY CONSTRUCTION and the task scores zero however good the model\n# is. It used to be min(931, MAX_LEN - longest prompt in batch), and prompts\n# are only guaranteed to leave 200, so the budget could bottom out at 200 --\n# enough for a 14x14 grid. Audited (audit_reachability.py): that starved 2 of\n# 172 evaluation outputs and 0 of 1076 training outputs. MAX_LEN is a cap on\n# TTT sequence construction, not a model limit (the base handles far more), so\n# decode is allowed to run past it; the batch sizer below now reserves KV for\n# the generated tokens too, which is strictly more conservative than before.\nGRID_MAX_TOK = 931\nBUDGET_STATS = {"raised": 0}\n# Per-phase wall-clock, so the cost side of any change is MEASURED rather\n# than estimated. The per-task budget is the binding resource here (a 25%\n# rise in it was worth +2.36 server points), so "what fraction of a task is\n# scoring?" decides whether a scoring change is affordable. Guessing it is\n# how a positive offline delta turns into a negative server result.\nPHASE_S = {"ttt": 0.0, "decode": 0.0, "score": 0.0}\nRANK = 0\n# The phase split is the single number the next config decision turns on, and\n# it was UNREADABLE on the last run: a log fetched through a budgeted API cut\n# off at ~170 s while the first per-task progress line lands past 1300 s. So\n# the beacon fires from INSIDE the first task, at each phase boundary, a fixed\n# handful of times. Printing the answer early beats printing it well.\nPHASE_BEACON = {"left": int(os.environ.get("PHASE_BEACONS", 8))}\n\n\ndef _beacon(where, extra=""):\n    if PHASE_BEACON["left"] <= 0:\n        return\n    PHASE_BEACON["left"] -= 1\n    t = sum(PHASE_S.values()) or 1.0\n    try:\n        mem = f" mem {torch.cuda.max_memory_allocated()/2**30:.1f}G"\n    except Exception:\n        mem = ""\n    print(f"[w{RANK}] PHASE@{where} ttt {PHASE_S[\'ttt\']:.0f}s "\n          f"dec {PHASE_S[\'decode\']:.0f}s sco {PHASE_S[\'score\']:.0f}s = "\n          f"{100*PHASE_S[\'ttt\']/t:.0f}/{100*PHASE_S[\'decode\']/t:.0f}/"\n          f"{100*PHASE_S[\'score\']/t:.0f}%{mem} {extra}", flush=True)\n\n# Commit-run gate. The competition ships a PLACEHOLDER test file that is 240\n# public *training* tasks (measured), so a commit run against it scores\n# plumbing and nothing else -- while burning a full-length run of the weekly\n# notebook GPU quota, which funds interactive runs only. On the placeholder\n# solve a probe and let the declared fallback cover the rest; the hidden set\n# always gets the full pipeline. PROBE_TASKS=0 disables the gate outright.\nPROBE_TASKS = int(os.environ.get("PROBE_TASKS", 8))\nPROBE_WALL_S = float(os.environ.get("PROBE_WALL_S", 45 * 60))\nPLACEHOLDER_IDS_SHA256 = (\n    "0425e31f8c028fd6d6c2953fd5dbed115c065b8c657375a57f8b477c6f1d650c")\n\n\n# ------------------------------------------------------------------ io\n# Where to look for the attached competition data and the model. On Kaggle every\n# attached source is mounted under /kaggle/input, which is what these default to,\n# so leaving them unset reproduces the scored run exactly.\n#\n# Offline they are the whole interface. The base checkpoint is 6.8 GB, so a run\n# that STAGES a copy of it to satisfy a hardcoded root pays that per shard, into\n# a run dir on the shared /project quota -- 62 such copies (420 GB) accumulated\n# before this existed. Point these at the real directories instead: no copy, and\n# no rewriting this file\'s source text to redirect it (a replacement keyed on a\n# literal is silently a no-op the moment the literal is reformatted).\n#\n# os.pathsep-separated, so several roots can be searched in order.\nINPUT_ROOTS = tuple(r for r in os.environ.get(\n    "ARC_INPUT_ROOTS", "/kaggle/input").split(os.pathsep) if r)\nMODEL_ROOTS = tuple(r for r in os.environ.get(\n    "ARC_MODEL_ROOTS", "/kaggle/input").split(os.pathsep) if r)\n\n\ndef find_file(name, roots=None):\n    for root in (INPUT_ROOTS + (".",) if roots is None else roots):\n        if not os.path.isdir(root):\n            continue\n        for dp, dn, fn in os.walk(root):\n            if name in fn:\n                return os.path.join(dp, name)\n    return None\n\n\ndef is_public_placeholder(challenges, ch_path):\n    """Is this challenge file the shipped placeholder, not a real test set?\n\n    Returns (verdict, reason). Two INDEPENDENT positive tests, either of which\n    identifies the placeholder exactly. Everything else -- no match, a missing\n    file, any exception -- returns False and the run does the full job, because\n    the two errors are not symmetric: a false negative costs quota, a false\n    positive ships a void submission. Never infer from the task count.\n    """\n    ids = sorted(challenges)\n    try:\n        if hashlib.sha256("\\n".join(ids).encode()).hexdigest() == \\\n                PLACEHOLDER_IDS_SHA256:\n            return True, f"id digest matches the shipped placeholder, {len(ids)} tasks"\n    except Exception as e:\n        print(f"gate: digest check errored: {type(e).__name__}: {e}", flush=True)\n    try:\n        tp = None\n        if ch_path:\n            cand = os.path.join(os.path.dirname(ch_path),\n                                "arc-agi_training_challenges.json")\n            tp = cand if os.path.exists(cand) else None\n        tp = tp or find_file("arc-agi_training_challenges.json")\n        if not tp:\n            return False, "no digest match and no public training file to compare against"\n        train = set(json.load(open(tp)))\n        if not train:\n            return False, "public training file present but empty"\n        inside = len(set(ids) & train)\n        frac = inside / max(len(ids), 1)\n        # >=90%, not 100%. The two errors are wildly asymmetric and this\n        # threshold moves the safety margin to the side that matters: a\n        # commit run the gate MISSES now costs a full-length run of quota\n        # (the wall is the competition\'s 12h, not a self-imposed 9.5h),\n        # while a false positive needs 90% of the 240 HIDDEN ids to also be\n        # public training ids -- the measured overlap there is zero, so the\n        # margin is enormous. Ids are unique 8-hex task names, not counts.\n        if frac >= 0.9:\n            return True, (f"{inside}/{len(ids)} ids are public training tasks "\n                          f"({frac:.0%}) ({tp})")\n        return False, (f"only {inside}/{len(ids)} ids ({frac:.0%}) lie inside "\n                       f"the public training set")\n    except Exception as e:\n        return False, f"gate errored ({type(e).__name__}: {e}), assuming real test data"\n\n\ndef is_grid(g):\n    if not isinstance(g, list) or not g or len(g) > MAX_SIDE:\n        return False\n    if not all(isinstance(r, list) for r in g):\n        return False\n    w = len(g[0])\n    if not (1 <= w <= MAX_SIDE):\n        return False\n    return all(len(r) == w and all(isinstance(c, int) and 0 <= c <= 9 for c in r)\n               for r in g)\n\n\ndef gstr(g):\n    return "\\n".join("".join(str(int(c)) for c in row) for row in g)\n\n\ndef enc(text):\n    ids, i = [], 0\n    sp = {"<|im_start|>": BOS, "<|im_end|>": EOS, "user": USER, "assistant": ASST}\n    while i < len(text):\n        for s, t in sp.items():\n            if text.startswith(s, i):\n                ids.append(t); i += len(s); break\n        else:\n            ch = text[i]\n            ids.append(NL if ch == "\\n" else int(ch))\n            i += 1\n    return ids\n\n\ndef pair_text(gi, go):\n    return ("<|im_start|>user\\n" + gstr(gi) + "<|im_end|>"\n            "<|im_start|>assistant\\n" + gstr(go) + "<|im_end|>")\n\n\ndef query_text(gi):\n    return "<|im_start|>user\\n" + gstr(gi) + "<|im_end|><|im_start|>assistant\\n"\n\n\ndef decode_grid(ids):\n    rows, cur = [], []\n    for t in ids:\n        if t == EOS:\n            break\n        if t == NL:\n            rows.append(cur); cur = []\n        elif t < 10:\n            cur.append(int(t))\n        if len(rows) > MAX_SIDE:\n            return None\n    if cur:\n        rows.append(cur)\n    rows = [r for r in rows if r]\n    if not rows:\n        return None\n    w = len(rows[0])\n    if not (1 <= w <= MAX_SIDE) or len(rows) > MAX_SIDE:\n        return None\n    if any(len(r) != w for r in rows):\n        return None\n    return rows\n\n\n# ------------------------------------------------------------------ augmentation\nD4 = [lambda a: a, lambda a: np.rot90(a, 1), lambda a: np.rot90(a, 2),\n      lambda a: np.rot90(a, 3), lambda a: a.T, lambda a: np.rot90(a.T, 1),\n      lambda a: np.rot90(a.T, 2), lambda a: np.rot90(a.T, 3)]\nD4_INV = [lambda a: a, lambda a: np.rot90(a, -1), lambda a: np.rot90(a, -2),\n          lambda a: np.rot90(a, -3), lambda a: a.T,\n          lambda a: np.rot90(a, -1).T, lambda a: np.rot90(a, -2).T,\n          lambda a: np.rot90(a, -3).T]\n\n\ndef apply_aug(grid, d, perm):\n    return np.asarray(perm)[D4[d](np.asarray(grid, dtype=int))].tolist()\n\n\ndef invert_aug(grid, d, perm):\n    inv = np.argsort(np.asarray(perm))\n    return np.asarray(D4_INV[d](inv[np.asarray(grid, dtype=int)])).tolist()\n\n\ndef make_views(n, rng, k_perms=0):\n    """n augmentation views: (dihedral index, colour permutation).\n\n    k_perms=0 (the default and every shipped build) draws a FRESH random colour\n    permutation per view, so 48 TTT views span 47 distinct palettes.\n\n    k_perms>0 instead cycles a pool of K palettes, holding the dihedral cycle\n    unchanged. The point is colour-perm DIVERSITY at a fixed view count and a\n    fixed cost. It became a live question only once E126 made embed_tokens and\n    lm_head adaptable: those are exactly the colour-token layers, so 47 random\n    palettes may now be asking the adapter to average over variation it could\n    instead specialise to this task\'s actual palette.\n    """\n    views = [(0, list(range(10)))]\n    if k_perms > 0:\n        pool = [list(rng.permutation(10)) for _ in range(k_perms)]\n        while len(views) < n:\n            views.append((len(views) % 8, list(pool[(len(views) - 1) % k_perms])))\n    else:\n        while len(views) < n:\n            views.append((len(views) % 8, list(rng.permutation(10))))\n    return views[:n]\n\n\n# ------------------------------------------------------------------ LoRA\nclass LoRALinear(nn.Module):\n    """y = base(x) + (alpha/sqrt(r)) * B(A(x)); only A,B train."""\n\n    def __init__(self, base, r, alpha):\n        super().__init__()\n        self.base = base\n        for p in self.base.parameters():\n            p.requires_grad_(False)\n        d_in, d_out = base.in_features, base.out_features\n        dt = base.weight.dtype\n        # Bind the adapter to ITS OWN base layer\'s device. Injection happens\n        # after the model has been moved, so omitting device= leaves A and B on\n        # CPU and every forward dies with a mat2-on-cpu RuntimeError -- which,\n        # caught per task, reads as 240 fallbacks rather than as a crash.\n        dv = base.weight.device\n        self.A = nn.Parameter(torch.zeros(r, d_in, dtype=dt, device=dv))\n        self.B = nn.Parameter(torch.zeros(d_out, r, dtype=dt, device=dv))\n        nn.init.normal_(self.A, std=1.0 / r)\n        self.scale = alpha / math.sqrt(r)\n\n    def forward(self, x):\n        return self.base(x) + torch.nn.functional.linear(\n            torch.nn.functional.linear(x, self.A), self.B) * self.scale\n\n    def reset(self):\n        with torch.no_grad():\n            nn.init.normal_(self.A, std=1.0 / self.A.shape[0])\n            self.B.zero_()\n\n\nclass LoRAEmbedding(nn.Module):\n    """LoRA over an nn.Embedding, in LoRALinear\'s exact (A, B, scale) convention.\n\n    The delta is the same low-rank product, with the one-hot token id standing in\n    for the dense input: A is (r, num_embeddings), B is (d_emb, r), and the lookup\n    F.embedding(x, A.T) IS the first factor applied to that one-hot. Exposing .A,\n    .B and .reset() with the same shapes is what lets every caller -- the optimizer\n    parameter sweep, the per-task reset, the DPO snapshot -- stay untouched.\n    """\n\n    def __init__(self, base, r, alpha):\n        super().__init__()\n        self.base = base\n        for p in self.base.parameters():\n            p.requires_grad_(False)\n        n_emb, d_emb = base.num_embeddings, base.embedding_dim\n        dt, dv = base.weight.dtype, base.weight.device\n        self.A = nn.Parameter(torch.zeros(r, n_emb, dtype=dt, device=dv))\n        self.B = nn.Parameter(torch.zeros(d_emb, r, dtype=dt, device=dv))\n        nn.init.normal_(self.A, std=1.0 / r)\n        self.scale = alpha / math.sqrt(r)\n\n    def forward(self, x):\n        delta = torch.nn.functional.embedding(x, self.A.transpose(0, 1))\n        return self.base(x) + torch.nn.functional.linear(delta, self.B) * self.scale\n\n    def reset(self):\n        with torch.no_grad():\n            nn.init.normal_(self.A, std=1.0 / self.A.shape[0])\n            self.B.zero_()\n\n\nTARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",\n           "gate_proj", "up_proj", "down_proj")\n# The reference solution adapts the token embedding and the output head as well\n# as the projections. On a 15-token cut vocabulary those two matrices are the\n# per-colour input and output distribution -- the part of the model a per-task\n# rule most directly re-specifies -- and they are nearly free: the adapters are\n# (r x 15) and (15 x r), against (r x 2560) for a single projection. Default OFF\n# so the shipped behaviour is unchanged until a server read says otherwise.\nEMBED_TARGETS = ("embed_tokens", "lm_head")\nLORA_EMBED = int(os.environ.get("LORA_EMBED", 0))  # projections only, as the 32.22 anchor\n# Global pre-adaptation across ALL tasks\' demonstrations, before any per-task\n# TTT. The pipeline otherwise resets the adapter to noise before every task, so\n# nothing at all is carried across the 240 tasks of a run. E109 is the reason to\n# test the opposite: on ARC-AGI-3 the same frontier model read 62.7% under a\n# stateless harness and 99.9% under one that reused prior work across problems.\n# Reads only the demo pairs already shipped in the challenges file -- the same\n# data per-task TTT consumes -- and never a hidden test output.\n# 0 = off, which is the behaviour every submission through v27 shipped.\nGLOBAL_VIEWS = int(os.environ.get("GLOBAL_VIEWS", 0))\nGLOBAL_LR = float(os.environ.get("GLOBAL_LR", 5e-5))\nGLOBAL_CAP_S = int(os.environ.get("GLOBAL_CAP_S", 2400))\n# MERGE the global delta into the frozen base instead of leaving it in the\n# adapter. Measured necessity, not a preference: with the delta left in A/B the\n# per-task TTT converges straight past it and the candidate pool came back\n# unchanged (pool 11.64 -> 11.76, oracle +0.00). Merging is the only way an\n# across-task prior survives per-task adaptation.\nGLOBAL_MERGE = int(os.environ.get("GLOBAL_MERGE", 1))\n\n\ndef inject_lora(model, r, alpha, embed=None):\n    embed = LORA_EMBED if embed is None else embed\n    mods, kinds = [], {"proj": 0, "embed": 0, "head": 0}\n    for name, mod in list(model.named_modules()):\n        for child_name, child in list(mod.named_children()):\n            if child_name in TARGETS and isinstance(child, nn.Linear):\n                setattr(mod, child_name, lin := LoRALinear(child, r, alpha))\n                mods.append(lin)\n                kinds["proj"] += 1\n            elif embed and child_name in EMBED_TARGETS:\n                if isinstance(child, nn.Embedding):\n                    setattr(mod, child_name, w := LoRAEmbedding(child, r, alpha))\n                    mods.append(w)\n                    kinds["embed"] += 1\n                elif isinstance(child, nn.Linear):\n                    setattr(mod, child_name, w := LoRALinear(child, r, alpha))\n                    mods.append(w)\n                    kinds["head"] += 1\n    # Its own artifact-side counter, because a target list that silently matches\n    # nothing is exactly the failure this build has been bitten by before: the\n    # run would look healthy and ship the old configuration.\n    print(f"LORA-TARGETS: r={r} alpha={alpha} embed={embed} -> "\n          f"{kinds[\'proj\']} proj + {kinds[\'embed\']} embed + {kinds[\'head\']} head "\n          f"= {len(mods)} adapted", flush=True)\n    if embed and (kinds["embed"] + kinds["head"]) == 0:\n        print("LORA-TARGETS: WARNING requested embed/head adaptation matched "\n              "NOTHING -- shipping projections only", flush=True)\n    return mods\n\n\n# The per-task init. Without global pre-adaptation this is a fresh random\n# adapter, exactly as before; with it, every task starts from the pre-adapted\n# weights instead. Kept as one function so the two paths cannot drift.\n_RESTORE = None\n\n\ndef merge_adapters(lmods):\n    """Fold the current adapter delta INTO the frozen base, then zero it.\n\n    Why this exists. Global pre-adaptation writes its result into the same A/B\n    the per-task TTT then trains, and per-task TTT converges hard -- loss ~0.02\n    against the global phase\'s ~0.08 -- so it simply overwrites the prior.\n    Measured: with a plain restore-point the candidate pool and its oracle came\n    back UNCHANGED (11.64 -> 11.76 mean pool, oracle +0.00), i.e. generation\n    never moved. Merging puts the global delta somewhere per-task TTT cannot\n    reach, which is the only way an across-task prior can survive.\n\n    Linear:    W += scale * (B @ A)            [A (r,d_in), B (d_out,r)]\n    Embedding: W += scale * (A.T @ B.T)        [A (r,n_emb), B (d_emb,r)]\n    """\n    n = 0\n    with torch.no_grad():\n        for m in lmods:\n            w = m.base.weight\n            if isinstance(m.base, nn.Embedding):\n                delta = (m.A.transpose(0, 1) @ m.B.transpose(0, 1))\n            else:\n                delta = (m.B @ m.A)\n            w.add_((m.scale * delta).to(w.dtype))\n            m.A.normal_(std=1.0 / m.A.shape[0])\n            m.B.zero_()\n            n += 1\n    return n\n\n\ndef set_restore_point(lmods):\n    global _RESTORE\n    _RESTORE = [(m.A.detach().clone(), m.B.detach().clone()) for m in lmods]\n\n\ndef reset_adapters(lmods):\n    if _RESTORE is None:\n        for m in lmods:\n            m.reset()\n        return\n    with torch.no_grad():\n        for m, (a, b) in zip(lmods, _RESTORE):\n            m.A.copy_(a)\n            m.B.copy_(b)\n\n\n# ------------------------------------------------------------------ TTT\ndef build_ttt(task, views, rng, max_len):\n    """One TTT sequence per view: the whole augmented task, demos shuffled.\n\n    TTT_MASK_FIRST drops the loss on the first N demonstration outputs. Those\n    targets are UNREACHABLE: under a causal mask the first pair\'s output must be\n    predicted from its own input alone, before any example of the rule has been\n    seen. No capacity makes that derivable, so its gradient can only teach\n    guessing -- while still lowering training loss, which is what makes it\n    invisible. Later pairs are reachable because the preceding pairs define the\n    rule.\n\n    Measured at mask1 - mask0 = +2.33 pts (E30), which is ~4x the 1-output\n    run-to-run noise floor, and non-monotone in the way the mechanism predicts:\n    mask2 is WORSE than mask1 because at a median of 3 demos it also deletes a\n    reachable target. Never actually shipped -- the parameter existed only in\n    the offline harness -- so every scored submission so far has run mask0.\n    """\n    seqs = []\n    for d, perm in views:\n        demos = [(apply_aug(p["input"], d, perm), apply_aug(p["output"], d, perm))\n                 for p in task["train"]]\n        rng.shuffle(demos)\n        while demos:\n            # never mask everything: keep at least one supervised pair\n            n_mask = min(TTT_MASK_FIRST, max(len(demos) - 1, 0))\n            ids, lab = [], []\n            for k, (gi, go) in enumerate(demos):\n                seg = enc(pair_text(gi, go))\n                a = seg.index(ASST)\n                l = [-100] * len(seg)\n                if k >= n_mask:\n                    for j in range(a + 2, len(seg)):\n                        l[j] = seg[j]\n                ids += seg; lab += l\n            if len(ids) <= max_len:\n                if any(x != -100 for x in lab):\n                    seqs.append((ids, lab))\n                break\n            demos = demos[1:]\n    return seqs\n\n\ndef run_ttt(model, lmods, seqs, dev, deadline):\n    """Per-task LoRA test-time training.\n\n    Gradient checkpointing is applied PER SEQUENCE rather than to the whole\n    loop. It is required only for the long tail -- a backward pass over an\n    8k-token sequence on a 4B model otherwise stores every layer\'s activations\n    and OOMs a 22 GiB card -- but it was being paid on every sequence, and it\n    costs a whole extra forward (fwd + recomputed fwd + bwd = 4 units against\n    3), i.e. ~25% of TTT.\n\n    That is worth attacking because TTT is 68% of a task on the scored\n    hardware (measured) while peak memory there is 10.3 of 22 GiB, and per-task\n    budget converts to score at ~0.096 points per 1%. Sequence lengths over the\n    eval set are median 1692 / p75 2538 / p90 3330 tokens, so a 2560-token\n    threshold leaves checkpointing on for the ~24% that actually need it.\n\n    This is a pure COST change: checkpointing is mathematically exact, so the\n    gradients, the optimizer trajectory and the resulting adapter are identical\n    either way. An OOM latches checkpointing back on for the rest of the task\n    and retries the sequence, so the worst case is the old behaviour.\n\n    enable_input_require_grads is what makes checkpointed blocks produce\n    gradients at all when only injected adapters are trainable.\n    """\n    params = [p for m in lmods for p in (m.A, m.B)]\n    opt = torch.optim.AdamW(params, lr=TTT_LR, weight_decay=0.0)\n    n = len(seqs)\n    sched = torch.optim.lr_scheduler.OneCycleLR(\n        opt, max_lr=TTT_LR, total_steps=max(n, 1), pct_start=0.1,\n        anneal_strategy="cos", cycle_momentum=False)\n    if hasattr(model, "enable_input_require_grads"):\n        model.enable_input_require_grads()\n    model.config.use_cache = False\n    model.train()\n    state = {"ck": None, "latched": False}\n\n    def _set_ck(on):\n        if state["ck"] == on:\n            return\n        if on:\n            model.gradient_checkpointing_enable(\n                gradient_checkpointing_kwargs={"use_reentrant": False})\n        else:\n            model.gradient_checkpointing_disable()\n        state["ck"] = on\n\n    def _step(ids, lab):\n        x = torch.tensor([ids], device=dev)\n        y = torch.tensor([lab], device=dev)\n        out = model(x, labels=y)\n        out.loss.backward()\n        torch.nn.utils.clip_grad_norm_(params, 1.0)\n        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)\n        v = float(out.loss.detach())\n        del out\n        return v\n\n    tot, done = 0.0, 0\n    try:\n        for ids, lab in seqs:\n            if time.time() > deadline:\n                break\n            want = state["latched"] or len(ids) > CKPT_MAX_TOK\n            _set_ck(want)\n            try:\n                tot += _step(ids, lab)\n                CKPT_STATS["ck" if want else "nock"] += 1\n            except Exception as e:\n                if "out of memory" not in str(e).lower():\n                    raise\n                # Latch, don\'t just retry: once one sequence overflows, the\n                # rest of this task is likely to as well, and thrashing the\n                # allocator costs more than the checkpointing does.\n                CKPT_STATS["oom"] += 1\n                state["latched"] = True\n                # Latch GLOBALLY after the second OOM. The threshold is sized\n                # from a measured memory fit, so repeated overflow means the\n                # fit is wrong on this hardware and every later task would pay\n                # the same failed forward before recovering.\n                if CKPT_STATS["oom"] >= 2:\n                    globals()["CKPT_MAX_TOK"] = 0\n                opt.zero_grad(set_to_none=True)\n                gc.collect(); torch.cuda.empty_cache()\n                _set_ck(True)\n                tot += _step(ids, lab)\n                CKPT_STATS["ck"] += 1\n            done += 1\n    finally:\n        _set_ck(True)\n        model.gradient_checkpointing_disable()\n        state["ck"] = None\n        model.config.use_cache = True\n        model.eval()\n        opt.zero_grad(set_to_none=True)\n        del opt\n    return tot / max(done, 1)\n\n\n# ------------------------------------------------------------------ decode\n@torch.no_grad()\ndef rollout_batch(model, pids, max_new, dev, deadline):\n    """Greedy-decode several view-prompts AT ONCE.\n\n    Decode diversity comes from views (distinct augmentations = distinct\n    conditioning), which is the one source of rule-level variation that has\n    measurably paid: within a single run, 3 -> 16 views is worth +6.4 pts with\n    11 outputs gained and 0 lost. But run one at a time each view is a batch-1\n    autoregressive decode, which leaves the GPU almost entirely idle -- so view\n    count was being limited by wall-clock rather than by whether it helps.\n    Batching the views turns that into roughly one decode\'s cost for all of\n    them, and buys back the budget to run more.\n\n    Prompts are LEFT-padded so every sequence\'s last token sits at the same\n    index; position_ids are derived from the mask, since a raw forward (unlike\n    generate()) will otherwise number the pad tokens as real positions.\n    Returns [(tokens, cum_nll), ...] aligned with `pids`.\n    """\n    n = len(pids)\n    if not n:\n        return []\n    L = max(len(p) for p in pids)\n    inp = torch.full((n, L), PAD, dtype=torch.long, device=dev)\n    att = torch.zeros((n, L), dtype=torch.long, device=dev)\n    for i, p in enumerate(pids):\n        inp[i, L - len(p):] = torch.tensor(p, device=dev)\n        att[i, L - len(p):] = 1\n    pos = (att.cumsum(-1) - 1).clamp(min=0)\n\n    out = model(inp, attention_mask=att, position_ids=pos, use_cache=True)\n    past, logits = out.past_key_values, out.logits[:, -1]\n    bad = torch.full((logits.shape[-1],), float("-inf"), device=dev)\n    bad[LEGAL] = 0.0\n\n    toks = [[] for _ in range(n)]\n    cum = torch.zeros(n, device=dev)\n    live = torch.ones(n, dtype=torch.bool, device=dev)\n    for _ in range(max_new):\n        lp = torch.log_softmax(logits.float() + bad, dim=-1)\n        t = torch.argmax(lp, dim=-1)\n        cum = cum - lp.gather(1, t[:, None]).squeeze(1) * live\n        for i in range(n):\n            if live[i]:\n                toks[i].append(int(t[i]))\n        live = live & (t != EOS)\n        if not bool(live.any()) or time.time() > deadline:\n            break\n        att = torch.cat([att, torch.ones((n, 1), dtype=torch.long, device=dev)], 1)\n        pos = pos[:, -1:] + 1\n        so = model(t[:, None], attention_mask=att, position_ids=pos,\n                   past_key_values=past, use_cache=True)\n        past, logits = so.past_key_values, so.logits[:, -1]\n    return [(toks[i], float(cum[i])) for i in range(n)]\n\n\n@torch.no_grad()\ndef dfs_decode_batch(model, pids, max_new, dev, deadline, budget, width,\n                     max_cands, max_rows):\n    """Threshold-DFS over every view\'s token tree AT ONCE.\n\n    Sampling redraws the same near-certain sequence over and over -- which is\n    why 4x the decode compute moved pass@2 by 0.00 -- but ENUMERATION returns\n    distinct high-mass sequences by construction, so it can enrich the pool\n    where sampling cannot. Each view is thresholded against ITS OWN best live\n    branch: a view the model is confident about stays one row, an ambiguous one\n    opens up, and no single ambiguous view can starve the others out of the row\n    budget.\n\n    Reduces to `rollout_batch` exactly at budget=0, width=1, max_cands=1 --\n    asserted at worker start, because a decoder that silently stopped\n    enumerating would read as a modelling result rather than a bug.\n\n    Returns [[(tokens, cum_nll), ...], ...] aligned with `pids`, best-first.\n    """\n    n = len(pids)\n    if not n:\n        return []\n    L = max(len(p) for p in pids)\n    inp = torch.full((n, L), PAD, dtype=torch.long, device=dev)\n    att = torch.zeros((n, L), dtype=torch.long, device=dev)\n    for i, p in enumerate(pids):\n        inp[i, L - len(p):] = torch.tensor(p, device=dev)\n        att[i, L - len(p):] = 1\n    pos = (att.cumsum(-1) - 1).clamp(min=0)\n\n    out = model(inp, attention_mask=att, position_ids=pos, use_cache=True)\n    past, logits = out.past_key_values, out.logits[:, -1]\n    bad = torch.full((logits.shape[-1],), float("-inf"), device=dev)\n    bad[LEGAL] = 0.0\n    legal = torch.tensor(LEGAL, device=dev, dtype=torch.long)\n\n    owner = list(range(n))\n    seqs = [[] for _ in range(n)]\n    cum = [0.0] * n\n    pos = pos[:, -1:]\n    finished = [[] for _ in range(n)]\n\n    for _ in range(max_new):\n        lp = torch.log_softmax(logits.float() + bad, dim=-1)[:, legal]\n        K = lp.shape[1]\n        flat = (torch.tensor(cum, device=dev)[:, None] - lp).reshape(-1).tolist()\n\n        by_view = {}\n        for r, v in enumerate(owner):\n            by_view.setdefault(v, []).append(r)\n        sel, ranked = [], []\n        for v, rows in by_view.items():\n            if len(finished[v]) >= max_cands:\n                continue\n            opts = sorted((flat[r * K + k], r, k) for r in rows for k in range(K))\n            keep = [o for o in opts if o[0] <= opts[0][0] + budget][:width] or opts[:1]\n            sel.append(keep[0])\n            ranked.extend(keep[1:])\n        ranked.sort()\n        room = max(0, max_rows - len(sel))\n        if len(ranked) > room:\n            DFS_STATS["rows_cap"] += 1\n        sel.extend(ranked[:room])\n        if not sel:\n            break\n\n        parent, n_owner, n_seqs, n_cum = [], [], [], []\n        for score, r, k in sel:\n            v, tok = owner[r], LEGAL[k]\n            if tok == EOS:\n                # Keep the trailing EOS. `rollout_batch` appends it before it\n                # marks the row dead, so dropping it here would make the two\n                # decoders differ by one token on exactly those views that\n                # FINISH -- which is why DFS-EQUIV passed on a worker whose\n                # task ran out of token budget and failed on one whose task\n                # completed. The grids decode the same either way; the point is\n                # that the equivalence check must compare like with like.\n                finished[v].append((seqs[r] + [EOS], score))\n                continue\n            parent.append(r)\n            n_owner.append(v)\n            n_seqs.append(seqs[r] + [tok])\n            n_cum.append(score)\n        if not parent or time.time() > deadline:\n            owner, seqs, cum = n_owner, n_seqs, n_cum\n            break\n\n        idx = torch.tensor(parent, device=dev, dtype=torch.long)\n        if not _reorder_cache(past, idx):\n            raise RuntimeError("KV cache reorder unsupported")\n        att = torch.cat([att.index_select(0, idx),\n                         torch.ones((len(parent), 1), dtype=torch.long, device=dev)], 1)\n        pos = pos.index_select(0, idx) + 1\n        nxt = torch.tensor([[s[-1]] for s in n_seqs], device=dev, dtype=torch.long)\n        so = model(nxt, attention_mask=att, position_ids=pos,\n                   past_key_values=past, use_cache=True)\n        past, logits = so.past_key_values, so.logits[:, -1]\n        owner, seqs, cum = n_owner, n_seqs, n_cum\n\n    # A view with nothing complete hands back its best partial row -- exactly\n    # what the greedy decoder this replaces would have returned, so a deadline\n    # hit degrades to the incumbent instead of to an empty pool.\n    best_live = {}\n    for r, v in enumerate(owner):\n        if v not in best_live or cum[r] < best_live[v][1]:\n            best_live[v] = (list(seqs[r]), cum[r])\n    res = []\n    for v in range(n):\n        f = sorted(finished[v], key=lambda z: z[1])[:max_cands]\n        if not f and v in best_live:\n            f = [best_live[v]]\n        DFS_STATS["seqs"] += len(f)\n        res.append(f)\n    return res\n\n\ndef _cache_len(past):\n    g = getattr(past, "get_seq_length", None)\n    if g is None:\n        return None\n    try:\n        return int(g())\n    except Exception:\n        return None\n\n\ndef _crop_cache(past, n):\n    """Truncate a KV cache to its first `n` positions, in place.\n\n    The cache class\'s OWN `crop` is preferred and the resulting length is always\n    verified: a cache reporting the wrong length mis-numbers positions, which\n    corrupts scores silently instead of raising. False means "use the uncached\n    path", never "carry on and hope".\n    """\n    fn = getattr(past, "crop", None)\n    if fn is None:\n        SCORE_CACHE["why"] = SCORE_CACHE["why"] or (\n            "no .crop on %s" % type(past).__name__)\n        return False\n    try:\n        fn(n)\n    except Exception as e:\n        SCORE_CACHE["why"] = SCORE_CACHE["why"] or (\n            "crop raised %s" % type(e).__name__)\n        return False\n    got = _cache_len(past)\n    if got is None or got == n:\n        return True\n    SCORE_CACHE["why"] = SCORE_CACHE["why"] or ("crop(%d) left len %d" % (n, got))\n    return False\n\n\ndef _snapshot_cache(past):\n    """Reference each layer\'s prompt K/V tensors, or None if the layout is new.\n\n    Why this exists, and it is a MEASURED cost bug rather than defensive\n    programming. `crop` is the only rewind the scorer had, and on the shipped\n    Kaggle runtime it refuses: the commit log reads `SC 0/16` (cached/fallback)\n    while the offline harness on transformers 4.57.6 reads `SC 36/0`. So every\n    scored submission has paid the UNCACHED price -- an O(prompt+grid) forward\n    per candidate per view instead of O(grid) -- while every offline cost\n    measurement paid the cached one. At a 1500-7000 token prompt against a\n    ~110-token grid that is ~5x on the scoring phase, and scoring is ~21-26% of\n    a real task, so it is worth ~17% of the per-task budget on the one side of\n    the pipeline where the budget actually binds (E212).\n\n    Rewinding by rebinding is exact and version-agnostic: every cache\n    implementation here APPENDS with `torch.cat`, which allocates a new tensor\n    and leaves the snapshot untouched, so putting the originals back restores\n    the prompt state bit-for-bit. It is still verified by length before use.\n    """\n    layers = getattr(past, "layers", None)          # transformers 5.x\n    if layers is not None:\n        try:\n            snap = [(l.keys, l.values) for l in layers]\n        except Exception:\n            return None\n        return ("layers", snap)\n    kc = getattr(past, "key_cache", None)            # transformers 4.x\n    vc = getattr(past, "value_cache", None)\n    if kc is not None and vc is not None and len(kc) == len(vc):\n        return ("lists", [(kc[i], vc[i]) for i in range(len(kc))])\n    return None\n\n\ndef _restore_cache(past, snap, n):\n    kind, items = snap\n    try:\n        if kind == "layers":\n            for l, (k, v) in zip(past.layers, items):\n                l.keys, l.values = k, v\n        else:\n            for i, (k, v) in enumerate(items):\n                past.key_cache[i], past.value_cache[i] = k, v\n        if hasattr(past, "_seen_tokens"):\n            past._seen_tokens = n\n    except Exception as e:\n        SCORE_CACHE["why"] = SCORE_CACHE["why"] or (\n            "restore raised %s" % type(e).__name__)\n        return False\n    got = _cache_len(past)\n    if got is None or got == n:\n        return True\n    SCORE_CACHE["why"] = SCORE_CACHE["why"] or ("restore left len %d" % got)\n    return False\n\n\ndef _make_rewind(past, n):\n    """A verified callable that rewinds `past` to `n` positions, or None.\n\n    Tried in order: the cache\'s own `crop`, then snapshot/restore. The first\n    call happens while the cache is still exactly `n` long, so it is a no-op\n    that costs nothing and proves the mechanism before any candidate is scored.\n    """\n    if past is None:\n        SCORE_CACHE["why"] = SCORE_CACHE["why"] or "past_key_values is None"\n        return None\n    if _crop_cache(past, n):\n        return lambda: _crop_cache(past, n)\n    snap = _snapshot_cache(past)\n    if snap is None:\n        SCORE_CACHE["why"] = SCORE_CACHE["why"] or (\n            "unknown cache layout %s" % type(past).__name__)\n        return None\n    if not _restore_cache(past, snap, n):\n        return None\n    SCORE_CACHE["snap"] += 1\n    return lambda: _restore_cache(past, snap, n)\n\n\n@torch.no_grad()\ndef _nll_cached(model, prompt_ids, resp_list, dev):\n    """Mean per-token NLL of each response under ONE shared prompt.\n\n    Every candidate for a given view conditions on a BYTE-IDENTICAL prompt --\n    the prompt is a function of (task, view) alone and only the target grid\n    varies -- so the uncached scorer was re-running the prefill once per\n    candidate. Prefilling once and rewinding the cache turns an O(P+m) forward\n    into an O(m) one; at a typical 1500-token prompt and 110-token grid that is\n    an order of magnitude off the scoring phase, which is what makes\n    SCORE_VIEWS -- the only selection lever that has ever paid here -- cheap\n    enough to raise.\n\n    Returns None if the runtime\'s cache class cannot be rewound.\n    """\n    if NO_SCORE_CACHE:      # ablation switch: reproduces the pre-cache build\n        return None\n    if not resp_list:\n        return []\n    P = len(prompt_ids)\n    out = model(torch.tensor([prompt_ids], device=dev), use_cache=True)\n    past = out.past_key_values\n    rewind = _make_rewind(past, P)\n    if rewind is None:\n        return None\n    head = torch.log_softmax(out.logits[0, -1].float(), dim=-1)\n    vals = []\n    for r_ids in resp_list:\n        tgt = torch.tensor(r_ids, device=dev)\n        tot = float(head[tgt[0]])\n        if len(r_ids) > 1:\n            lg = model(tgt[:-1][None, :], past_key_values=past,\n                       use_cache=True).logits[0].float()\n            tot += float(torch.log_softmax(lg, dim=-1)\n                         .gather(1, tgt[1:, None]).sum())\n            if not rewind():\n                return None\n        vals.append(-tot / len(r_ids))\n    return vals\n\n\n@torch.no_grad()\ndef score_pool(model, task, tin, grids, views, dev, max_len, deadline):\n    """aug_nll for every candidate, with a CHEAP pass pruning the pool first.\n\n    Scoring is ~26% of a real task (measured on the public eval set, not on the\n    placeholder) and costs |pool| x SCORE_VIEWS long forwards. But only the HEAD\n    of the ranking is ever read -- pass@2 looks at two candidates -- so paying\n    the full view budget to precisely rank a candidate that will finish tenth is\n    waste. Rank everything on SCORE_CHEAP views, keep the top SCORE_KEEP, and\n    spend the full budget only on those.\n\n    Sized by exact re-scoring of stored per-view vectors, so generation is\n    pinned and any difference is the cascade\'s own: on the shipped greedy pool\n    cheap=2/keep=6 reproduces the full ranking EXACTLY (0 up / 0 down over 169\n    outputs) at 0.66x the forwards. On a wider DFS pool the same idea runs at\n    0.34x, which is what would make a richer generator affordable.\n\n    Pruned candidates are returned at 99.0 so they sort last without having been\n    scored -- they were never going to be picked.\n    """\n    ng = len(grids)\n    idxs = list(range(ng))\n    if SCORE_CHEAP > 0 and 0 < SCORE_KEEP < ng:\n        pre = _score_subset(model, task, tin, grids, views[:SCORE_CHEAP], idxs,\n                            dev, max_len, deadline)\n        idxs = sorted(idxs, key=lambda i: pre[i])[:SCORE_KEEP]\n        SCORE_CASCADE["pruned"] += ng - len(idxs)\n    SCORE_CASCADE["scored"] += len(idxs)\n    full = _score_subset(model, task, tin, grids, views, idxs, dev, max_len,\n                         deadline)\n    return [full[i] if i in full else 99.0 for i in range(ng)]\n\n\n@torch.no_grad()\ndef _score_subset(model, task, tin, grids, views, idxs, dev, max_len, deadline):\n    """aug_nll for `idxs`, looping VIEWS outer so the prompt is shared.\n\n    Same arithmetic as scoring each candidate separately -- asserted at worker\n    start -- with two deliberate consequences of the reordering:\n      * the prefill is paid once per view instead of once per (view, candidate);\n      * a deadline hit now costs every candidate the same tail of views, rather\n        than giving early candidates a full estimate and late ones nothing.\n        Ranking on a uniformly noisier estimate beats ranking on a mix.\n    """\n    sums = {i: 0.0 for i in idxs}\n    cnts = {i: 0 for i in idxs}\n    for d, perm in views:\n        if time.time() > deadline:\n            break\n        demos = [(apply_aug(p["input"], d, perm), apply_aug(p["output"], d, perm))\n                 for p in task["train"]]\n        p_ids = enc("".join(pair_text(a, b) for a, b in demos)\n                    + query_text(apply_aug(tin, d, perm)))\n        rs, sel = [], []\n        for i in idxs:\n            r = enc(gstr(apply_aug(grids[i], d, perm)) + "<|im_end|>")\n            if len(p_ids) + len(r) <= max_len:\n                rs.append(r)\n                sel.append(i)\n        if not rs:\n            continue\n        vals = _nll_cached(model, p_ids, rs, dev)\n        if vals is None:\n            SCORE_CACHE["fallback"] += 1\n            vals = []\n            for r in rs:\n                x = torch.tensor([p_ids + r], device=dev)\n                lg = model(x).logits[0].float()\n                lp = torch.log_softmax(\n                    lg[len(p_ids) - 1:len(p_ids) + len(r) - 1], -1)\n                t = torch.tensor(r, device=dev)\n                vals.append(float(-lp.gather(1, t[:, None]).mean()))\n        else:\n            SCORE_CACHE["cached"] += 1\n        for i, v in zip(sel, vals):\n            sums[i] += v\n            cnts[i] += 1\n    return {i: (sums[i] / cnts[i] if cnts[i] else 99.0) for i in idxs}\n\n\n@torch.no_grad()\ndef rollout(model, pid, max_new, dev, temperature, deadline):\n    x = torch.tensor([pid], device=dev)\n    out = model(x, use_cache=True)\n    past, logits = out.past_key_values, out.logits[0, -1]\n    mask = torch.full((logits.numel(),), float("-inf"), device=dev)\n    mask[LEGAL] = 0.0\n    toks, cum = [], 0.0\n    for _ in range(max_new):\n        lp = torch.log_softmax(logits.float() + mask, dim=-1)\n        if temperature <= 0:\n            t = int(torch.argmax(lp))\n        else:\n            t = int(torch.multinomial(torch.softmax(\n                (logits.float() + mask) / temperature, dim=-1), 1))\n        cum -= float(lp[t])\n        toks.append(t)\n        if t == EOS or time.time() > deadline:\n            break\n        step = model(torch.tensor([[t]], device=dev),\n                     past_key_values=past, use_cache=True)\n        past, logits = step.past_key_values, step.logits[0, -1]\n    return toks, cum\n\n\ndef _sum_logp(model, pid, rid, dev):\n    """Sum log p(response | prompt). Differentiable; the DPO objective needs it."""\n    ids = pid + rid\n    out = model(torch.tensor([ids], device=dev))\n    lp = torch.log_softmax(out.logits[0, len(pid) - 1:len(ids) - 1].float(), dim=-1)\n    tgt = torch.tensor(rid, device=dev)\n    return lp.gather(1, tgt[:, None]).sum()\n\n\ndef _cell_agree(a, b):\n    """Fraction of cells equal, 0 unless the shapes match. The grading signal."""\n    if not a or not b or len(a) != len(b) or len(a[0]) != len(b[0]):\n        return 0.0\n    n = ok = 0\n    for ra, rb in zip(a, b):\n        for x, y in zip(ra, rb):\n            n += 1\n            ok += (x == y)\n    return ok / max(1, n)\n\n\ndef mine_ttdpo_pairs(model, task, dev, deadline, rng):\n    """Leave-one-out preference pairs from the DEMONSTRATIONS only.\n\n    `chosen` is the held-out demo\'s GOLD output, which always exists, so pair\n    coverage does not depend on the model already being right. `rejected` is the\n    model\'s own highest-likelihood rollout that is NOT the gold -- the hard\n    negative that actually competes with the truth, which is the negative worth\n    training against. Graded by per-cell agreement so ties are broken toward the\n    most confusable wrong answer rather than an arbitrary one.\n    """\n    demos = task["train"]\n    if len(demos) < 2:\n        return []\n    pairs, jobs = [], []\n    folds = list(range(len(demos)))\n    rng.shuffle(folds)\n    for k in folds[:TTDPO_FOLDS]:\n        if time.time() > deadline:\n            break\n        d, perm = make_views(1, np.random.default_rng(SEED + 7919 * k))[0]\n        ctx = [(apply_aug(p["input"], d, perm), apply_aug(p["output"], d, perm))\n               for i, p in enumerate(demos) if i != k]\n        qin = apply_aug(demos[k]["input"], d, perm)\n        gold = apply_aug(demos[k]["output"], d, perm)\n        pid = enc("".join(pair_text(a, b) for a, b in ctx) + query_text(qin))\n        gid = enc(gstr(gold) + "<|im_end|>")\n        if len(pid) + len(gid) > MAX_LEN:\n            continue\n        jobs.append((pid, gid, gold))\n    if not jobs:\n        return []\n    # ONE batched greedy decode for every fold, not a sequential rollout per\n    # fold per temperature. The first version mined 3 folds x 4 samples one at a\n    # time, which is 12 batch-1 autoregressive decodes: it burned the entire\n    # stage budget and produced ONE pair across eight tasks, so zero optimizer\n    # steps ever ran. Views are independent, so batching costs nothing in\n    # quality -- the same argument that makes the main decode path batched.\n    outs = rollout_batch(model, [j[0] for j in jobs],\n                         max(len(j[1]) for j in jobs) + 16, dev, deadline)\n    for (pid, gid, gold), (toks, cum) in zip(jobs, outs):\n        g = decode_grid(toks)\n        # A fold where the model already reproduces the held-out gold offers no\n        # competing wrong answer, so it yields no pair. That is the published\n        # pilot\'s `no_same_shape_negative` bucket and it is genuinely empty --\n        # unlike its missing-POSITIVE bucket, which taking the gold as `chosen`\n        # removes entirely.\n        # Separate the two ways a fold yields nothing, because they demand\n        # OPPOSITE responses and the bare pair count cannot tell them apart:\n        #   nog  -- the rollout did not parse as a grid at all. A BUG (budget,\n        #           padding, truncation) and fixable.\n        #   same -- the model reproduced the held-out gold, so no wrong answer\n        #           competes. A FINDING about test-time training, not a defect.\n        if g is None:\n            TTDPO_STATS["nog"] += 1\n        elif g == gold:\n            TTDPO_STATS["same"] += 1\n        else:\n            pairs.append((pid, gid, toks, _cell_agree(g, gold)))\n    return pairs\n\n\ndef run_ttdpo(model, lmods, task, dev, deadline):\n    """One short preference-optimisation phase on top of the CE-trained adapter.\n\n    Reference is the adapter as it stands on entry, so the reference log-probs\n    are computed once up front and cached -- no second model copy on a 22 GiB\n    card. At step 0 policy and reference are identical, so the loss must equal\n    log 2; that is asserted as a fail-closed self-check, because a reference\n    that silently differs turns DPO into unconstrained likelihood surgery.\n\n    Every guard rolls the adapter back rather than shipping a worse one.\n    """\n    TTDPO_STATS["tasks"] += 1\n    dl = min(deadline, time.time() + TTDPO_S)\n    model.eval()\n    with torch.no_grad():\n        pairs = mine_ttdpo_pairs(model, task, dev, dl, random.Random(SEED))\n    if not pairs:\n        TTDPO_STATS["nopair"] += 1\n        return\n    TTDPO_STATS["pairs"] += len(pairs)\n    steps_before = TTDPO_STATS["steps"]\n    before = [{k: v.detach().clone() for k, v in\n               (("A", m.A), ("B", m.B))} for m in lmods]\n\n    with torch.no_grad():\n        ref = [(float(_sum_logp(model, p, c, dev)),\n                float(_sum_logp(model, p, r, dev))) for p, c, r, _ in pairs]\n\n    params = [p for m in lmods for p in (m.A, m.B)]\n    opt = torch.optim.AdamW(params, lr=TTDPO_LR, weight_decay=0.0)\n    model.train()\n    model.config.use_cache = False\n    ok = True\n    try:\n        for step in range(max(1, TTDPO_STEPS)):\n            if time.time() > dl:\n                break\n            tot = None\n            for (p, c, r, _), (rc, rr) in zip(pairs, ref):\n                pc = _sum_logp(model, p, c, dev)\n                pr = _sum_logp(model, p, r, dev)\n                margin = (pc - rc) - (pr - rr)\n                l = -torch.nn.functional.logsigmoid(TTDPO_BETA * margin)\n                tot = l if tot is None else tot + l\n            tot = tot / len(pairs)\n            if step == 0:\n                # policy == reference here, so this MUST be log 2\n                if abs(float(tot) - math.log(2.0)) > 0.02:\n                    TTDPO_STATS["refchk"] += 1\n                    ok = False\n                    break\n            tot.backward()\n            torch.nn.utils.clip_grad_norm_(params, 0.5)\n            opt.step(); opt.zero_grad(set_to_none=True)\n            TTDPO_STATS["steps"] += 1\n    except torch.cuda.OutOfMemoryError:\n        ok = False\n    except Exception:\n        ok = False\n    finally:\n        model.eval()\n        model.config.use_cache = True\n        opt.zero_grad(set_to_none=True)\n\n    if TTDPO_STATS["steps"] == steps_before:\n        # No optimizer step ran (mining ate the budget, or the deadline passed).\n        # This is a NO-OP, not a success: counting it as fired is how a stage\n        # that never executed comes to look like a stage that did not help.\n        TTDPO_STATS["nostep"] += 1\n        ok = False\n    if ok:\n        # Guard: the adapter must not get WORSE at reproducing the demo golds.\n        with torch.no_grad():\n            after = sum(float(_sum_logp(model, p, c, dev)) for p, c, _, _ in pairs)\n            base = sum(rc for rc, _ in ref)\n        if after < base - abs(base) * 0.02:\n            ok = False\n    if not ok:\n        TTDPO_STATS["rollback"] += 1\n        with torch.no_grad():\n            for m, sv in zip(lmods, before):\n                m.A.copy_(sv["A"]); m.B.copy_(sv["B"])\n    else:\n        TTDPO_STATS["fired"] += 1\n\n\n@torch.no_grad()\ndef score_cand(model, task, tin, grid, views, dev, max_len):\n    vals = []\n    for d, perm in views:\n        demos = [(apply_aug(p["input"], d, perm), apply_aug(p["output"], d, perm))\n                 for p in task["train"]]\n        q = apply_aug(tin, d, perm)\n        tgt = apply_aug(grid, d, perm)\n        p_ids = enc("".join(pair_text(a, b) for a, b in demos) + query_text(q))\n        r_ids = enc(gstr(tgt) + "<|im_end|>")\n        if len(p_ids) + len(r_ids) > max_len:\n            continue\n        x = torch.tensor([p_ids + r_ids], device=dev)\n        lg = model(x).logits[0].float()\n        lp = torch.log_softmax(lg[len(p_ids) - 1:len(p_ids) + len(r_ids) - 1], -1)\n        t = torch.tensor(r_ids, device=dev)\n        vals.append(float(-lp.gather(1, t[:, None]).mean()))\n    return float(np.mean(vals)) if vals else 99.0\n\n\n# ------------------------------------- constrained beam over the pool support\n# The sampled candidates each get a DIFFERENT sparse subset of cells wrong, so\n# the pool\'s per-cell union contains the truth far more often than any single\n# grid does (measured on the public eval set: per-cell oracle 44.3% vs 24.2%\n# exact-match, and 0.00% when the same pools are scored against a mismatched\n# ground truth of the same shape). Decoding left-to-right but restricting each\n# cell\'s vocabulary to the colours the pool actually offers there therefore\n# reaches grids no single decode produced -- which reranking, capped at the\n# exact-match oracle, cannot. Separators are forced, so only contested cells\n# branch (~1.4 options each). Measured +6.4 pts of pool oracle (11 outputs\n# gained, 0 lost) and +4.07 pts pass@2 over the same run\'s sampled-only arm.\ndef _reorder_cache(past, idx):\n    """Reindex a KV cache along the batch dim across transformers versions."""\n    for meth in ("reorder_cache", "batch_select_indices"):\n        fn = getattr(past, meth, None)\n        if fn is not None:\n            fn(idx)\n            return True\n    layers = getattr(past, "layers", None)\n    if layers is not None:\n        for lay in layers:\n            lay.keys = lay.keys.index_select(0, idx)\n            lay.values = lay.values.index_select(0, idx)\n        return True\n    kc, vc = getattr(past, "key_cache", None), getattr(past, "value_cache", None)\n    if kc is not None and vc is not None:\n        for i in range(len(kc)):\n            kc[i] = kc[i].index_select(0, idx)\n            vc[i] = vc[i].index_select(0, idx)\n        return True\n    return False\n\n\ndef cell_support(grids, shape):\n    h, w = shape\n    sup = [[set() for _ in range(w)] for _ in range(h)]\n    used = 0\n    for g in grids:\n        if len(g) != h or any(len(r) != w for r in g):\n            continue\n        used += 1\n        for i in range(h):\n            for j in range(w):\n                sup[i][j].add(int(g[i][j]))\n    return (sup, used) if used else (None, 0)\n\n\n@torch.no_grad()\ndef support_beam(model, pid, sup, h, w, dev, beam, opt_cap, max_tokens):\n    sch = []\n    for i in range(h):\n        for j in range(w):\n            o = sorted(sup[i][j])\n            sch.append(o[:opt_cap])\n        sch.append([NL] if i < h - 1 else [EOS])\n    if len(sch) > max_tokens:\n        return []\n    out = model(torch.tensor([pid], device=dev), use_cache=True)\n    past, logits = out.past_key_values, out.logits[:, -1]\n    seqs, scores = [[]], torch.zeros(1, device=dev, dtype=torch.float32)\n    for step, allowed in enumerate(sch):\n        lp = torch.log_softmax(logits.float(), dim=-1)\n        b = lp.shape[0]\n        if len(allowed) == 1:\n            t = allowed[0]\n            scores = scores + lp[:, t]\n            seqs = [s + [t] for s in seqs]\n            bsel = torch.arange(b, device=dev)\n            nxt = torch.full((b, 1), t, device=dev, dtype=torch.long)\n        else:\n            a = torch.tensor(allowed, device=dev, dtype=torch.long)\n            flat = (scores[:, None] + lp[:, a]).reshape(-1)\n            k = min(beam, flat.numel())\n            top, idx = torch.topk(flat, k)\n            bsel = torch.div(idx, len(allowed), rounding_mode="floor")\n            tsel = a[idx % len(allowed)]\n            seqs = [seqs[int(x)] + [int(y)] for x, y in zip(bsel, tsel)]\n            scores, nxt = top, tsel[:, None]\n        if step == len(sch) - 1:\n            break\n        if not _reorder_cache(past, bsel):\n            return []\n        so = model(nxt, past_key_values=past, use_cache=True)\n        past, logits = so.past_key_values, so.logits[:, -1]\n    order = torch.argsort(scores, descending=True)\n    return [seqs[int(i)] for i in order]\n\n\n# ------------------------------------------------------------------ worker\ndef _unclaimed_weight(qdir, weights, total):\n    """Total weight of tasks nobody has claimed yet."""\n    try:\n        taken = os.listdir(qdir)\n    except OSError:\n        return total\n    spent = 0.0\n    for f in taken:\n        try:\n            spent += weights[int(f[1:])]\n        except (ValueError, IndexError):\n            pass\n    return max(total - spent, 1.0)\n\n\ndef _claim_next(qdir, n):\n    """Atomically claim the lowest unclaimed task index, or None if exhausted.\n\n    O_CREAT|O_EXCL is the lock: the filesystem guarantees exactly one creator,\n    so two workers can never take the same task and no lock file needs cleaning\n    up. Scanning from 0 each time keeps the global order LONGEST-FIRST, which is\n    the LPT schedule -- the greedy makespan-optimal rule for exactly this\n    problem, and strictly better than the static round-robin it replaces.\n    """\n    for i in range(n):\n        try:\n            fd = os.open(os.path.join(qdir, f"c{i}"),\n                         os.O_CREAT | os.O_EXCL | os.O_WRONLY)\n            os.close(fd)\n            return i\n        except FileExistsError:\n            continue\n        except OSError:\n            return None\n    return None\n\n\ndef solve_task(model, lmods, task, dev, cap_deadline):\n    rng = np.random.default_rng(SEED)\n    reset_adapters(lmods)\n    seqs = build_ttt(task, make_views(TTT_VIEWS, rng, TTT_COLOR_PERMS),\n                     random.Random(SEED), MAX_LEN)\n    if seqs and time.time() < cap_deadline:\n        _t = time.time()\n        run_ttt(model, lmods, seqs, dev, cap_deadline)\n        PHASE_S["ttt"] += time.time() - _t\n        _beacon("ttt", f"{len(seqs)} seqs ck{CKPT_STATS[\'ck\']}/"\n               f"nock{CKPT_STATS[\'nock\']}/oom{CKPT_STATS[\'oom\']}")\n    if TTDPO and time.time() < cap_deadline:\n        _t = time.time()\n        run_ttdpo(model, lmods, task, dev, cap_deadline)\n        PHASE_S["ttt"] += time.time() - _t\n        _beacon("dpo", f"pairs{TTDPO_STATS[\'pairs\']} steps{TTDPO_STATS[\'steps\']} "\n                       f"fired{TTDPO_STATS[\'fired\']} rb{TTDPO_STATS[\'rollback\']} "\n                       f"np{TTDPO_STATS[\'nopair\']} ns{TTDPO_STATS[\'nostep\']} "\n                       f"nog{TTDPO_STATS[\'nog\']} same{TTDPO_STATS[\'same\']}")\n\n    preds = []\n    for tin in task["test"]:\n        pool = {}\n        # Build every view\'s prompt first, then decode them in batches. Views\n        # are independent, so batching costs nothing in candidate quality and\n        # near-linearly reduces decode wall-time; DECODE_BATCH bounds the KV\n        # cache (a 22 GiB card holds ~8 long prompts alongside a 4B model).\n        jobs = []\n        for d, perm in make_views(DECODE_VIEWS, rng):\n            demos = [(apply_aug(p["input"], d, perm),\n                      apply_aug(p["output"], d, perm)) for p in task["train"]]\n            q = apply_aug(tin["input"], d, perm)\n            # Over-length views used to be SKIPPED outright, which silently\n            # gives a task with big demos an empty pool and a guaranteed zero.\n            # Measured on the public eval set: 0.6% of views and exactly one\n            # task of 120 lose every view that way. Small, but that task\n            # scores zero BY CONSTRUCTION, so drop the earliest demos until\n            # the prompt fits rather than dropping the view -- the reference\n            # implementation does the same. Keeping the LAST demos matters:\n            # they sit nearest the query, and a rule shown once is still a\n            # rule, whereas no prompt at all is nothing to condition on.\n            kept = list(demos)\n            while True:\n                pid = enc("".join(pair_text(a, b) for a, b in kept)\n                          + query_text(q))\n                if len(pid) <= MAX_LEN - 200 or len(kept) <= 1:\n                    break\n                kept = kept[1:]\n            if len(pid) > MAX_LEN - 200:\n                continue                      # even one demo will not fit\n            if len(kept) < len(demos):\n                TRUNC_STATS["views"] += 1\n            for si in range(SAMPLES_PER_VIEW):\n                jobs.append((d, perm, pid, si))\n\n        bsz = _decode_batch_for([j[2] for j in jobs])\n        DECODE_STATS["batches"] += 1\n        DECODE_STATS["views"] += bsz\n        i = 0\n        while i < len(jobs):\n            if time.time() > cap_deadline:\n                break\n            chunk = jobs[i:i + bsz]\n            i += bsz\n            greedy = [c for c in chunk if c[3] == 0]\n            other = [c for c in chunk if c[3] != 0]\n            results = []\n            if greedy:\n                budget = GRID_MAX_TOK\n                if MAX_LEN - max(len(c[2]) for c in greedy) < GRID_MAX_TOK:\n                    BUDGET_STATS["raised"] += 1\n                try:\n                    _t = time.time()\n                    if DFS_BUDGET > 0:\n                        DFS_STATS["outputs"] += 1\n                        per_view = dfs_decode_batch(\n                            model, [c[2] for c in greedy], budget, dev,\n                            cap_deadline, DFS_BUDGET, DFS_WIDTH,\n                            DFS_MAX_CANDS, DFS_ROWS)\n                        # one view can now yield several sequences, so the\n                        # (job, output) pairing stops being one-to-one\n                        outs = [(c, o) for c, vs in zip(greedy, per_view)\n                                for o in vs]\n                    else:\n                        outs = list(zip(greedy,\n                                        rollout_batch(model,\n                                                      [c[2] for c in greedy],\n                                                      budget, dev, cap_deadline)))\n                except Exception as e:\n                    print(f"  batch decode fell back ({type(e).__name__}: {e})",\n                          flush=True)\n                    outs = [(c, rollout(model, c[2], budget, dev, 0.0,\n                                        cap_deadline)) for c in greedy]\n                results += outs\n                PHASE_S["decode"] += time.time() - _t\n            _t = time.time()\n            for c in other:                       # sampled views stay per-item\n                if time.time() > cap_deadline:\n                    break\n                results.append((c, rollout(model, c[2], GRID_MAX_TOK,\n                                           dev, TEMP, cap_deadline)))\n            PHASE_S["decode"] += time.time() - _t\n            for (d, perm, _pid, _si), (toks, cum) in results:\n                g = decode_grid(toks)\n                if g is None:\n                    continue\n                try:\n                    gc_ = invert_aug(g, d, perm)\n                except Exception:\n                    continue\n                if not is_grid(gc_):\n                    continue\n                k = (len(gc_), len(gc_[0]), tuple(x for r in gc_ for x in r))\n                e = pool.setdefault(k, dict(grid=gc_, hits=0))\n                e["hits"] += 1\n        # --- second candidate source: constrained beam over this pool\'s support\n        if BEAM_VIEWS and pool and time.time() < cap_deadline:\n            BEAM_STATS["outputs"] += 1\n            _before = len(pool)\n            try:\n                grids = [e["grid"] for e in pool.values()]\n                tally = {}\n                for e in pool.values():\n                    s = (len(e["grid"]), len(e["grid"][0]))\n                    tally[s] = tally.get(s, 0) + e["hits"]\n                shape = max(tally.items(), key=lambda kv: kv[1])[0]\n                sup, used = cell_support(grids, shape)\n                if sup:\n                    for d, perm in make_views(BEAM_VIEWS, rng):\n                        if time.time() > cap_deadline:\n                            break\n                        demos = [(apply_aug(p["input"], d, perm),\n                                  apply_aug(p["output"], d, perm))\n                                 for p in task["train"]]\n                        q = apply_aug(tin["input"], d, perm)\n                        pid = enc("".join(pair_text(a, b) for a, b in demos)\n                                  + query_text(q))\n                        if len(pid) > MAX_LEN - 200:\n                            continue\n                        # map the canonical support into this view\'s frame by\n                        # augmenting one full grid per option index, so the\n                        # geometry and the colour permutation are applied by\n                        # exactly the same code path as the data itself\n                        mx = max(len(sup[i][j]) for i in range(shape[0])\n                                 for j in range(shape[1]))\n                        layers = [apply_aug(\n                            [[sorted(sup[i][j])[min(o, len(sup[i][j]) - 1)]\n                              for j in range(shape[1])]\n                             for i in range(shape[0])], d, perm)\n                            for o in range(mx)]\n                        ah, aw = len(layers[0]), len(layers[0][0])\n                        asup = [[set() for _ in range(aw)] for _ in range(ah)]\n                        for lay in layers:\n                            for i in range(ah):\n                                for j in range(aw):\n                                    asup[i][j].add(int(lay[i][j]))\n                        for toks in support_beam(model, pid, asup, ah, aw, dev,\n                                                 BEAM_WIDTH, BEAM_OPT_CAP,\n                                                 min(1200, MAX_LEN - len(pid))):\n                            g = decode_grid(toks)\n                            if g is None:\n                                continue\n                            try:\n                                gc_ = invert_aug(g, d, perm)\n                            except Exception:\n                                continue\n                            if not is_grid(gc_):\n                                continue\n                            kk = (len(gc_), len(gc_[0]),\n                                  tuple(x for r in gc_ for x in r))\n                            pool.setdefault(kk, dict(grid=gc_, hits=0))\n            except Exception as e:\n                BEAM_STATS["skipped"] += 1\n                print(f"  beam skipped: {type(e).__name__}: {e}", flush=True)\n            added = len(pool) - _before\n            BEAM_STATS["novel"] += added\n            BEAM_STATS["ran"] += int(added > 0)\n\n        _beacon("decode", f"pool={len(pool)}")\n        sv = make_views(SCORE_VIEWS, rng)\n        _t = time.time()\n        # Rank on augmentation NLL ALONE. Measured equal-or-better than\n        # hits-aug_nll on 1440 pooled rows (21.46% vs 20.83%), and it is\n        # REQUIRED once a second candidate source exists: a DFS or beam\n        # candidate has hits=0 by construction, so under the hit-count rule it\n        # always ranks last and its entire contribution measures exactly zero.\n        _grids = [e["grid"] for e in pool.values()]\n        scored = list(zip([-a for a in score_pool(model, task, tin["input"],\n                                                  _grids, sv, dev, MAX_LEN,\n                                                  cap_deadline)], _grids))\n        PHASE_S["score"] += time.time() - _t\n        _beacon("score", f"pool={len(pool)} SV={SCORE_VIEWS}")\n        scored.sort(key=lambda z: -z[0])\n        cands = [g for _, g in scored[:2]]\n        fb = [list(r) for r in tin["input"]] if is_grid(tin["input"]) else [[0]]\n        while len(cands) < 2:\n            cands.append(fb)\n        preds.append({"attempt_1": cands[0], "attempt_2": cands[1]})\n    return preds\n\n\n# ------------------------------------------------------------ ensemble path\n_CUR_TID = ""\n# Global-slack guard, set by the worker before each ensemble task: seconds of wall\n# left AFTER reserving 1.2x the remaining tasks\' measured single-adapter cost. An\n# extra adapter is fitted only out of this, so the ensemble can never starve a\n# LATER task\'s adapter 0 -- it spends idle wall and nothing else.\n_ENS_SLACK = float("inf")\n_ENS_COST0 = []\n# Weight of each task whose adapter-0 cost is in _ENS_COST0, and of the task in\n# hand. The guard projects remaining work as (seconds per unit weight) x (weight\n# still unclaimed). It used to project from the MEAN per-task cost so far -- but\n# tasks run longest-first, so that mean is dominated by the biggest tasks and\n# overstates everything still to come, and the guard refused the second adapter\n# on a large share of tasks while leaving wall unused (E235).\n_ENS_W = []\n_ENS_WCUR = 0.0\n# "weight" (default) or "mean" (the v45 rule, kept for the replay/ablation).\nENS_GUARD = os.environ.get("ENS_GUARD", "weight")\nENS_GUARD_MARGIN = float(os.environ.get("ENS_GUARD_MARGIN", 1.25))\n\n\ndef _save_adapter(lmods):\n    return [(m.A.detach().clone(), m.B.detach().clone()) for m in lmods]\n\n\ndef _load_adapter(lmods, st):\n    with torch.no_grad():\n        for m, (a, b) in zip(lmods, st):\n            m.A.copy_(a)\n            m.B.copy_(b)\n\n\ndef _ens_decode(model, task, tin, dviews, dev, deadline, pool, k, K):\n    """One adapter\'s decode over its own views into the SHARED pool.\n\n    The prompt construction and batching are the single-adapter path\'s, line\n    for line (greedy, or threshold-DFS when DFS_BUDGET > 0), so adapter 0\'s\n    candidates are exactly the ones the K=1 build would produce.\n    """\n    jobs = []\n    for d, perm in dviews:\n        demos = [(apply_aug(p["input"], d, perm),\n                  apply_aug(p["output"], d, perm)) for p in task["train"]]\n        q = apply_aug(tin["input"], d, perm)\n        kept = list(demos)\n        while True:\n            pid = enc("".join(pair_text(a, b) for a, b in kept) + query_text(q))\n            if len(pid) <= MAX_LEN - 200 or len(kept) <= 1:\n                break\n            kept = kept[1:]\n        if len(pid) > MAX_LEN - 200:\n            continue\n        if len(kept) < len(demos):\n            TRUNC_STATS["views"] += 1\n        for si in range(SAMPLES_PER_VIEW):\n            jobs.append((d, perm, pid, si))\n    bsz = _decode_batch_for([j[2] for j in jobs])\n    DECODE_STATS["batches"] += 1\n    DECODE_STATS["views"] += bsz\n    i = 0\n    nb = 0\n    while i < len(jobs):\n        grace = k == 0 and nb < ENS_MIN_DECODE_BATCHES and not pool\n        bdl = max(deadline, time.time() + ENS_DECODE_GRACE_S) if grace else deadline\n        if time.time() > bdl:\n            break\n        if grace and time.time() > deadline:\n            STARVE_STATS["grace_batches"] += 1\n        nb += 1\n        chunk = jobs[i:i + bsz]\n        i += bsz\n        greedy = [c for c in chunk if c[3] == 0]\n        other = [c for c in chunk if c[3] != 0]\n        results = []\n        _t = time.time()\n        if greedy:\n            budget = GRID_MAX_TOK\n            if MAX_LEN - max(len(c[2]) for c in greedy) < GRID_MAX_TOK:\n                BUDGET_STATS["raised"] += 1\n            try:\n                if DFS_BUDGET > 0:\n                    DFS_STATS["outputs"] += 1\n                    per_view = dfs_decode_batch(\n                        model, [c[2] for c in greedy], budget, dev, bdl,\n                        DFS_BUDGET, DFS_WIDTH, DFS_MAX_CANDS, DFS_ROWS)\n                    outs = [(c, o) for c, vs in zip(greedy, per_view) for o in vs]\n                else:\n                    outs = list(zip(greedy, rollout_batch(\n                        model, [c[2] for c in greedy], budget, dev, bdl)))\n            except Exception as e:\n                print(f"  batch decode fell back ({type(e).__name__}: {e})",\n                      flush=True)\n                outs = [(c, rollout(model, c[2], budget, dev, 0.0, bdl))\n                        for c in greedy]\n            results += outs\n        for c in other:\n            if time.time() > bdl:\n                break\n            results.append((c, rollout(model, c[2], GRID_MAX_TOK, dev, TEMP,\n                                       bdl)))\n        PHASE_S["decode"] += time.time() - _t\n        for (d, perm, _pid, _si), (toks, cum) in results:\n            g = decode_grid(toks)\n            if g is None:\n                continue\n            try:\n                gc_ = invert_aug(g, d, perm)\n            except Exception:\n                continue\n            if not is_grid(gc_):\n                continue\n            key = (len(gc_), len(gc_[0]), tuple(x for r in gc_ for x in r))\n            e = pool.setdefault(key, dict(grid=gc_, hits=[0] * K))\n            e["hits"][k] += 1\n\n\ndef _ens_score(model, lmods, task, tin, grids, states, svs, dev, deadline):\n    """Joint aug_nll: each adapter scores on ITS OWN score views, then average.\n\n    Cascade as the single path: a cheap joint pass on SCORE_CHEAP views per\n    adapter prunes to the head, and only the head gets every adapter\'s full view\n    set. With one adapter this is score_pool exactly (same views, same keep).\n    Returns (joint, per_adapter_dicts).\n    """\n    n = len(grids)\n    idxs = list(range(n))\n    keep = SCORE_KEEP if len(states) == 1 else ENS_KEEP\n    if not ENS_FULLSCORE and SCORE_CHEAP > 0 and 0 < keep < n:\n        tot, cnt = [0.0] * n, [0] * n\n        for st, sv in zip(states, svs):\n            if len(states) > 1:\n                _load_adapter(lmods, st)\n            r = _score_subset(model, task, tin, grids, sv[:SCORE_CHEAP], idxs,\n                              dev, MAX_LEN, deadline)\n            for j in idxs:\n                if r[j] < 99.0:\n                    tot[j] += r[j]\n                    cnt[j] += 1\n        pre = [tot[j] / cnt[j] if cnt[j] else 99.0 for j in range(n)]\n        idxs = sorted(idxs, key=lambda j: pre[j])[:keep]\n        SCORE_CASCADE["pruned"] += n - len(idxs)\n    SCORE_CASCADE["scored"] += len(idxs)\n    mat = []\n    for st, sv in zip(states, svs):\n        if len(states) > 1:\n            _load_adapter(lmods, st)\n        mat.append(_score_subset(model, task, tin, grids, sv, idxs, dev, MAX_LEN,\n                                 deadline))\n    joint = []\n    for j in range(n):\n        vals = [m[j] for m in mat if j in m and m[j] < 99.0]\n        joint.append(sum(vals) / len(vals) if vals else 99.0)\n    return joint, mat\n\n\ndef solve_task_ens(model, lmods, task, dev, cap_deadline):\n    t0 = time.time()\n    K = max(TTT_ENSEMBLE, 1)\n    # Every adapter\'s views are drawn up front in the SAME order the single path\n    # draws them (TTT views, then per test input: decode views, score views), so\n    # adapter k is bit-for-bit the K=1 build at SEED + k*stride. BEAM_VIEWS must\n    # be 0 here: the single path draws beam views conditionally.\n    plans = []\n    for k in range(K):\n        sk = SEED + k * ENS_SEED_STRIDE\n        rng = np.random.default_rng(sk)\n        tv = make_views(TTT_VIEWS, rng, TTT_COLOR_PERMS)\n        per = [(make_views(DECODE_VIEWS, rng), make_views(SCORE_VIEWS, rng))\n               for _ in task["test"]]\n        plans.append((sk, tv, per))\n    ext_deadline = max(cap_deadline,\n                       min(t0 + ENS_CAP_MULT * max(cap_deadline - t0, 0.0),\n                           DEADLINE - 60))\n    reserve = ENS_SCORE_RESERVE * max(ext_deadline - t0, 0.0)\n    pools = [dict() for _ in task["test"]]\n    states, ran = [], []\n    cost0 = 0.0\n    for k, (sk, tv, per) in enumerate(plans):\n        tk = time.time()\n        if k == 0:\n            dl = cap_deadline                 # adapter 0 is never squeezed\n        else:\n            dl = ext_deadline - reserve\n            if tk + 1.15 * cost0 > dl or 1.15 * cost0 > _ENS_SLACK:\n                ENS_STATS["skipped"] += K - k\n                break\n        reset_adapters(lmods)\n        seqs = build_ttt(task, tv, random.Random(sk), MAX_LEN)\n        tdl = dl\n        if k == 0 and ENS_TRAIN_FRAC < 1.0:\n            tdl = min(dl, t0 + ENS_TRAIN_FRAC * max(dl - t0, 0.0))\n        if seqs and time.time() < tdl:\n            _t = time.time()\n            run_ttt(model, lmods, seqs, dev, tdl)\n            PHASE_S["ttt"] += time.time() - _t\n            if tdl < dl and time.time() >= tdl:\n                STARVE_STATS["train_cut"] += 1\n        for t, tin in enumerate(task["test"]):\n            _ens_decode(model, task, tin, per[t][0], dev, dl, pools[t], k, K)\n        states.append(_save_adapter(lmods) if K > 1 else None)\n        ran.append(k)\n        ENS_STATS["adapters"] += 1\n        if k == 0:\n            cost0 = time.time() - t0\n            _ENS_COST0.append(cost0)\n            _ENS_W.append(_ENS_WCUR)\n        else:\n            globals()["_ENS_SLACK"] = _ENS_SLACK - (time.time() - tk)\n    ENS_STATS["tasks"] += 1\n    ENS_STATS["joint"] += int(len(states) > 1)\n    _beacon("decode", f"ens={len(states)}/{K} pool={sum(len(p) for p in pools)}")\n\n    preds = []\n    for t, tin in enumerate(task["test"]):\n        pool = pools[t]\n        grids = [e["grid"] for e in pool.values()]\n        _t = time.time()\n        svs = [plans[k][2][t][1] for k in ran]\n        joint, mat = _ens_score(model, lmods, task, tin["input"], grids, states,\n                                svs, dev,\n                                ext_deadline if len(states) > 1 else cap_deadline)\n        PHASE_S["score"] += time.time() - _t\n        order = sorted(range(len(grids)), key=lambda j: joint[j])\n        if ENS_MIN_DECODE_BATCHES > 0 and grids and all(v >= 99.0 for v in joint):\n            hits = [sum(e["hits"]) for e in pool.values()]\n            order = sorted(range(len(grids)), key=lambda j: -hits[j])\n            STARVE_STATS["vote_order"] += 1\n        cands = [grids[j] for j in order[:2]]\n        if ENS_DUMP:\n            try:\n                rec = {"tid": _CUR_TID, "t": t, "seeds": [plans[k][0] for k in ran],\n                       "pick": order[:2],\n                       "cands": [{"g": grids[j],\n                                  "h": list(e["hits"]),\n                                  "s": [(m[j] if j in m else None) for m in mat]}\n                                 for j, e in enumerate(pool.values())]}\n                with open(f"_ens_{os.getpid()}.jsonl", "a") as fh:\n                    fh.write(json.dumps(rec) + "\\n")\n            except Exception as e:\n                print(f"  ENS_DUMP failed: {type(e).__name__}: {e}", flush=True)\n        fb = [list(r) for r in tin["input"]] if is_grid(tin["input"]) else [[0]]\n        while len(cands) < 2:\n            cands.append(fb)\n        preds.append({"attempt_1": cands[0], "attempt_2": cands[1]})\n    if len(states) > 1:\n        _load_adapter(lmods, states[0])       # leave the model as K=1 would\n    return preds\n\n\ndef _install_flash(rank):\n    """Install flash-attn from an ATTACHED dataset. There is no internet here.\n\n    The wheel must be installed before transformers is asked for\n    `flash_attention_2`; merely attaching the dataset does nothing, which is\n    exactly the packaging failure that later reads as "the kernel did not\n    help". Every step is logged and every failure is non-fatal -- the caller\n    falls back to sdpa and the run still produces a complete submission.\n    """\n    import glob as _g\n    try:\n        import flash_attn                      # already in the image\n        print(f"[w{rank}] FLASH: already importable", flush=True)\n        return\n    except Exception:\n        pass\n    whls = sorted(set(_g.glob("/kaggle/input/*/**/flash_attn*.whl",\n                              recursive=True)))\n    if not whls:\n        if rank == 0:\n            print(f"[w{rank}] FLASH: no flash_attn wheel among the attached "\n                  f"inputs -- staying on sdpa", flush=True)\n        return\n    d = os.path.dirname(whls[0])\n    r = subprocess.run(\n        [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",\n         f"--find-links={d}", whls[0]], capture_output=True, text=True)\n    ok = r.returncode == 0\n    print(f"[w{rank}] FLASH: pip {\'ok\' if ok else \'FAILED\'} "\n          f"({os.path.basename(whls[0])})"\n          f"{\'\' if ok else \': \' + r.stderr[-200:]}", flush=True)\n\n\ndef worker(rank, model_dir, task_items, ret):\n    global RANK\n    RANK = rank\n    # `rank` selects the GPU only when several are visible; a subprocess\n    # launched with CUDA_VISIBLE_DEVICES already sees exactly one.\n    dev = "cuda:0" if torch.cuda.device_count() == 1 else f"cuda:{rank}"\n    torch.cuda.set_device(dev)\n    from transformers import AutoModelForCausalLM\n    # low_cpu_mem_usage streams the shards straight to the device: four\n    # concurrent workers each materialising a full bf16 copy in host RAM first\n    # is the one way this fan-out can OOM the box rather than a GPU.\n    # Prefer flash-attention when an attached wheel provides it, else fall back\n    # to sdpa. Attention is the dominant cost of both TTT and decoding, and the\n    # budget is per-task wall-clock, so a faster kernel converts directly into\n    # more views -- the one axis measured to pay (+6.4 pts, 11 gained/0 lost).\n    # Which kernel actually loaded is LOGGED: an unresolved offline package is\n    # dropped silently by the platform, so "flash-attn made no difference"\n    # must never be confusable with "flash-attn was never installed".\n    if USE_FLASH:\n        _install_flash(rank)\n    attn, model = None, None\n    for cand in (["flash_attention_2", "sdpa"] if USE_FLASH else ["sdpa"]):\n        try:\n            model = AutoModelForCausalLM.from_pretrained(\n                model_dir, torch_dtype=torch.bfloat16,\n                attn_implementation=cand,\n                low_cpu_mem_usage=True, local_files_only=True).to(dev)\n            attn = cand\n            break\n        except Exception as e:\n            if rank == 0:\n                print(f"[w{rank}] attn={cand} unavailable "\n                      f"({type(e).__name__})", flush=True)\n    if model is None:\n        raise RuntimeError("no usable attention implementation")\n    print(f"[w{rank}] ATTN: {attn}", flush=True)\n    model.config.use_cache = True\n    global KV_BYTES_PER_TOK\n    _c = model.config\n    _kvh = getattr(_c, "num_key_value_heads", None) or _c.num_attention_heads\n    _hd = getattr(_c, "head_dim", None) or _c.hidden_size // _c.num_attention_heads\n    KV_BYTES_PER_TOK = 2 * _kvh * _hd * 2 * _c.num_hidden_layers\n    print(f"[w{rank}] KV {KV_BYTES_PER_TOK/1024:.0f} KiB/token -> decode batch "\n          f"{_decode_batch_for([[0]*1700])} at 1700 tok, "\n          f"{_decode_batch_for([[0]*8192])} at 8192 tok", flush=True)\n    lmods = inject_lora(model, LORA_R, LORA_ALPHA)\n    # Self-check BEFORE the first task. Every per-task failure is caught and\n    # turned into a fallback, so a systematic fault (wrong device, wrong dtype,\n    # a kernel the card lacks) would otherwise be invisible until the score\n    # comes back looking like a bad model. One forward pass settles it.\n    bad = [n for n, p in model.named_parameters()\n           if p.device != next(model.parameters()).device]\n    try:\n        with torch.no_grad():\n            model(torch.tensor([[0, 1, 2]], device=dev))\n        print(f"[w{rank}] SELF-CHECK PASS  (adapted forward ran on {dev}"\n              f"{\', \' + str(len(bad)) + \' off-device params\' if bad else \'\'})",\n              flush=True)\n    except Exception as e:\n        print(f"[w{rank}] SELF-CHECK FAIL  {type(e).__name__}: {e}", flush=True)\n        raise\n    # SEED-PROBE. Every other knob in this build reports a counter that proves it\n    # took effect in the ARTIFACT rather than only in the offline harness; SEED\n    # had none, and a knob with no counter is exactly how `mask_first` sat in the\n    # ledger as "adopted" while every scored submission ran without it. A seed\n    # that silently failed to apply would make a decorrelated second nominee a\n    # byte-identical clone of the first -- the one property its value depends on.\n    # The digest is over the first TTT sequence\'s token ids, so it moves if and\n    # only if the seed actually reached the augmentation draw.\n    try:\n        import hashlib\n        _t0 = task_items[len(task_items) // 2][1]\n        # Digest the VIEW LIST, not just the first sequence. The first version\n        # hashed the first 64 tokens of sequence 0, and view 0 is hardcoded to\n        # the identity augmentation -- so its only seed-dependence was a\n        # shuffle of the task\'s handful of demos, and consecutive integer seeds\n        # shuffle a 2- or 3-element list IDENTICALLY. It duly reported the same\n        # digest for seed 1 and seed 2 on a 3-demo task while the runs genuinely\n        # differed, i.e. a false "unchanged" on a knob that had in fact applied.\n        # The view list is what test-time training actually consumes and it\n        # varies with the seed throughout.\n        _v = make_views(TTT_VIEWS, np.random.default_rng(SEED))\n        _dv = hashlib.sha256(str(_v).encode()).hexdigest()[:12]\n        _s = build_ttt(_t0, _v, random.Random(SEED), MAX_LEN)\n        _d = hashlib.sha256(str(_s[0][0][:64]).encode()).hexdigest()[:12] if _s else "empty"\n        print(f"[w{rank}] SEED-PROBE seed={SEED} views-digest={_dv} seq-digest={_d}",\n              flush=True)\n    except Exception as e:\n        print(f"[w{rank}] SEED-PROBE unavailable ({type(e).__name__}: {e})", flush=True)\n    if TTT_ENSEMBLE > 1:\n        print(f"[w{rank}] ENSEMBLE: K={TTT_ENSEMBLE} seeds="\n              f"{[SEED + k * ENS_SEED_STRIDE for k in range(TTT_ENSEMBLE)]} "\n              f"reserve={ENS_SCORE_RESERVE} keep={ENS_KEEP} capx={ENS_CAP_MULT:g} "\n              f"guard={ENS_GUARD}x{ENS_GUARD_MARGIN:g} "\n              f"fullscore={ENS_FULLSCORE} dump={ENS_DUMP} beam={BEAM_VIEWS}"\n              f"{\' (BEAM IGNORED on this path)\' if BEAM_VIEWS else \'\'}", flush=True)\n    print(f"[w{rank}] ready, {len(task_items)} tasks, "\n          f"{len(lmods)} adapted layers", flush=True)\n\n    # Batched decoding is a pure throughput change, so it is only valid if it\n    # is a NO-OP on the output. Left-padding plus explicit position_ids is the\n    # part that can silently break that. Verify it HERE, on this machine and\n    # this checkpoint, before any task is solved -- and fall back to\n    # sequential decoding automatically if it diverges, so a wrong answer can\n    # never be the failure mode. A silent divergence would look exactly like a\n    # weaker model, which is the trap this run has hit twice already.\n    # Inference must run grad-free. A decode builds an autograd graph across\n    # every step otherwise, holding all activations: measured at >21 GiB on a\n    # 22 GiB card BEFORE the first task, which OOMs the run. This regressed\n    # once when a newly inserted function silently took the decorator off the\n    # one below it, so it is asserted rather than assumed.\n    import inspect as _inspect\n    try:\n        _ungrad = [n for n, f in (("rollout", rollout), ("rollout_batch", rollout_batch),\n                                  ("score_cand", score_cand), ("score_pool", score_pool),\n                                  ("dfs_decode_batch", dfs_decode_batch))\n                   if "no_grad" not in _inspect.getsource(f).split("def ")[0]]\n    except OSError:\n        # source is unavailable whenever __main__ is synthetic (exec\'d source).\n        # A GUARD MUST NOT KILL THE RUN IT GUARDS -- this crashed all 8 shards\n        # of an offline eval. Report that it could not run and continue.\n        _ungrad = None\n        print(f"[w{rank}] GRAD-GUARD SKIPPED (source unavailable)", flush=True)\n    if _ungrad is None:\n        pass                      # already reported SKIPPED; never claim PASS\n    elif _ungrad:\n        print(f"[w{rank}] GRAD-GUARD FAIL: {_ungrad} not grad-free", flush=True)\n    else:\n        print(f"[w{rank}] GRAD-GUARD PASS (all decode/score paths grad-free)",\n              flush=True)\n\n    global DECODE_BATCH\n    if DECODE_BATCH > 1 and task_items:\n        try:\n            t0 = task_items[len(task_items) // 2][1]\n            rng = np.random.default_rng(0)\n            pids = []\n            for d, perm in make_views(2, rng):\n                demos = [(apply_aug(p["input"], d, perm),\n                          apply_aug(p["output"], d, perm)) for p in t0["train"]]\n                q = apply_aug(t0["test"][0]["input"], d, perm)\n                p_ = enc("".join(pair_text(a, b) for a, b in demos) + query_text(q))\n                if len(p_) <= MAX_LEN - 200:\n                    pids.append(p_)\n            if len(pids) >= 2:\n                bud = min(160, MAX_LEN - max(len(p) for p in pids))\n                dl = time.time() + 300\n                seq = [decode_grid(rollout(model, p, bud, dev, 0.0, dl)[0])\n                       for p in pids]\n                bat = [decode_grid(t) for t, _ in\n                       rollout_batch(model, pids, bud, dev, dl)]\n                if seq == bat:\n                    print(f"[w{rank}] BATCH-EQUIV PASS (batched decode matches "\n                          f"sequential on {len(pids)} views)", flush=True)\n                else:\n                    DECODE_BATCH = 1\n                    print(f"[w{rank}] BATCH-EQUIV FAIL -- falling back to "\n                          f"sequential decoding (DECODE_BATCH=1)", flush=True)\n        except Exception as e:\n            DECODE_BATCH = 1\n            print(f"[w{rank}] BATCH-EQUIV errored ({type(e).__name__}: {e}) -- "\n                  f"falling back to sequential", flush=True)\n\n    # DFS is a strict superset of greedy, so at budget=0/width=1 it must return\n    # the greedy sequence token-for-token. That is the one property that makes\n    # the enumeration safe to ship: if it holds, DFS can only ADD candidates.\n    # Checked on this checkpoint and this card, not just in a unit test.\n    global DFS_BUDGET\n    if DFS_BUDGET > 0 and task_items:\n        try:\n            t0 = task_items[len(task_items) // 2][1]\n            rng = np.random.default_rng(0)\n            pids = []\n            for d, perm in make_views(2, rng):\n                demos = [(apply_aug(p["input"], d, perm),\n                          apply_aug(p["output"], d, perm)) for p in t0["train"]]\n                q = apply_aug(t0["test"][0]["input"], d, perm)\n                p_ = enc("".join(pair_text(a, b) for a, b in demos) + query_text(q))\n                if len(p_) <= MAX_LEN - 200:\n                    pids.append(p_)\n            if len(pids) >= 2:\n                bud = min(160, MAX_LEN - max(len(p) for p in pids))\n                dl = time.time() + 300\n                ref = [t for t, _ in rollout_batch(model, pids, bud, dev, dl)]\n                got = dfs_decode_batch(model, pids, bud, dev, dl, 0.0, 1, 1, 8)\n                same = len(got) == len(ref) and all(\n                    len(v) == 1 and v[0][0] == r for v, r in zip(got, ref))\n                if same:\n                    wide = dfs_decode_batch(model, pids, bud, dev, dl,\n                                            DFS_BUDGET, DFS_WIDTH,\n                                            DFS_MAX_CANDS, DFS_ROWS)\n                    print(f"[w{rank}] DFS-EQUIV PASS (budget=0 reproduces greedy "\n                          f"on {len(pids)} views; at budget={DFS_BUDGET} they "\n                          f"yield {sum(len(v) for v in wide)} sequences)",\n                          flush=True)\n                else:\n                    DFS_BUDGET = 0.0\n                    print(f"[w{rank}] DFS-EQUIV FAIL -- disabling DFS, "\n                          f"reverting to greedy decoding", flush=True)\n        except Exception as e:\n            DFS_BUDGET = 0.0\n            print(f"[w{rank}] DFS-EQUIV errored ({type(e).__name__}: {e}) -- "\n                  f"reverting to greedy decoding", flush=True)\n\n    # The cached scorer is an exact-arithmetic speedup: same views, same\n    # targets, same mean, one prefill instead of |pool|. Verify the arithmetic\n    # rather than assume the runtime\'s cache class rewinds the way this one\n    # needs -- a wrong NLL would silently reorder the ranking, which is the\n    # single thing the whole pipeline turns on.\n    if task_items:\n        try:\n            t0 = task_items[len(task_items) // 2][1]\n            d, perm = make_views(1, np.random.default_rng(0))[0]\n            demos = [(apply_aug(p["input"], d, perm),\n                      apply_aug(p["output"], d, perm)) for p in t0["train"]]\n            p_ids = enc("".join(pair_text(a, b) for a, b in demos)\n                        + query_text(apply_aug(t0["test"][0]["input"], d, perm)))\n            outs = [apply_aug(p["output"], d, perm) for p in t0["train"]][:3]\n            rs = [enc(gstr(g) + "<|im_end|>") for g in outs]\n            rs = [r for r in rs if len(p_ids) + len(r) <= MAX_LEN]\n            fast = _nll_cached(model, p_ids, rs, dev) if rs else None\n            if fast is None:\n                print(f"[w{rank}] SCORE-CACHE UNAVAILABLE -- scoring will use "\n                      f"the uncached path (correct, just slower)", flush=True)\n            else:\n                slow = []\n                with torch.no_grad():\n                    for r in rs:\n                        lg = model(torch.tensor([p_ids + r], device=dev)).logits[0].float()\n                        lp = torch.log_softmax(\n                            lg[len(p_ids) - 1:len(p_ids) + len(r) - 1], -1)\n                        t = torch.tensor(r, device=dev)\n                        slow.append(float(-lp.gather(1, t[:, None]).mean()))\n                err = max(abs(a - b) for a, b in zip(fast, slow))\n                SCORE_CACHE["err"] = float(err)\n                mech = "snapshot" if SCORE_CACHE["snap"] else "crop"\n                SCORE_CACHE.update(cached=0, fallback=0, snap=0)\n                if err < SCORE_CACHE_TOL:\n                    print(f"[w{rank}] SCORE-CACHE PASS via {mech} "\n                          f"(max |Δnll| {err:.2e} < tol {SCORE_CACHE_TOL:.1e} "\n                          f"over {len(rs)} candidates)", flush=True)\n                else:\n                    globals()["_nll_cached"] = lambda *a, **k: None\n                    print(f"[w{rank}] SCORE-CACHE FAIL via {mech} "\n                          f"(max |Δnll| {err:.2e} >= tol {SCORE_CACHE_TOL:.1e}) "\n                          f"-- forcing the uncached path", flush=True)\n        except Exception as e:\n            globals()["_nll_cached"] = lambda *a, **k: None\n            print(f"[w{rank}] SCORE-CACHE errored ({type(e).__name__}: {e}) -- "\n                  f"forcing the uncached path", flush=True)\n\n    # ---- GLOBAL PRE-ADAPTATION (E109). One LoRA pass over EVERY task\'s\n    # demonstrations before any per-task work, after which reset_adapters()\n    # restores these weights instead of noise. Runs before the loop so its cost\n    # comes out of the wall the adaptive per-task allocator then divides up,\n    # rather than silently overrunning it. Its own counter, because a target\n    # list that trains on nothing would otherwise look identical to success.\n    if GLOBAL_VIEWS > 0:\n        try:\n            _g0 = time.time()\n            _grng = np.random.default_rng(SEED)\n            gseqs = []\n            for _tid, _t in task_items:\n                gseqs += build_ttt(_t, make_views(GLOBAL_VIEWS, _grng),\n                                   random.Random(SEED), MAX_LEN)\n            random.Random(SEED).shuffle(gseqs)\n            print(f"[w{rank}] GLOBAL-PREADAPT: {len(gseqs)} seqs over "\n                  f"{len(task_items)} tasks x {GLOBAL_VIEWS} views", flush=True)\n            if gseqs:\n                _gl = run_ttt(model, lmods, gseqs, dev,\n                              min(time.time() + GLOBAL_CAP_S, DEADLINE - 60))\n                if GLOBAL_MERGE:\n                    _nm = merge_adapters(lmods)\n                    _how = f"MERGED into {_nm} base weights"\n                else:\n                    set_restore_point(lmods)\n                    _how = "kept as adapter restore point"\n                print(f"[w{rank}] GLOBAL-PREADAPT: done in {time.time()-_g0:.0f}s, "\n                      f"loss {_gl:.4f}, {_how}", flush=True)\n            else:\n                print(f"[w{rank}] GLOBAL-PREADAPT: NO SEQUENCES BUILT -- "\n                      f"falling back to per-task init", flush=True)\n        except Exception as e:\n            # Never let this cost the run: without a restore point every task\n            # simply starts from a fresh adapter, i.e. the shipped behaviour.\n            print(f"[w{rank}] GLOBAL-PREADAPT FAILED ({type(e).__name__}: {e}) -- "\n                  f"continuing with per-task init", flush=True)\n\n    out = {}\n    n_failed = 0\n    _costs = []\n    _wtotal = max(sum(WEIGHTS), 1.0) if WEIGHTS else 1.0\n    i = -1\n    while True:\n        i += 1\n        if QDIR:\n            # SHARED QUEUE. Static round-robin sharding left whichever GPU drew\n            # the cheap tasks idle for the rest of the run: measured across four\n            # equal shards of the public eval the walls were 57/71/66/55 min, so\n            # 12.3% of total GPU time was spent doing nothing while the slowest\n            # shard finished. Per-task budget is worth ~0.096 pts per 1%, and\n            # the reference implementation runs one shared queue across its four\n            # workers for exactly this reason.\n            idx = _claim_next(QDIR, len(task_items))\n            if idx is None:\n                break\n            tid, task = task_items[idx]\n            try:\n                claimed = len(os.listdir(QDIR))\n            except OSError:\n                claimed = idx + 1\n            remaining = max(1.0, (len(task_items) - claimed) / max(NWORKERS, 1) + 1.0)\n        else:\n            if i >= len(task_items):\n                break\n            tid, task = task_items[i]\n            remaining = len(task_items) - i\n        left = DEADLINE - time.time()\n        uniform = max(left / max(remaining, 1), 20.0)\n        cap = min(TASK_CAP, uniform)\n        if BUDGET_BY_SIZE and WEIGHTS:\n            try:\n                if QDIR:\n                    w_i = WEIGHTS[idx]\n                    w_rem = _unclaimed_weight(QDIR, WEIGHTS, _wtotal) + w_i\n                    share = left * w_i / max(w_rem / max(NWORKERS, 1), 1e-9)\n                else:\n                    w_i = WEIGHTS[i]\n                    w_rem = max(sum(WEIGHTS[i:]), 1e-9)\n                    share = left * w_i / w_rem\n                cap = min(TASK_CAP,\n                          max(20.0, min(2.0 * uniform, max(0.5 * uniform, share))))\n            except Exception:\n                pass\n        # Spend enumeration only out of measured slack (see DFS_ADAPTIVE).\n        # `global DFS_BUDGET` is already declared by the DFS-EQUIV self-check.\n        if DFS_ADAPTIVE and len(_costs) >= DFS_MIN_TASKS:\n            mean_cost = sum(_costs) / len(_costs)\n            projected = mean_cost * max(remaining, 1)\n            DFS_BUDGET = (DFS_ADAPTIVE_BUDGET\n                          if left - projected > DFS_SLACK_FRAC * projected\n                          else 0.0)\n            ADAPT_STATS["dfs_on" if DFS_BUDGET > 0 else "dfs_off"] += 1\n        _t0 = time.time()\n        try:\n            if TTT_ENSEMBLE > 1:\n                globals()["_CUR_TID"] = tid\n                _w_i, _w_after = 1.0, max(remaining - 1.0, 0.0)\n                if WEIGHTS:\n                    try:\n                        if QDIR:\n                            _w_i = WEIGHTS[idx]\n                            _w_after = max(_unclaimed_weight(QDIR, WEIGHTS, _wtotal)\n                                           / max(NWORKERS, 1), 0.0)\n                        else:\n                            _w_i = WEIGHTS[i]\n                            _w_after = max(sum(WEIGHTS[i + 1:]), 0.0)\n                    except Exception:\n                        _w_i, _w_after = 1.0, max(remaining - 1.0, 0.0)\n                globals()["_ENS_WCUR"] = _w_i\n                if len(_ENS_COST0) < 3:\n                    globals()["_ENS_SLACK"] = float("inf")\n                elif ENS_GUARD == "weight" and WEIGHTS and sum(_ENS_W) > 0:\n                    _u = sum(_ENS_COST0) / sum(_ENS_W)        # s per unit weight\n                    globals()["_ENS_SLACK"] = (left - _u * _w_i\n                                               - ENS_GUARD_MARGIN * _u * _w_after)\n                else:\n                    _mk1 = 1.15 * sum(_ENS_COST0) / len(_ENS_COST0)\n                    globals()["_ENS_SLACK"] = left - 1.2 * max(remaining, 1) * _mk1\n                out[tid] = solve_task_ens(model, lmods, task, dev,\n                                          time.time() + cap)\n            else:\n                out[tid] = solve_task(model, lmods, task, dev,\n                                      time.time() + cap)\n        except Exception as e:\n            n_failed += 1\n            print(f"[w{rank}] {tid} failed: {type(e).__name__}: {e}", flush=True)\n            # An OOM leaves the caching allocator fragmented and can cascade\n            # into CUBLAS_STATUS_ALLOC_FAILED on every later task, so reclaim\n            # before moving on rather than every fifth task.\n            gc.collect(); torch.cuda.empty_cache()\n            out[tid] = [{"attempt_1": ([list(r) for r in x["input"]]\n                                       if is_grid(x["input"]) else [[0]]),\n                         "attempt_2": ([list(r) for r in x["input"]]\n                                       if is_grid(x["input"]) else [[0]])}\n                        for x in task["test"]]\n        # Measure GREEDY cost only: a DFS task is ~24% dearer, and folding that\n        # back into the mean would make the projection chase its own tail.\n        if DFS_BUDGET <= 0:\n            _costs.append(time.time() - _t0)\n        if (i + 1) % 5 == 0 or i == 0:\n            el = time.time() - START\n            # PHASE rides the progress line, not the worker summary. A log read\n            # back through a budgeted API is truncated from the HEAD, so\n            # anything printed only at the end is exactly what gets cut -- the\n            # same failure that once cost a completed 2-hour probe its result.\n            # Interleaving it means a truncated head still carries the answer.\n            _tot = sum(PHASE_S.values()) or 1.0\n            print(f"[w{rank}] {len(out)}/{len(task_items)}  {el/60:.0f}min  "\n                  f"cap={cap:.0f}s  left={(DEADLINE-time.time())/60:.0f}min  "\n                  f"PHASE ttt {100*PHASE_S[\'ttt\']/_tot:.0f}% "\n                  f"dec {100*PHASE_S[\'decode\']/_tot:.0f}% "\n                  f"sco {100*PHASE_S[\'score\']/_tot:.0f}% (SV={SCORE_VIEWS}) "\n                  f"DFS {DFS_STATS[\'seqs\']}/{max(DFS_STATS[\'outputs\'],1)} "\n                  f"cap{DFS_STATS[\'rows_cap\']} "\n                  f"SC {SCORE_CACHE[\'cached\']}/{SCORE_CACHE[\'fallback\']}"\n                  f"{\'/snap\' + str(SCORE_CACHE[\'snap\']) if SCORE_CACHE[\'snap\'] else \'\'}"\n                  f"{\'/why=\' + SCORE_CACHE[\'why\'] if SCORE_CACHE[\'why\'] else \'\'}"\n                  f"{\' SCERR %.1e\' % SCORE_CACHE[\'err\'] if SCORE_CACHE[\'err\'] >= 0 else \'\'} "\n                  f"CK {CKPT_STATS[\'ck\']}/{CKPT_STATS[\'nock\']}/{CKPT_STATS[\'oom\']} "\n                  f"CAS {SCORE_CASCADE[\'scored\']}/{SCORE_CASCADE[\'pruned\']} "\n                  f"DB {DECODE_STATS[\'views\']/max(DECODE_STATS[\'batches\'],1):.1f} "\n                  f"ADFS {ADAPT_STATS[\'dfs_on\']}/{ADAPT_STATS[\'dfs_off\']} "\n                  + (f"ENS {ENS_STATS[\'joint\']}/{ENS_STATS[\'tasks\']} "\n                     f"ad{ENS_STATS[\'adapters\']} sk{ENS_STATS[\'skipped\']} "\n                     if TTT_ENSEMBLE > 1 else "") +\n                  # Peak VRAM is a SHIP constraint, not a curiosity: the wider\n                  # decode batch and the retained prompt cache both grow it,\n                  # and the scored run\'s card has 22 GiB. Measured here so the\n                  # headroom is a number rather than an assumption.\n                  f"mem {torch.cuda.max_memory_allocated()/2**30:.1f}G",\n                  flush=True)\n            gc.collect(); torch.cuda.empty_cache()\n    ret[rank] = out\n    # The degradation counter. Without it a run where most tasks silently took\n    # the fallback path is indistinguishable from a genuinely weak model.\n    print(f"[w{rank}] done {len(out)} tasks, {n_failed} FELL BACK "\n          f"({100.0*n_failed/max(len(out),1):.0f}%)", flush=True)\n    print(f"[w{rank}] BEAM: {BEAM_STATS[\'ran\']}/{BEAM_STATS[\'outputs\']} outputs "\n          f"gained candidates, {BEAM_STATS[\'novel\']} novel total, "\n          f"{BEAM_STATS[\'skipped\']} skipped", flush=True)\n    print(f"[w{rank}] BUDGET: {BUDGET_STATS[\'raised\']} decode batches whose "\n          f"budget would have been below a full grid under the old rule",\n          flush=True)\n    print(f"[w{rank}] TRUNC: {TRUNC_STATS[\'views\']} views needed demo dropping "\n          f"to fit the context", flush=True)\n    print(f"[w{rank}] STARVE: train_frac={ENS_TRAIN_FRAC} min_dec_batches={ENS_MIN_DECODE_BATCHES} "\n          f"{json.dumps(STARVE_STATS)}", flush=True)\n    _tot = sum(PHASE_S.values()) or 1.0\n    print(f"[w{rank}] PHASE: ttt {PHASE_S[\'ttt\']:.0f}s ({100*PHASE_S[\'ttt\']/_tot:.0f}%)  "\n          f"decode {PHASE_S[\'decode\']:.0f}s ({100*PHASE_S[\'decode\']/_tot:.0f}%)  "\n          f"score {PHASE_S[\'score\']:.0f}s ({100*PHASE_S[\'score\']/_tot:.0f}%)  "\n          f"[SCORE_VIEWS={SCORE_VIEWS}]", flush=True)\n\n\ndef worker_entry():\n    """Re-entry point when this file is launched as a shard worker."""\n    r = int(os.environ["ARC_WORKER_RANK"])\n    global DEADLINE\n    DEADLINE = float(os.environ.get("ARC_DEADLINE", DEADLINE))\n    raw = json.load(open(QUEUE_FILE if QDIR else f"_shard{r}.json"))\n    items = [(e[0], e[1]) for e in raw]\n    global WEIGHTS\n    WEIGHTS = [1.0] * len(raw) if BUDGET_FLAT_WEIGHTS else \\\n               [float(e[2]) if len(e) > 2 else 1.0 for e in raw]\n    ret = {}\n    try:\n        worker(r, os.environ["ARC_MODEL_DIR"], items, ret)\n    except Exception as e:\n        print(f"[w{r}] fatal: {type(e).__name__}: {e}", flush=True)\n    out = ret.get(r, {})\n    # Fill gaps ONLY under static sharding, where `items` is this worker\'s own\n    # partition and unanswered means unanswered. Under a SHARED QUEUE `items` is\n    # the whole task list, so this loop would emit a fallback for every task the\n    # OTHER workers solved -- and main() merges parts in rank order, so the last\n    # worker\'s fallbacks would overwrite three workers\' real predictions and the\n    # run would score like a broken model. main()\'s own missing-id pass covers\n    # the queue case and counts what it filled.\n    if not QDIR:\n        for tid, task in items:\n            if tid not in out:\n                out[tid] = [{"attempt_1": ([list(rr) for rr in x["input"]]\n                                           if is_grid(x["input"]) else [[0]]),\n                             "attempt_2": ([list(rr) for rr in x["input"]]\n                                           if is_grid(x["input"]) else [[0]])}\n                            for x in task["test"]]\n    json.dump(out, open(f"_part{r}.json", "w"))\n    print(f"[w{r}] wrote {len(out)} tasks", flush=True)\n\n\ndef main():\n    global DEADLINE\n    # ARC_FORCE_NGPU exists ONLY so the multi-worker fan-out can be exercised on\n    # a single-GPU development box: every offline A/B ever run here took the\n    # in-process branch below, while every scored run takes the shared-queue\n    # branch, so the two have never been compared. Unset (the scored run always\n    # is, since there is no env there) this is exactly torch\'s own count, and\n    # ARC_FORCE_CVD leaves the per-worker device mapping untouched.\n    ngpu = int(os.environ.get("ARC_FORCE_NGPU", 0)) or torch.cuda.device_count()\n    print("=" * 60, flush=True)\n    print(f"HARDWARE: {ngpu} GPU(s)"\n          f"{\' [FORCED]\' if os.environ.get(\'ARC_FORCE_NGPU\') else \'\'}", flush=True)\n    for i in range(min(ngpu, torch.cuda.device_count())):\n        p = torch.cuda.get_device_properties(i)\n        print(f"  [{i}] {p.name}  {p.total_memory/2**30:.1f} GiB  "\n              f"sm_{p.major}{p.minor}", flush=True)\n    print(f"torch {torch.__version__}  wall budget {WALL/3600:.1f}h", flush=True)\n    print("=" * 60, flush=True)\n\n    # Fail LOUDLY and early on a missing attachment. Offline, an unresolved\n    # source is dropped silently by the platform, and a None here would surface\n    # as an opaque TypeError deep in the run instead of naming what is missing.\n    ch_path = find_file("arc-agi_test_challenges.json")\n    assert ch_path, "challenges file not found -- competition data not attached"\n    challenges = json.load(open(ch_path))\n    print(f"challenges: {ch_path}  tasks={len(challenges)}", flush=True)\n\n    # SELF-TRAINING PROBE (offline only; PSEUDO_DEMOS is unset in every shipped\n    # build, so this block is inert on Kaggle). Appends one (test_input,\n    # predicted_output) pair to a task\'s demonstrations, so the pseudo-label is\n    # both trained on and visible in-context. The pseudo-label is the pipeline\'s\n    # own attempt_1, which is right ~28% of the time -- that is the point of the\n    # test, not an oversight.\n    _pd = os.environ.get("PSEUDO_DEMOS", "")\n    if _pd and os.path.exists(_pd):\n        extra = json.load(open(_pd))\n        n_add = 0\n        for _tid, _pairs in extra.items():\n            t = challenges.get(_tid)\n            if not t:\n                continue\n            for _in, _out in _pairs:\n                t["train"].append({"input": _in, "output": _out})\n                n_add += 1\n        print(f"PSEUDO-DEMOS: appended {n_add} pair(s) across "\n              f"{len(extra)} task(s) from {_pd}", flush=True)\n    cfg_path = find_file("config.json", roots=MODEL_ROOTS)\n    assert cfg_path, "model config.json not found -- model source not attached"\n    mdir = os.path.dirname(cfg_path)\n    cfg = json.load(open(os.path.join(mdir, "config.json")))\n    assert cfg.get("vocab_size") == 16, "expected the 16-token grid vocabulary"\n    print(f"model: {mdir}", flush=True)\n\n    # longest-first so big tasks do not all land at the end of one worker\n    items = sorted(challenges.items(),\n                   key=lambda kv: -sum(len(p["input"]) * len(p["input"][0])\n                                       for p in kv[1]["train"]))\n    placeholder, why = is_public_placeholder(challenges, ch_path)\n    if placeholder and PROBE_TASKS > 0:\n        # Deterministic spread across the size-sorted list, so the probe covers\n        # the largest tasks (the ones that OOM) as well as ordinary ones and\n        # still lands work on every worker.\n        step = max(1, len(items) // PROBE_TASKS)\n        items = items[::step][:PROBE_TASKS]\n        DEADLINE = min(DEADLINE, START + PROBE_WALL_S)\n        mode = "PROBE"\n        print(f"GATE: PLACEHOLDER -- {why}", flush=True)\n        print(f"GATE: solving {len(items)} probe tasks, the remaining "\n              f"{len(challenges) - len(items)} take the declared fallback; wall "\n              f"capped at {(DEADLINE - START) / 60:.0f}min", flush=True)\n    else:\n        mode = "FULL"\n        print(f"GATE: FULL RUN -- {why}"\n              f"{\' (gate disabled: PROBE_TASKS=0)\' if placeholder else \'\'}",\n              flush=True)\n    # Per-task cost weight, used to size each task\'s slice of the wall. Built\n    # with the SAME build_ttt the workers run, so it is exact rather than a\n    # proxy; 8.3s for 240 tasks, i.e. 0.02% of the run.\n    _t = time.time()\n    tw = {}\n    for tid, task in items:\n        try:\n            tw[tid] = float(max(1, sum(len(i) for i, _ in build_ttt(\n                task, make_views(TTT_VIEWS, np.random.default_rng(0)),\n                random.Random(0), MAX_LEN))))\n        except Exception:\n            tw[tid] = 1.0\n    _v = sorted(tw.values())\n    print(f"task cost weights: {time.time()-_t:.1f}s  p10 {_v[len(_v)//10]:,.0f}  "\n          f"p50 {_v[len(_v)//2]:,.0f}  p90 {_v[9*len(_v)//10]:,.0f}  "\n          f"spread {_v[9*len(_v)//10]/max(_v[len(_v)//10],1):.1f}x", flush=True)\n    shards = [items[i::max(ngpu, 1)] for i in range(max(ngpu, 1))]\n\n    # Fan out with SUBPROCESSES, not multiprocessing.\n    # mp.spawn has to pickle the worker function by qualified name, which fails\n    # whenever __main__ is synthetic (exec\'d source, notebook conversion) --\n    # and it fails by killing the workers, which silently yields an all-fallback\n    # submission that scores like a bad model rather than like a crash.\n    # A subprocess re-executes this same file, so nothing is pickled.\n    submission = {}\n    # A converted notebook has no __file__ and may have an argv[0] that is not\n    # this source, so probe both and tolerate either being absent -- a bare\n    # `__file__` reference would raise NameError and lose the whole run.\n    src = None\n    for cand in ((sys.argv[0] if sys.argv else None), globals().get("__file__")):\n        if cand and os.path.exists(cand):\n            src = os.path.abspath(cand)\n            break\n    print(f"self-path for worker re-exec: {src}", flush=True)\n    launched = []\n    if ngpu > 1 and src:\n        qdir = os.path.abspath("_queue")\n        os.makedirs(qdir, exist_ok=True)\n        for f in os.listdir(qdir):          # a re-run must not inherit claims\n            os.remove(os.path.join(qdir, f))\n        json.dump([[k, v, tw[k]] for k, v in items], open("_queue.json", "w"))\n        for r in range(ngpu):\n            json.dump([[k, v, tw[k]] for k, v in shards[r]], open(f"_shard{r}.json", "w"))\n            env = dict(os.environ)\n            env["ARC_QDIR"] = qdir\n            env["ARC_NWORKERS"] = str(ngpu)\n            env["CUDA_VISIBLE_DEVICES"] = os.environ.get("ARC_FORCE_CVD", (os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[r] if os.environ.get("CUDA_VISIBLE_DEVICES") else str(r)))\n            env["ARC_WORKER_RANK"] = str(r)\n            env["ARC_MODEL_DIR"] = mdir\n            env["ARC_DEADLINE"] = str(DEADLINE)\n            launched.append(subprocess.Popen(\n                [sys.executable, "-u", src], env=env))\n        for p in launched:\n            p.wait()\n        seen_from = {}\n        for r in range(ngpu):\n            fp = f"_part{r}.json"\n            if os.path.exists(fp):\n                try:\n                    part = json.load(open(fp))\n                except Exception as e:\n                    print(f"part {r} unreadable: {e}", flush=True)\n                    continue\n                for tid in part:\n                    seen_from.setdefault(tid, []).append(r)\n                submission.update(part)\n        # Under a shared queue every task is claimed by exactly one worker, so\n        # any id appearing in two parts means a later part OVERWROTE an earlier\n        # worker\'s real prediction -- the exact way a queue regression would\n        # score like a broken model rather than raise. Cheap, permanent tripwire.\n        dupes = {t: w for t, w in seen_from.items() if len(w) > 1}\n        print(f"collected {len(submission)} tasks from {ngpu} workers; "\n              f"per-worker {[sum(1 for v in seen_from.values() if v == [r]) for r in range(ngpu)]}; "\n              f"DUPLICATE-PARTS {len(dupes)}", flush=True)\n        if dupes:\n            print(f"  !! ids returned by several workers: "\n                  f"{list(dupes.items())[:5]}", flush=True)\n\n    # Single-GPU, or every worker failed: solve in-process so the run still\n    # produces real predictions instead of falling back wholesale.\n    if not submission:\n        print("running in-process (single GPU or worker launch failed)", flush=True)\n        ret = {}\n        globals()["WEIGHTS"] = ([1.0] * len(items) if BUDGET_FLAT_WEIGHTS\n                                else [tw[k] for k, _ in items])\n        worker(0, mdir, items, ret)\n        submission.update(ret.get(0, {}))\n\n    # any task a worker never returned still gets a valid fallback\n    n_missing = 0\n    for tid, task in challenges.items():\n        if tid not in submission:\n            n_missing += 1\n            submission[tid] = [\n                {"attempt_1": ([list(r) for r in x["input"]]\n                               if is_grid(x["input"]) else [[0]]),\n                 "attempt_2": ([list(r) for r in x["input"]]\n                               if is_grid(x["input"]) else [[0]])}\n                for x in task["test"]]\n\n    bad = 0\n    for tid, task in challenges.items():\n        p = submission.get(tid)\n        if not isinstance(p, list) or len(p) != len(task["test"]):\n            bad += 1; continue\n        for d in p:\n            for k in ("attempt_1", "attempt_2"):\n                if not is_grid(d.get(k)):\n                    bad += 1\n    assert bad == 0, f"{bad} malformed entries"\n    assert set(submission) == set(challenges), "task id mismatch"\n    with open("submission.json", "w") as f:\n        json.dump(submission, f)\n    print(f"wrote submission.json — mode={mode}, {len(submission)} tasks, "\n          f"{n_missing} fallback-only, {(time.time()-START)/3600:.2f}h", flush=True)\n\n\nif __name__ == "__main__":\n    if os.environ.get("ARC_WORKER_RANK") is not None:\n        worker_entry()\n    else:\n        main()\n'
DSL_SOURCE = ''  # re-arc dsl.py (Michael Hodel, MIT) when prog_dsl is on
VERIFIERS_SOURCE = ''  # re-arc verifiers.py: human-written solvers of public training tasks
ICECUBER_FILES = {}  # icecuber 2020 DSL search (top-quarks/ARC-solution, MIT): C++ sources
ICECUBER_PY = ''  # its stdlib runner (build + per-test-input search + parse)
CORPUS_PY = ''  # retrieval-by-execution runner over a solver-program corpus (corpus_exec.py)
CORPUS_PROGS = 'eNqLjgUAARUAuQ=='  # b64(zlib(json)) of demo-verified programs written for PUBLIC tasks: [{"task", "code"}]
AGENT_SOURCE = ''  # llm_agent.py (multi-turn Python-REPL agent client) when llm_task_mode == "agent"
WALL = float(CFG.get("wall_s", 12 * 3600 - 600))
DEADLINE = T0 + WALL
MODE = CFG.get("mode", "sequential")
PROBE_TASKS = int(CFG.get("probe_tasks", 8))
LLM_PROBE_WALL_S = float(CFG.get("llm_probe_wall_s", 25 * 60))
OUTP = "/kaggle/working/submission.json"
LOGD = "/kaggle/working/combo_run"
TTT_DIR = "/tmp/ttt_run"
os.makedirs(LOGD, exist_ok=True)
PLACEHOLDER_IDS_SHA256 = "0425e31f8c028fd6d6c2953fd5dbed115c065b8c657375a57f8b477c6f1d650c"
print("CFG", json.dumps(CFG), "WALL", WALL, "MODE", MODE, "TTT_SOURCE", len(TTT_SOURCE or ""), "chars", flush=True)


def sh(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"ERR {e}"


def find_file(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    return hits[0] if hits else None


print("HARDWARE", json.dumps({
    "gpus": sh("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader"),
    "mem": sh("grep -E 'MemTotal|MemAvailable' /proc/meminfo"), "nproc": os.cpu_count()}), flush=True)

# ------------------------------------------------------------------ data + placeholder gate
CH = find_file("arc-agi_test_challenges.json")
tasks = json.load(open(CH))
print("TEST", CH, len(tasks), "tasks", sum(len(t["test"]) for t in tasks.values()), "outputs", flush=True)


def is_public_placeholder(ids):
    try:
        if hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest() == PLACEHOLDER_IDS_SHA256:
            return True, "digest"
        tp = find_file("arc-agi_training_challenges.json")
        if tp:
            train = set(json.load(open(tp)))
            frac = len(set(ids) & train) / max(len(ids), 1)
            if frac >= 0.9:
                return True, f"{frac:.0%} of ids are public training tasks"
            return False, f"{frac:.0%} in training"
        return False, "no training file"
    except Exception as e:
        return False, f"gate error {e}"


placeholder, why = is_public_placeholder(list(tasks))
print("PLACEHOLDER", placeholder, why, flush=True)

# ------------------------------------------------------------------ submission state + merge
llm_ans = {}      # (task, k) -> parsed grids in arrival order
llm_forced = {}   # (task, k) -> grids from budget-forced continuations of truncated replies
prog_ver = {}     # program mode: (task, k) -> test grid of a program that reproduced EVERY demonstration
prog_unv = {}     # program mode: (task, k) -> (demos right, test grid) of the best program that failed a demo
AGENT_MODE = CFG.get("llm_task_mode", "grid") in ("agent", "agent_grid")  # the REPL agent fills the program slots
# agent_grid (llm_only): the REPL agent on every task first, then direct-grid samples on the SAME server only for
# outputs without a demo-verified agent transform; merge = the program branch (verified agent answer first)
AGENT_GRID = CFG.get("llm_task_mode", "grid") == "agent_grid"
PROG_MODE = CFG.get("llm_task_mode", "grid") in ("program", "agent", "cascade", "agent_grid")
# cascade: one program pass over every task, then direct-grid requests only for outputs with no demo-verified
# program (program jobs are (t, None, s), grid jobs (t, k, s)); merge = the program branch (verified first)
CASCADE = CFG.get("llm_task_mode", "grid") == "cascade"
CASC = CFG.get("cascade") or {}
ttt_sub = {}      # task -> [{"attempt_1", "attempt_2"}] from the TTT subprocess, once it finishes
ice_ver = {}      # icecuber: (task, k) -> top-1 candidate that reproduces EVERY demonstration
cor_ver = {}      # corpus: (task, k) -> test grid of a corpus program that reproduces EVERY demonstration
ans_lock = threading.Lock()
merge_stats = {}
nat_stop = {}     # grid mode: (task, k) -> fewest completion tokens of a reply that stopped on its own
seq_rates = []    # adaptive_tb: observed per-request decode rates (completion tokens / wall s)


def seq_rate():
    """Conservative per-sequence decode rate for the adaptive budget: the 25th percentile of the last 32
    finished requests once >= 8 exist, else the configured prior (never above 1.5x the prior)."""
    a = CFG.get("adaptive_tb") or {}
    r0 = float(a.get("rate0", 6.5))
    with ans_lock:
        xs = sorted(seq_rates[-32:])
    if len(xs) < 8:
        return r0
    return min(1.5 * r0, xs[len(xs) // 4])


def llm_ranked(t, k):
    gs = llm_ans.get((t, k), [])
    cnt = {}
    for i, g in enumerate(gs):
        cnt.setdefault(json.dumps(g), [0, i])[0] += 1
    return [json.loads(x) for x, _ in sorted(cnt.items(), key=lambda kv: (-kv[1][0], kv[1][1]))]


def ttt_attempts(t, k):
    p = ttt_sub.get(t)
    if not isinstance(p, list) or k >= len(p) or not isinstance(p[k], dict):
        return []
    echo = tasks[t]["test"][k]["input"]
    out = [p[k].get("attempt_1"), p[k].get("attempt_2")]
    return [g for g in out if isinstance(g, list) and g and g != echo]  # an echo is the TTT fallback


def merged(t, k):
    lc, tc = llm_ranked(t, k), ttt_attempts(t, k)
    fc = llm_forced.get((t, k), [])
    pos = CFG.get("forced_pos", "last")  # where a budget-forced answer ranks: first | after_ttt1 | last
    if pos == "first":
        lc = lc + [g for g in fc if g not in lc]
    rule = CFG.get("merge", "llm_first")
    pv = [prog_ver[(t, k)]] if (t, k) in prog_ver else []
    pu = [prog_unv[(t, k)][1]] if (t, k) in prog_unv else []
    upos = CFG.get("prog_unverified_pos", "last")
    near_k = int(CFG.get("prog_near_miss", 0))  # 0 = inert; k>0: a program failing <=k demos ranks after TTT1
    near = []
    if PROG_MODE and near_k > 0 and pu and upos == "last" \
            and len(tasks[t]["train"]) - prog_unv[(t, k)][0] <= near_k:
        near, pu = pu, []
    ic = [ice_ver[(t, k)]] if (t, k) in ice_ver else []
    ipos = (CFG.get("icecuber") or {}).get("pos", "after_ttt1")
    cc = [cor_ver[(t, k)]] if (t, k) in cor_ver else []
    cpos = (CFG.get("corpus") or {}).get("pos", "after_ttt1")
    if PROG_MODE:  # a demo-verified program first; an unverified one only after the TTT's first attempt
        seq = pv + lc[:1] + (cc if cpos == "before_ttt1" else []) + (ic if ipos == "before_ttt1" else []) + tc[:1] \
            + (cc if cpos == "after_ttt1" else []) + (ic if ipos == "after_ttt1" else []) \
            + (pu if upos == "after_ttt1" else []) + near + tc[1:2] + (ic if ipos == "after_ttt2" else []) \
            + (cc if cpos == "after_ttt2" else []) + lc[1:] + tc[2:]
        pu = pu if upos == "last" else []
    elif rule == "llm_first":
        seq = lc[:1] + tc[:1] + (fc[:1] if pos == "after_ttt1" else []) + lc[1:2] + tc[1:2] + lc[2:] + tc[2:]
    elif rule == "ttt_first":
        seq = tc[:1] + lc[:1] + tc[1:2] + lc[1:2] + tc[2:] + lc[2:]
    elif rule == "llm_then_ttt":
        seq = lc[:1] + tc + lc[1:]
    else:
        raise ValueError(rule)
    if pos in ("after_ttt1", "last"):
        seq = seq + fc
    seq = seq + pu
    out = []
    for g in seq:
        if g not in out:
            out.append(g)
        if len(out) == 2:
            break
    src = ("P" if pv else "") + ("C" if cc else "") + ("I" if ic else "") + ("U" if (t, k) in prog_unv else "") + ("L" if llm_ans.get((t, k)) else "") \
        + ("F" if fc else "") + ("T" if tc else "")
    echo = tasks[t]["test"][k]["input"]
    while len(out) < 2:
        out.append(out[0] if out else echo)
    return out, src or "-"


def write_submission(final=False):
    sub, stats = {}, {}
    for t, task in tasks.items():
        outs = []
        for k in range(len(task["test"])):
            (a1, a2), src = merged(t, k)
            stats[src] = stats.get(src, 0) + 1
            outs.append({"attempt_1": a1, "attempt_2": a2})
        sub[t] = outs
    tmp = OUTP + ".tmp"
    json.dump(sub, open(tmp, "w"))
    os.replace(tmp, OUTP)
    merge_stats.clear()
    merge_stats.update(stats)
    if final:
        print("SUBMISSION-WRITTEN", len(sub), "tasks  sources", json.dumps(stats), flush=True)


write_submission()

# ------------------------------------------------------------------ embedded client helpers
"""ARC-AGI-2 solver client for an OpenAI-compatible chat server (llama.cpp llama-server).

Standalone (stdlib only) so the same file runs on the cluster and inside a Kaggle
notebook. Each test input of each task is sent as an independent chat request
(`--samples` per output); the final answer grid is parsed from the message CONTENT
only (a reply cut by the token cap is UNPARSEABLE, never read out of its reasoning).

Writes:
  <out>/calls.jsonl   one row per request: task, test index, sample, parsed grid,
                      hit (when solutions are given), tokens, seconds, finish reason
  <out>/summary.json  pass@1 / pass@2 per output, token and time totals, error counts
  <out>/submission.json (optional, --submission) Kaggle-format attempts
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request

PROMPT_HEAD = (
    "You are participating in a puzzle solving competition. You are an expert at solving puzzles.\n\n"
    "Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern "
    "or transformation in the training examples that maps the input to the output, then apply that "
    "pattern to the test input to give a final output.\n\n"
    "Respond in the format of the training output examples: give the final output grid as a JSON "
    "list of lists of integers, and make it the last thing in your answer.\n\n"
)


# Grid encoding shown to the LLM: "json" = the ARC Prize harness format ([0, 0, 5] rows);
# "digits" = one row per line, digits only (about half the tokens per cell for a digit-per-token vocab).
GRID_FORMAT = os.environ.get("ARC_GRID_FORMAT", "json")
PROMPT_HEAD_DIGITS = (
    "You are participating in a puzzle solving competition. You are an expert at solving puzzles.\n\n"
    "Below is a list of input and output pairs with a pattern. Each grid is written one row per line, "
    "one digit (0-9) per cell, with no separators. Your goal is to identify the pattern "
    "or transformation in the training examples that maps the input to the output, then apply that "
    "pattern to the test input to give a final output.\n\n"
    "Respond in the format of the training output examples: give the final output grid one row per line, "
    "digits only, and make it the last thing in your answer.\n\n"
)


def grid_text(g):
    if GRID_FORMAT == "digits":
        return "\n".join("".join(str(v) for v in r) for r in g)
    return "[" + ",\n".join(json.dumps(r, separators=(",", ", ")) for r in g) + "]"


SG_MAX_OBJECTS = 30


def scene_graph(g):
    """Deterministic object list for one grid: 4-connected same-colour components over the background
    (the most frequent colour), each with colour, size, bounding box, a filled-rectangle flag and a
    shape label shared by components whose normalised masks are identical (repetition)."""
    h, w = len(g), len(g[0])
    cnt = {}
    for r in g:
        for v in r:
            cnt[v] = cnt.get(v, 0) + 1
    bg = max(cnt, key=lambda c: (cnt[c], -c))
    seen = [[False] * w for _ in range(h)]
    objs = []
    for i in range(h):
        for j in range(w):
            if seen[i][j] or g[i][j] == bg:
                continue
            col, stack, cells = g[i][j], [(i, j)], []
            seen[i][j] = True
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, b + db
                    if 0 <= x < h and 0 <= y < w and not seen[x][y] and g[x][y] == col:
                        seen[x][y] = True
                        stack.append((x, y))
            r0, r1 = min(a for a, _ in cells), max(a for a, _ in cells)
            c0, c1 = min(b for _, b in cells), max(b for _, b in cells)
            mask = tuple(sorted((a - r0, b - c0) for a, b in cells))
            objs.append((r0, c0, r1, c1, col, len(cells), mask))
    objs.sort()
    shapes, lines = {}, []
    for r0, c0, r1, c1, col, n, mask in objs:
        if mask not in shapes:
            shapes[mask] = "S%d" % (len(shapes) + 1)
    for r0, c0, r1, c1, col, n, mask in objs[:SG_MAX_OBJECTS]:
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        tag = " filled-rect" if n == bh * bw else ""
        lines.append(f"- colour {col}: {n} cells, rows {r0}-{r1}, cols {c0}-{c1} ({bh}x{bw}{tag}), shape {shapes[mask]}")
    more = f"\n- ... {len(objs) - SG_MAX_OBJECTS} more objects" if len(objs) > SG_MAX_OBJECTS else ""
    counts = ", ".join(f"{c}:{cnt[c]}" for c in sorted(cnt))
    return (f"Objects ({h}x{w} grid, background {bg}, colour counts {counts}; {len(objs)} objects, "
            f"{len(shapes)} distinct shapes):\n" + ("\n".join(lines) if lines else "- none") + more)


DIFF_MAX_CELLS = 40


def pair_diff(a, b):
    """Deterministic description of how a demo output differs from its input: changed cells (or colour-change
    counts) for equal shapes; size relation, an exact crop location or an integer scale for different shapes."""
    h, w, H, W = len(a), len(a[0]), len(b), len(b[0])
    if (h, w) == (H, W):
        ch = [(i, j, a[i][j], b[i][j]) for i in range(h) for j in range(w) if a[i][j] != b[i][j]]
        if not ch:
            return "Changes: output identical to input."
        if len(ch) <= DIFF_MAX_CELLS:
            return (f"Changes ({len(ch)} of {h * w} cells; all others unchanged): "
                    + ", ".join(f"({i},{j}) {x}->{y}" for i, j, x, y in ch))
        pc = {}
        for _, _, x, y in ch:
            pc[(x, y)] = pc.get((x, y), 0) + 1
        return (f"Changes ({len(ch)} of {h * w} cells; all others unchanged), by colour: "
                + ", ".join(f"{x}->{y} x{n}" for (x, y), n in sorted(pc.items(), key=lambda kv: -kv[1])))
    rel = f"Output is {H}x{W}, input is {h}x{w}"
    if H <= h and W <= w:
        for i in range(h - H + 1):
            for j in range(w - W + 1):
                if all(a[i + r][j:j + W] == b[r] for r in range(H)):
                    return rel + f"; output equals input rows {i}-{i + H - 1}, cols {j}-{j + W - 1}."
    if H % h == 0 and W % w == 0:
        return rel + f" ({H // h}x vertically, {W // w}x horizontally)."
    if h % H == 0 and w % W == 0:
        return rel + f" (1/{h // H} vertically, 1/{w // W} horizontally)."
    return rel + "."


def _sym_flags(g):
    """Exact symmetries of a grid: LR / UD mirror, 180-degree rotation, transpose (square only)."""
    h, w = len(g), len(g[0])
    f = []
    if all(r == r[::-1] for r in g):
        f.append("left-right mirror")
    if g == g[::-1]:
        f.append("top-bottom mirror")
    if [r[::-1] for r in g[::-1]] == g:
        f.append("180-degree rotation")
    if h == w and all(g[i][j] == g[j][i] for i in range(h) for j in range(w)):
        f.append("transpose")
    return f


def _size_rule(task):
    """Output-size relation shared by every demo, and the test output size it implies (None if no rule)."""
    tr = task["train"]
    io = [((len(p["input"]), len(p["input"][0])), (len(p["output"]), len(p["output"][0]))) for p in tr]
    if all(i == o for i, o in io):
        return "every output has the same size as its input", (lambda h, w: (h, w))
    outs = {o for _, o in io}
    if len(outs) == 1:
        o = next(iter(outs))
        return f"every output is {o[0]}x{o[1]}", (lambda h, w: o)
    if all(o[0] % i[0] == 0 and o[1] % i[1] == 0 for i, o in io):
        fs = {(o[0] // i[0], o[1] // i[1]) for i, o in io}
        if len(fs) == 1:
            a, b = next(iter(fs))
            return f"every output is {a}x taller and {b}x wider than its input", (lambda h, w: (h * a, w * b))
    if all(i[0] % o[0] == 0 and i[1] % o[1] == 0 for i, o in io):
        fs = {(i[0] // o[0], i[1] // o[1]) for i, o in io}
        if len(fs) == 1:
            a, b = next(iter(fs))
            return f"every output is 1/{a} the height and 1/{b} the width of its input", (lambda h, w: (h // a, w // b))
    return None, None


def hints2(task, test_idx):
    """Deterministic task-level facts: per-demo sizes and colour sets, output symmetries, the size rule and the
    test output size it implies."""
    lines = ["Automatic analysis (hints, may be incomplete):"]
    for i, p in enumerate(task["train"]):
        a, b = p["input"], p["output"]
        ca, cb = sorted({v for r in a for v in r}), sorted({v for r in b for v in r})
        new, gone = sorted(set(cb) - set(ca)), sorted(set(ca) - set(cb))
        sy = _sym_flags(b)
        lines.append(f"- Example {i}: input {len(a)}x{len(a[0])} colours {ca}; output {len(b)}x{len(b[0])} colours {cb}"
                     + (f"; new colours in output {new}" if new else "") + (f"; colours removed {gone}" if gone else "")
                     + (f"; output symmetric under {', '.join(sy)}" if sy else ""))
    rule, fn = _size_rule(task)
    ti = task["test"][test_idx]["input"]
    if rule:
        th, tw = fn(len(ti), len(ti[0]))
        lines.append(f"- Size rule: {rule}; if it holds, the test output is {th}x{tw}.")
    else:
        lines.append("- Size rule: no single size relation holds across the examples.")
    lines.append(f"- Test input: {len(ti)}x{len(ti[0])} colours {sorted({v for r in ti for v in r})}.")
    return "\n".join(lines)


def _bg(g):
    cnt = {}
    for r in g:
        for v in r:
            cnt[v] = cnt.get(v, 0) + 1
    return max(cnt, key=lambda c: (cnt[c], -c))


def _dihedral(g):
    """The 8 dihedral images of a grid, keyed by name."""
    t = lambda x: [list(r) for r in zip(*x)]
    r90 = lambda x: [list(r) for r in zip(*x[::-1])]
    a = [list(r) for r in g]
    out = {"identity": a, "rot90": r90(a)}
    out["rot180"] = r90(out["rot90"])
    out["rot270"] = r90(out["rot180"])
    out["flip-lr"] = [r[::-1] for r in a]
    out["flip-ud"] = a[::-1]
    out["transpose"] = t(a)
    out["anti-transpose"] = r90(r90(t(a)))
    return out


def _period_mask(g, max_mask_frac=0.35):
    """Smallest (row period, col period) under which the grid is periodic once the cells of ONE colour are
    treated as unknown (a masked region), or exactly periodic (mask None). None if no period below half size."""
    h, w = len(g), len(g[0])
    cnt = {}
    for r in g:
        for v in r:
            cnt[v] = cnt.get(v, 0) + 1
    masks = [None]
    for c in sorted(cnt):  # a mask = one colour filling >= 80% of its bounding box (a rectangular hole)
        bb = _bbox(g, c)
        if cnt[c] <= max_mask_frac * h * w and cnt[c] >= 0.8 * (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1):
            masks.append(c)
    best = None
    for m in masks:
        for py in range(1, h // 2 + 1):
            for px in range(1, w // 2 + 1):
                if best and py * px >= best[0] * best[1]:
                    break
                cls, ok = {}, True
                for i in range(h):
                    gi = g[i]
                    for j in range(w):
                        v = gi[j]
                        if v == m:
                            continue
                        key = (i % py, j % px)
                        u = cls.get(key)
                        if u is None:
                            cls[key] = v
                        elif u != v:
                            ok = False
                            break
                    if not ok:
                        break
                if ok and len(cls) == py * px and py * px >= 2 and len(set(cls.values())) >= 2 and (py < h or px < w):
                    if best is None or py * px < best[0] * best[1]:
                        best = (py, px, m)
                    break
        if best and best[2] is None:
            return best
    return best


def _bbox(g, c):
    cells = [(i, j) for i, r in enumerate(g) for j, v in enumerate(r) if v == c]
    if not cells:
        return None
    return (min(i for i, _ in cells), min(j for _, j in cells), max(i for i, _ in cells), max(j for _, j in cells))


def _sym_mask(g):
    """A mirror symmetry (left-right or top-bottom about any axis covering >= 60% of the grid, or a transpose of a
    square grid) that holds on every cell pair once ONE colour is treated as unknown; None if none holds."""
    h, w = len(g), len(g[0])
    cnt = {}
    for r in g:
        for v in r:
            cnt[v] = cnt.get(v, 0) + 1
    bgc = max(cnt, key=lambda c: (cnt[c], -c))
    masks = []
    for c in sorted(cnt):  # a mask = one colour filling >= 80% of its bounding box (a rectangular hole)
        bb = _bbox(g, c)
        if c != bgc and cnt[c] <= 0.35 * h * w and cnt[c] >= 0.8 * (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1):
            masks.append(c)
    for m in masks:
        mcells = [(i, j) for i in range(h) for j in range(w) if g[i][j] == m]
        for kind, n in (("left-right", w), ("top-bottom", h)):
            for s2 in range(n - 1, 2 * n - 2):  # mirror axis at s2 / 2: j <-> s2 - j
                lo, hi = max(0, s2 - (n - 1)), min(n - 1, s2)
                if hi - lo + 1 < 0.6 * n:
                    continue
                ok, pairs, fill = True, 0, 0
                for i in range(h if kind == "left-right" else w):
                    for j in range(lo, hi + 1):
                        jj = s2 - j
                        if jj <= j:
                            continue
                        a = g[i][j] if kind == "left-right" else g[j][i]
                        b = g[i][jj] if kind == "left-right" else g[jj][i]
                        if a == m or b == m:
                            fill += (a == m) != (b == m)
                            continue
                        pairs += a != bgc
                        if a != b:
                            ok = False
                            break
                    if not ok:
                        break
                if ok and fill and pairs >= 0.15 * h * w:
                    return kind, s2 / 2, m, fill
        if h == w:
            ok, pairs, fill = True, 0, 0
            for i in range(h):
                for j in range(i + 1, w):
                    a, b = g[i][j], g[j][i]
                    if a == m or b == m:
                        fill += (a == m) != (b == m)
                        continue
                    pairs += a != bgc
                    if a != b:
                        ok = False
                        break
                if not ok:
                    break
            if ok and fill and pairs >= 0.15 * h * w:
                return "transpose", None, m, fill
    return None


def hints3(task, test_idx):
    """Deterministic structural facts beyond hints2: which input colours are always preserved / which cells
    change (same-size tasks), crop-at-colour-bounding-box and tiling-by-symmetry relations (size-changing
    tasks), and periodic patterns with a masked colour region (pattern completion)."""
    tr, ti = task["train"], task["test"][test_idx]["input"]
    lines = ["More automatic analysis (hints, may be incomplete):"]
    same = all(len(p["input"]) == len(p["output"]) and len(p["input"][0]) == len(p["output"][0]) for p in tr)
    if same:
        cols = sorted({v for p in tr for r in p["input"] for v in r})
        kept, frm, to = [], set(), set()
        for c in cols:
            seen, pres = 0, True
            for p in tr:
                a, b = p["input"], p["output"]
                for i, r in enumerate(a):
                    for j, v in enumerate(r):
                        if v == c:
                            seen += 1
                            if b[i][j] != c:
                                pres = False
            if seen and pres:
                kept.append(c)
        for p in tr:
            a, b = p["input"], p["output"]
            for i, r in enumerate(a):
                for j, v in enumerate(r):
                    if b[i][j] != v:
                        frm.add(v)
                        to.add(b[i][j])
        if frm:
            lines.append(f"- In every example, input cells of colours {kept} are never changed; changed cells had "
                         f"input colours {sorted(frm)} and became colours {sorted(to)}.")
            tk = sum(1 for r in ti for v in r if v in kept)
            if kept and tk:
                lines.append(f"- If that holds for the test, the {tk} test-input cells of colours {kept} keep "
                             f"their colour in the test output; only cells of colours {sorted(frm)} may change.")
    else:
        crops = []
        for p in tr:
            a, b = p["input"], p["output"]
            h, w, H, W = len(a), len(a[0]), len(b), len(b[0])
            locs = []
            if H <= h and W <= w:
                for i in range(h - H + 1):
                    for j in range(w - W + 1):
                        if all(a[i + r][j:j + W] == b[r] for r in range(H)):
                            locs.append((i, j))
            crops.append(locs)
        if all(crops):
            lines.append("- In every example the output is an exact sub-grid (crop) of the input.")
            for c in sorted({v for p in tr for r in p["input"] for v in r}):
                good = None
                for mode in ("box", "inside"):
                    ok = True
                    for p, locs in zip(tr, crops):
                        bb = _bbox(p["input"], c)
                        H, W = len(p["output"]), len(p["output"][0])
                        if bb is None:
                            ok = False
                            break
                        r0, c0, r1, c1 = bb if mode == "box" else (bb[0] + 1, bb[1] + 1, bb[2] - 1, bb[3] - 1)
                        if not ((r0, c0) in locs and r1 - r0 + 1 == H and c1 - c0 + 1 == W):
                            ok = False
                            break
                    if ok:
                        good = mode
                        break
                if good:
                    bb = _bbox(ti, c)
                    where = (f"; in the test input that region is rows {bb[0] + (good == 'inside')}-"
                             f"{bb[2] - (good == 'inside')}, cols {bb[1] + (good == 'inside')}-{bb[3] - (good == 'inside')}"
                             if bb else "")
                    lines.append(f"- The crop is exactly the {'bounding box of' if good == 'box' else 'region inside the bounding box of'} "
                                 f"the colour-{c} cells in every example{where}.")
                    break
        tiles = []
        for p in tr:
            a, b = p["input"], p["output"]
            h, w, H, W = len(a), len(a[0]), len(b), len(b[0])
            if H % h or W % w or (H == h and W == w):
                tiles = None
                break
            d = _dihedral(a)
            grid = []
            for bi in range(H // h):
                row = []
                for bj in range(W // w):
                    blk = [r[bj * w:(bj + 1) * w] for r in b[bi * h:(bi + 1) * h]]
                    name = next((k for k, v in d.items() if v == blk), None)
                    row.append(name or "?")
                grid.append(row)
            tiles.append(grid)
        if tiles and all("?" not in sum(g, []) for g in tiles) and all(g == tiles[0] for g in tiles):
            lines.append("- Every output is a tiling of transformed copies of its input; block layout (rows of blocks): "
                         + " / ".join(", ".join(r) for r in tiles[0]) + ".")
    for nm, g in [(f"Example {i} input", p["input"]) for i, p in enumerate(tr)] + \
                 [(f"Example {i} output", p["output"]) for i, p in enumerate(tr)] + [("Test input", ti)]:
        pm = _period_mask(g)
        if pm:
            py, px, m = pm
            if m is None:
                lines.append(f"- {nm} is exactly periodic: it repeats every {py} rows and {px} columns.")
            else:
                bb = _bbox(g, m)
                lines.append(f"- {nm} is periodic (repeats every {py} rows and {px} columns) except the colour-{m} "
                             f"cells (rows {bb[0]}-{bb[2]}, cols {bb[1]}-{bb[3]}), which may hide part of the pattern.")
    for nm, g in [(f"Example {i} input", p["input"]) for i, p in enumerate(tr)] + [("Test input", ti)]:
        sm = _sym_mask(g)
        if sm:
            kind, ax, m, fill = sm
            axis = (f" about the axis between columns {int(ax)} and {int(ax) + 1}" if kind == "left-right" and ax != int(ax)
                    else f" about column {int(ax)}" if kind == "left-right"
                    else f" about the axis between rows {int(ax)} and {int(ax) + 1}" if kind == "top-bottom" and ax != int(ax)
                    else f" about row {int(ax)}" if kind == "top-bottom" else "")
            lines.append(f"- {nm} is {kind} symmetric{axis} except the colour-{m} cells; {fill} of those cells have a "
                         f"non-colour-{m} mirror partner (the symmetry could fill them).")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def build_prompt_aug(task, test_idx, sg=False, diff=False, h2=False, h3=False):
    """build_prompt with optional per-grid object lists (scene_graph) and per-demo change summaries (pair_diff)."""
    head = PROMPT_HEAD_DIGITS if GRID_FORMAT == "digits" else PROMPT_HEAD
    if sg:
        head += ("After each grid you also get an automatically extracted list of its objects (connected "
                 "same-colour regions); use it as a hint, but the grids are the ground truth.\n\n")
    if diff:
        head += ("After each training example you also get an automatically computed summary of how its output "
                 "differs from its input; use it as a hint, but the grids are the ground truth.\n\n")
    x = (lambda g: "\n\n" + scene_graph(g)) if sg else (lambda g: "")
    parts = [head, "--Training Examples--\n"]
    for i, p in enumerate(task["train"]):
        parts.append(f"--Example {i}--\n\nINPUT:\n\n{grid_text(p['input'])}{x(p['input'])}\n\n"
                     f"OUTPUT:\n\n{grid_text(p['output'])}{x(p['output'])}\n\n")
        if diff:
            parts.append(pair_diff(p["input"], p["output"]) + "\n\n")
    parts.append("--End of Training Examples--\n\n--Test Input--\n")
    ti = task["test"][test_idx]["input"]
    parts.append(grid_text(ti) + x(ti) + "\n--End of Test Input--\n\n")
    if h2:
        parts.append(hints2(task, test_idx) + "\n\n")
    if h3:
        _h3 = hints3(task, test_idx)
        if _h3:
            parts.append(_h3 + "\n\n")
    parts.append("Your response:")
    return "".join(parts)


REFINE_TEXT = ("A previous attempt produced this output grid for the test input:\n\n{grid}\n\nIt may contain "
               "mistakes. Re-derive the transformation rule from the training examples, check the previous output "
               "cell by cell against that rule, and reply with the corrected output grid (repeat it unchanged if it "
               "is already right).\n\nYour response:")


CONCISE_TEXT = ("Think efficiently: work out the rule from the examples, then construct the output. While "
                "reasoning, refer to cells and regions by coordinates instead of rewriting whole grids, and write the "
                "full output grid only once, at the end.\n\nYour response:")


TWO_TEXT = ("Give your best output grid as your answer. If, after your analysis, a DIFFERENT output grid is also "
            "plausible, add it after your best grid: a line 'ALTERNATIVE:' followed by that second grid. The best "
            "grid must come first.\n\nYour response:")


def parse_two(content):
    """(best, alternative) grids of a --two-answers reply: the last grid before 'ALTERNATIVE:' and the last after
    it; without the marker the whole reply's last grid and None."""
    if "ALTERNATIVE:" in content:
        head, tail = content.rsplit("ALTERNATIVE:", 1)
        b = parse_grid(head)
        a = parse_grid(tail)
        if b is None:
            return a, None
        return b, (a if a != b else None)
    return parse_grid(content), None

def build_prompt_sg(task, test_idx):
    """build_prompt with a deterministic scene graph (scene_graph) after every grid."""
    head = PROMPT_HEAD_DIGITS if GRID_FORMAT == "digits" else PROMPT_HEAD
    head += ("After each grid you also get an automatically extracted list of its objects (connected "
             "same-colour regions); use it as a hint, but the grids are the ground truth.\n\n")
    parts = [head, "--Training Examples--\n"]
    for i, p in enumerate(task["train"]):
        parts.append(f"--Example {i}--\n\nINPUT:\n\n{grid_text(p['input'])}\n\n{scene_graph(p['input'])}\n\n"
                     f"OUTPUT:\n\n{grid_text(p['output'])}\n\n{scene_graph(p['output'])}\n\n")
    parts.append("--End of Training Examples--\n\n--Test Input--\n")
    ti = task["test"][test_idx]["input"]
    parts.append(grid_text(ti) + "\n\n" + scene_graph(ti) + "\n--End of Test Input--\n\nYour response:")
    return "".join(parts)


def build_prompt(task, test_idx):
    parts = [PROMPT_HEAD_DIGITS if GRID_FORMAT == "digits" else PROMPT_HEAD, "--Training Examples--\n"]
    for i, p in enumerate(task["train"]):
        parts.append(f"--Example {i}--\n\nINPUT:\n\n{grid_text(p['input'])}\n\nOUTPUT:\n\n{grid_text(p['output'])}\n\n")
    parts.append("--End of Training Examples--\n\n--Test Input--\n")
    parts.append(grid_text(task["test"][test_idx]["input"]) + "\n--End of Test Input--\n\nYour response:")
    return "".join(parts)


# Qwen3.8 chat template, rendered by hand for raw /v1/completions continuations (budget forcing):
# a system line only for xhigh (the template default) and low; medium renders none.
QWEN38_EFFORT_SYS = {
    "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
             "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
    "low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
           "without unnecessary elaboration.",
    "medium": "",
}
FORCE_TEXT = ("\n\nConsidering the limited time, I have to give the solution based on the thinking directly now."
              "\n</think>\n\n")


def raw_prompt_qwen38(task, test_idx, effort):
    sys_line = QWEN38_EFFORT_SYS[effort or "xhigh"]
    head = f"<|im_start|>system\n{sys_line}<|im_end|>\n" if sys_line else ""
    return (head + "<|im_start|>user\n" + build_prompt(task, test_idx).strip()
            + "<|im_end|>\n<|im_start|>assistant\n<think>\n")


_LIST_RE = re.compile(r"\[\s*\[[\d,\s\[\]]*\]\s*\]", re.S)


def parse_grid(text):
    """Last well-formed rectangular 2-D int grid (values 0-9, 1..30 per side) in text."""
    if not text:
        return None
    best = None
    for m in _LIST_RE.finditer(text):
        try:
            g = json.loads(m.group(0))
        except Exception:
            continue
        if (isinstance(g, list) and g and all(isinstance(r, list) and r for r in g)
                and len({len(r) for r in g}) == 1 and len(g) <= 30 and len(g[0]) <= 30
                and all(isinstance(v, int) and 0 <= v <= 9 for r in g for v in r)):
            best = g
    if best is None and GRID_FORMAT == "digits":  # json mode stays byte-for-byte the old parser
        best = _parse_digit_block(text)
    return best


_DIGIT_LINE = re.compile(r"^\s*`*\s*([0-9]{1,30})\s*`*\s*$")


def _parse_digit_block(text):
    """Last run of consecutive digit-only lines of equal length (<=30x30) -> grid; None if absent."""
    blocks, cur = [], []
    for line in text.splitlines():
        m = _DIGIT_LINE.match(line)
        if m and (not cur or len(m.group(1)) == len(cur[0])):
            cur.append(m.group(1))
        else:
            if cur:
                blocks.append(cur)
            cur = [m.group(1)] if m else []
    if cur:
        blocks.append(cur)
    blocks = [b for b in blocks if len(b) <= 30]
    if not blocks:
        return None
    return [[int(c) for c in row] for row in blocks[-1]]


INT4_DEQ_PATCH = r"""
# --- int4_deq: dequantize selected compressed-tensors pack-quantized int4 modules to bf16 at load ---
import os as _deq_os, re as _deq_re
_DEQ_RE = _deq_re.compile(_deq_os.environ.get("INT4_DEQ_RE", r"\.linear_attn\.in_proj_[ab]$"))


def _deq_int4(packed, scale, shape):
    import torch
    out, inn = int(shape[0]), int(shape[1])
    p = packed.to(torch.int64) & 0xFFFFFFFF
    v = torch.stack([(p >> (4 * i)) & 0xF for i in range(8)], dim=-1).reshape(out, -1)[:, :inn]
    ng = scale.shape[1]
    w = (v - 8).to(torch.float32).reshape(out, ng, inn // ng) * scale.to(torch.float32).unsqueeze(-1)
    return w.reshape(out, inn).to(torch.bfloat16)


def _deq_wrap(fn):
    def it(*a, **k):
        buf = {}
        for name, t in fn(*a, **k):
            mod, _, leaf = name.rpartition(".")
            if leaf in ("weight_packed", "weight_scale", "weight_shape") and _DEQ_RE.search(mod):
                d = buf.setdefault(mod, {})
                d[leaf] = t
                if len(d) == 3:
                    _DEQ_COUNT[0] += 1
                    yield mod + ".weight", _deq_int4(d["weight_packed"], d["weight_scale"], d["weight_shape"].tolist())
                    del buf[mod]
                continue
            yield name, t
        if buf:
            raise RuntimeError("int4_deq: incomplete modules " + ",".join(sorted(buf)[:5]))
    return it


_DEQ_COUNT = [0]
for _n in ("safetensors_weights_iterator", "multi_thread_safetensors_weights_iterator"):
    if _n in globals():
        globals()[_n] = _deq_wrap(globals()[_n])
# --- end int4_deq ---
"""


def int4_prepare(src_dir, dst_dir="/tmp/int4_model", vllm_dir=None, pattern=r"\.linear_attn\.in_proj_[ab]$"):
    """Make a compressed-tensors int4 checkpoint servable when some quantized modules have no kernel (e.g. a
    96-row fused in_proj_ba on sm89): patch vLLM's safetensors iterator to dequantize modules matching
    `pattern` to bf16 at load, and point vLLM at a shadow dir (symlinks + a config whose quantization
    `ignore` list names those modules). Returns (dst_dir, number of modules ignored)."""
    import json as _j, os as _o, re as _r
    if vllm_dir:
        wu = _o.path.join(vllm_dir, "model_executor", "model_loader", "weight_utils.py")
        src = open(wu).read()
        if "--- int4_deq:" not in src:
            with open(wu, "a") as f:
                f.write("\n" + INT4_DEQ_PATCH.replace('r"\\.linear_attn\\.in_proj_[ab]$"', repr(pattern)))
    _o.makedirs(dst_dir, exist_ok=True)
    for fn in _o.listdir(src_dir):
        d = _o.path.join(dst_dir, fn)
        if fn != "config.json" and not _o.path.lexists(d):
            _o.symlink(_o.path.join(src_dir, fn), d)
    idx = _j.load(open(_o.path.join(src_dir, "model.safetensors.index.json")))["weight_map"]
    rx = _r.compile(pattern)
    mods = sorted({k[:-len(".weight_packed")] for k in idx if k.endswith(".weight_packed") and rx.search(k[:-len(".weight_packed")])})
    cfg = _j.load(open(_o.path.join(src_dir, "config.json")))
    qc = cfg["quantization_config"]
    qc["ignore"] = list(qc.get("ignore") or []) + [m for m in mods if m not in (qc.get("ignore") or [])]
    with open(_o.path.join(dst_dir, "config.json"), "w") as f:
        _j.dump(cfg, f, indent=1)
    return dst_dir, len(mods)


def post(url, payload, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--solutions", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--task-ids", default="", help="comma list; default all (after --limit/--offset)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--shuffle-seed", type=int, default=-1)
    ap.add_argument("--samples", type=int, default=1, help="requests per test output")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--min-p", type=float, default=0.01)
    ap.add_argument("--effort", default="low", help="low = thinking on, no effort prefix; high/max = DSV4 template effort prefix")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--force-answer", action="store_true",
                    help="on a truncated reply, continue raw prompt+thinking+forced </think> for an answer (Qwen3.8 template)")
    ap.add_argument("--force-max-tokens", type=int, default=4096)
    ap.add_argument("--max-model-len", type=int, default=65536, help="bounds the forced continuation")
    ap.add_argument("--grid-format", default="", help="json (ARC Prize harness, default) | digits")
    ap.add_argument("--reasoning-effort", default="", help="chat_template_kwargs.reasoning_effort (Qwen3.8: xhigh [template default] / medium / low)")
    ap.add_argument("--model-name", default="", help="OpenAI `model` field (vLLM needs its served name)")
    ap.add_argument("--top-k", type=int, default=0, help="0 = omit")
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--deadline-s", type=float, default=0, help="stop issuing new requests after this many seconds")
    ap.add_argument("--submission", action="store_true")
    ap.add_argument("--scene-graph", action="store_true", help="append a deterministic object list after every grid (build_prompt_sg)")
    ap.add_argument("--diff", action="store_true", help="append a per-demo input->output change summary (build_prompt_aug)")
    ap.add_argument("--refine-from", default="", help="calls.jsonl of an earlier run: each output whose earlier reply "
                    "parsed gets ONE refine request (the earlier grid shown, asked to check and correct it); others skipped")
    ap.add_argument("--hints2", action="store_true", help="append task-level size/colour/symmetry facts (hints2) before the answer cue")
    ap.add_argument("--save-text", default="", help="jsonl path: full user prompt, reasoning and content of every reply "
                    "that stopped on its own (for rejection-sampling fine-tuning)")
    ap.add_argument("--concise", action="store_true", help="append CONCISE_TEXT (reason without re-writing whole grids) "
                    "before the answer cue")
    ap.add_argument("--two-answers", action="store_true", help="ask for a best grid plus an optional ALTERNATIVE grid "
                    "(attempt 2); rows get grid2/hit2")
    ap.add_argument("--hints3", action="store_true", help="append structural facts (hints3: preserved colours, crop/tiling relations, masked periodic patterns)")
    args = ap.parse_args()
    global GRID_FORMAT
    if args.grid_format:
        GRID_FORMAT = args.grid_format

    os.makedirs(args.out, exist_ok=True)
    tasks = json.load(open(args.challenges))
    sols = json.load(open(args.solutions)) if args.solutions else {}
    ids = sorted(tasks)
    if args.shuffle_seed >= 0:
        random.Random(args.shuffle_seed).shuffle(ids)
    if args.task_ids:
        ids = [t for t in args.task_ids.split(",") if t]
    ids = ids[args.offset:]
    if args.limit:
        ids = ids[:args.limit]

    jobs = [(t, k, s) for t in ids for k in range(len(tasks[t]["test"])) for s in range(args.samples)]
    prev = {}
    if args.refine_from:  # first parsed, naturally stopped grid per output
        for ln in open(args.refine_from):
            r_ = json.loads(ln)
            key = (r_["task"], int(r_["test"]))
            if r_.get("grid") is not None and r_.get("finish") == "stop" and key not in prev:
                prev[key] = (r_["grid"], bool(r_.get("hit")))
        jobs = [j for j in jobs if (j[0], j[1]) in prev]
    print(f"LLMARC tasks={len(ids)} outputs={sum(len(tasks[t]['test']) for t in ids)} requests={len(jobs)} "
          f"parallel={args.parallel} max_tokens={args.max_tokens} effort={args.effort or '-'} grid_format={GRID_FORMAT}", flush=True)

    calls_f = open(os.path.join(args.out, "calls.jsonl"), "a")
    text_f = open(args.save_text, "a") if args.save_text else None
    lock = threading.Lock()
    t0 = time.time()
    counters = {"http_err": 0, "unparsed": 0, "truncated": 0, "skipped_deadline": 0, "done": 0, "hits": 0}

    def run(job):
        t, k, s = job
        if args.deadline_s and time.time() - t0 > args.deadline_s:
            with lock:
                counters["skipped_deadline"] += 1
            return None
        user = (build_prompt_aug(tasks[t], k, sg=args.scene_graph, diff=args.diff, h2=args.hints2, h3=args.hints3) if (args.diff or args.hints2 or args.hints3)
                                                     else (build_prompt_sg if args.scene_graph else build_prompt)(tasks[t], k))
        if args.refine_from:
            assert user.endswith("Your response:")
            user = user[:-len("Your response:")] + REFINE_TEXT.format(grid=grid_text(prev[(t, k)][0]))
        if args.concise:
            assert user.endswith("Your response:")
            user = user[:-len("Your response:")] + CONCISE_TEXT
        if args.two_answers:
            assert user.endswith("Your response:")
            user = user[:-len("Your response:")] + TWO_TEXT
        payload = {
            "messages": [{"role": "user", "content": user}],
            "max_tokens": args.max_tokens, "temperature": args.temperature,
            "top_p": args.top_p, "min_p": args.min_p, "seed": random.randint(0, 2**31 - 1),
        }
        if args.model_name:
            payload["model"] = args.model_name
        if args.top_k:
            payload["top_k"] = args.top_k
        kw = {"enable_thinking": not args.no_think}
        if args.effort in ("high", "max") and not args.no_think:
            kw["reasoning_effort"] = args.effort
        if args.reasoning_effort and not args.no_think:
            kw["reasoning_effort"] = args.reasoning_effort
        if kw:
            payload["chat_template_kwargs"] = kw
        ts = time.time()
        row = {"task": t, "test": k, "sample": s}
        if args.refine_from:
            row["prev_hit"] = prev[(t, k)][1]
        try:
            r = post(args.url, payload, args.timeout)
            ch = r["choices"][0]
            msg = ch.get("message", {})
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""  # vLLM >=0.1x renamed the field
            fin = ch.get("finish_reason")
            g = parse_grid(content) if fin != "length" else None
            if args.two_answers and fin != "length":
                g, g2 = parse_two(content)
                row["grid2"] = g2
            u = r.get("usage", {})
            tm = r.get("timings", {})
            if text_f is not None and fin == "stop":
                row["_text"] = {"prompt": user, "reasoning": reasoning, "content": content}
            row.update(finish=fin, grid=g, n_content=len(content), n_reasoning=len(reasoning),
                       prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                       gen_tps=tm.get("predicted_per_second"), pp_tps=tm.get("prompt_per_second"),
                       content_tail=content[-300:])
            if fin == "length":
                with lock:
                    counters["truncated"] += 1
                if args.force_answer and reasoning:
                    room = args.max_model_len - (u.get("prompt_tokens") or 0) - (u.get("completion_tokens") or 0) - 64
                    fp = {"prompt": raw_prompt_qwen38(tasks[t], k, args.reasoning_effort) + reasoning + FORCE_TEXT,
                          "max_tokens": max(256, min(args.force_max_tokens, room)), "temperature": args.temperature,
                          "top_p": args.top_p, "min_p": args.min_p}
                    if args.model_name:
                        fp["model"] = args.model_name
                    if args.top_k:
                        fp["top_k"] = args.top_k
                    try:
                        r2 = post(args.url.replace("/chat/completions", "/completions"), fp, args.timeout)
                        c2 = r2["choices"][0]
                        txt = c2.get("text") or ""
                        fg = parse_grid(txt.split("</think>")[-1])
                        row.update(forced_grid=fg, forced_finish=c2.get("finish_reason"),
                                   forced_tokens=(r2.get("usage") or {}).get("completion_tokens"),
                                   forced_prompt_tokens=(r2.get("usage") or {}).get("prompt_tokens"),
                                   forced_tail=txt[-200:])
                        with lock:
                            counters["forced"] = counters.get("forced", 0) + 1
                            counters["forced_parsed"] = counters.get("forced_parsed", 0) + (fg is not None)
                    except Exception as e:  # noqa: BLE001 - counted
                        row["forced_error"] = f"{type(e).__name__}: {str(e)[:200]}"
                        with lock:
                            counters["forced_err"] = counters.get("forced_err", 0) + 1
            elif g is None:
                with lock:
                    counters["unparsed"] += 1
        except Exception as e:  # noqa: BLE001 - counted and reported, never silent
            row.update(error=f"{type(e).__name__}: {str(e)[:300]}", grid=None)
            with lock:
                counters["http_err"] += 1
        row["sec"] = round(time.time() - ts, 1)
        if sols and row.get("grid") is not None:
            row["hit"] = row["grid"] == sols[t][k]
        elif sols:
            row["hit"] = False
        if sols and args.two_answers:
            row["hit2"] = row.get("grid2") is not None and row["grid2"] == sols[t][k]
            row["hit_any"] = bool(row.get("hit")) or row["hit2"]
        if args.refine_from:
            row["changed"] = row.get("grid") is not None and row["grid"] != prev[(t, k)][0]
        if sols and "forced_grid" in row:
            row["forced_hit"] = row["forced_grid"] == sols[t][k]
            with lock:
                counters["forced_hits"] = counters.get("forced_hits", 0) + bool(row["forced_hit"])
        txt = row.pop("_text", None)
        with lock:
            counters["done"] += 1
            counters["hits"] += 1 if row.get("hit") else 0
            if txt is not None:
                text_f.write(json.dumps({"task": t, "test": k, "sample": s, "hit": row.get("hit"),
                                         "completion_tokens": row.get("completion_tokens"), **txt}) + "\n")
                text_f.flush()
            calls_f.write(json.dumps(row) + "\n")
            calls_f.flush()
            n = counters["done"]
            if n % max(1, len(jobs) // 20) == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"PROGRESS {n}/{len(jobs)} el={el:.0f}s hits={counters['hits']} "
                      f"err={counters['http_err']} trunc={counters['truncated']} unparsed={counters['unparsed']}",
                      flush=True)
        return row

    rows = []
    with cf.ThreadPoolExecutor(args.parallel) as ex:
        for r in ex.map(run, jobs):
            if r is not None:
                rows.append(r)
    calls_f.close()
    if text_f is not None:
        text_f.close()
    wall = time.time() - t0

    # per-output aggregation
    by = {}
    for r in rows:
        by.setdefault((r["task"], r["test"]), []).append(r)
    n_out = sum(len(tasks[t]["test"]) for t in ids)
    p1 = p2 = 0.0
    anyk = 0
    for key, rs in by.items():
        hits = [bool(r.get("hit")) for r in rs]
        n = len(hits)
        c = sum(hits)
        p1 += c / n if n else 0
        # unbiased pass@2 from n samples
        if n >= 2:
            p2 += 1 - ((n - c) * (n - c - 1)) / (n * (n - 1))
        else:
            p2 += c / max(n, 1)
        anyk += 1 if c else 0
    comp = [r.get("completion_tokens") or 0 for r in rows]
    summ = {
        "n_tasks": len(ids), "n_outputs": n_out, "n_requests": len(jobs), "n_rows": len(rows),
        "pass1_pct": round(100 * p1 / max(n_out, 1), 3), "pass2_pct": round(100 * p2 / max(n_out, 1), 3),
        "anyk_pct": round(100 * anyk / max(n_out, 1), 3),
        "completion_tokens_total": sum(comp), "completion_tokens_mean": round(sum(comp) / max(len(comp), 1), 1),
        "completion_tokens_max": max(comp) if comp else 0,
        "wall_s": round(wall, 1), "agg_tok_per_s": round(sum(comp) / max(wall, 1e-9), 2),
        "gen_tps_mean": round(sum((r.get("gen_tps") or 0) for r in rows) / max(len(rows), 1), 2),
        "counters": counters, "args": vars(args),
    }
    json.dump(summ, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print("SUMMARY " + json.dumps({k: v for k, v in summ.items() if k != "args"}), flush=True)

    if args.submission:
        sub = {}
        for t in ids:
            outs = []
            for k in range(len(tasks[t]["test"])):
                gs = [r["grid"] for r in by.get((t, k), []) if r.get("grid") is not None]
                a1 = gs[0] if gs else [[0]]
                a2 = next((g for g in gs[1:] if g != a1), a1)
                outs.append({"attempt_1": a1, "attempt_2": a2})
            sub[t] = outs
        json.dump(sub, open(os.path.join(args.out, "submission.json"), "w"))



GRID_FORMAT = CFG.get("grid_format", "json")  # prompt grid encoding (json = ARC harness; digits = one row per line)
print("GRID-FORMAT", GRID_FORMAT, flush=True)

# ------------------------------------------------------------------ program mode helpers
PROG_HEAD = (
    "You are participating in a puzzle solving competition. You are an expert at solving puzzles "
    "and at writing Python.\n\n"
    "Below is a list of input and output pairs that share ONE hidden transformation rule, followed "
    "by the test input(s) the same rule must be applied to. Grids are lists of lists of integers "
    "0-9 (colours).\n\n")
PROG_TAIL = (
    "Work out the rule, then write a Python function\n\n"
    "def transform(grid: list[list[int]]) -> list[list[int]]:\n\n"
    "that implements it. Your function will be executed on every training input and must reproduce "
    "each training output EXACTLY; it will then be run on the test input(s). It must work for any "
    "input of this puzzle, so do not hard-code the examples. You may use the standard library and "
    "numpy. Give the complete function (with its imports and any helpers) in a single ```python code "
    "block as the last thing in your answer.")
PROG_RUNNER = r'''
import json, sys, resource
try:
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
except Exception:
    pass
d = json.loads(sys.stdin.read())
out = []
if d.get("dsl"):  # the human-authored DSL importable as `dsl`
    import types
    _m = types.ModuleType("dsl")
    exec(compile(d["dsl"], "dsl.py", "exec"), _m.__dict__)
    sys.modules["dsl"] = _m
env = {"__name__": "__cand__"}
try:
    exec(compile(d["src"], "<cand>", "exec"), env)
    fn = env.get("transform")
    if fn is None:
        raise NameError("no function named transform")
except BaseException as e:
    print(json.dumps({"load_error": (type(e).__name__ + ": " + str(e))[:300]}))
    sys.exit(0)
for g in d["inputs"]:
    try:
        r = fn([list(row) for row in g])
        if hasattr(r, "tolist"):
            r = r.tolist()
        r = [[int(v) for v in (row.tolist() if hasattr(row, "tolist") else row)] for row in r]
        ok = (0 < len(r) <= 30 and 0 < len(r[0]) <= 30 and len({len(x) for x in r}) == 1
              and all(0 <= v <= 9 for x in r for v in x))
        out.append(r if ok else "invalid grid")
    except BaseException as e:
        out.append((type(e).__name__ + ": " + str(e))[:300])
print(json.dumps({"outs": out}))
'''
_CODE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)


def _pgrid(g):
    return "[" + ",\n".join(json.dumps(r, separators=(",", ", ")) for r in g) + "]"


PROG_HINT_TEXT = ("A different solver (a pattern-matching model, NOT a program) predicted the following output for "
                  "this test input. It is often close to correct but may contain errors, so use it only as a clue "
                  "and trust the training examples:\n")


DSL_HEAD = ("You may (optionally) use the human-written ARC DSL below: `from dsl import *` inside your code. Its "
            "Grid type is a tuple of tuples, so convert first with `I = tuple(tuple(r) for r in grid)`; objects are "
            "frozensets of (colour, (row, col)) cells, indices are frozensets of (row, col). Returning a tuple "
            "of tuples is fine. Plain Python/numpy is equally fine.\n\n--DSL functions (name(args) -> type  # "
            "what it does)--\n")
DSL_DIGEST = [None]
DSL_LOCK = threading.Lock()
DSL_EXAMPLES = []  # (feature vector, task id, human solver source)


def _task_feats(task):
    """Crude task descriptor for retrieving similar human solvers."""
    tr = task["train"]
    same = all(len(p["input"]) == len(p["output"]) and len(p["input"][0]) == len(p["output"][0]) for p in tr)
    rh = sum(len(p["output"]) / len(p["input"]) for p in tr) / len(tr)
    rw = sum(len(p["output"][0]) / len(p["input"][0]) for p in tr) / len(tr)
    ci = sum(len({v for r in p["input"] for v in r}) for p in tr) / len(tr)
    co = sum(len({v for r in p["output"] for v in r}) for p in tr) / len(tr)
    newc = sum(bool({v for r in p["output"] for v in r} - {v for r in p["input"] for v in r}) for p in tr) / len(tr)
    chg = 0.0
    if same:
        chg = sum(sum(a != b for ra, rb in zip(p["input"], p["output"]) for a, b in zip(ra, rb))
                  / (len(p["input"]) * len(p["input"][0])) for p in tr) / len(tr)
    return (float(same) * 2, min(rh, 4), min(rw, 4), ci / 3, co / 3, newc, chg * 3)


def _dsl_init():
    import ast
    tr_ch = {}
    try:
        tp = find_file("arc-agi_training_challenges.json")
        tr_ch = json.load(open(tp)) if tp else {}
    except Exception as e:  # counted: retrieval degrades to the digest alone
        counters["dsl_train_err"] = counters.get("dsl_train_err", 0) + 1
        print("DSL-TRAIN-ERR", e, flush=True)
    lines = []
    mod = ast.parse(DSL_SOURCE)
    for f in mod.body:
        if isinstance(f, ast.FunctionDef):
            args = ", ".join(a.arg for a in f.args.args)
            ret = " -> " + ast.unparse(f.returns) if f.returns else ""
            doc = (ast.get_docstring(f) or "").split("\n")[0].strip()
            lines.append(f"{f.name}({args}){ret}  # {doc}")
    consts = [n.targets[0].id for n in mod.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id.isupper()]
    for f in ast.parse(VERIFIERS_SOURCE).body:
        if isinstance(f, ast.FunctionDef) and f.name.startswith("verify_") and f.name[7:] in tr_ch:
            tid = f.name[7:]
            body = ast.get_source_segment(VERIFIERS_SOURCE, f).replace(
                f"def verify_{tid}(I: Grid) -> Grid:", "def solve(I):")
            DSL_EXAMPLES.append((_task_feats(tr_ch[tid]), tid, body))
    DSL_DIGEST[0] = DSL_HEAD + "\n".join(lines) + "\nConstants: " + ", ".join(consts) + "\n--End of DSL--\n\n"
    if not CFG.get("prog_dsl_digest", True):  # light form: the import note only; retrieved solvers show the API
        DSL_DIGEST[0] = DSL_HEAD.split("\n\n--DSL functions")[0].replace(
            "the human-written ARC DSL below", "Michael Hodel's human-written ARC DSL (arc-dsl; the solutions "
            "below show it in use)") + "\n\n"
    print("DSL-READY digest", len(DSL_DIGEST[0]), "chars, solvers", len(DSL_EXAMPLES), flush=True)


def dsl_prompt(task, t=None):
    """DSL digest + the k nearest human solvers (never the task's own) -- '' unless prog_dsl is on."""
    if not CFG.get("prog_dsl") or not DSL_SOURCE:
        return ""
    with DSL_LOCK:  # 64 request threads arrive at once: build the digest + retrieval index exactly once
        if DSL_DIGEST[0] is None:
            _dsl_init()
    k = int(CFG.get("prog_dsl_k", 2))
    ex = ""
    if DSL_EXAMPLES and k > 0:
        q = _task_feats(task)
        near = sorted((e for e in DSL_EXAMPLES if e[1] != t), key=lambda e: sum(abs(a - b) for a, b in zip(e[0], q)))[:k]
        ex = ("--Human-written DSL solutions of some OTHER, possibly related puzzles (for style; each maps the input "
              "Grid I to the output Grid)--\n" + "\n\n".join(e[2] for e in near) + "\n--End of DSL solutions--\n\n")
    return DSL_DIGEST[0] + ex


IMG_OFF = [False]  # set when image requests keep failing (4xx): later requests go text-only
ARC_RGB = [(0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0), (170, 170, 170),
           (240, 18, 190), (255, 133, 27), (127, 219, 255), (135, 12, 37)]
IMG_NOTE = ("Every grid is also shown as an image right after its list form: one coloured square per cell "
            "(0 black, 1 blue, 2 red, 3 green, 4 yellow, 5 grey, 6 magenta, 7 orange, 8 light blue, 9 maroon), "
            "separated by thin grey grid lines. The lists are authoritative.\n\n")


def grid_png(g, min_side=256, max_cell=48, min_cell=16):
    """Pure-python PNG (no PIL): integer cell size, 1-px grey lines between cells."""
    H, W = len(g), len(g[0])
    cs = max(min_cell, min(max_cell, -(-min_side // max(H, W))))
    line = (80, 80, 80)
    ph, pw = H * cs + 1, W * cs + 1
    raw = bytearray()
    for y in range(ph):
        raw.append(0)
        on_h = y % cs == 0
        i = min(y // cs, H - 1)
        for x in range(pw):
            if on_h or x % cs == 0:
                raw.extend(line)
            else:
                raw.extend(ARC_RGB[g[i][min(x // cs, W - 1)] % 10])

    def chunk(tag, d):
        return struct.pack(">I", len(d)) + tag + d + struct.pack(">I", zlib.crc32(tag + d) & 0xffffffff)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", pw, ph, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def build_content_prog_images(task, t=None):
    """build_prompt_prog as OpenAI multimodal parts: an image after every grid (vision-tower ladder steps)."""
    parts = []

    def txt(x):
        if parts and parts[-1]["type"] == "text":
            parts[-1]["text"] += x
        else:
            parts.append({"type": "text", "text": x})

    def img(g):
        parts.append({"type": "image_url", "image_url": {"url": grid_png(g)}})
    txt(PROG_HEAD + dsl_prompt(task, t) + IMG_NOTE + "--Training Examples--\n")
    for i, p in enumerate(task["train"]):
        txt(f"--Example {i}--\n\nINPUT:\n\n{_pgrid(p['input'])}\n\n")
        img(p["input"])
        txt(f"\n\nOUTPUT:\n\n{_pgrid(p['output'])}\n\n")
        img(p["output"])
        txt("\n\n")
    txt("--End of Training Examples--\n\n")
    for k, p in enumerate(task["test"]):
        txt(f"--Test Input {k}--\n{_pgrid(p['input'])}\n\n")
        img(p["input"])
        txt("\n\n")
        if t is not None and CFG.get("prog_hint") == "ttt1":
            h = ttt_attempts(t, k)
            if h:
                txt(f"--Other Solver's Prediction for Test Input {k}--\n{PROG_HINT_TEXT}{_pgrid(h[0])}\n\n")
                counters["prog_hinted"] = counters.get("prog_hinted", 0) + 1
    txt("--End of Test Inputs--\n\n" + PROG_TAIL)
    return parts


def build_prompt_prog(task, t=None):
    parts = [PROG_HEAD, dsl_prompt(task, t), "--Training Examples--\n"]
    for i, p in enumerate(task["train"]):
        parts.append(f"--Example {i}--\n\nINPUT:\n\n{_pgrid(p['input'])}\n\nOUTPUT:\n\n{_pgrid(p['output'])}\n\n")
    parts.append("--End of Training Examples--\n\n")
    for k, p in enumerate(task["test"]):
        parts.append(f"--Test Input {k}--\n{_pgrid(p['input'])}\n\n")
        if t is not None and CFG.get("prog_hint") == "ttt1":  # E284: the TTT's attempt_1 as a clue (TTT runs first)
            h = ttt_attempts(t, k)
            if h:
                parts.append(f"--Other Solver's Prediction for Test Input {k}--\n{PROG_HINT_TEXT}{_pgrid(h[0])}\n\n")
                counters["prog_hinted"] = counters.get("prog_hinted", 0) + 1
    parts.append("--End of Test Inputs--\n\n" + PROG_TAIL)
    return "".join(parts)


def extract_code(text):
    if not text:
        return None
    blocks = _CODE_RE.findall(text)
    tail = text.rsplit("```", 1)
    if len(tail) == 2 and text.count("```") % 2 == 1:
        blocks.append(re.sub(r"^(?:python|py)?[ \t]*\n", "", tail[1]))
    for b in reversed(blocks):
        if "def transform" in b:
            return b
    return None


def run_program(src, inputs, timeout):
    try:
        p = subprocess.run([sys.executable, "-c", PROG_RUNNER], input=json.dumps({"src": src, "inputs": inputs,
                                                                      "dsl": DSL_SOURCE if CFG.get("prog_dsl") else ""}),
                           capture_output=True, text=True, timeout=timeout)
        line = (p.stdout.strip().splitlines() or [""])[-1]
        return json.loads(line) if line else {"load_error": "no output"}
    except subprocess.TimeoutExpired:
        return {"load_error": "timeout"}
    except Exception as e:  # counted by the caller
        return {"load_error": f"{type(e).__name__}: {str(e)[:120]}"}


def prog_take(t, code, row):
    """Execute a candidate program on the demos; store its test outputs as verified / best-unverified."""
    task = tasks[t]
    tmo = float(CFG.get("exec_timeout", 20))
    d = run_program(code, [p["input"] for p in task["train"]], tmo)
    if "outs" not in d:
        row["prog_err"] = d.get("load_error", "?")[:120]
        with ans_lock:
            counters["prog_load_err"] = counters.get("prog_load_err", 0) + 1
        return
    n_right = sum(o == p["output"] for o, p in zip(d["outs"], task["train"]))
    row["n_right"] = n_right
    ntr = len(task["train"])
    if n_right < ntr and n_right == 0:
        return
    dt = run_program(code, [p["input"] for p in task["test"]], tmo)
    grids = [o if isinstance(o, list) else None for o in dt.get("outs", [])]
    with ans_lock:
        counters["prog_ran"] = counters.get("prog_ran", 0) + 1
        if n_right == ntr:
            counters["prog_verified"] = counters.get("prog_verified", 0) + 1
        for k, g in enumerate(grids):
            if g is None:
                continue
            if n_right == ntr:
                prog_ver.setdefault((t, k), g)
            elif (t, k) not in prog_unv or n_right > prog_unv[(t, k)][0]:
                prog_unv[(t, k)] = (n_right, g)
    row["verified"] = n_right == ntr

# ------------------------------------------------------------------ TTT subprocess
ttt_proc = [None]


def start_ttt(gpus, wall_s):
    if not TTT_SOURCE:
        print("TTT-SKIP no source embedded", flush=True)
        return
    os.makedirs(TTT_DIR, exist_ok=True)
    src = os.path.join(TTT_DIR, "ttt_solver.py")
    open(src, "w").write(TTT_SOURCE)
    env = dict(os.environ)
    env["WALL_S"] = str(max(120, int(wall_s)))
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    # the solver takes the FIRST config.json under its model roots; with an LLM attached that can be the
    # LLM's. Point it at the 16-token grid checkpoint explicitly.
    grid_cfgs = []
    for c in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        try:
            if json.load(open(c)).get("vocab_size") == 16:
                grid_cfgs.append(c)
        except Exception:
            pass
    if grid_cfgs:
        env["ARC_MODEL_ROOTS"] = os.path.dirname(grid_cfgs[0])
    print("TTT-MODEL", grid_cfgs[:2], flush=True)
    env.update({k: str(v) for k, v in CFG.get("ttt_env", {}).items()})
    print(f"TTT-START gpus={gpus or 'all'} WALL_S={int(wall_s)} t={time.time() - T0:.0f}s", flush=True)
    ttt_proc[0] = subprocess.Popen([sys.executable, "-u", src], cwd=TTT_DIR, env=env, start_new_session=True)


def finish_ttt(wait_until=None):
    p = ttt_proc[0]
    if p is None:
        return
    try:
        p.wait(timeout=max(1, (wait_until or DEADLINE + 60) - time.time()))
    except subprocess.TimeoutExpired:
        print("TTT-TIMEOUT killing its process group", flush=True)
        import signal
        try:
            os.killpg(p.pid, signal.SIGKILL)  # the solver's per-GPU workers hold VRAM otherwise
        except Exception:
            p.kill()
        try:
            p.wait(timeout=60)
        except Exception:
            pass
    fp = os.path.join(TTT_DIR, "submission.json")
    try:
        s = json.load(open(fp))
        with ans_lock:
            ttt_sub.update(s)
        n = sum(1 for t in tasks for k in range(len(tasks[t]["test"])) if ttt_attempts(t, k))
        print(f"TTT-DONE rc={p.returncode} tasks={len(s)} outputs_with_attempts={n} t={time.time() - T0:.0f}s", flush=True)
    except Exception as e:
        print(f"TTT-NO-OUTPUT rc={p.returncode} {type(e).__name__}: {e}", flush=True)


# ------------------------------------------------------------------ llama.cpp server
def stage_bin_dir(src_dir, dst_dir):
    import shutil
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    for f in os.listdir(dst_dir):
        fp = os.path.join(dst_dir, f)
        if os.path.isfile(fp):
            os.chmod(fp, 0o755)
        m = re.match(r"^(lib[\w+-]+\.so)((?:\.\d+)+)$", f)
        if m:
            base, ver = m.group(1), m.group(2).split(".")[1:]
            for i in range(len(ver)):
                ln = os.path.join(dst_dir, base + "".join("." + v for v in ver[:i]))
                if not os.path.exists(ln):
                    os.symlink(f, ln)
    return dst_dir


def find_server():
    cands = sorted(glob.glob("/kaggle/input/**/src_build/llama-server", recursive=True))
    for i, p in enumerate(cands):
        d = stage_bin_dir(os.path.dirname(p), f"/tmp/llbin{i}")
        envv = dict(os.environ, LD_LIBRARY_PATH=d + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
        out = sh(f"LD_LIBRARY_PATH={d}:$LD_LIBRARY_PATH {d}/llama-server --version")
        print("TRY-BIN", p, out[-200:].replace("\n", " | "), flush=True)
        if "version" in out.lower():
            return os.path.join(d, "llama-server"), envv
    return None, None


SERVER_KIND = CFG.get("server", "llamacpp")
PORT = 8080
URL = f"http://127.0.0.1:{PORT}"
counters = {"requests": 0, "parsed": 0, "truncated": 0, "unparsed": 0, "http_err": 0, "skipped_time": 0,
            "server_launch_fail": 0, "pip_fail": 0, "fp8_table_fail": 0}


def libcuda_stub():
    """flashinfer JIT links with -lcuda; the image has only the driver's libcuda.so.1, so ld fails
    ('cannot find -lcuda'). Point LIBRARY_PATH at a dir holding a libcuda.so symlink."""
    d = "/tmp/libcuda_stub"
    os.makedirs(d, exist_ok=True)
    cands = re.findall(r"=>\s*(\S*libcuda\.so\.1)", sh("ldconfig -p | grep libcuda.so.1"))
    cands += sorted(glob.glob("/usr/lib/x86_64-linux-gnu/libcuda.so.1") + glob.glob("/usr/local/nvidia/lib64/libcuda.so.1"))
    if cands and not os.path.exists(f"{d}/libcuda.so"):
        os.symlink(cands[0], f"{d}/libcuda.so")
    print("LIBCUDA-STUB", cands[:2], os.path.exists(f"{d}/libcuda.so"), flush=True)
    return d


# a dead engine can leave the API process hanging for the whole load_timeout: stop waiting on these
FATAL = ("Engine core initialization failed", "WorkerProc initialization failed", "Ninja build failed",
         "CUDA out of memory", "CUDA error: out of memory", "Worker failed with error")


RT_ENV = {}


def setup_image_runtime():
    """Image-layer runtime (adapted from lordhansolo/taaf-kaggle-source, Apache-2.0): a vLLM nightly Docker image's
    layers shipped as a dataset are unpacked under /tmp and run with this notebook's python 3.12 on PYTHONPATH.
    Fills RT_ENV (the server's environment); True when `import vllm` works under it."""
    import shutil
    from pathlib import Path
    rcfg = CFG["image_runtime"]
    man_p = sorted(glob.glob(rcfg.get("manifest_glob", "/kaggle/input/**/runtime-manifest.json"), recursive=True))
    print("RUNTIME-MANIFEST", man_p[:2], flush=True)
    RT_DS = Path(man_p[0]).parent
    print("DISK", sh("df -h / /tmp /kaggle/working /dev/shm 2>&1"), flush=True)

    def _free_gb(d):
        try:
            os.makedirs(d, exist_ok=True)
            st = os.statvfs(d)
            return st.f_bavail * st.f_frsize / 2**30
        except Exception:
            return -1
    _cands = [rcfg.get("root", "/tmp/vllm-image-runtime"), "/kaggle/tmp/vllm-image-runtime", "/kaggle/working/vllm-image-runtime"]
    _ok = [c for c in _cands if _free_gb(os.path.dirname(c)) >= float(rcfg.get("need_gb", 21))]
    RT_ROOT = Path(_ok[0] if _ok else max(_cands, key=lambda c: _free_gb(os.path.dirname(c))))
    print("RUNTIME-ROOT", RT_ROOT, {c: round(_free_gb(os.path.dirname(c)), 1) for c in _cands}, flush=True)
    man = json.loads((RT_DS / "runtime-manifest.json").read_text())
    print("RUNTIME", {k: man.get(k) for k in ("image", "vllm_version", "torch_version", "dist_packages", "nvcc")}, flush=True)
    shutil.rmtree(RT_ROOT, ignore_errors=True)
    RT_ROOT.mkdir(parents=True, exist_ok=True)
    t = time.time()
    dec = ["--use-compress-program=pigz"] if shutil.which("pigz") else ["-z"]
    for layer in sorted(man["selected_layers"], key=lambda l: int(l["index"])):
        blob = RT_DS / layer["file"]
        assert blob.stat().st_size == int(layer["size"]), f"layer size mismatch {blob}"
        for w in layer.get("whiteouts", []):
            if w["kind"] == "opaque" and w["target"] not in ("", "."):
                shutil.rmtree(RT_ROOT / w["target"], ignore_errors=True)
        subprocess.run(["tar", "-xf", str(blob), "-C", str(RT_ROOT), "--no-same-owner", *dec], check=True)
        for w in layer.get("whiteouts", []):
            (RT_ROOT / w["marker"]).unlink(missing_ok=True)
            if w["kind"] == "file":
                tg = RT_ROOT / w["target"]
                if tg.is_dir() and not tg.is_symlink():
                    shutil.rmtree(tg)
                else:
                    tg.unlink(missing_ok=True)
        print(f"LAYER {layer['index']} {time.time() - t:.0f}s", flush=True)
    dist = RT_ROOT / man["dist_packages"]
    if man.get("patch") and rcfg.get("apply_patch", True):
        r = subprocess.run([sys.executable, str(RT_DS / man["patch"]["applier"]), str(dist)], capture_output=True, text=True)
        print("RUNTIME-PATCH rc", r.returncode, r.stdout[-800:], r.stderr[-1500:], flush=True)
    for tp in rcfg.get("text_patches", []):  # exact-text source edits under dist-packages (counted, non-fatal)
        try:
            fp = dist / tp["file"]
            src = fp.read_text()
            n = src.count(tp["old"])
            if n == 1:
                fp.write_text(src.replace(tp["old"], tp["new"]))
                for pyc in (fp.parent / "__pycache__").glob(fp.stem + ".*.pyc"):
                    pyc.unlink(missing_ok=True)
            print("TEXT-PATCH", tp["file"], "applied" if n == 1 else f"MISS count={n}", flush=True)
        except Exception as e:
            print("TEXT-PATCH-FAIL", tp.get("file"), repr(e)[:300], flush=True)
    cuda_home = RT_ROOT / Path(man["nvcc"]).parent.parent
    libs = [cuda_home / "targets" / "x86_64-linux" / "lib", dist / "torch" / "lib"] + sorted(dist.glob("nvidia/*/lib")) \
        + sorted(dist.glob("nvidia/cu13/*/lib")) + [Path("/usr/local/nvidia/lib64")]
    cache = Path(str(RT_ROOT) + "-cache")
    RT_ENV.clear()
    RT_ENV.update(os.environ)
    RT_ENV.pop("PYTORCH_ALLOC_CONF", None)
    RT_ENV["PYTHONPATH"] = os.pathsep.join([str(dist / "nvidia_cutlass_dsl" / "dsl_packages"), str(dist)]
                                           + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    RT_ENV["PATH"] = os.pathsep.join([str(cuda_home / "bin"), str(RT_ROOT / "usr" / "local" / "bin"), os.environ.get("PATH", "")])
    RT_ENV["LD_LIBRARY_PATH"] = os.pathsep.join([str(x) for x in libs if x.is_dir()] + [os.environ.get("LD_LIBRARY_PATH", "")])
    for k, sub in (("VLLM_CACHE_ROOT", "vllm"), ("TRITON_CACHE_DIR", "triton"), ("FLASHINFER_WORKSPACE_BASE", "flashinfer"),
                   ("TMPDIR", "tmp")):
        (cache / sub).mkdir(parents=True, exist_ok=True)
        RT_ENV[k] = str(cache / sub)
    RT_ENV.update({"USE_TF": "0", "TRANSFORMERS_NO_TF": "1", "TRANSFORMERS_NO_TORCHVISION": "1", "VLLM_NO_USAGE_STATS": "1",
                   "CUDA_HOME": str(cuda_home), "CUDACXX": str(cuda_home / "bin" / "nvcc"), "TORCH_CUDA_ARCH_LIST": "8.9",
                   "VLLM_ENABLE_CUDA_COMPATIBILITY": "0", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
    RT_ENV.update(rcfg.get("env", {}))
    print("RUNTIME-READY", f"{time.time() - t:.0f}s", sh("df -h /tmp | tail -1"), flush=True)
    _r = subprocess.run([sys.executable, "-c", "import vllm, torch; print('vllm', vllm.__version__, torch.__version__, "
                         "torch.cuda.is_available())"], capture_output=True, text=True, env=RT_ENV, timeout=600)
    ver = (_r.stdout + _r.stderr).strip()[-1500:]
    print("VERSIONS", ver, flush=True)
    return "vllm " in _r.stdout


def install_vllm():
    """pip-install vLLM from an attached wheel payload (no internet). True when `import vllm` works."""
    if CFG.get("image_runtime"):
        try:
            ok = setup_image_runtime()
        except Exception as e:
            print("RUNTIME-FAIL", repr(e)[:500], flush=True)
            ok = False
        if not ok:
            counters["pip_fail"] += 1
        print("VLLM-IMPORT", ok, "image_runtime", flush=True)
        return ok
    whl = sorted(glob.glob(CFG.get("payload_glob", "/kaggle/input/**/vllm-*.whl"), recursive=True))
    if not whl:
        print("VLLM-NO-PAYLOAD", flush=True)
        return False
    d = os.path.dirname(whl[0])
    t = time.time()
    log = sh(f"{sys.executable} -m pip install --no-index --find-links {d} {' '.join(CFG['pip_pkgs'])} 2>&1 | tail -6", 1800)
    print("PIP", f"{time.time() - t:.0f}s", log[-1500:], flush=True)
    ok = "vllm" in sh(f"{sys.executable} -c 'import vllm; print(\"vllm\", vllm.__version__)' 2>&1 | tail -1")
    if not ok:
        counters["pip_fail"] += 1
        wh = [w for w in glob.glob(os.path.join(d, "*.whl")) if not re.match(r".*/(torch|triton)-", w)]
        print("PIP-NODEPS", sh(f"{sys.executable} -m pip install --no-index --no-deps {' '.join(wh)} 2>&1 | tail -4", 1800)[-1000:], flush=True)
        ok = "vllm" in sh(f"{sys.executable} -c 'import vllm; print(\"vllm\", vllm.__version__)' 2>&1 | tail -1")
    print("VLLM-IMPORT", ok, flush=True)
    if ok:
        install_fp8_tables()
    return ok


def install_fp8_tables():
    """Write device-tuned Triton block-FP8 GEMM tables (cfg `fp8_tables`: {vLLM json filename: {M: config}})
    into the installed vLLM's configs dir; with no table vLLM runs one untuned default config on this GPU."""
    tabs = CFG.get("fp8_tables") or {}
    if not tabs:
        return 0
    cdir = sh(f"{sys.executable} -c 'import os, vllm; print(os.path.join(os.path.dirname(vllm.__file__), "
              f"\"model_executor/layers/quantization/utils/configs\"))' 2>/dev/null").strip().split("\n")[-1].strip().strip()
    n = 0
    if not os.path.isdir(cdir):
        counters["fp8_table_fail"] += 1
        print("FP8-TABLE-NO-DIR", repr(cdir[:200]), flush=True)
        return 0
    for name, tab in tabs.items():
        try:
            with open(os.path.join(cdir, name), "w") as f:
                json.dump(tab, f)
            n += 1
        except Exception as e:
            counters["fp8_table_fail"] += 1
            print("FP8-TABLE-FAIL", name, repr(e)[:200], flush=True)
    print("FP8-TABLES", n, "of", len(tabs), cdir, flush=True)
    return n


def kill_server(p):
    """Kill the server's whole process group (vLLM leaves TP worker processes holding VRAM otherwise)
    and wait until the GPUs are released."""
    import signal
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass
    sh("pkill -9 -f vllm.entrypoints; pkill -9 -f 'VLLM::'; pkill -9 -f llama-server", 60)
    t = time.time()
    while time.time() - t < 180:
        used = [int(x) for x in re.findall(r"\d+", sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"))]
        if not used or max(used) < 1500:  # no readable GPU (nothing to wait for) or released
            break
        time.sleep(5)
    print("SERVER-KILLED", f"{time.time() - t:.0f}s", "GPU", sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader"), flush=True)


REQ_MODEL = ["m"]  # the served name every LLM request uses: "m" = base, or the LoRA adapter the sentinel picked


def prep_lora(lcfg):
    """Copy a PEFT adapter (trained on the text-only Qwen3_5ForCausalLM) into two keyings: `arc` re-keyed onto the
    multimodal checkpoint class (model.layers -> model.language_model.layers), `arcplain` = original keys.
    Returns the vLLM server args ([] when no adapter is attached)."""
    try:
        from safetensors.torch import load_file, save_file
        src = None
        for g in lcfg["globs"]:
            hits = sorted(glob.glob(g, recursive=True))
            if hits:
                src = os.path.dirname(hits[0])
                break
        if not src:
            print("LORA-NO-ADAPTER", lcfg["globs"], flush=True)
            counters["lora_missing"] = 1
            return []
        sd = load_file(os.path.join(src, "adapter_model.safetensors"))
        acfg = json.load(open(os.path.join(src, "adapter_config.json")))
        if isinstance(acfg.get("target_modules"), (list, tuple)):
            acfg["target_modules"] = sorted({t.split(".")[-1] for t in acfg["target_modules"]})
        mods, pre = [], "base_model.model.model.layers."
        for name, rekey in (("arc", True), ("arcplain", False)):
            d = f"{LOGD}/lora_{name}"
            os.makedirs(d, exist_ok=True)
            save_file({(("base_model.model.model.language_model.layers." + k[len(pre):]) if rekey and k.startswith(pre)
                        else k): v.contiguous() for k, v in sd.items()}, os.path.join(d, "adapter_model.safetensors"))
            json.dump(acfg, open(os.path.join(d, "adapter_config.json"), "w"))
            mods.append(f"{name}={d}")
        print("LORA-PREP", src, len(sd), "tensors r", acfg.get("r"), flush=True)
        return ["--enable-lora", "--max-lora-rank", str(lcfg.get("max_rank", 16)), "--max-loras", "2",
                "--lora-modules"] + mods
    except Exception as e:
        print("LORA-PREP-ERR", repr(e)[:400], flush=True)
        counters["lora_prep_err"] = 1
        return []


def lora_sentinel(text):
    """Teacher-forced logprobs of one fixed text under m / arc / arcplain: an adapter whose logprobs equal the
    base's is NOT applied, whatever its load message said. Returns the first applied adapter name or None."""
    res = {}
    for m in ("m", "m", "arc", "arcplain"):
        try:
            body = json.dumps({"model": m, "prompt": text, "max_tokens": 1, "temperature": 0, "echo": True,
                               "logprobs": 0}).encode()
            req = urllib.request.Request(URL + "/v1/completions", body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                lp = json.loads(r.read())["choices"][0]["logprobs"]
            res.setdefault(m, []).append([x or 0.0 for x in lp["token_logprobs"]])
        except Exception as e:
            print("LORA-SENT-ERR", m, repr(e)[:300], flush=True)
    base = res.get("m", [None])[0]
    if base is None:
        return None
    floor = 0.0
    if len(res["m"]) > 1:
        floor = sum(abs(a - b) for a, b in zip(res["m"][0], res["m"][1])) / max(len(base), 1)
    for m in ("arc", "arcplain"):
        if m not in res or len(res[m][0]) != len(base):
            continue
        d = sum(abs(a - b) for a, b in zip(res[m][0], base)) / max(len(base), 1)
        ok = d > max(5 * floor, float(CFG["lora"].get("min_diff", 1e-3)))
        print("LORA-SENT", m, json.dumps({"mean_abs_diff": round(d, 6), "floor": round(floor, 6), "applied": ok}), flush=True)
        if ok:
            return m
    return None


LORA_ARGS = []


def server_cmd(step, model, server):
    if SERVER_KIND == "vllm":
        extra = []
        if CFG.get("lora") and step.get("lora", True):
            if not LORA_ARGS:
                LORA_ARGS.append(prep_lora(CFG["lora"]))
            extra = LORA_ARGS[0]
        if RT_ENV:
            return [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model,
                    "--served-model-name", "m", "--host", "127.0.0.1", "--port", str(PORT)] + step.get("args", []) + extra
        return [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
                "--served-model-name", "m", "--host", "127.0.0.1", "--port", str(PORT)] + step.get("args", []) + extra
    return [server, "-m", model, "--host", "127.0.0.1", "--port", str(PORT), "-np", str(step["np"]),
            "-c", str(step["ctx"]), "--jinja", "--metrics", "--no-webui"] + step.get("extra", [])


def int4_deq_model(step, mdir):
    """Ladder step option `int4_deq` (regex, or true for the GDN in_proj_a/b default): dequantize those int4
    modules to bf16 at load (patched vLLM iterator + shadow dir). Falls back to the original dir on error."""
    pat = step.get("int4_deq")
    if not pat:
        return mdir
    try:
        vdir = sh(f"{sys.executable} -c 'import importlib.util as u; print(u.find_spec(\"vllm\").submodule_search_locations[0])'").strip().split("\n")[-1]
        kw = {} if pat is True else {"pattern": pat}
        d, n = int4_prepare(mdir, "/tmp/int4_model_" + step["name"], vllm_dir=vdir, **kw)
        print("INT4-DEQ", d, "modules", n, "vllm", vdir, flush=True)
        return d
    except Exception as e:
        print("INT4-DEQ-FAIL", repr(e)[:300], flush=True)
        return mdir


SANITY = [("What is 12 + 30? Reply with only the number.", "42"),
          ("What is the capital of France? Reply with one word.", "paris"),
          ("Spell the word cat backwards. Reply with only the result.", "tac"),
          ("Which number is larger, 7 or 3? Reply with only the number.", "7")]


def sanity_ok(name):
    """Known-answer probe of a freshly loaded engine (thinking off, greedy): >=3 of 4 right or it is refused."""
    ok, outs = 0, []
    for q, a in SANITY:
        try:
            r = post(URL + "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": q}],
                                                     "max_tokens": 24, "temperature": 0.0,
                                                     "chat_template_kwargs": {"enable_thinking": False}}, 120)
            _m = r["choices"][0].get("message") or {}
            txt = (_m.get("content") or _m.get("reasoning_content") or _m.get("reasoning") or "").strip()
        except Exception as e:
            txt = f"ERR {type(e).__name__}"
        outs.append(txt[:40])
        ok += a in txt.lower()
    print("SANITY", name, f"{ok}/{len(SANITY)}", json.dumps(outs), flush=True)
    counters["sanity_" + ("ok" if ok >= 3 else "fail")] = counters.get("sanity_" + ("ok" if ok >= 3 else "fail"), 0) + 1
    return ok >= 3


VLLM_OK = []


def launch_server(gpus, give_up_at, start=0):
    """Try the ladder from step `start`; returns (process, step index) or (None, None)."""
    if SERVER_KIND == "vllm":
        if not VLLM_OK:
            VLLM_OK.append(install_vllm())
        if not VLLM_OK[0]:
            return None, None
        # the image ships TensorFlow; transformers imports it and TF -> oauth2client -> pyOpenSSL crashed vLLM
        server, senv = None, dict(RT_ENV or os.environ, **{"USE_TF": "0", "TRANSFORMERS_NO_TF": "1", "USE_JAX": "0", "USE_FLAX": "0",
                                                 "LIBRARY_PATH": libcuda_stub() + ":" + os.environ.get("LIBRARY_PATH", "")})
    else:
        server, senv = find_server()
        if not server:
            print("NO-SERVER-BINARY", flush=True)
            return None, None
    if gpus:
        senv["CUDA_VISIBLE_DEVICES"] = gpus
    for idx in range(start, len(CFG["ladder"])):
        step = CFG["ladder"][idx]
        if time.time() >= give_up_at:
            print("LADDER-STOP no time to load", step["name"], flush=True)
            break
        model = None
        for g in step["model_globs"]:
            hits = sorted(glob.glob(g, recursive=True))
            if hits:
                model = os.path.dirname(hits[0]) if SERVER_KIND == "vllm" else hits[0]
                break
        if not model:
            print("LADDER-SKIP no model", step["name"], flush=True)
            continue
        if SERVER_KIND == "vllm":
            model = int4_deq_model(step, model)
        args = server_cmd(step, model, server)
        print("LAUNCH", " ".join(args), flush=True)
        logp = f"{LOGD}/server_{step['name']}.log"
        logf = open(logp, "w")
        p = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT, env=dict(senv, **step.get("env", {})),
                             start_new_session=True)
        t = time.time()
        fatal = None
        while time.time() - t < step.get("load_timeout", 1800) and time.time() < give_up_at:
            if p.poll() is not None:
                break
            logf.flush()
            fatal = next((m for m in FATAL if m in open(logp, errors="replace").read()[-20000:]), None)
            if fatal and time.time() - t > 30:
                print("SERVER-FATAL", repr(fatal), flush=True)
                break
            try:
                with urllib.request.urlopen(URL + "/health", timeout=5) as r:
                    body = r.read()
                    if r.status == 200 and (SERVER_KIND == "vllm" or json.loads(body).get("status") == "ok"):
                        print(f"SERVER-UP {step['name']} load_s={time.time() - t:.0f}", flush=True)
                        print("GPU", sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"), flush=True)
                        print("KV-LINES", sh(f"grep -iE 'KV cache size|maximum concurrency|speculative|backend' {logp} | tail -8"), flush=True)
                        REQ_MODEL[0] = "m"
                        if step.get("sanity") and not sanity_ok(step["name"]):
                            break  # garbage-producing engine (e.g. a mis-dequantized checkpoint): next ladder step
                        if "--enable-lora" in args:
                            print("LORA-LOG", sh(f"grep -iE 'lora' {logp} | grep -viE 'POST|GET' | tail -6"), flush=True)
                            first = tasks[sorted(tasks)[0]]
                            pick = lora_sentinel((build_prompt_prog(first) if PROG_MODE else build_prompt(first, 0))[:6000])
                            REQ_MODEL[0] = pick or "m"
                            counters["lora_applied"] = int(pick is not None)
                            print("LORA-REQUEST-MODEL", REQ_MODEL[0], flush=True)
                        return p, idx
            except Exception:
                pass
            time.sleep(10)
        counters["server_launch_fail"] += 1
        kill_server(p)
        print(f"LAUNCH-FAIL {step['name']}", sh(f"tail -c 2000 {LOGD}/server_{step['name']}.log"), flush=True)
    return None, None


def server_alive(p):
    if p.poll() is not None:
        return False
    try:
        with urllib.request.urlopen(URL + "/health", timeout=15) as r:
            return r.status == 200
    except Exception:
        return False


def run_llm(ids, llm_deadline, gpus, proc0=None):
    """Serve the LLM until llm_deadline. If the engine dies mid-phase (a runtime OOM answers every request
    with a 500), the in-flight jobs are re-queued and the NEXT ladder step is launched (max 2 relaunches)."""
    outputs = [(t, k) for t in ids for k in range(len(tasks[t]["test"]))]
    jobs = [(t, k, s) for s in range(CFG["samples"]) for (t, k) in outputs]
    if PROG_MODE and not AGENT_GRID:  # one request per TASK: the program answers every test input at once
        jobs = [(t, None, s) for s in range(CFG["samples"]) for t in ids]
    if CASCADE:  # program pass first, then grid samples (dispatched only once the task's programs are done)
        jobs = [(t, None, s) for s in range(int(CASC.get("prog_samples", 1))) for t in ids] \
            + [(t, k, s) for s in range(int(CASC.get("grid_samples", 1))) for (t, k) in outputs]
    prog_left = {}  # cascade: task -> program jobs not yet finished
    for (t, k, s) in jobs:
        if k is None:
            prog_left[t] = prog_left.get(t, 0) + 1
    grid_sem = threading.Semaphore(int(CASC.get("grid_parallel", CFG["parallel"])))
    cursor = [0]
    retry = []
    lock = threading.Lock()
    last_write = [time.time()]
    start, launches = 0, 0
    counters.setdefault("server_deaths", 0)
    counters.setdefault("requeued", 0)
    while True:
        # stop trying to load once less than min_useful (at most half the remaining phase) would be left
        give_up = llm_deadline - min(float(CFG.get("llm_min_useful_s", 600)), 0.5 * (llm_deadline - time.time()))
        if proc0 is not None and launches == 0 and proc0[0] is not None and proc0[0].poll() is None:
            proc, idx = proc0  # agent_grid: keep the agent phase's server (no reload)
            print("LLM-REUSE-SERVER", CFG["ladder"][idx]["name"], flush=True)
        else:
            proc, idx = launch_server(gpus, give_up, start)
        if proc is None:
            print("LLM-SKIP no server", flush=True)
            break
        launches += 1
        state = {"dead": False}

        def next_job():
            with lock:
                if retry:
                    return retry.pop()
                if cursor[0] >= len(jobs):
                    return None
                if CASCADE and jobs[cursor[0]][1] is not None and prog_left.get(jobs[cursor[0]][0], 0) > 0:
                    # a grid job whose task still has a program in flight: take the first grid job whose
                    # task is done instead, else tell the worker to wait (counted)
                    for i in range(cursor[0], len(jobs)):
                        if jobs[i][1] is not None and prog_left.get(jobs[i][0], 0) == 0:
                            jobs.insert(cursor[0], jobs.pop(i))
                            break
                    else:
                        counters["casc_wait"] = counters.get("casc_wait", 0) + 1
                        return "WAIT"
                if PROG_MODE and not CASCADE and CFG.get("resample_near_first") and jobs[cursor[0]][2] > 0:
                    # later samples: a task whose best program fails exactly ONE demo goes first (E281 prior)
                    for i in range(cursor[0], len(jobs)):
                        tt = jobs[i][0]
                        nt = len(tasks[tt]["train"])
                        if any((tt, kk) in prog_unv and nt - prog_unv[(tt, kk)][0] == 1
                               for kk in range(len(tasks[tt]["test"]))):
                            if i != cursor[0]:
                                jobs.insert(cursor[0], jobs.pop(i))
                                counters["resample_near_first"] = counters.get("resample_near_first", 0) + 1
                            break
                cursor[0] += 1
                return jobs[cursor[0] - 1]

        def worker(wid, proc=proc, state=state):
            while not state["dead"]:
                job = next_job()
                if job is None:
                    return
                if job == "WAIT":
                    time.sleep(10)
                    continue
                t, k, s = job
                jprog = PROG_MODE and k is None  # this job asks for a program (cascade: per job)
                if AGENT_GRID and (t, k) in prog_ver:  # a demo-verified agent transform already answers it
                    with ans_lock:
                        counters["ag_skip_verified"] = counters.get("ag_skip_verified", 0) + 1
                    continue
                if CASCADE and not jprog:
                    if (t, k) in prog_ver:  # a demo-verified program already answers this output
                        with ans_lock:
                            counters["casc_skip_verified"] = counters.get("casc_skip_verified", 0) + 1
                        continue
                    grid_sem.acquire()
                try:
                    run_job(t, k, s, jprog, proc, state)
                finally:
                    if CASCADE and not jprog:
                        grid_sem.release()
                    if jprog:
                        with lock:
                            prog_left[t] = prog_left.get(t, 1) - 1

        def run_job(t, k, s, jprog, proc, state):
            if True:
                if jprog and s > 0 and any((t, kk) in prog_ver for kk in range(len(tasks[t]["test"]))):
                    return  # a later sample only for tasks still without a verified program
                if not jprog and s > 0 and CFG.get("resample_skip_tok") \
                        and nat_stop.get((t, k), 1e12) < float(CFG["resample_skip_tok"]):
                    with ans_lock:  # a reply that stopped on its own early is ~0.77 precise (K35): spend elsewhere
                        counters["resample_skipped"] = counters.get("resample_skipped", 0) + 1
                    return
                left = llm_deadline - time.time()
                if left < CFG.get("min_request_s", 300):
                    with lock:
                        counters["skipped_time"] += 1
                    return
                tb = CFG.get("commit_thinking_budget") if placeholder else None
                tb = tb or CFG.get("thinking_budget")
                mt = int(CFG["max_tokens"])
                if CASCADE and jprog:  # the program pass keeps its own (shorter) budget
                    tb = (CFG.get("commit_thinking_budget") if placeholder else None) or CASC.get("prog_thinking_budget", tb)
                    mt = int(CASC.get("prog_max_tokens", mt))
                if tb and CFG.get("adaptive_tb"):  # shrink the request so it can FINISH before the deadline
                    a = CFG["adaptive_tb"]
                    allowed = int((left - float(a.get("margin_s", 300))) * seq_rate())
                    if allowed < int(a.get("min_tokens", 4096)):
                        with lock:
                            counters["skipped_time"] += 1
                        return
                    if allowed < mt:
                        room = min(mt - int(tb), max(int(a.get("min_room", 2048)), allowed // 3))
                        mt = allowed
                        tb = max(256, min(int(tb), mt - room))
                        with ans_lock:
                            counters["tb_shrunk"] = counters.get("tb_shrunk", 0) + 1
                used_img = bool(jprog and CFG["ladder"][idx].get("images") and not IMG_OFF[0])
                if used_img:  # vision-tower step: grids also as PNGs
                    user = build_content_prog_images(tasks[t], t)
                    with ans_lock:
                        counters["prog_images"] = counters.get("prog_images", 0) + 1
                else:
                    user = build_prompt_prog(tasks[t], t) if jprog else (
                        build_prompt_sg(tasks[t], k) if s in (CFG.get("sg_samples") or []) else build_prompt(tasks[t], k))
                    pv = (CFG.get("prompt_variants") or [])
                    if not jprog and pv:  # per-sample hint variant (eval: SG+diff 74, diff 65, SG 69, plain 59 of 172)
                        v = pv[min(s, len(pv) - 1)]
                        user = build_prompt_aug(tasks[t], k, sg=bool(v.get("sg")), diff=bool(v.get("diff")), h2=bool(v.get("h2")),
                                                h3=bool(v.get("h3")))
                payload = {"model": REQ_MODEL[0], "messages": [{"role": "user", "content": user}],
                           "max_tokens": mt, "temperature": CFG["temperature"],
                           "top_p": CFG["top_p"], "min_p": CFG["min_p"],
                           "chat_template_kwargs": {"enable_thinking": CFG["effort"] != "none"}}
                if CFG.get("top_k"):
                    payload["top_k"] = CFG["top_k"]
                if SERVER_KIND == "llamacpp":
                    payload["t_max_predict_ms"] = int(max(60, left - 90) * 1000)
                if CFG["effort"] in ("high", "max"):
                    payload["chat_template_kwargs"]["reasoning_effort"] = CFG["effort"]
                if CFG.get("reasoning_effort"):  # Qwen3.8 template: xhigh (its default) / medium / low
                    payload["chat_template_kwargs"]["reasoning_effort"] = CFG["reasoning_effort"]
                if tb:  # vLLM forces </think> in-request at the budget (no re-prefill); the commit run uses a
                    payload["thinking_token_budget"] = int(tb)  # small one so the forced path is exercised there
                ts = time.time()
                row = {"task": t, "test": k, "sample": s, "t": round(ts - T0), "launch": launches}
                if not jprog and s in (CFG.get("sg_samples") or []):
                    row["sg"] = 1
                if not jprog and CFG.get("prompt_variants"):
                    _v = CFG["prompt_variants"][min(s, len(CFG["prompt_variants"]) - 1)]
                    row["pv"] = ("sg" if _v.get("sg") else "") + ("diff" if _v.get("diff") else "") + ("h2" if _v.get("h2") else "") \
                        + ("h3" if _v.get("h3") else "") or "plain"
                if tb:
                    row["tb"] = int(tb)
                    row["mt"] = mt
                try:
                    try:
                        r = post(URL + "/v1/chat/completions", payload, max(60, left))
                        if used_img:
                            with ans_lock:
                                counters["img_ok"] = counters.get("img_ok", 0) + 1
                    except Exception as e:  # a 4xx on an image request: resend as text, and after 3 image
                        if not used_img or not 400 <= int(getattr(e, "code", 0) or 0) < 500:  # failures with
                            raise  # no image success, stop sending images at all (counted)
                        with ans_lock:
                            counters["img_err"] = counters.get("img_err", 0) + 1
                            if counters["img_err"] >= 3 and not counters.get("img_ok"):
                                IMG_OFF[0] = True
                        row["img_fallback"] = f"{type(e).__name__}: {str(e)[:120]}"
                        payload["messages"] = [{"role": "user", "content": build_prompt_prog(tasks[t], t)}]
                        r = post(URL + "/v1/chat/completions", payload, max(60, left))
                    ch = r["choices"][0]
                    content = (ch.get("message") or {}).get("content") or ""
                    fin = ch.get("finish_reason")
                    if jprog:
                        code = extract_code(content)
                        _m = ch.get("message") or {}
                        row.update(finish=fin, code=code is not None, ncontent=len(content),
                                   nreason=len(_m.get("reasoning_content") or _m.get("reasoning") or ""),
                                   ctok=(r.get("usage") or {}).get("completion_tokens"))
                        with ans_lock:
                            counters["requests"] += 1
                            counters["prog_code" if code else "prog_nocode"] = \
                                counters.get("prog_code" if code else "prog_nocode", 0) + 1
                        if code:
                            prog_take(t, code, row)
                        raise StopIteration  # program rows skip the grid-answer path below
                    g = parse_grid(content) if fin != "length" else None
                    row.update(finish=fin, parsed=g is not None,
                               ctok=(r.get("usage") or {}).get("completion_tokens"))
                    with ans_lock:
                        counters["requests"] += 1
                        if g is not None:
                            llm_ans.setdefault((t, k), []).append(g)
                            counters["parsed"] += 1
                            _ct = row.get("ctok") or 0
                            if fin == "stop" and (not tb or _ct < int(tb)):  # stopped before any thinking budget
                                nat_stop[(t, k)] = min(nat_stop.get((t, k), 1e12), _ct)
                        elif fin == "length":
                            counters["truncated"] += 1
                        else:
                            counters["unparsed"] += 1
                    msg = ch.get("message") or {}
                    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                    left2 = llm_deadline - time.time()
                    if fin == "length" and CFG.get("force_answer") and reasoning and left2 > 90:
                        u = r.get("usage") or {}
                        room = int(CFG.get("max_model_len", 65536)) - (u.get("prompt_tokens") or 0) \
                            - (u.get("completion_tokens") or 0) - 64
                        fp = {"model": REQ_MODEL[0], "max_tokens": max(256, min(int(CFG.get("force_max_tokens", 4096)), room)),
                              "prompt": raw_prompt_qwen38(tasks[t], k, CFG.get("reasoning_effort") or "xhigh")
                              + reasoning + FORCE_TEXT,
                              "temperature": CFG["temperature"], "top_p": CFG["top_p"], "min_p": CFG["min_p"]}
                        if CFG.get("top_k"):
                            fp["top_k"] = CFG["top_k"]
                        try:
                            r2 = post(URL + "/v1/completions", fp, max(30, left2 - 30))
                            txt = r2["choices"][0].get("text") or ""
                            fg = parse_grid(txt.split("</think>")[-1])
                            row.update(forced=fg is not None, ftok=(r2.get("usage") or {}).get("completion_tokens"))
                            with ans_lock:
                                counters["forced"] = counters.get("forced", 0) + 1
                                if fg is not None:
                                    counters["forced_parsed"] = counters.get("forced_parsed", 0) + 1
                                    if fg not in llm_forced.setdefault((t, k), []):
                                        llm_forced[(t, k)].append(fg)
                        except Exception as e:  # counted; the natural answer path is unaffected
                            row["forced_error"] = f"{type(e).__name__}: {str(e)[:120]}"
                            with ans_lock:
                                counters["forced_err"] = counters.get("forced_err", 0) + 1
                except StopIteration:
                    pass
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                    with ans_lock:
                        counters["http_err"] += 1
                    if llm_deadline - time.time() > 60 and not server_alive(proc):
                        with lock:
                            retry.append(job)
                            counters["requeued"] += 1
                            state["dead"] = True
                        row["requeued"] = True
                row["sec"] = round(time.time() - ts)
                if row.get("ctok") and time.time() - ts > 30:
                    with ans_lock:
                        seq_rates.append(row["ctok"] / (time.time() - ts))
                with lock:
                    with open(f"{LOGD}/calls.jsonl", "a") as f:
                        f.write(json.dumps(row) + "\n")
                    if time.time() - last_write[0] > 60:
                        with ans_lock:
                            write_submission()
                        last_write[0] = time.time()
                        print(f"PROGRESS t={time.time() - T0:.0f}s cursor={cursor[0]}/{len(jobs)} retry={len(retry)} "
                              f"{json.dumps(counters)}", flush=True)

        npar = int(CFG["ladder"][idx].get("parallel") or CFG["parallel"])  # ladder steps may cap client concurrency
        print("LLM-PARALLEL", npar, CFG["ladder"][idx]["name"], flush=True)
        ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(npar)]
        for th in ths:
            th.start()
        for th in ths:
            th.join(timeout=max(1, llm_deadline + 120 - time.time()))
        kill_server(proc)  # whole process group; waits for the VRAM to be released
        if not state["dead"]:
            break
        counters["server_deaths"] += 1
        print(f"SERVER-DIED step={CFG['ladder'][idx]['name']} t={time.time() - T0:.0f}s retry={len(retry)} "
              f"cursor={cursor[0]}/{len(jobs)}", flush=True)
        if launches > 2:
            break
        start = idx + 1 if idx + 1 < len(CFG["ladder"]) else idx
    n_ans = sum(1 for o in outputs if llm_ans.get(o) or o in prog_ver or o in prog_unv)
    print("LLM-END", json.dumps(counters), f"answered_outputs={n_ans}/{len(outputs)}",
          f"t={time.time() - T0:.0f}s", "GPU", sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader"),
          flush=True)


def wait_vram_free(limit_s=300):
    t = time.time()
    while time.time() - t < limit_s:
        used = [int(x) for x in re.findall(r"\d+", sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"))]
        if not used or max(used) < 1500:
            break
        time.sleep(5)
    print("VRAM-FREE-WAIT", f"{time.time() - t:.0f}s", sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader"), flush=True)


def run_icecuber(ids, deadline):
    """CPU-only enumerative DSL search, run in a thread beside the GPU phases; fills ice_ver (counted, never fatal)."""
    ic = CFG["icecuber"]
    try:
        ns = {"__name__": "icecuber_run"}
        exec(compile(ICECUBER_PY, "icecuber_run.py", "exec"), ns)
        src = "/tmp/icecuber_src"
        for rel, txt in ICECUBER_FILES.items():
            os.makedirs(os.path.dirname(os.path.join(src, rel)) or src, exist_ok=True)
            open(os.path.join(src, rel), "w").write(txt)
        res, C = ns["solve_all"]({t: tasks[t] for t in ids}, src, "/tmp/icecuber_work",
                                 depth=int(ic.get("depth", 3)), fallback_depth=int(ic.get("fallback_depth", 2)),
                                 timeout=int(ic.get("timeout", 300)), parallel=int(ic.get("parallel", 16)),
                                 mem_gb=int(ic.get("mem_gb", 6)), deadline=deadline,
                                 log=lambda m: print(m, flush=True))
        got = {}
        for t, per in res.items():
            for k, cands in enumerate(per):
                if cands and cands[0].get("verified"):
                    got[(t, k)] = cands[0]["grid"]
        with ans_lock:
            ice_ver.update(got)
            counters["ice_verified"] = len(got)
            write_submission()
        print("ICECUBER-END", json.dumps(C), "verified_outputs", len(got), flush=True)
    except Exception as e:  # counted: the other generators are untouched
        counters["ice_err"] = counters.get("ice_err", 0) + 1
        print("ICECUBER-ERR", type(e).__name__, str(e)[:500], flush=True)


def run_corpus(ids, deadline):
    """CPU retrieval-by-execution: every corpus program is run on every task's demonstrations (subprocess,
    per-call timeout); a program reproducing EVERY demo output fills cor_ver (counted, never fatal)."""
    cf = CFG["corpus"]
    try:
        d = "/tmp/corpus"
        os.makedirs(d + "/lib", exist_ok=True)
        open(d + "/corpus_exec.py", "w").write(CORPUS_PY)
        open(d + "/dsl.py", "w").write(DSL_SOURCE)
        open(d + "/verifiers.py", "w").write(VERIFIERS_SOURCE)
        with open(d + "/lib/calls.jsonl", "w") as f:
            for x in json.loads(zlib.decompress(base64.b64decode(CORPUS_PROGS))):
                f.write(json.dumps({"task": x["task"], "code": x["code"], "verified": True}) + "\n")
        json.dump({t: tasks[t] for t in ids}, open(d + "/tasks.json", "w"))
        cmd = [sys.executable, d + "/corpus_exec.py", "--challenges", d + "/tasks.json", "--out", d + "/out",
               "--dsl", d + "/dsl.py", "--verifiers", d + "/verifiers.py", "--llm-runs", d + "/lib",
               "--workers", str(int(cf.get("workers", 8))), "--timeout", str(float(cf.get("timeout", 2.0)))]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=max(60.0, deadline - time.time()))
        print("CORPUS-LOG", (r.stdout + r.stderr)[-800:].replace("\n", " | "), flush=True)
        got = {}
        for line in open(d + "/out/cands.jsonl"):
            x = json.loads(line)
            if x["cands"]:
                got[(x["task"], int(x["k"]))] = x["cands"][0]["grid"]
        with ans_lock:
            cor_ver.update(got)
            counters["corpus_verified"] = len(got)
            write_submission()
        print("CORPUS-END verified_outputs", len(got), "tasks", len({t for t, _ in got}), flush=True)
    except Exception as e:  # counted: the other generators are untouched
        counters["corpus_err"] = counters.get("corpus_err", 0) + 1
        print("CORPUS-ERR", type(e).__name__, str(e)[:500], flush=True)


JOIN = []  # (thread, its own deadline) of CPU generators the final write must not outrun


def join_cpu_generators(margin):
    for th, dl in JOIN:
        th.join(timeout=max(0.0, min(dl, DEADLINE - margin / 2) - time.time()))
        if th.is_alive():
            counters["cpu_gen_unjoined"] = counters.get("cpu_gen_unjoined", 0) + 1


AGENT_DIR = f"{LOGD}/agent"


def ingest_agent():
    """Agent records (final rows + partial bests) -> prog_ver / prog_unv, ranked as the agent-only build ranks
    them: verified first, then most demonstrations right, earlier sample, final row before partial."""
    recs = []
    for fn, kind in ((f"{AGENT_DIR}/calls.jsonl", 0), (f"{AGENT_DIR}/partial.jsonl", 1)):
        if not os.path.exists(fn):
            continue
        for line in open(fn):
            try:
                r = json.loads(line)
            except Exception:
                counters["agent_read_err"] = counters.get("agent_read_err", 0) + 1  # a line being written now
                continue
            if r.get("test") is None or r.get("task") not in tasks:
                continue
            recs.append((not r.get("verified"), -int(r.get("n_right", -1)), int(r.get("sample", 0)), kind, r))
    recs.sort(key=lambda x: x[:4])
    ver, unv = {}, {}
    for *_, r in recs:
        t = r["task"]
        for k, g in enumerate(r["test"][:len(tasks[t]["test"])]):
            if not (isinstance(g, list) and g and isinstance(g[0], list)):
                continue
            if r.get("verified"):
                ver.setdefault((t, k), g)
            elif (t, k) not in unv:
                unv[(t, k)] = (int(r.get("n_right", 0)), g)
    with ans_lock:
        prog_ver.update(ver)
        prog_unv.update(unv)
    return len(ver), len(unv)


def run_agent(ids, llm_deadline, gpus, keep_server=False):
    """The REPL agent (llm_agent.py, a subprocess) against the ladder's vLLM server until llm_deadline; its
    verified / best-unverified test grids feed the program-mode merge (verified -> TTT1 -> near-miss -> TTT2)."""
    os.makedirs(AGENT_DIR, exist_ok=True)
    give_up = llm_deadline - min(float(CFG.get("llm_min_useful_s", 1800)), 0.5 * (llm_deadline - time.time()))
    proc, idx = launch_server(gpus, give_up, 0)
    if proc is None:
        print("AGENT-SKIP no server", flush=True)
        return
    agent_py = f"{AGENT_DIR}/llm_agent.py"
    open(agent_py, "w").write(AGENT_SOURCE)
    stop_new = llm_deadline - float(CFG.get("agent_stop_before_s", 1500))
    cmd = [sys.executable, "-u", agent_py, "--challenges", CH, "--out", AGENT_DIR, "--url", URL + "/v1/completions",
           "--partial", "--task-ids", ",".join(ids), "--deadline-s", str(max(60.0, stop_new - time.time()))] \
        + [str(a) for a in CFG["agent_args"]]
    if placeholder:
        cmd += [str(a) for a in CFG.get("probe_agent_args", [])]
    print("AGENT", " ".join(cmd[2:]), flush=True)
    ta = time.time()
    logf = open(f"{AGENT_DIR}/agent.log", "w")
    ap = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    last, n = 0, 0
    while ap.poll() is None and time.time() < llm_deadline:
        time.sleep(15)
        if time.time() - last > 60:
            nv, nu = ingest_agent()
            with ans_lock:
                write_submission()
            last, n = time.time(), n + 1
            if n % 30 == 0:
                print(f"[{time.time() - T0:6.0f}s] agent verified_outputs={nv} unverified_outputs={nu} "
                      + sh(f"tail -1 {AGENT_DIR}/agent.log")[-200:], flush=True)
    if ap.poll() is None:
        print("AGENT-HARD-STOP at the LLM deadline", flush=True)
        try:
            os.killpg(ap.pid, signal.SIGKILL)
        except Exception:
            pass
    nv, nu = ingest_agent()
    counters.update(agent_s=round(time.time() - ta), agent_verified_outputs=nv, agent_unverified_outputs=nu)
    print("AGENT-TAIL", sh(f"tail -3 {AGENT_DIR}/agent.log")[-1500:], flush=True)
    if keep_server:
        return proc, idx
    kill_server(proc)


METRIC_NAMES = {"gen": ["vllm:generation_tokens_total"], "prompt": ["vllm:prompt_tokens_total"],
                "acc": ["vllm:spec_decode_num_accepted_tokens_total"],
                "draft": ["vllm:spec_decode_num_draft_tokens_total"], "drafts": ["vllm:spec_decode_num_drafts_total"],
                "preempt": ["vllm:num_preemptions_total"], "kv": ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"],
                "running": ["vllm:num_requests_running"], "waiting": ["vllm:num_requests_waiting"]}
METRICS = []  # one scrape per minute while a vLLM server answers (cumulative counters + gauges)


def scrape_metrics():
    """Each tracked vLLM Prometheus metric summed over its label sets; None while no server answers."""
    try:
        with urllib.request.urlopen(URL + "/metrics", timeout=10) as r:
            txt = r.read().decode("utf-8", "replace")
    except Exception:
        return None
    out = {}
    for key, names in METRIC_NAMES.items():
        for name in names:
            vals = [float(m.group(1)) for m in
                    re.finditer(rf"^{re.escape(name)}(?:{{[^}}]*}})?\s+([0-9.eE+-]+)", txt, re.M)]
            if vals:
                out[key] = sum(vals)
                break
    return out


def metrics_poller(period_s=60.0):
    n = 0
    while True:
        m = scrape_metrics()
        if m:
            m["t"] = round(time.time() - T0)
            METRICS.append(m)
            try:
                with open(f"{LOGD}/metrics.jsonl", "a") as f:
                    f.write(json.dumps(m) + "\n")
            except Exception:
                pass
            n += 1
            if n % 10 == 1:
                print("METRICS", json.dumps(m), flush=True)
        time.sleep(period_s)


def metrics_summary(t_from=None):
    """Window read from the per-minute scrapes: decode tok/s, MTP acceptance, preemptions, KV use, running."""
    ms = [m for m in METRICS if "gen" in m and (t_from is None or m["t"] >= t_from)]
    if len(ms) < 2:
        return {"scrapes": len(ms)}
    a, b = ms[0], ms[-1]
    dt = max(1, b["t"] - a["t"])
    s = {"scrapes": len(ms), "window_t0": a["t"], "window_s": dt, "gen_tok_s": round((b["gen"] - a["gen"]) / dt, 1)}
    d_draft = b.get("draft", 0) - a.get("draft", 0)
    if d_draft > 0:
        s["mtp_accept_rate"] = round((b.get("acc", 0) - a.get("acc", 0)) / d_draft, 4)
        d_steps = b.get("drafts", 0) - a.get("drafts", 0)
        if d_steps > 0:
            s["mtp_accepted_per_step"] = round((b.get("acc", 0) - a.get("acc", 0)) / d_steps, 3)
    s["preemptions"] = b.get("preempt", 0) - a.get("preempt", 0) if "preempt" in b else None
    for key in ("kv", "running", "waiting"):
        v = [m[key] for m in ms if key in m]
        if v:
            s[f"{key}_mean"], s[f"{key}_min"], s[f"{key}_max"] = round(sum(v) / len(v), 3), min(v), max(v)
    return s


def measure_commit(win_s):
    """Commit-run measurement (the scored re-run returns only its score): serve the shipped ladder at the shipped
    concurrency on the public EVALUATION tasks for win_s, log steady-state decode tok/s, MTP acceptance,
    preemptions and KV use plus first-sample hits on the outputs that finish, then write the placeholder file."""
    ec, es = find_file("arc-agi_evaluation_challenges.json"), find_file("arc-agi_evaluation_solutions.json")
    if not ec:
        print("MEASURE-SKIP no evaluation file", flush=True)
        with ans_lock:
            write_submission(final=True)
        return
    ev = json.load(open(ec))
    sol = json.load(open(es)) if es else {}
    keep = dict(tasks)
    tasks.clear()
    tasks.update(ev)
    CFG["commit_thinking_budget"] = None  # measure the scored run's request shape, not the commit's short one
    ids = sorted(ev, key=lambda t: len(build_prompt(ev[t], 0)))
    print(f"COMMIT-MEASURE {len(ids)} eval tasks, {sum(len(ev[t]['test']) for t in ids)} outputs, "
          f"window to t={win_s:.0f}s", flush=True)
    try:
        run_llm(ids, T0 + win_s, "")
    except Exception as e:  # counted; the placeholder submission is still written below
        counters["measure_err"] = f"{type(e).__name__}: {str(e)[:200]}"
    hits = sum(1 for (t, k), gs in llm_ans.items()
               if gs and t in sol and k < len(sol[t]) and gs[0] == sol[t][k])
    s = {"outputs_answered": len(llm_ans), "first_sample_hits": hits, "requests": counters.get("requests"),
         "truncated": counters.get("truncated"), "http_err": counters.get("http_err"),
         "server_deaths": counters.get("server_deaths"), "measure_err": counters.get("measure_err")}
    t_up = next((m["t"] for m in METRICS if m.get("gen")), None)
    s["steady"] = metrics_summary(None if t_up is None else t_up + float(CFG.get("measure_skip_s", 600)))  # skip the prefill burst
    s["whole"] = metrics_summary(None)
    if s["steady"].get("gen_tok_s"):  # samples per hidden output if the scored run decodes at this rate
        s["proj_samples_per_output_260x60.8k"] = round(s["steady"]["gen_tok_s"] * (WALL - 900) / (260 * 60800), 3)
    print("MEASURE-SUMMARY", json.dumps(s), flush=True)
    try:
        json.dump(s, open(f"{LOGD}/measure_summary.json", "w"))
    except Exception:
        pass
    tasks.clear()
    tasks.update(keep)
    llm_ans.clear()
    llm_forced.clear()
    nat_stop.clear()
    with ans_lock:
        write_submission(final=True)
    print("RUN-END", json.dumps(counters), f"elapsed={time.time() - T0:.0f}s commit_measure", flush=True)


def main():
    ids = sorted(tasks, key=lambda t: len(build_prompt(tasks[t], 0)))
    if CFG.get("server", "vllm") == "vllm":
        threading.Thread(target=metrics_poller, args=(float(CFG.get("metrics_period_s", 60)),), daemon=True).start()
    if placeholder and CFG.get("commit_measure_s"):  # commit run = an eval-120 throughput/MTP measurement window
        measure_commit(float(CFG["commit_measure_s"]))
        return
    if placeholder and CFG.get("commit_skip_llm"):  # quota-lean commit: the scored re-run is the only real run
        with ans_lock:
            write_submission(final=True)
        print("RUN-END", json.dumps(counters), "merge_sources", json.dumps(merge_stats),
              f"elapsed={time.time() - T0:.0f}s commit_skip_llm", flush=True)
        return
    llm_deadline = DEADLINE if MODE in ("concurrent", "llm_only") else T0 + float(CFG.get("llm_wall_s", WALL / 2))
    if placeholder and PROBE_TASKS > 0:
        step = max(1, len(ids) // PROBE_TASKS)
        ids = ids[::step][:PROBE_TASKS]
        llm_deadline = min(llm_deadline, T0 + LLM_PROBE_WALL_S)
        print(f"COMMIT-RUN: LLM solves {len(ids)} probe tasks by t={llm_deadline - T0:.0f}s "
              f"(the TTT solver applies its own placeholder gate)", flush=True)
    margin = float(CFG.get("merge_margin_s", 300))
    if CFG.get("corpus"):  # CPU retrieval-by-execution beside the TTT phase (seconds to minutes)
        cor_dl = T0 + float(CFG["corpus"].get("wall_s", 3600))
        cor_th = threading.Thread(target=run_corpus, args=(ids, cor_dl), daemon=True)
        cor_th.start()
        JOIN.append((cor_th, cor_dl))
    if CFG.get("icecuber"):  # CPU search beside the TTT phase; must end before the LLM phase needs the CPUs
        ice_dl = T0 + float(CFG["icecuber"].get("wall_s", 14400))
        if placeholder:
            ice_dl = min(ice_dl, T0 + 1800)
        threading.Thread(target=run_icecuber, args=(ids, ice_dl), daemon=True).start()
    if MODE == "ttt_then_llm":
        # TTT first on the stock image (the vLLM pip install happens only after it, so it cannot change the
        # TTT's transformers), with a fixed wall; the LLM then runs to the deadline and its answers go first.
        ttt_wall = float(CFG["ttt_wall_s"]) if not placeholder else DEADLINE - time.time() - margin
        start_ttt("", ttt_wall)
        # the commit run's TTT gates itself to ~45 min; never let a hung solver hold the GPUs past that
        finish_ttt(wait_until=time.time() + (min(ttt_wall, 3600) if placeholder else ttt_wall)
                   + float(CFG.get("ttt_grace_s", 900)))
        with ans_lock:
            write_submission()
        if CFG.get("skip_llm"):  # decomposition build: this combo's exact TTT phase (+ CPU generators), no LLM
            join_cpu_generators(margin)
            with ans_lock:
                write_submission(final=True)
            print("RUN-END", json.dumps(counters), "merge_sources", json.dumps(merge_stats),
                  f"elapsed={time.time() - T0:.0f}s skip_llm", flush=True)
            return
        wait_vram_free()
        llm_deadline = DEADLINE - margin
        if placeholder and PROBE_TASKS > 0:
            llm_deadline = min(llm_deadline, time.time() + LLM_PROBE_WALL_S)
        (run_agent if AGENT_MODE else run_llm)(ids, llm_deadline, "")
        join_cpu_generators(margin)
        with ans_lock:
            write_submission(final=True)
        print("RUN-END", json.dumps(counters), "merge_sources", json.dumps(merge_stats),
              f"elapsed={time.time() - T0:.0f}s", flush=True)
        return
    if MODE == "concurrent":
        start_ttt(CFG.get("ttt_gpus", ""), DEADLINE - time.time() - margin)
    if MODE != "ttt_only" and AGENT_GRID:
        agent_dl = min(llm_deadline, T0 + float(CFG.get("agent_wall_s", 0.5 * WALL)))
        if placeholder and PROBE_TASKS > 0:
            agent_dl = min(agent_dl, T0 + 0.5 * LLM_PROBE_WALL_S)
        srv = run_agent(ids, agent_dl, "", keep_server=True)
        print(f"AGENT-GRID agent phase done at t={time.time() - T0:.0f}s verified_outputs={len(prog_ver)}", flush=True)
        with ans_lock:
            write_submission()
        run_llm(ids, llm_deadline, "", proc0=srv)
    elif MODE != "ttt_only":
        run_llm(ids, llm_deadline, CFG.get("llm_gpus", "") if MODE == "concurrent" else "")
        with ans_lock:
            write_submission()
    if MODE in ("sequential", "ttt_only"):
        start_ttt("", DEADLINE - time.time() - margin)
    finish_ttt()
    join_cpu_generators(margin)
    with ans_lock:
        write_submission(final=True)
    print("RUN-END", json.dumps(counters), "merge_sources", json.dumps(merge_stats),
          f"elapsed={time.time() - T0:.0f}s", flush=True)


main()
