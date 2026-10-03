#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export PYTHONUNBUFFERED=1
# PLE shares CUDA tensors across sibling worker processes. Expandable segments
# require pidfd_getfd for that IPC path, which is denied with ptrace_scope=1.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
# Triton's bundled Blackwell ptxas is CUDA 13.1, which produces cubins that the
# installed 570 driver cannot load. The cu129 wheel provides an SM120-capable
# CUDA 12.9 ptxas that has been exercised against Qwen's vision kernels here.
export TRITON_PTXAS_BLACKWELL_PATH="${TRITON_PTXAS_BLACKWELL_PATH:-/home/bince/venvs/vllm-qwen38/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin/ptxas}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/home/bince/.cache/triton-qwen38-sm120-cu129}"
export VLLM_PLE_CPU_OFFLOAD="${VLLM_PLE_CPU_OFFLOAD:-1}"
# FlashInfer 0.6.x misclassifies SM120 during its sampling JIT warmup. The
# native vLLM sampler is semantically equivalent and works on this toolchain.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

vllm_args=(
  /home/bince/models/models--Qwen--Qwen3.8-Flash-Next-FP8
  --served-model-name Qwen3.8-Flash-Next-FP8
  --host 0.0.0.0
  --port "${QWEN_PORT:-31000}"
  --tensor-parallel-size 2
  --max-model-len "${QWEN_MAX_MODEL_LEN:-262144}"
  --gpu-memory-utilization "${QWEN_GPU_MEMORY_UTILIZATION:-0.88}"
  # The precompiled vLLM FlashAttention vision extension carries PTX newer
  # than the installed 570 driver can JIT. PyTorch SDPA includes working SM120
  # kernels in this cu129 environment and is used only by the vision encoder.
  --mm-encoder-attn-backend TORCH_SDPA
  --enable-prefix-caching
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --max-num-seqs "${QWEN_MAX_NUM_SEQS:-4}"
)

exec /home/bince/venvs/vllm-qwen38/bin/vllm serve "${vllm_args[@]}"
