#!/usr/bin/env bash
# 在独立 conda 环境启动本地 vLLM OpenAI-compatible 推理服务
set -euo pipefail

MODEL="${LLM_WEIGHTS:-/data/zch/models/Qwen3.8-27B-FP8}"
PORT="${VLLM_PORT:-8001}"
HOST="${VLLM_HOST:-127.0.0.1}"
NAME="${LLM_MODEL:-Qwen3.8-27B-FP8}"
VLLM_BIN="${VLLM_BIN:-/opt/conda/envs/qwen3vl-sft/bin/vllm}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

if [[ ! -x "$VLLM_BIN" ]]; then
  echo "找不到 vLLM 可执行文件: $VLLM_BIN" >&2
  exit 1
fi

exec "$VLLM_BIN" serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$NAME" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" \
  --gpu-memory-utilization "${VLLM_GPU_MEM:-0.90}" \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --trust-remote-code
