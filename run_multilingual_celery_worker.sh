#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GPU_PHYSICAL_INDEX="${1:-${GPU_PHYSICAL_INDEX:-2}}"
PY_ENV_PREFIX="${PY_ENV_PREFIX:-/home/op/miniconda3/envs/chatterbox-turbo-api/bin}"
CELERY_BIN="${CELERY_BIN:-$PY_ENV_PREFIX/celery}"

export ENABLE_CELERY=0
export DEVICE="${DEVICE:-cuda}"
export REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
export EXPECTED_GPU_NAME="${EXPECTED_GPU_NAME:-RTX 3060}"
export CELERY_BROKER_URL="${CELERY_BROKER_URL:-redis://127.0.0.1:6379/14}"
export CELERY_RESULT_BACKEND="${CELERY_RESULT_BACKEND:-$CELERY_BROKER_URL}"
export CELERY_QUEUE="${CELERY_QUEUE:-chatterbox_mtl_3060x2}"
export LAZY_LOAD_MODEL="${LAZY_LOAD_MODEL:-1}"
export MODEL_IDLE_UNLOAD_SECONDS="${MODEL_IDLE_UNLOAD_SECONDS:-300}"
export MODEL_IDLE_CHECK_INTERVAL_SECONDS="${MODEL_IDLE_CHECK_INTERVAL_SECONDS:-30}"
export MIN_FREE_VRAM_MB="${MIN_FREE_VRAM_MB:-4500}"
export MODEL_LOAD_WAIT_TIMEOUT_SECONDS="${MODEL_LOAD_WAIT_TIMEOUT_SECONDS:-60}"
export STARTUP_WARMUP="${STARTUP_WARMUP:-0}"
export MAX_INPUT_CHARS="${MAX_INPUT_CHARS:-12000}"
export AUTO_CHUNK_ENABLED="${AUTO_CHUNK_ENABLED:-1}"
export AUTO_CHUNK_HARD_LIMIT="${AUTO_CHUNK_HARD_LIMIT:-580}"
export MIN_CHUNK_CHARS="${MIN_CHUNK_CHARS:-0}"
export CHUNK_PAUSE_MS="${CHUNK_PAUSE_MS:-140}"
export DEFAULT_RESPONSE_FORMAT="${DEFAULT_RESPONSE_FORMAT:-wav}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_INFO="$(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader | awk -F', ' -v idx="$GPU_PHYSICAL_INDEX" '$1 == idx {print $2 "|" $3; exit}')"
if [[ -z "$GPU_INFO" ]]; then
  echo "ERROR: Could not resolve physical GPU index $GPU_PHYSICAL_INDEX via nvidia-smi." >&2
  exit 1
fi

GPU_UUID="${GPU_INFO%%|*}"
GPU_NAME="${GPU_INFO#*|}"
if [[ -n "$EXPECTED_GPU_NAME" && "$GPU_NAME" != *"$EXPECTED_GPU_NAME"* ]]; then
  echo "ERROR: Physical GPU index $GPU_PHYSICAL_INDEX resolved to '$GPU_NAME', expected '$EXPECTED_GPU_NAME'." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_UUID"

exec "$CELERY_BIN" -A real_chatterbox_multilingual_8010.celery_app worker \
  --loglevel="${CELERY_LOGLEVEL:-INFO}" \
  --pool=solo \
  --concurrency=1 \
  --queues="$CELERY_QUEUE" \
  --hostname="chatterbox-mtl-gpu${GPU_PHYSICAL_INDEX}@%h"
