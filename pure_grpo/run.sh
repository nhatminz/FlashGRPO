#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL:-$ROOT/models/Qwen2.5-7B-Instruct}"
EXP="${EXP:-pure_grpo_b200_qwen25_7b}"
CONFIG="${CONFIG:-pure_grpo/configs/grpo_b200_qwen25_7b_simplelrabel3to5.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

BATCH_SIZE="${BATCH_SIZE:-8}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_TRAINING_TOKEN="${MAX_TRAINING_TOKEN:-4096}"
MAX_TRAINING_PADDING_GAP="${MAX_TRAINING_PADDING_GAP:-4096}"
LOGPS_CHUNK_SIZE="${LOGPS_CHUNK_SIZE:-512}"

"$PYTHON_BIN" pure_grpo/scripts/train_grpo.py \
  --config "$CONFIG" \
  --set run_name="$EXP" \
  --set model.model_dir="$MODEL" \
  --set generation.max_length="$MAX_LENGTH" \
  --set training.batch_size="$BATCH_SIZE" \
  --set training.accumulation_steps="$ACCUMULATION_STEPS" \
  --set training.max_training_token="$MAX_TRAINING_TOKEN" \
  --set training.max_training_padding_gap="$MAX_TRAINING_PADDING_GAP" \
  --set training.logps_chunk_size="$LOGPS_CHUNK_SIZE" \
  --set logging.log_dir="logs/pure_grpo_b200/$EXP" \
  --set training.saved_model_dir="outputs/$EXP/target_lora"
