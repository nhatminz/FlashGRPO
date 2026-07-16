#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL:-$ROOT/models/Qwen2.5-7B-Instruct}"
DATA="${DATA:-$ROOT/data/sharegpt/ShareGPT_V4.3_unfiltered_cleaned_split.json}"
CONFIG="${CONFIG:-flashgrpo_b200/configs/pretrain_medusa_heads_sharegpt_qwen25_7b_b200.yaml}"
OUT="${OUT:-outputs/flashgrpo_b200_medusa_sharegpt_qwen25_7b}"

PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-4}"
PRETRAIN_GRAD_ACCUM="${PRETRAIN_GRAD_ACCUM:-4}"
PRETRAIN_LR="${PRETRAIN_LR:-3e-4}"
PRETRAIN_MAX_SEQ_LEN="${PRETRAIN_MAX_SEQ_LEN:-1024}"
PRETRAIN_NUM_SAMPLES="${PRETRAIN_NUM_SAMPLES:-0}"
PRETRAIN_SAVE_STEPS="${PRETRAIN_SAVE_STEPS:-500}"

python flashgrpo_b200/scripts/pretrain_medusa_heads.py \
  --config "$CONFIG" \
  --model_name_or_path "$MODEL" \
  --dataset_path "$DATA" \
  --output_dir "$OUT" \
  --num_train_epochs "$PRETRAIN_EPOCHS" \
  --batch_size "$PRETRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$PRETRAIN_GRAD_ACCUM" \
  --learning_rate "$PRETRAIN_LR" \
  --max_seq_len "$PRETRAIN_MAX_SEQ_LEN" \
  --num_samples "$PRETRAIN_NUM_SAMPLES" \
  --save_steps "$PRETRAIN_SAVE_STEPS" \
  --dtype bf16 \
  --head_dtype fp32 \
  --chain_loss_weight 0.0
