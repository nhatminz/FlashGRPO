#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${PYTHONPATH:-$ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL:-$ROOT/models/Qwen2.5-7B-Instruct}"
HEADS="${HEADS:-outputs/flashgrpo_b200_medusa_sharegpt_qwen25_7b}"
EXP="${EXP:-reflexgrpo_b200_qwen25_7b}"
CONFIG="${CONFIG:-flashgrpo_b200/configs/reflexgrpo_optimized_b200_qwen25_7b_simplelrabel3to5.yaml}"

BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-2}"
MAX_TRAINING_TOKEN="${MAX_TRAINING_TOKEN:-8192}"
MAX_TRAINING_PADDING_GAP="${MAX_TRAINING_PADDING_GAP:-1024}"
LOGPS_CHUNK_SIZE="${LOGPS_CHUNK_SIZE:-512}"
CPEAK_NODES="${CPEAK_NODES:-768}"
MAX_TREE_NODES_PER_SEQ="${MAX_TREE_NODES_PER_SEQ:-32}"

python flashgrpo_b200/scripts/train_flashgrpo_b200.py \
  --config "$CONFIG" \
  --set run_name="$EXP" \
  --set model.model_dir="$MODEL" \
  --set flashgrpo.medusa_heads_checkpoint="$HEADS" \
  --set aux_head_checkpoint="$HEADS" \
  --set training.batch_size="$BATCH_SIZE" \
  --set training.accumulation_steps="$ACCUMULATION_STEPS" \
  --set training.max_training_token="$MAX_TRAINING_TOKEN" \
  --set training.max_training_padding_gap="$MAX_TRAINING_PADDING_GAP" \
  --set training.logps_chunk_size="$LOGPS_CHUNK_SIZE" \
  --set flashgrpo.cpeak_nodes="$CPEAK_NODES" \
  --set flashgrpo.max_tree_nodes_per_seq="$MAX_TREE_NODES_PER_SEQ" \
  --set logging.log_dir="logs/flashgrpo_b200/$EXP" \
  --set training.saved_model_dir="outputs/$EXP/target_lora" \
  --set training.saved_medusa_dir="outputs/$EXP/medusa_heads"
