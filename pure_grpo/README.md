# Pure GRPO Baseline

This folder contains a target-only GRPO baseline for timing comparisons against
FlashGRPO.

The baseline intentionally keeps the same data, prompt template, LoRA setup,
GRPO loss, reward functions, optimizer, gradient accumulation, logging format,
and sequence length semantics as `flashgrpo`. The rollout path is the only major
difference:

- FlashGRPO: target root token + MEDUSA/Chain-MEDUSA candidate tree + target verification.
- Pure GRPO: target model autoregressive decoding only, one token per KV-cache step.

`generation.max_length` is treated as total context length, matching FlashGRPO.
Because there is no speculative draft, `average_accept_length` is always 1.0 and
`medusa_acceptance_rate` / `draft_acceptance_rate` are 0.0.

## Run

```bash
EXP=grpo_pure_baseline_chain_adaptive_match

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python pure_grpo/scripts/train_grpo.py \
  --config pure_grpo/configs/grpo_qwen25_1p5b_simplelrabel3to5.yaml \
  --set run_name=${EXP} \
  --set model.model_dir=${MODEL} \
  --set logging.log_dir=logs/pure_grpo/${EXP} \
  --set training.saved_model_dir=outputs/${EXP}/target_lora
```

Metrics are saved to:

- `logs/pure_grpo/${EXP}/metrics.jsonl`
- `logs/pure_grpo/${EXP}/metrics.csv`
- `logs/pure_grpo/${EXP}/summary.txt`

