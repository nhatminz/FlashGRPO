# FlashGRPO B200

`flashgrpo_b200` is an independent copy of the FlashGRPO code path for the
B200 environment. Its Python imports point to `flashgrpo_b200.*`, so changes
here do not affect the original `flashgrpo` package used for the 3090 setup.

Main B200 defaults:

- target model dtype is loaded from config and defaults to `bf16`;
- Flash Attention 2 is requested;
- rollout `batch_size` is 16 and `accumulation_steps` is 2, keeping the GRPO
  effective prompt-group batch at 32;
- GRPO training token budget is larger: `max_training_token: 8192`;
- speculative tree budget is larger: `cpeak_nodes: 768`;
- `empty_cache_after_target_train` is disabled to avoid unnecessary cache churn
  on high-memory GPUs.

Default command:

```bash
MODEL=/mnt/hdd/nhatminh/FastGRPO-main/models/Qwen2.5-1.5B-Instruct
HEADS=outputs/flashgrpo_pretrained_extractkv/medusa_heads/step50
EXP=reflexgrpo_b200_step50

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
python flashgrpo_b200/scripts/train_flashgrpo_b200.py \
  --config flashgrpo_b200/configs/reflexgrpo_optimized_b200_qwen25_1p5b_simplelrabel3to5.yaml \
  --set run_name=${EXP} \
  --set model.model_dir=${MODEL} \
  --set flashgrpo.medusa_heads_checkpoint=${HEADS} \
  --set aux_head_checkpoint=${HEADS} \
  --set logging.log_dir=logs/flashgrpo_b200/${EXP} \
  --set training.saved_model_dir=outputs/${EXP}/target_lora \
  --set training.saved_medusa_dir=outputs/${EXP}/medusa_heads
```
