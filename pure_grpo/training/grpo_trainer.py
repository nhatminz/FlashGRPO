from __future__ import annotations

import json
import importlib.util
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from helper.get_QAs import get_train_QAs
from helper.rewards import accuracy_reward_func, format_reward_func
from pure_grpo.model_utils.qwen_wrapper import autocast_dtype, unwrap_causal_lm
from pure_grpo.training.pure_generate import PureGenerateConfig, pure_target_generate
from pure_grpo.utils.config import save_resolved_config
from pure_grpo.utils.gpu_monitor import GpuMonitor
from pure_grpo.utils.metrics import MetricsLogger
from pure_grpo.utils.seed import seed_everything
from pure_grpo.utils.timing import format_duration


def _get(mapping: dict[str, Any], key: str, default=None):
    cur = mapping
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _dtype_from_name(name: str, default: torch.dtype = torch.float32) -> torch.dtype:
    name = str(name).lower()
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    return default


def _resolve_attn_implementation(requested: str | None) -> str:
    requested = str(requested or "eager")
    if requested == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        print(
            "Warning: model.attn_implementation=flash_attention_2 was requested, "
            "but flash_attn is not installed in this environment. Falling back "
            "to eager because Qwen sliding-window attention is not implemented for sdpa."
        )
        return "eager"
    return requested


class TrainDataCollator:
    def __init__(self, tokenizer, max_prompt_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_prompt_length = int(max_prompt_length)
        self.system_prompt = "You are a math problem assistant."
        self.user_prompt = """Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            Your response should include your thought process enclosed within <think></think> tags
            and the final answer enclosed within <answer></answer> tags (Just put a number between the tags).\n
            ### Instruction:\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}"""

    def __call__(self, batch):
        messages = []
        answers = []
        for example in batch:
            messages.append(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.user_prompt.format_map({"instruction": example["question"]})},
                ]
            )
            answers.append(example["answer"])
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokenized = self.tokenizer(
            text=text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_prompt_length,
            padding_side="left",
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "messages": messages,
            "answers": answers,
        }


def token_logps_from_hidden(hidden_states, lm_head, labels, chunk_size):
    hidden_states = hidden_states[:, :-1, :]
    labels = labels[:, 1:].to(hidden_states.device)
    seq_len = hidden_states.shape[1]
    chunks = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        logits = lm_head(hidden_states[:, start:end, :]).float()
        cur_labels = labels[:, start:end]
        selected = torch.gather(logits, dim=-1, index=cur_labels.unsqueeze(-1)).squeeze(-1)
        chunks.append(selected - torch.logsumexp(logits, dim=-1))
        del logits, selected
    return torch.cat(chunks, dim=1) if chunks else hidden_states.new_zeros((hidden_states.shape[0], 0))


def compute_model_token_logps(causal_lm, input_ids, attention_mask, chunk_size):
    base_model = unwrap_causal_lm(causal_lm)
    device = input_ids.device
    device_type = "cuda" if device.type == "cuda" else device.type
    with torch.amp.autocast(device_type, dtype=autocast_dtype(base_model), enabled=(device.type == "cuda")):
        outputs = base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    return token_logps_from_hidden(hidden_states, base_model.lm_head, input_ids, chunk_size)


def compute_target_loss_and_backward(
    target_model,
    input_ids,
    attention_mask,
    mask,
    reward,
    epsilon,
    beta,
    grpo_iteration,
    old_logps=None,
    ref_logps=None,
    chunk_size=256,
    loss_scale=1.0,
):
    device = input_ids.device
    token_mask = mask[:, :-1].to(device=device, dtype=torch.float32)
    denom = token_mask.sum(-1).clamp_min(1.0)
    reward = reward.to(device=device, dtype=torch.float32)
    seq_len = token_mask.shape[1]

    if grpo_iteration == 0:
        target_model.disable_adapter_layers()
        with torch.no_grad():
            ref_logps_gpu = compute_model_token_logps(target_model, input_ids, attention_mask, chunk_size).detach()
        target_model.enable_adapter_layers()
        ref_logps_for_loss = ref_logps_gpu
        old_logps_for_loss = None
    else:
        old_logps_for_loss = old_logps.to(device, non_blocking=True)
        ref_logps_for_loss = ref_logps.to(device, non_blocking=True)

    target_model.enable_adapter_layers()
    base_model = unwrap_causal_lm(target_model)
    device_type = "cuda" if device.type == "cuda" else device.type
    with torch.amp.autocast(device_type, dtype=autocast_dtype(base_model), enabled=(device.type == "cuda")):
        outputs = base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

    policy_hidden = hidden_states[:, :-1, :]
    labels = input_ids[:, 1:].to(device)
    old_chunks = []
    loss_value = 0.0
    abs_loss1_value = 0.0
    loss2_value = 0.0
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        logits = base_model.lm_head(policy_hidden[:, start:end, :]).float()
        cur_labels = labels[:, start:end]
        logps = torch.gather(logits, dim=-1, index=cur_labels.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        cur_mask = token_mask[:, start:end]
        if grpo_iteration == 0:
            cur_old_logps = logps.detach()
            old_chunks.append(cur_old_logps.detach().cpu())
        else:
            cur_old_logps = old_logps_for_loss[:, start:end]
        cur_ref_logps = ref_logps_for_loss[:, start:end]
        coef1 = torch.exp(logps - cur_old_logps)
        coef2 = torch.clamp(coef1, 1 - epsilon, 1 + epsilon)
        loss1 = torch.min(coef1 * reward, coef2 * reward)
        coef3 = cur_ref_logps - logps
        kl = torch.exp(coef3) - coef3 - 1
        token_loss = -(loss1 - beta * kl)
        chunk_loss = ((token_loss * cur_mask).sum(-1) / denom).sum()
        (chunk_loss * loss_scale).backward(retain_graph=(end < seq_len))
        with torch.no_grad():
            loss_value += float(chunk_loss.detach().cpu())
            abs_loss1_value += float(torch.abs((loss1 * cur_mask).sum(-1) / denom).sum().detach().cpu())
            loss2_value += float(((kl * cur_mask).sum(-1) / denom).sum().detach().cpu())
        del logits, logps, coef1, coef2, loss1, coef3, kl, token_loss, chunk_loss

    if grpo_iteration == 0:
        old_logps_out = torch.cat(old_chunks, dim=1) if old_chunks else torch.empty((input_ids.shape[0], 0))
        ref_logps_out = ref_logps_for_loss.detach().cpu()
    else:
        old_logps_out = old_logps
        ref_logps_out = ref_logps
    return loss_value, abs_loss1_value, loss2_value, old_logps_out, ref_logps_out


def maybe_empty_cuda_cache(config: dict[str, Any], *, force: bool = False) -> bool:
    if not torch.cuda.is_available():
        return False
    rt = config.get("runtime", {})
    if not bool(rt.get("empty_cache_enabled", True)):
        return False
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    threshold_mb = float(rt.get("empty_cache_threshold_mb", 18_000))
    ratio = float(rt.get("empty_cache_reserved_ratio", 1.4))
    should_clear = force or reserved / 1024**2 >= threshold_mb or reserved >= max(allocated * ratio, allocated + 1024**3)
    if should_clear:
        torch.cuda.empty_cache()
        return True
    return False


def _merge_int_dicts(rows: list[dict], key: str) -> dict[int, int]:
    merged: dict[int, int] = {}
    for row in rows:
        for raw_k, raw_v in row.get(key, {}).items():
            k = int(raw_k)
            merged[k] = merged.get(k, 0) + int(raw_v)
    return merged


def _weighted_average(rows: list[dict], value_key: str, weight_key: str) -> float:
    total_weight = sum(float(row.get(weight_key, 0) or 0) for row in rows)
    if total_weight <= 0:
        return 0.0
    return sum(float(row.get(value_key, 0.0) or 0.0) * float(row.get(weight_key, 0) or 0) for row in rows) / total_weight


def _merge_generation_outputs(rows: list[dict]) -> dict:
    if len(rows) == 1:
        return rows[0]
    generated = []
    for row in rows:
        generated.extend(row["generated_token_ids"])
    total_acc_length = sum(int(row.get("total_acc_length", 0)) for row in rows)
    total_decoded = sum(int(row.get("total_decoded_token_num", 0)) for row in rows)
    total_verify = sum(int(row.get("total_verify_rounds", 0)) for row in rows)
    total_time = sum(float(row.get("total_time_cost", 0.0) or 0.0) for row in rows)
    return {
        "generated_token_ids": generated,
        "max_sequence_length": max((int(row.get("max_sequence_length", 0)) for row in rows), default=0),
        "total_acc_length": int(total_acc_length),
        "average_accept_length": total_acc_length / max(total_decoded, 1),
        "accepted_tokens_per_medusa_step": total_acc_length / max(total_decoded, 1),
        "total_decoded_token_num": int(total_decoded),
        "total_accepted_draft_tokens": 0,
        "total_proposed_draft_tokens": 0,
        "total_accepted_medusa_tokens": 0,
        "total_proposed_medusa_tokens": 0,
        "draft_acceptance_rate": 0.0,
        "medusa_acceptance_rate": 0.0,
        "total_verify_rounds": int(total_verify),
        "average_active_batch_size": _weighted_average(rows, "average_active_batch_size", "total_verify_rounds"),
        "average_tree_nodes_per_seq": 1.0,
        "accept_length_histogram": _merge_int_dicts(rows, "accept_length_histogram"),
        "medusa_accept_by_depth": {},
        "medusa_proposed_by_depth": {},
        "last_tree_plan": rows[-1].get("last_tree_plan", {}),
        "cache_update_mode": "target_only",
        "kv_extraction_success_count": 0,
        "kv_extraction_fallback_count": 0,
        "kv_extraction_time": 0.0,
        "recompute_fallback_time": 0.0,
        "oom_count": sum(int(row.get("oom_count", 0)) for row in rows),
        "oom_split_count": sum(int(row.get("oom_split_count", 0)) for row in rows),
        "total_time_cost": total_time,
        "prefill_time_cost": sum(float(row.get("prefill_time_cost", 0.0) or 0.0) for row in rows),
        "target_time_cost": sum(float(row.get("target_time_cost", 0.0) or 0.0) for row in rows),
        "tree_verify_time_cost": 0.0,
        "cache_update_time_cost": sum(float(row.get("cache_update_time_cost", 0.0) or 0.0) for row in rows),
        "medusa_head_time_cost": 0.0,
        "draft_time_cost": 0.0,
        "check_time_cost": 0.0,
    }


def generate_with_oom_retry(
    target_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    *,
    repeated_generate_nums: int,
    max_length: int,
    gen_config: PureGenerateConfig,
    statistical_time: bool,
    enabled: bool,
    max_splits: int,
    split_depth: int = 0,
) -> dict:
    try:
        return pure_target_generate(
            target_model,
            input_ids,
            attention_mask,
            tokenizer,
            repeated_generate_nums=repeated_generate_nums,
            max_length=max_length,
            config=gen_config,
            statistical_time=statistical_time,
        )
    except RuntimeError as exc:
        if (not enabled) or (not _is_cuda_oom(exc)) or input_ids.shape[0] <= 1 or split_depth >= max_splits:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mid = max(1, input_ids.shape[0] // 2)
        outputs = []
        for chunk_ids, chunk_mask in [(input_ids[:mid], attention_mask[:mid]), (input_ids[mid:], attention_mask[mid:])]:
            if chunk_ids.shape[0] == 0:
                continue
            outputs.append(
                generate_with_oom_retry(
                    target_model,
                    chunk_ids,
                    chunk_mask,
                    tokenizer,
                    repeated_generate_nums=repeated_generate_nums,
                    max_length=max_length,
                    gen_config=gen_config,
                    statistical_time=statistical_time,
                    enabled=enabled,
                    max_splits=max_splits,
                    split_depth=split_depth + 1,
                )
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        merged = _merge_generation_outputs(outputs)
        merged["oom_count"] = int(merged.get("oom_count", 0)) + 1
        merged["oom_split_count"] = int(merged.get("oom_split_count", 0)) + 1
        return merged


@dataclass
class Accumulator:
    messages: list
    rewards: list
    std_rewards: list
    used_items: int = 0
    used_items_at_last_update: int = 0


def run_training(config: dict[str, Any]) -> None:
    seed_everything(int(config.get("seed", 42)))
    run_name = str(config.get("run_name", "pure_grpo_run"))
    log_dir = Path(config.get("logging", {}).get("log_dir", f"logs/pure_grpo/{run_name}"))
    logger = MetricsLogger(log_dir, append=bool(_get(config, "logging.append", False)))
    save_resolved_config(config, log_dir / "config_resolved.yaml")
    gpu_monitor = GpuMonitor(
        enabled=bool(_get(config, "runtime.log_gpu_metrics", True)),
        min_interval_s=float(_get(config, "runtime.gpu_poll_interval_s", 10.0)),
    )

    model_dir = str(config["model"]["model_dir"])
    attn_impl_requested = str(_get(config, "model.attn_implementation", "eager"))
    attn_impl = _resolve_attn_implementation(attn_impl_requested)
    model_dtype_name = str(_get(config, "model.dtype", "auto")).lower()
    model_torch_dtype = (
        _dtype_from_name(model_dtype_name)
        if model_dtype_name in {"fp16", "bf16", "fp32"}
        else "auto"
    )
    hf_config = AutoConfig.from_pretrained(model_dir)
    target_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=model_torch_dtype,
        config=hf_config,
        attn_implementation=attn_impl,
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_cfg = config.get("lora", {})
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg.get("r", 64)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
    )
    target_model = get_peft_model(target_model, lora_config)
    load_lora_path = str(config.get("training", {}).get("load_lora_path", ""))
    if load_lora_path:
        target_model.load_adapter(load_lora_path, adapter_name="default")
    target_model.print_trainable_parameters()
    target_model.train()
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=float(_get(config, "training.target_lr", 1e-6)))

    qas = get_train_QAs(str(_get(config, "data.train_option", "simplelr_abel_level3to5")))
    max_train_samples = int(_get(config, "training.max_train_samples", 0))
    if max_train_samples > 0:
        qas = qas[:max_train_samples]
    batch_size = int(_get(config, "training.batch_size", 4))
    num_workers = int(_get(config, "training.num_workers", 4))
    dataloader = DataLoader(
        qas,
        collate_fn=TrainDataCollator(tokenizer, max_prompt_length=int(_get(config, "generation.max_prompt_length", 4096))),
        num_workers=num_workers,
        persistent_workers=bool(_get(config, "training.persistent_workers", True)) and num_workers > 0,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    gen_config = PureGenerateConfig(
        do_sample=bool(_get(config, "generation.do_sample", True)),
        temperature=float(_get(config, "generation.temperature", 1.0)),
        top_p=float(_get(config, "generation.top_p", 0.95)),
        top_k=_get(config, "generation.top_k", None),
    )
    num_epochs = int(_get(config, "training.num_epochs", 1))
    start_epoch = int(_get(config, "training.start_epoch", 0))
    start_batch = int(_get(config, "training.start_batch", 0))
    start_used_items = int(_get(config, "training.start_used_items", 0))
    start_rollout_count = int(_get(config, "training.start_rollout_count", 0))
    if start_batch > 0 and start_rollout_count <= 0:
        start_rollout_count = start_batch
    repeated_generate_nums = int(_get(config, "generation.repeated_generate_nums", 8))
    max_length = int(_get(config, "generation.max_length", 2048))
    accumulation_steps = int(_get(config, "training.accumulation_steps", 4))
    grpo_iteration_num = int(_get(config, "training.grpo_iteration_num", 1))
    max_training_token = int(_get(config, "training.max_training_token", 2048))
    max_training_padding_gap = int(_get(config, "training.max_training_padding_gap", 256))
    logps_chunk_size = int(_get(config, "training.logps_chunk_size", 256))
    beta = float(_get(config, "training.beta", 0.04))
    epsilon = float(_get(config, "training.epsilon", 0.1))
    statistical_time = bool(_get(config, "logging.statistical_time", False))
    generation_oom_split_retry = bool(_get(config, "runtime.generation_oom_split_retry", True))
    generation_oom_max_splits = int(_get(config, "runtime.generation_oom_max_splits", 3))
    save_steps = int(_get(config, "training.save_steps", 500))
    saved_model_dir = Path(_get(config, "training.saved_model_dir", f"outputs/{run_name}/target_lora"))
    saved_model_dir.mkdir(parents=True, exist_ok=True)

    logger.log({
        "phase": "run_config",
        "run_name": run_name,
        "model_dir": model_dir,
        "target_trainable_params": sum(p.numel() for p in target_model.parameters() if p.requires_grad),
        "model_torch_dtype": str(model_torch_dtype).replace("torch.", ""),
        "attn_implementation": attn_impl,
        "start_epoch": start_epoch,
        "start_batch": start_batch,
        "start_used_items": start_used_items,
        "start_rollout_count": start_rollout_count,
        "method": "pure_grpo",
        "baseline_source": "target_only",
    })

    acc = Accumulator(
        messages=[],
        rewards=[],
        std_rewards=[],
        used_items=start_used_items,
        used_items_at_last_update=start_used_items,
    )
    batch_old_logps = []
    batch_ref_logps = []
    total_generate_time = 0.0
    total_train_time = 0.0
    total_rollout_tokens = 0
    rollout_count = start_rollout_count
    start_time = time.time()

    epoch_bar = tqdm(range(start_epoch, num_epochs), desc="Epoch", dynamic_ncols=True)
    for epoch in epoch_bar:
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        batch_iter = enumerate(dataloader)
        if epoch_start_batch > 0:
            for _ in range(epoch_start_batch):
                next(batch_iter, None)
        batch_bar = tqdm(
            batch_iter,
            total=len(dataloader),
            initial=epoch_start_batch,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            dynamic_ncols=True,
            leave=False,
        )
        ignored_correct = 0
        ignored_incorrect = 0
        for batch_idx, batch in batch_bar:
            if batch["input_ids"].shape[-1] >= max_length:
                continue
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            messages = batch["messages"]
            answers = batch["answers"]

            target_model.eval()
            try:
                with torch.inference_mode():
                    outputs = generate_with_oom_retry(
                        target_model,
                        input_ids,
                        attention_mask,
                        tokenizer,
                        repeated_generate_nums=repeated_generate_nums,
                        max_length=max_length,
                        gen_config=gen_config,
                        statistical_time=statistical_time,
                        enabled=generation_oom_split_retry,
                        max_splits=generation_oom_max_splits,
                    )
            except RuntimeError as exc:
                if _is_cuda_oom(exc) and bool(_get(config, "training.save_on_exception", True)):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    emergency_step = max(0, acc.used_items // max(1, batch_size * accumulation_steps))
                    emergency_tag = f"oom_step{emergency_step}_batch{batch_idx}"
                    target_model.save_pretrained(saved_model_dir / emergency_tag)
                    logger.log({
                        "phase": "exception_checkpoint",
                        "epoch": epoch + 1,
                        "batch": batch_idx,
                        "step": emergency_step,
                        "tag": emergency_tag,
                        "error": "cuda_oom",
                    })
                raise
            rollout_count += 1
            target_model.train()

            total_generate_time += outputs["total_time_cost"]
            total_rollout_tokens += sum(len(x) for x in outputs["generated_token_ids"])
            decoded_sequences = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs["generated_token_ids"]]
            token_lengths = [len(item) for item in outputs["generated_token_ids"]]
            length_stdev = stdev(token_lengths) if len(token_lengths) > 1 else 0.0
            length_mean = mean(token_lengths) if token_lengths else 0.0

            for prompt_idx in range(len(answers)):
                decoded_for_prompt = []
                new_messages = []
                ground_truth = answers[prompt_idx]
                for repeat_idx in range(repeated_generate_nums):
                    seq_idx = prompt_idx * repeated_generate_nums + repeat_idx
                    decoded = decoded_sequences[seq_idx]
                    decoded_for_prompt.append(decoded)
                    new_messages.append(messages[prompt_idx] + [{"role": "assistant", "content": decoded}])
                format_rewards = format_reward_func(decoded_for_prompt)
                answer_rewards = accuracy_reward_func(decoded_for_prompt, [ground_truth] * repeated_generate_nums)
                rewards = np.array([0.2 * f + a for f, a in zip(format_rewards, answer_rewards)])
                if rewards.std() == 0:
                    if rewards[0] >= 1.0:
                        ignored_correct += 1
                    else:
                        ignored_incorrect += 1
                    continue
                std_rewards = (rewards - rewards.mean()) / rewards.std()
                acc.messages += new_messages
                acc.rewards += rewards.tolist()
                acc.std_rewards += std_rewards.tolist()
                acc.used_items += 1

            gpu = gpu_monitor.sample()
            rollout_log = {
                "phase": "rollout",
                "epoch": epoch + 1,
                "batch": batch_idx,
                "rollout_count": rollout_count,
                "used_items": acc.used_items,
                "pending_used_items": acc.used_items - acc.used_items_at_last_update,
                "generation_time": outputs["total_time_cost"],
                "prefill_time": outputs.get("prefill_time_cost", 0.0),
                "target_time": outputs.get("target_time_cost", 0.0),
                "tree_verify_time": 0.0,
                "cache_update_time": outputs.get("cache_update_time_cost", 0.0),
                "medusa_head_time": 0.0,
                "draft_time": 0.0,
                "tokens_per_sec_generation": sum(token_lengths) / max(outputs["total_time_cost"], 1e-9),
                "average_accept_length": outputs["average_accept_length"],
                "accepted_tokens_per_medusa_step": outputs["accepted_tokens_per_medusa_step"],
                "medusa_acceptance_rate": 0.0,
                "draft_acceptance_rate": 0.0,
                "total_verify_rounds": outputs.get("total_verify_rounds", 0),
                "accepted_medusa_tokens": 0,
                "proposed_medusa_tokens": 0,
                "tree_nodes_per_seq": 1.0,
                "B_cur_avg": outputs["average_active_batch_size"],
                "mean_response_length": length_mean,
                "response_length_variance": float(np.var(token_lengths)) if token_lengths else 0.0,
                "response_length_stdev": length_stdev,
                "total_rollout_tokens": total_rollout_tokens,
                "eos_count": sum(1 for seq in outputs["generated_token_ids"] if seq and seq[-1] == tokenizer.eos_token_id),
                "cache_update_mode": outputs["cache_update_mode"],
                "oom_count": outputs["oom_count"],
                "oom_split_count": outputs.get("oom_split_count", 0),
                "last_tree_plan": outputs["last_tree_plan"],
                "gpu": gpu.__dict__,
                "baseline_source": "target_only",
            }
            logger.log(rollout_log)

            batch_bar.set_postfix(
                phase="rollout",
                gen=format_duration(outputs["total_time_cost"]),
                acc=f"{outputs['average_accept_length']:.3f}",
                pending=f"{acc.used_items - acc.used_items_at_last_update}/{batch_size * accumulation_steps}",
                refresh=False,
            )

            if not acc.messages:
                continue
            pending = acc.used_items - acc.used_items_at_last_update
            required = max(1, batch_size * accumulation_steps)
            if pending < required:
                continue

            texts = tokenizer.apply_chat_template(acc.messages, tokenize=False, add_generation_prompt=False)
            tokenized = tokenizer(texts, padding=False)
            loss_mask = []
            for message, ids in zip(acc.messages, tokenized.input_ids):
                prompt_text = tokenizer.apply_chat_template(message[:-1], tokenize=False, add_generation_prompt=True)
                prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
                loss_mask.append([0] * max(prompt_len - 1, 0) + [1] * max(len(ids) - prompt_len + 1, 0))

            sorted_pairs = sorted(zip(tokenized.input_ids, tokenized.attention_mask, loss_mask, acc.std_rewards), key=lambda x: len(x[0]))
            all_input_ids, all_attention_mask, all_loss_mask, all_rewards = map(list, zip(*sorted_pairs))
            step = acc.used_items // required
            batch_old_logps.clear()
            batch_ref_logps.clear()

            grpo_bar = tqdm(range(grpo_iteration_num), desc="GRPO", dynamic_ncols=True, leave=False)
            for grpo_iteration in grpo_bar:
                if statistical_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                train_start = time.time()
                target_optimizer.zero_grad(set_to_none=True)
                microbatch_index = 0
                cur_ids = []
                cur_attn = []
                cur_mask = []
                cur_rewards = []
                cur_max_len = 0

                def flush_microbatch():
                    nonlocal microbatch_index, cur_ids, cur_attn, cur_mask, cur_rewards, cur_max_len
                    if not cur_ids:
                        return
                    for row_idx in range(len(cur_ids)):
                        pad = cur_max_len - len(cur_ids[row_idx])
                        if pad > 0:
                            cur_ids[row_idx] += [0] * pad
                            cur_attn[row_idx] += [0] * pad
                            cur_mask[row_idx] += [0] * pad
                    device = next(target_model.parameters()).device
                    mb_ids = torch.tensor(cur_ids, device=device)
                    mb_attn = torch.tensor(cur_attn, device=device)
                    mb_mask = torch.tensor(cur_mask, device=device)
                    mb_rewards = torch.tensor(cur_rewards, device=device).unsqueeze(-1)
                    old_logps = None if grpo_iteration == 0 else batch_old_logps[microbatch_index]
                    ref_logps = None if grpo_iteration == 0 else batch_ref_logps[microbatch_index]
                    loss, abs_loss1, loss2, old_logps, ref_logps = compute_target_loss_and_backward(
                        target_model,
                        mb_ids,
                        mb_attn,
                        mb_mask,
                        mb_rewards,
                        epsilon,
                        beta,
                        grpo_iteration,
                        old_logps=old_logps,
                        ref_logps=ref_logps,
                        chunk_size=logps_chunk_size,
                        loss_scale=1.0 / max(len(acc.messages), 1),
                    )
                    if grpo_iteration == 0:
                        batch_old_logps.append(old_logps)
                        batch_ref_logps.append(ref_logps)
                    microbatch_index += 1
                    cur_ids, cur_attn, cur_mask, cur_rewards = [], [], [], []
                    cur_max_len = 0

                for ids, attn, mask, reward in zip(all_input_ids, all_attention_mask, all_loss_mask, all_rewards):
                    proposed_max = max(cur_max_len, len(ids))
                    fits_token_budget = proposed_max * (len(cur_ids) + 1) <= max_training_token
                    fits_gap = (len(ids) - cur_max_len) * len(cur_ids) <= max_training_padding_gap
                    if cur_ids and not (fits_token_budget and fits_gap):
                        flush_microbatch()
                    cur_max_len = max(cur_max_len, len(ids))
                    cur_ids.append(list(ids))
                    cur_attn.append(list(attn))
                    cur_mask.append(list(mask))
                    cur_rewards.append(float(reward))
                flush_microbatch()
                target_optimizer.step()
                target_optimizer.zero_grad(set_to_none=True)
                cache_cleared = maybe_empty_cuda_cache(
                    config,
                    force=bool(_get(config, "runtime.empty_cache_after_target_train", True)),
                )
                if statistical_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                train_elapsed = time.time() - train_start
                total_train_time += train_elapsed
                grpo_log = {
                    "phase": "target_train",
                    "epoch": epoch + 1,
                    "step": step,
                    "grpo_iteration": grpo_iteration + 1,
                    "used_items": acc.used_items,
                    "pending_used_items": pending,
                    "train_time": train_elapsed,
                    "total_generate_time": total_generate_time,
                    "total_train_time": total_train_time,
                    "mean_reward": float(np.mean(acc.rewards)) if acc.rewards else 0.0,
                    "reward_variance": float(np.var(acc.rewards)) if acc.rewards else 0.0,
                    "ignore_due_correct_cur_epoch": ignored_correct,
                    "ignore_due_incorrect_cur_epoch": ignored_incorrect,
                    "used_time_min": (time.time() - start_time) / 60.0,
                    "baseline_source": "target_only",
                    "empty_cache_after_train": cache_cleared,
                }
                logger.log(grpo_log)
                grpo_bar.set_postfix(step=step, train=format_duration(train_elapsed), reward=f"{grpo_log['mean_reward']:.3f}", refresh=False)

            acc.used_items_at_last_update = acc.used_items
            acc.messages.clear()
            acc.rewards.clear()
            acc.std_rewards.clear()
            batch_old_logps.clear()
            batch_ref_logps.clear()
            if save_steps > 0 and step > 0 and step % save_steps == 0:
                target_model.save_pretrained(saved_model_dir / f"step{step}")

        epoch_bar.set_postfix(
            used=acc.used_items,
            elapsed=format_duration(time.time() - start_time),
            gen=format_duration(total_generate_time),
            train=format_duration(total_train_time),
            refresh=False,
        )

    final_step = max(0, acc.used_items // max(1, batch_size * accumulation_steps))
    target_model.save_pretrained(saved_model_dir / f"step{final_step}")
    summary = {
        "run_name": run_name,
        "final_step": final_step,
        "used_items": acc.used_items,
        "total_generate_time_s": total_generate_time,
        "total_train_time_s": total_train_time,
        "metrics_jsonl": str(logger.jsonl_path),
        "metrics_csv": str(logger.csv_path),
        "saved_model_dir": str(saved_model_dir / f"step{final_step}"),
    }
    logger.write_summary(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
