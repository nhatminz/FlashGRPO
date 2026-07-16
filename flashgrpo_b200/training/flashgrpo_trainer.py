from __future__ import annotations

import json
import os
import time
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from flashgrpo_b200.decoding.flash_medusa_decoder import FlashMedusaConfig, FlashMedusaDecoder
from flashgrpo_b200.models.medusa_heads import MedusaHeads
from flashgrpo_b200.models.qwen_flashgrpo_wrapper import autocast_dtype, unwrap_causal_lm
from flashgrpo_b200.training.online_medusa_trainer import OnlineMedusaConfig, OnlineMedusaTrainer
from flashgrpo_b200.training.reflex_aux import AuxiliaryHeadRefresher, ReliabilityDecision, ReliabilityTracker
from flashgrpo_b200.utils.config import save_resolved_config
from flashgrpo_b200.utils.gpu_monitor import GpuMonitor
from flashgrpo_b200.utils.metrics import MetricsLogger
from flashgrpo_b200.utils.seed import seed_everything
from flashgrpo_b200.utils.timing import format_duration
from helper.get_QAs import get_train_QAs
from helper.rewards import accuracy_reward_func, format_reward_func


def _get(mapping: dict[str, Any], key: str, default=None):
    cur = mapping
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


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
            "but flash_attn is not installed in this environment. Falling back to sdpa."
        )
        return "sdpa"
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


def build_medusa_update_batch(prompt_input_ids, prompt_attention_mask, generated_token_ids, repeated_generate_nums, pad_token_id):
    rows = []
    masks = []
    loss_masks = []
    batch = prompt_input_ids.shape[0]
    for prompt_idx in range(batch):
        prompt_ids = prompt_input_ids[prompt_idx][prompt_attention_mask[prompt_idx].bool()].detach().cpu().tolist()
        for repeat_idx in range(repeated_generate_nums):
            gen_idx = prompt_idx * repeated_generate_nums + repeat_idx
            gen_ids = [int(x) for x in generated_token_ids[gen_idx]]
            ids = prompt_ids + gen_ids
            rows.append(ids)
            masks.append([1] * len(ids))
            loss_masks.append([0] * len(prompt_ids) + [1] * len(gen_ids))
    max_len = max((len(row) for row in rows), default=0)
    for idx in range(len(rows)):
        pad = max_len - len(rows[idx])
        rows[idx] += [pad_token_id] * pad
        masks[idx] += [0] * pad
        loss_masks[idx] += [0] * pad
    return (
        torch.tensor(rows, dtype=torch.long),
        torch.tensor(masks, dtype=torch.long),
        torch.tensor(loss_masks, dtype=torch.long),
    )


@dataclass
class Accumulator:
    token_ids: list
    prompt_lens: list
    rewards: list
    std_rewards: list
    used_items: int = 0
    used_items_at_last_update: int = 0


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


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
    total_accepted = sum(int(row.get("total_accepted_medusa_tokens", row.get("total_accepted_draft_tokens", 0))) for row in rows)
    total_proposed = sum(int(row.get("total_proposed_medusa_tokens", row.get("total_proposed_draft_tokens", 0))) for row in rows)
    total_verify = sum(int(row.get("total_verify_rounds", 0)) for row in rows)
    total_time = sum(float(row.get("total_time_cost", 0.0) or 0.0) for row in rows)
    reflex_metrics = _merge_reflex_metrics([row.get("reflex_metrics", {}) for row in rows])
    out = {
        "generated_token_ids": generated,
        "max_sequence_length": max((int(row.get("max_sequence_length", 0)) for row in rows), default=0),
        "total_acc_length": int(total_acc_length),
        "average_accept_length": total_acc_length / max(total_decoded, 1),
        "accepted_tokens_per_medusa_step": total_acc_length / max(total_decoded, 1),
        "total_decoded_token_num": int(total_decoded),
        "total_accepted_draft_tokens": int(total_accepted),
        "total_proposed_draft_tokens": int(total_proposed),
        "total_accepted_medusa_tokens": int(total_accepted),
        "total_proposed_medusa_tokens": int(total_proposed),
        "draft_acceptance_rate": total_accepted / max(total_proposed, 1),
        "medusa_acceptance_rate": total_accepted / max(total_proposed, 1),
        "total_verify_rounds": int(total_verify),
        "average_active_batch_size": _weighted_average(rows, "average_active_batch_size", "total_verify_rounds"),
        "average_tree_nodes_per_seq": _weighted_average(rows, "average_tree_nodes_per_seq", "total_verify_rounds"),
        "accept_length_histogram": _merge_int_dicts(rows, "accept_length_histogram"),
        "medusa_accept_by_depth": _merge_int_dicts(rows, "medusa_accept_by_depth"),
        "medusa_proposed_by_depth": _merge_int_dicts(rows, "medusa_proposed_by_depth"),
        "last_tree_plan": rows[-1].get("last_tree_plan", {}),
        "cache_update_mode": rows[-1].get("cache_update_mode", "extract_path"),
        "kv_extraction_success_count": sum(int(row.get("kv_extraction_success_count", 0)) for row in rows),
        "kv_extraction_fallback_count": sum(int(row.get("kv_extraction_fallback_count", 0)) for row in rows),
        "kv_extraction_time": sum(float(row.get("kv_extraction_time", 0.0) or 0.0) for row in rows),
        "recompute_fallback_time": sum(float(row.get("recompute_fallback_time", 0.0) or 0.0) for row in rows),
        "oom_count": sum(int(row.get("oom_count", 0)) for row in rows),
        "oom_split_count": sum(int(row.get("oom_split_count", 0)) for row in rows),
        "total_time_cost": total_time,
        "prefill_time_cost": sum(float(row.get("prefill_time_cost", 0.0) or 0.0) for row in rows),
        "target_time_cost": sum(float(row.get("target_time_cost", 0.0) or 0.0) for row in rows),
        "tree_verify_time_cost": sum(float(row.get("tree_verify_time_cost", 0.0) or 0.0) for row in rows),
        "cache_update_time_cost": sum(float(row.get("cache_update_time_cost", 0.0) or 0.0) for row in rows),
        "medusa_head_time_cost": sum(float(row.get("medusa_head_time_cost", 0.0) or 0.0) for row in rows),
        "draft_time_cost": sum(float(row.get("draft_time_cost", 0.0) or 0.0) for row in rows),
        "check_time_cost": sum(float(row.get("check_time_cost", 0.0) or 0.0) for row in rows),
        "reflex_metrics": reflex_metrics,
        "reflex_head_metrics": reflex_metrics.get("per_head", {}),
        "reflex_aux_records": _merge_reflex_aux_records([row.get("reflex_aux_records", {}) for row in rows]),
    }
    return out


def _merge_reflex_aux_records(rows: list[dict]) -> dict:
    rows = [row for row in rows if row and row.get("hidden") is not None and int(row["hidden"].shape[0]) > 0]
    if not rows:
        return {}
    max_prev = max((int(row.get("prev_tokens", torch.empty(0, 0)).shape[1]) for row in rows), default=0)
    merged = {
        "hidden": torch.cat([row["hidden"] for row in rows], dim=0),
        "fast_state": torch.cat([row["fast_state"] for row in rows], dim=0),
        "labels": torch.cat([row["labels"] for row in rows], dim=0),
        "horizons": torch.cat([row["horizons"] for row in rows], dim=0),
        "prev_lens": torch.cat([row["prev_lens"] for row in rows], dim=0),
        "reflex_scale": torch.cat(
            [row.get("reflex_scale", torch.ones((int(row["hidden"].shape[0]),), dtype=torch.float32)) for row in rows],
            dim=0,
        ),
        "target_logsumexp": torch.cat(
            [row.get("target_logsumexp", torch.zeros((int(row["hidden"].shape[0]),), dtype=torch.float32)) for row in rows],
            dim=0,
        ),
        "old_logsumexp": torch.cat(
            [row.get("old_logsumexp", torch.zeros((int(row["hidden"].shape[0]),), dtype=torch.float32)) for row in rows],
            dim=0,
        ),
        "has_sparse_teacher": torch.cat(
            [row.get("has_sparse_teacher", torch.zeros((int(row["hidden"].shape[0]),), dtype=torch.bool)) for row in rows],
            dim=0,
        ),
    }
    prev_chunks = []
    for row in rows:
        prev = row.get("prev_tokens")
        if prev is None:
            prev = torch.full((int(row["hidden"].shape[0]), 0), -1, dtype=torch.long)
        if int(prev.shape[1]) < max_prev:
            pad = torch.full((int(prev.shape[0]), max_prev - int(prev.shape[1])), -1, dtype=torch.long)
            prev = torch.cat([prev, pad], dim=1)
        prev_chunks.append(prev)
    merged["prev_tokens"] = torch.cat(prev_chunks, dim=0) if prev_chunks else torch.empty((0, max_prev), dtype=torch.long)
    for prefix, id_dtype, value_dtype in (
        ("target", torch.int32, torch.float16),
        ("old", torch.int32, torch.float16),
    ):
        id_key = f"{prefix}_top_ids"
        value_key = f"{prefix}_top_logits"
        width = max((int(row.get(id_key, torch.empty(0, 0)).shape[1]) for row in rows), default=0)
        id_chunks = []
        value_chunks = []
        for row in rows:
            count = int(row["hidden"].shape[0])
            ids = row.get(id_key, torch.full((count, 0), -1, dtype=id_dtype))
            values = row.get(value_key, torch.zeros((count, 0), dtype=value_dtype))
            if int(ids.shape[1]) < width:
                ids = torch.cat([ids, torch.full((count, width - int(ids.shape[1])), -1, dtype=id_dtype)], dim=1)
                values = torch.cat([values, torch.zeros((count, width - int(values.shape[1])), dtype=value_dtype)], dim=1)
            id_chunks.append(ids)
            value_chunks.append(values)
        merged[id_key] = torch.cat(id_chunks, dim=0)
        merged[value_key] = torch.cat(value_chunks, dim=0)
    return merged


def _merge_reflex_record_batches(batches: list[dict], max_records: int) -> dict:
    valid = [batch for batch in batches if batch and torch.is_tensor(batch.get("hidden"))]
    if not valid:
        return {}
    # Prefixes and sparse teacher supports can have different widths across
    # rollout batches. The auxiliary merger pads those fields before concat.
    merged = _merge_reflex_aux_records(valid)
    total = int(merged.get("hidden", torch.empty(0)).shape[0])
    limit = max(0, int(max_records))
    if limit > 0 and total > limit:
        # Keep recent records; they best match the current policy/head pair.
        start = total - limit
        merged = {key: value[start:].contiguous() for key, value in merged.items()}
    return merged


def _merge_reflex_metrics(rows: list[dict]) -> dict:
    rows = [row for row in rows if row]
    if not rows:
        return {}
    per_head: dict[str, dict] = {}
    total_updates = sum(int(row.get("num_reflex_updates", 0) or 0) for row in rows)
    feedback_weight = total_updates if total_updates > 0 else 1
    feedback_mean = sum(float(row.get("feedback_rms_mean", row.get("feedback_norm_mean", 0.0)) or 0.0) * int(row.get("num_reflex_updates", 0) or 0) for row in rows) / max(feedback_weight, 1)
    feedback_p95 = max(float(row.get("feedback_rms_p95", row.get("feedback_norm_p95", 0.0)) or 0.0) for row in rows)
    fast_norm_mean = sum(float(row.get("fast_state_rms_mean", row.get("fast_state_norm_mean", 0.0)) or 0.0) for row in rows) / max(len(rows), 1)
    fast_norm_p95 = max(float(row.get("fast_state_rms_p95", row.get("fast_state_norm_p95", 0.0)) or 0.0) for row in rows)
    for row in rows:
        for head, metrics in (row.get("per_head") or {}).items():
            out = per_head.setdefault(
                str(head),
                {"mature": 0, "accepted": 0, "ce_sum": 0.0, "tv_sum": 0.0, "gated": 0.0, "depth_buckets": {}},
            )
            mature = int(metrics.get("mature", 0) or 0)
            out["mature"] += mature
            out["accepted"] += int(metrics.get("accepted", 0) or 0)
            out["ce_sum"] += float(metrics.get("mature_ce", 0.0) or 0.0) * mature
            out["tv_sum"] += float(metrics.get("sparse_tv", 0.0) or 0.0) * mature
            out["gated"] += float(metrics.get("nonzero_gate_fraction", 0.0) or 0.0) * mature
            for bucket, values in (metrics.get("depth_buckets") or {}).items():
                count = int(values.get("mature", 0) or 0)
                bucket_out = out["depth_buckets"].setdefault(
                    str(bucket),
                    {"mature": 0, "accepted": 0.0, "tv_sum": 0.0},
                )
                bucket_out["mature"] += count
                bucket_out["accepted"] += float(values.get("acceptance_rate", 0.0) or 0.0) * count
                bucket_out["tv_sum"] += float(values.get("sparse_tv", 0.0) or 0.0) * count
    for head, metrics in per_head.items():
        mature = int(metrics.pop("mature", 0))
        accepted = int(metrics.pop("accepted", 0))
        ce_sum = float(metrics.pop("ce_sum", 0.0))
        tv_sum = float(metrics.pop("tv_sum", 0.0))
        gated = float(metrics.pop("gated", 0.0))
        raw_buckets = metrics.pop("depth_buckets", {})
        acc = accepted / max(mature, 1)
        buckets = {
            bucket: {
                "mature": int(values["mature"]),
                "acceptance_rate": float(values["accepted"]) / max(int(values["mature"]), 1),
                "sparse_tv": float(values["tv_sum"]) / max(int(values["mature"]), 1),
            }
            for bucket, values in raw_buckets.items()
        }
        per_head[head] = {
            "mature": mature,
            "accepted": accepted,
            "acceptance_rate": acc,
            "rejection_rate": 1.0 - acc if mature else 0.0,
            "mature_ce": ce_sum / max(mature, 1),
            "sparse_tv": tv_sum / max(mature, 1),
            "nonzero_gate_fraction": gated / max(mature, 1),
            "depth_buckets": buckets,
        }
    feedback_collection_rounds = sum(int(row.get("feedback_collection_rounds", 0) or 0) for row in rows)
    return {
        "enabled": any(bool(row.get("enabled", False)) for row in rows),
        "feedback_enabled": any(bool(row.get("feedback_enabled", False)) for row in rows),
        "proposal_injection_enabled": any(bool(row.get("proposal_injection_enabled", False)) for row in rows),
        "num_reflex_updates": int(total_updates),
        "feedback_rms_mean": feedback_mean,
        "feedback_rms_p95": feedback_p95,
        "feedback_norm_mean": feedback_mean,
        "feedback_norm_p95": feedback_p95,
        "fast_state_rms_mean": fast_norm_mean,
        "fast_state_rms_p95": fast_norm_p95,
        "fast_state_norm_mean": fast_norm_mean,
        "fast_state_norm_p95": fast_norm_p95,
        "effective_feedback_updates_mean": sum(float(row.get("effective_feedback_updates_mean", 0.0) or 0.0) for row in rows) / max(len(rows), 1),
        "nonzero_gate_fraction": sum(float(row.get("nonzero_gate_fraction", 0.0) or 0.0) for row in rows) / max(len(rows), 1),
        "feedback_collection_rounds": feedback_collection_rounds,
        "pending_prediction_records": sum(int(row.get("pending_prediction_records", 0) or 0) for row in rows),
        "numerical_reset_count": sum(int(row.get("numerical_reset_count", 0) or 0) for row in rows),
        "per_head": per_head,
    }


def generate_with_oom_retry(
    decoder: FlashMedusaDecoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    repeated_generate_nums: int,
    max_length: int,
    statistical_time: bool,
    generation_step: int,
    enabled: bool,
    max_splits: int,
    collect_reflex_aux_cache: bool = False,
    split_depth: int = 0,
) -> dict:
    try:
        return decoder.generate(
            input_ids,
            attention_mask,
            repeated_generate_nums=repeated_generate_nums,
            max_length=max_length,
            statistical_time=statistical_time,
            generation_step=generation_step,
            collect_reflex_aux_cache=collect_reflex_aux_cache,
        )
    except RuntimeError as exc:
        if (not enabled) or (not _is_cuda_oom(exc)) or input_ids.shape[0] <= 1 or split_depth >= max_splits:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mid = max(1, input_ids.shape[0] // 2)
        chunks = [
            (input_ids[:mid], attention_mask[:mid]),
            (input_ids[mid:], attention_mask[mid:]),
        ]
        outputs = []
        for chunk_ids, chunk_mask in chunks:
            if chunk_ids.shape[0] == 0:
                continue
            outputs.append(
                generate_with_oom_retry(
                    decoder,
                    chunk_ids,
                    chunk_mask,
                    repeated_generate_nums=repeated_generate_nums,
                    max_length=max_length,
                    statistical_time=statistical_time,
                    generation_step=generation_step,
                    enabled=enabled,
                    max_splits=max_splits,
                    collect_reflex_aux_cache=collect_reflex_aux_cache,
                    split_depth=split_depth + 1,
                )
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        merged = _merge_generation_outputs(outputs)
        merged["oom_count"] = int(merged.get("oom_count", 0)) + 1
        merged["oom_split_count"] = int(merged.get("oom_split_count", 0)) + 1
        return merged


def _make_flash_config(config: dict[str, Any]) -> FlashMedusaConfig:
    fg = config.get("flashgrpo", {})
    gen = config.get("generation", {})
    reflex = config.get("reflex", {})
    aux = config.get("aux_update", {})
    return FlashMedusaConfig(
        num_medusa_heads=int(fg.get("num_medusa_heads", 3)),
        tree_mode=fg.get("tree_mode", "concurrency_aware"),
        tree_layout=fg.get("tree_layout", "dense"),
        acceptance=fg.get("acceptance", "exact_target"),
        cache_update_mode=fg.get("cache_update_mode", "extract_path"),
        allow_recompute_fallback=bool(fg.get("allow_recompute_fallback", True)),
        cpeak_nodes=int(fg.get("cpeak_nodes", 64)),
        min_tree_nodes_per_seq=int(fg.get("min_tree_nodes_per_seq", 1)),
        max_tree_nodes_per_seq=int(fg.get("max_tree_nodes_per_seq", 16)),
        max_tree_depth=int(fg.get("max_tree_depth", int(fg.get("num_medusa_heads", 3)) + 1)),
        fixed_tree_topk_by_depth=tuple(fg.get("fixed_tree_topk_by_depth", [4, 3, 2])),
        do_sample=bool(gen.get("do_sample", True)),
        temperature=float(gen.get("temperature", 1.0)),
        top_p=float(gen.get("top_p", 0.95)),
        top_k=gen.get("top_k", None),
        clone_tree_cache=bool(fg.get("clone_tree_cache", True)),
        enable_medusa_spec_after=int(fg.get("enable_medusa_spec_after", 0)),
        proposal_mode=str(fg.get("proposal_mode", "medusa")),
        chain_enable_after=int(fg.get("chain_enable_after", 0)),
        chain_bootstrap_from_medusa=bool(fg.get("chain_bootstrap_from_medusa", True)),
        adaptive_tree_enabled=bool(fg.get("adaptive_tree_enabled", False)),
        adaptive_confidence_metric=str(fg.get("adaptive_confidence_metric", "top1_prob")),
        adaptive_confidence_quantile=float(fg.get("adaptive_confidence_quantile", 0.25)),
        adaptive_confidence_low=float(fg.get("adaptive_confidence_low", 0.15)),
        adaptive_confidence_high=float(fg.get("adaptive_confidence_high", 0.45)),
        adaptive_min_topk_by_depth=tuple(fg.get("adaptive_min_topk_by_depth", [1, 1, 1])),
        reflex_enabled=bool(reflex.get("enabled", False)),
        reflex_state_space=str(reflex.get("state_space", "projected")),
        reflex_fast_state_dim=int(reflex.get("fast_state_dim", 128)),
        reflex_beta=float(reflex.get("beta", 0.95)),
        reflex_eta=float(reflex.get("eta", 0.1)),
        reflex_top_m_feedback=int(reflex.get("proposal_topk", reflex.get("top_m_feedback", 64))),
        reflex_feedback_stride=int(reflex.get("feedback_stride", 1)),
        reflex_feedback_stride_min=int(reflex.get("feedback_stride_min", 1)),
        reflex_target_topk=int(reflex.get("target_topk", 32)),
        reflex_feedback_union_cap=int(reflex.get("feedback_union_cap", 96)),
        reflex_tv_gate_low=float(reflex.get("tv_gate_low", 0.05)),
        reflex_tv_gate_high=float(reflex.get("tv_gate_high", 0.20)),
        reflex_horizon_weight_decay=float(reflex.get("horizon_weight_decay", 0.85)),
        reflex_half_life_tokens=float(reflex.get("half_life_tokens", 48.0)),
        reflex_feedback_variance_beta=float(reflex.get("feedback_variance_beta", 0.99)),
        reflex_feedback_rms_clip=float(reflex.get("feedback_rms_clip", 3.0)),
        reflex_state_rms_clip=float(reflex.get("state_rms_clip", 2.0)),
        reflex_numerical_reset_rms=float(reflex.get("numerical_reset_rms", 2.5)),
        reflex_relative_rms_delta_base=float(reflex.get("relative_rms_delta_base", 0.01)),
        reflex_warmup_effective_updates=float(reflex.get("warmup_effective_updates", 16.0)),
        reflex_magnitude_gate_floor=float(reflex.get("magnitude_gate_floor", 0.25)),
        reflex_guard_calibration_rollouts=int(reflex.get("guard_calibration_rollouts", 20)),
        reflex_guard_aal_drop_fraction=float(reflex.get("guard_aal_drop_fraction", 0.05)),
        reflex_guard_patience=int(reflex.get("guard_patience", 2)),
        reflex_guard_disable_rollouts=int(reflex.get("guard_disable_rollouts", 50)),
        reflex_feedback_clip_norm=float(reflex.get("feedback_clip_norm", 2.0)),
        reflex_hidden_feedback_clip_norm=float(reflex.get("hidden_feedback_clip_norm", 0.0)),
        reflex_fast_state_clip_norm=float(reflex.get("fast_state_clip_norm", 8.0)),
        reflex_correction_clip_norm=float(reflex.get("correction_clip_norm", 1.0)),
        reflex_normalize_correction=bool(reflex.get("normalize_correction", True)),
        reflex_feedback_ce_gate=bool(reflex.get("feedback_ce_gate", True)),
        reflex_feedback_ce_tau=float(reflex.get("feedback_ce_tau", 4.0)),
        reflex_feedback_ce_threshold=float(reflex.get("feedback_ce_threshold", 0.4)),
        reflex_normalize_feedback=bool(reflex.get("normalize_feedback", True)),
        reflex_feedback_enabled=bool(reflex.get("feedback_enabled", False)),
        reflex_proposal_injection_enabled=bool(reflex.get("proposal_injection_enabled", False)),
        reflex_proposal_injection_scale=float(reflex.get("proposal_injection_scale", 0.0)),
        reflex_proposal_injection_after=int(reflex.get("proposal_injection_after", 0)),
        reflex_proposal_injection_warmup=int(reflex.get("proposal_injection_warmup", 0)),
        reflex_aux_cache_enabled=bool(aux.get("reflex_cache_enabled", False)),
        reflex_aux_cache_max_records=int(aux.get("max_cached_records", fg.get("medusa_max_tokens_per_update", 8192))),
        reflex_aux_cache_stride=int(aux.get("cache_stride", 1)),
        reflex_aux_store_fast_state=bool(aux.get("update_fast_state_injections", False)),
    )


def maybe_empty_cuda_cache(config: dict[str, Any], *, force: bool = False) -> bool:
    if not torch.cuda.is_available():
        return False
    fg = config.get("flashgrpo", {})
    if not bool(fg.get("empty_cache_enabled", True)):
        return False
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    threshold_mb = float(fg.get("empty_cache_threshold_mb", 18_000))
    ratio = float(fg.get("empty_cache_reserved_ratio", 1.4))
    should_clear = force or reserved / 1024**2 >= threshold_mb or reserved >= max(allocated * ratio, allocated + 1024**3)
    if should_clear:
        torch.cuda.empty_cache()
        return True
    return False


def _aux_refresh_allowed_for_generation_step(config: dict[str, Any], generation_step: int) -> tuple[bool, str]:
    aux_cfg = config.get("aux_update", {})
    if not bool(aux_cfg.get("defer_until_reflex_warmup_complete", False)):
        return True, ""
    reflex_cfg = config.get("reflex", {})
    if not bool(reflex_cfg.get("enabled", False)) or not bool(reflex_cfg.get("proposal_injection_enabled", False)):
        return True, ""
    after = int(reflex_cfg.get("proposal_injection_after", 0))
    warmup = int(reflex_cfg.get("proposal_injection_warmup", 0))
    ready_step = after + max(0, warmup)
    if int(generation_step) < ready_step:
        return False, f"reflex_warmup_until_step_{ready_step}"
    return True, ""


def run_training(config: dict[str, Any]) -> None:
    seed_everything(int(config.get("seed", 42)))
    run_name = str(config.get("run_name", "flashgrpo_run"))
    log_dir = Path(config.get("logging", {}).get("log_dir", f"logs/flashgrpo/{run_name}"))
    logger = MetricsLogger(log_dir, append=bool(_get(config, "logging.append", False)))
    save_resolved_config(config, log_dir / "config_resolved.yaml")
    gpu_monitor = GpuMonitor(
        enabled=bool(_get(config, "flashgrpo.log_gpu_metrics", True)),
        min_interval_s=float(_get(config, "flashgrpo.gpu_poll_interval_s", 10.0)),
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

    base = unwrap_causal_lm(target_model)
    fg = config.get("flashgrpo", {})
    reflex_cfg = config.get("reflex", {})
    reflex_enabled = bool(reflex_cfg.get("enabled", False))
    reflex_state_space = str(reflex_cfg.get("state_space", "projected"))
    reflex_fast_state_dim = (
        int(hf_config.hidden_size)
        if reflex_enabled and reflex_state_space == "hidden"
        else int(reflex_cfg.get("fast_state_dim", 128)) if reflex_enabled else 0
    )
    medusa_reflex_fast_state_dim = 0 if reflex_state_space == "hidden" else reflex_fast_state_dim
    medusa_dtype = _dtype_from_name(str(fg.get("head_dtype", "fp32")), default=torch.float32)
    medusa_heads = MedusaHeads(
        hidden_size=hf_config.hidden_size,
        vocab_size=hf_config.vocab_size,
        num_heads=int(fg.get("num_medusa_heads", 3)),
        dtype=medusa_dtype,
        tie_lm_head=bool(fg.get("tie_lm_head", True)),
        lm_head=base.lm_head,
        medusa_loss_decay=float(fg.get("medusa_loss_decay", 0.8)),
        chain_bottleneck_ratio=int(fg.get("chain_bottleneck_ratio", 8)),
        chain_gate_init=float(fg.get("chain_gate_init", -3.0)),
        reflex_fast_state_dim=medusa_reflex_fast_state_dim,
        reflex_init_scale=float(reflex_cfg.get("init_scale", 0.0)),
    ).cuda()
    medusa_checkpoint = str(
        fg.get("medusa_heads_checkpoint", "")
        or config.get("aux_head_checkpoint", "")
        or fg.get("load_medusa_path", "")
    )
    require_pretrained_heads = bool(fg.get("require_pretrained_heads", False))
    allow_random_init = bool(fg.get("allow_random_init", not require_pretrained_heads))
    loaded_reflex_up = False
    if medusa_checkpoint:
        medusa_heads = MedusaHeads.from_pretrained(
            medusa_checkpoint,
            map_location="cpu",
            dtype=medusa_dtype,
            lm_head=base.lm_head,
            chain_bottleneck_ratio=int(fg.get("chain_bottleneck_ratio", 8)),
            chain_gate_init=float(fg.get("chain_gate_init", -3.0)),
            reflex_fast_state_dim=medusa_reflex_fast_state_dim,
            reflex_init_scale=float(reflex_cfg.get("init_scale", 0.0)),
        ).cuda()
        print(f"Loaded MEDUSA heads from {medusa_checkpoint}")
    elif require_pretrained_heads and not allow_random_init:
        raise FileNotFoundError(
            "flashgrpo.require_pretrained_heads=true but flashgrpo.medusa_heads_checkpoint is empty. "
            "Run flashgrpo/scripts/pretrain_medusa_heads.py first or set allow_random_init=true for debugging."
        )
    elif not medusa_checkpoint:
        print("Warning: MEDUSA heads are randomly initialized.")
    loaded_keys = getattr(medusa_heads, "_loaded_compatible_keys", set())
    loaded_reflex_up = any(str(key).startswith("reflex_up.") for key in loaded_keys)
    reflex_injection_enabled = bool(reflex_enabled and reflex_cfg.get("proposal_injection_enabled", False))
    if reflex_injection_enabled and reflex_state_space != "hidden":
        allow_random_reflex = bool(reflex_cfg.get("allow_random_reflex", False))
        if (not medusa_checkpoint or not loaded_reflex_up) and not allow_random_reflex:
            raise RuntimeError(
                "Reflex proposal injection is enabled, but the loaded auxiliary checkpoint does not contain "
                "compatible reflex_up.* weights. This would run with randomly initialized/zero Reflex correction. "
                "Pretrain Reflex first, pass that checkpoint via flashgrpo.medusa_heads_checkpoint or "
                "aux_head_checkpoint, or disable reflex.proposal_injection_enabled for a Medusa-only baseline."
            )
        if getattr(medusa_heads, "reflex_up", None):
            with torch.no_grad():
                reflex_up_abs_sum = sum(float(up.weight.detach().float().abs().sum().cpu()) for up in medusa_heads.reflex_up)
            if reflex_up_abs_sum <= 0.0:
                raise RuntimeError(
                    "Reflex proposal injection is enabled but all reflex_up weights are zero. "
                    "Run flashgrpo/scripts/pretrain_medusa_heads.py with the Reflex warm-start config first, "
                    "or disable reflex.proposal_injection_enabled for a baseline run."
                )

    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=float(_get(config, "training.target_lr", 1e-6)))
    medusa_optimizer = torch.optim.AdamW(
        medusa_heads.parameters(),
        lr=float(fg.get("medusa_lr", 5e-4)),
        weight_decay=float(fg.get("medusa_weight_decay", 0.0)),
        eps=float(fg.get("medusa_adam_eps", 1e-6)),
    )
    medusa_trainer = OnlineMedusaTrainer(
        target_model,
        medusa_heads,
        medusa_optimizer,
        OnlineMedusaConfig(
            medusa_lr=float(fg.get("medusa_lr", 5e-4)),
            medusa_weight_decay=float(fg.get("medusa_weight_decay", 0.0)),
            medusa_train_every=int(fg.get("medusa_train_every", 1)),
            medusa_update_steps_per_iter=int(fg.get("medusa_update_steps_per_iter", 1)),
            medusa_microbatch_size=int(fg.get("medusa_microbatch_size", 1)),
            medusa_max_tokens_per_update=int(fg.get("medusa_max_tokens_per_update", 8192)),
            medusa_loss_decay=float(fg.get("medusa_loss_decay", 0.8)),
            medusa_loss_chunk_size=int(fg.get("medusa_loss_chunk_size", 64)),
            chain_loss_weight=float(fg.get("chain_loss_weight", 0.0)),
            chain_loss_max_depth=int(fg.get("chain_loss_max_depth", int(fg.get("num_medusa_heads", 3)))),
            chain_bootstrap_from_medusa=bool(fg.get("chain_bootstrap_from_medusa", True)),
            reflex_record_microbatch_size=int(config.get("aux_update", {}).get("reflex_record_microbatch_size", 256)),
            reflex_correction_clip_norm=float(reflex_cfg.get("correction_clip_norm", 1.0)),
            reflex_normalize_correction=bool(reflex_cfg.get("normalize_correction", True)),
            rollback_nonfinite_update=bool(config.get("aux_update", {}).get("rollback_nonfinite_update", True)),
            refresh_distill_weight=float(config.get("aux_update", {}).get("distill_weight", 0.7)),
            refresh_hard_token_weight=float(config.get("aux_update", {}).get("hard_token_weight", 0.3)),
            refresh_proximal_weight=float(config.get("aux_update", {}).get("proximal_kl_weight", 0.1)),
        ),
    )
    aux_cfg = config.get("aux_update", {})
    default_aux_mode = "reliability_triggered" if reflex_enabled else "always"
    aux_tracker = ReliabilityTracker(
        mode=str(aux_cfg.get("mode", default_aux_mode)),
        calibration_iterations=int(aux_cfg.get("calibration_iterations", 20)),
        check_interval=int(aux_cfg.get("check_interval", aux_cfg.get("interval", 20))),
        min_records_per_head=int(aux_cfg.get("min_records_per_head", aux_cfg.get("min_mature_records", 1024))),
        trigger_z_high=float(aux_cfg.get("trigger_z_high", aux_cfg.get("drift_threshold", 2.5))),
        acceptance_drop_min=float(aux_cfg.get("acceptance_drop_min", 0.05)),
        patience=int(aux_cfg.get("patience", 2)),
        cooldown_iterations=int(aux_cfg.get("cooldown_iterations", 50)),
        max_heads_per_event=int(aux_cfg.get("max_heads_per_event", 2)),
        mad_floor=float(aux_cfg.get("mad_floor", 0.01)),
    )
    decoder = FlashMedusaDecoder(target_model, medusa_heads, tokenizer, _make_flash_config(config))

    qas = get_train_QAs(str(_get(config, "data.train_option", "simplelr_abel_level3to5")))
    # Keep FastGRPO semantics: sample_num is a logging window, not a dataset
    # limiter. Use max_train_samples for smoke/debug subset runs.
    sample_num = int(_get(config, "training.sample_num", 100))
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
    generation_oom_split_retry = bool(_get(config, "flashgrpo.generation_oom_split_retry", True))
    generation_oom_max_splits = int(_get(config, "flashgrpo.generation_oom_max_splits", 3))
    save_steps = int(_get(config, "training.save_steps", 500))
    saved_model_dir = Path(_get(config, "training.saved_model_dir", f"outputs/{run_name}/flashgrpo_target"))
    saved_medusa_dir = Path(_get(config, "training.saved_medusa_dir", f"outputs/{run_name}/flashgrpo_medusa"))
    saved_model_dir.mkdir(parents=True, exist_ok=True)
    saved_medusa_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cfg = config.get("checkpoint", {})
    aux_refresher = AuxiliaryHeadRefresher.from_config(
        medusa_heads=medusa_heads,
        trainer=medusa_trainer,
        optimizer=medusa_optimizer,
        save_dir=saved_medusa_dir,
        aux_cfg=aux_cfg,
        checkpoint_cfg=checkpoint_cfg,
        fallback_steps=int(fg.get("medusa_update_steps_per_iter", 1)),
    )
    logger.log({
        "phase": "run_config",
        "run_name": run_name,
        "model_dir": model_dir,
        "target_trainable_params": sum(p.numel() for p in target_model.parameters() if p.requires_grad),
        "medusa_params": sum(p.numel() for p in medusa_heads.parameters() if p.requires_grad),
        "start_epoch": start_epoch,
        "start_batch": start_batch,
        "start_used_items": start_used_items,
        "start_rollout_count": start_rollout_count,
        "baseline_source": "not_available",
        "method": "flashgrpo",
        "model_dtype_requested": model_dtype_name,
        "model_torch_dtype": str(model_torch_dtype).replace("torch.", ""),
        "attn_implementation_requested": attn_impl_requested,
        "attn_implementation_resolved": attn_impl,
        "medusa_head_dtype": str(medusa_dtype).replace("torch.", ""),
        "reflex_enabled": bool(reflex_enabled),
        "reflex_state_space": reflex_state_space,
        "reflex_fast_state_dim": int(reflex_fast_state_dim),
        "reflex_feedback_enabled": bool(reflex_cfg.get("feedback_enabled", False)),
        "reflex_proposal_injection_enabled": bool(reflex_cfg.get("proposal_injection_enabled", False)),
        "reflex_proposal_injection_scale": float(reflex_cfg.get("proposal_injection_scale", 0.0)),
        "reflex_proposal_injection_warmup": int(reflex_cfg.get("proposal_injection_warmup", 0)),
        "reflex_feedback_ce_gate": bool(reflex_cfg.get("feedback_ce_gate", True)),
        "aux_head_checkpoint": medusa_checkpoint,
        "aux_checkpoint_has_reflex_up": bool(loaded_reflex_up),
        "aux_update_mode": str(aux_cfg.get("mode", default_aux_mode)),
        "aux_calibration_iterations": int(aux_cfg.get("calibration_iterations", 20)),
        "aux_update_interval": int(aux_cfg.get("check_interval", aux_cfg.get("interval", 20))),
        "aux_update_steps": aux_refresher.config.steps,
        "aux_update_before_policy_step": True,
        "aux_update_defer_until_reflex_warmup_complete": bool(aux_cfg.get("defer_until_reflex_warmup_complete", False)),
        "medusa_lr": float(fg.get("medusa_lr", 5e-4)),
        "medusa_adam_eps": float(fg.get("medusa_adam_eps", 1e-6)),
        "save_aux_every_grpo_iters": int(checkpoint_cfg.get("save_aux_every_grpo_iters", 0)),
        "save_aux_on_triggered_update": bool(checkpoint_cfg.get("save_aux_on_triggered_update", False)),
    })

    acc = Accumulator(
        token_ids=[],
        prompt_lens=[],
        rewards=[],
        std_rewards=[],
        used_items=start_used_items,
        used_items_at_last_update=start_used_items,
    )
    batch_old_logps = []
    batch_ref_logps = []
    total_generate_time = 0.0
    total_train_time = 0.0
    total_head_update_time = 0.0
    total_rollout_tokens = 0
    total_accepted_length = 0
    total_decoded_steps = 0
    total_accepted_medusa_tokens = 0
    total_proposed_medusa_tokens = 0
    total_verify_rounds = 0
    rollout_count = start_rollout_count
    required_used_items = max(1, batch_size * accumulation_steps)
    pending_aux_record_batches: list[dict] = []
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
            answers = batch["answers"]
            online_aux_enabled = bool(fg.get("online_medusa", True))
            generation_step_for_batch = int(rollout_count)
            next_grpo_step = acc.used_items // required_used_items + 1
            aux_refresh_allowed, aux_refresh_skip_reason = _aux_refresh_allowed_for_generation_step(config, generation_step_for_batch)
            collect_reflex_aux_cache = bool(
                online_aux_enabled
                and aux_refresher.config.reflex_cache_enabled
                and aux_refresh_allowed
                and aux_tracker.should_evaluate(next_grpo_step)
            )

            target_model.eval()
            medusa_heads.eval()
            try:
                with torch.inference_mode():
                    outputs = generate_with_oom_retry(
                        decoder,
                        input_ids,
                        attention_mask,
                        repeated_generate_nums=repeated_generate_nums,
                        max_length=max_length,
                        statistical_time=statistical_time,
                        generation_step=generation_step_for_batch,
                        enabled=generation_oom_split_retry,
                        max_splits=generation_oom_max_splits,
                        collect_reflex_aux_cache=collect_reflex_aux_cache,
                    )
            except RuntimeError as exc:
                if _is_cuda_oom(exc) and bool(_get(config, "training.save_on_exception", True)):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    emergency_step = max(0, acc.used_items // max(1, batch_size * accumulation_steps))
                    emergency_tag = f"oom_step{emergency_step}_batch{batch_idx}"
                    target_model.save_pretrained(saved_model_dir / emergency_tag)
                    medusa_heads.save_pretrained(saved_medusa_dir / emergency_tag)
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
            medusa_heads.train()
            total_generate_time += outputs["total_time_cost"]
            total_rollout_tokens += sum(len(x) for x in outputs["generated_token_ids"])
            total_accepted_length += int(outputs.get("total_acc_length", 0))
            total_decoded_steps += int(outputs.get("total_decoded_token_num", 0))
            total_accepted_medusa_tokens += int(outputs.get("total_accepted_medusa_tokens", 0))
            total_proposed_medusa_tokens += int(outputs.get("total_proposed_medusa_tokens", 0))
            total_verify_rounds += int(outputs.get("total_verify_rounds", 0))
            decoded_sequences = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs["generated_token_ids"]]
            prompt_ids_cpu = input_ids.detach().cpu()
            prompt_mask_cpu = attention_mask.detach().cpu().bool()
            prompt_token_rows = [
                prompt_ids_cpu[row][prompt_mask_cpu[row]].tolist()
                for row in range(prompt_ids_cpu.shape[0])
            ]
            token_lengths = [len(item) for item in outputs["generated_token_ids"]]
            length_stdev = stdev(token_lengths) if len(token_lengths) > 1 else 0.0
            length_mean = mean(token_lengths) if token_lengths else 0.0

            head_stats = {"medusa_loss": 0.0, "head_update_time": 0.0, "head_update_tokens": 0, "aux_update_deferred": False}
            aux_tracker.observe(outputs.get("reflex_metrics", {}))
            reflex_records = outputs.get("reflex_aux_records", {})
            if collect_reflex_aux_cache and reflex_records.get("hidden") is not None:
                pending_aux_record_batches.append(reflex_records)
            aux_decision = ReliabilityDecision(
                evaluated=False,
                triggered=False,
                reason=aux_refresh_skip_reason or "await_grpo_boundary",
                head_metrics=(outputs.get("reflex_metrics", {}) or {}).get("per_head", {}),
            )
            head_stats["aux_update_evaluated"] = bool(aux_decision.evaluated)
            head_stats["aux_update_triggered"] = bool(aux_decision.triggered)
            head_stats["aux_update_reason"] = aux_decision.reason
            head_stats["aux_triggered_heads"] = aux_decision.triggered_heads
            head_stats["aux_drift_scores"] = aux_decision.drift_scores

            for prompt_idx in range(len(answers)):
                decoded_for_prompt = []
                ground_truth = answers[prompt_idx]
                for repeat_idx in range(repeated_generate_nums):
                    seq_idx = prompt_idx * repeated_generate_nums + repeat_idx
                    decoded = decoded_sequences[seq_idx]
                    decoded_for_prompt.append(decoded)
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
                prompt_tokens = prompt_token_rows[prompt_idx]
                for repeat_idx in range(repeated_generate_nums):
                    seq_idx = prompt_idx * repeated_generate_nums + repeat_idx
                    acc.token_ids.append(prompt_tokens + [int(token) for token in outputs["generated_token_ids"][seq_idx]])
                    acc.prompt_lens.append(len(prompt_tokens))
                acc.rewards += rewards.tolist()
                acc.std_rewards += std_rewards.tolist()
                acc.used_items += 1

            gpu = gpu_monitor.sample()
            rollout_log = {
                "phase": "rollout",
                "epoch": epoch + 1,
                "batch": batch_idx,
                "used_items": acc.used_items,
                "pending_used_items": acc.used_items - acc.used_items_at_last_update,
                "generation_time": outputs["total_time_cost"],
                "prefill_time": outputs.get("prefill_time_cost", 0.0),
                "target_time": outputs.get("target_time_cost", 0.0),
                "tree_verify_time": outputs.get("tree_verify_time_cost", 0.0),
                "cache_update_time": outputs.get("cache_update_time_cost", 0.0),
                "medusa_head_time": outputs.get("medusa_head_time_cost", 0.0),
                "draft_time": outputs.get("draft_time_cost", 0.0),
                "tokens_per_sec_generation": sum(token_lengths) / max(outputs["total_time_cost"], 1e-9),
                "average_accept_length": outputs["average_accept_length"],
                "accepted_tokens_per_medusa_step": outputs["accepted_tokens_per_medusa_step"],
                "medusa_acceptance_rate": outputs["medusa_acceptance_rate"],
                "draft_acceptance_rate": outputs["draft_acceptance_rate"],
                "total_verify_rounds": outputs.get("total_verify_rounds", 0),
                "accepted_medusa_tokens": outputs.get("total_accepted_medusa_tokens", 0),
                "proposed_medusa_tokens": outputs.get("total_proposed_medusa_tokens", 0),
                "tree_nodes_per_seq": outputs["average_tree_nodes_per_seq"],
                "B_cur_avg": outputs["average_active_batch_size"],
                "medusa_loss": head_stats.get("medusa_loss", 0.0),
                "parallel_medusa_loss": head_stats.get("parallel_medusa_loss", 0.0),
                "chain_loss": head_stats.get("chain_loss", 0.0),
                "chain_loss_weight": head_stats.get("chain_loss_weight", float(fg.get("chain_loss_weight", 0.0))),
                "head_update_time": head_stats.get("head_update_time", 0.0),
                "head_update_tokens": head_stats.get("head_update_tokens", 0),
                "head_update_tokens_per_sec": head_stats.get("head_update_tokens_per_sec", 0.0),
                "head_update_time_ratio_vs_total": head_stats.get("head_update_time", 0.0) / max(outputs["total_time_cost"] + head_stats.get("head_update_time", 0.0), 1e-9),
                "aux_update_evaluated": head_stats.get("aux_update_evaluated", False),
                "aux_update_triggered": head_stats.get("aux_update_triggered", False),
                "aux_update_reason": head_stats.get("aux_update_reason", ""),
                "aux_triggered_heads": head_stats.get("aux_triggered_heads", []),
                "aux_drift_scores": head_stats.get("aux_drift_scores", {}),
                "aux_update_deferred": head_stats.get("aux_update_deferred", False),
                "aux_pending_jobs": 0,
                "aux_pending_jobs_dropped": head_stats.get("aux_pending_jobs_dropped", 0),
                "reflex_aux_cache_collected": bool(collect_reflex_aux_cache),
                "reflex_aux_cached_records": int((outputs.get("reflex_aux_records", {}).get("hidden").shape[0]) if outputs.get("reflex_aux_records", {}).get("hidden") is not None else 0),
                "reflex": outputs.get("reflex_metrics", {}),
                "mean_response_length": length_mean,
                "response_length_variance": float(np.var(token_lengths)) if token_lengths else 0.0,
                "response_length_stdev": length_stdev,
                "total_rollout_tokens": total_rollout_tokens,
                "eos_count": sum(1 for seq in outputs["generated_token_ids"] if seq and seq[-1] == tokenizer.eos_token_id),
                "cache_update_mode": outputs["cache_update_mode"],
                "kv_extraction_success_count": outputs.get("kv_extraction_success_count", 0),
                "kv_extraction_fallback_count": outputs.get("kv_extraction_fallback_count", 0),
                "kv_extraction_time": outputs.get("kv_extraction_time", 0.0),
                "recompute_fallback_time": outputs.get("recompute_fallback_time", 0.0),
                "oom_count": outputs["oom_count"],
                "oom_split_count": outputs.get("oom_split_count", 0),
                "last_tree_plan": outputs["last_tree_plan"],
                "gpu": gpu.__dict__,
                "baseline_source": "not_available",
            }
            logger.log(rollout_log)

            batch_bar.set_postfix(
                phase="rollout",
                gen=format_duration(outputs["total_time_cost"]),
                acc=f"{outputs['average_accept_length']:.3f}",
                macc=f"{outputs['medusa_acceptance_rate']:.3f}",
                mloss=f"{head_stats.get('medusa_loss', 0.0):.3f}",
                pending=f"{acc.used_items - acc.used_items_at_last_update}/{batch_size * accumulation_steps}",
                refresh=False,
            )

            if not acc.token_ids:
                continue
            pending = acc.used_items - acc.used_items_at_last_update
            required = required_used_items
            if pending < required:
                continue

            step = acc.used_items // required
            aux_decision = aux_tracker.update(None, step)
            merged_reflex_records = _merge_reflex_record_batches(
                pending_aux_record_batches,
                aux_refresher.config.max_cached_records,
            )
            pending_aux_record_batches.clear()
            boundary_aux_stats: dict[str, Any] = {
                "phase": "auxiliary_decision",
                "epoch": epoch + 1,
                "step": step,
                "rollout_count": rollout_count,
                "aux_update_evaluated": bool(aux_decision.evaluated),
                "aux_update_triggered": bool(aux_decision.triggered),
                "aux_update_reason": aux_decision.reason,
                "aux_triggered_heads": list(aux_decision.triggered_heads),
                "aux_drift_scores": dict(aux_decision.drift_scores),
            }
            if online_aux_enabled and aux_decision.triggered:
                refresh_start = time.time()
                refresh_stats = aux_refresher.maybe_update(
                    decision=aux_decision,
                    head_ids=None,
                    head_mask=None,
                    head_loss_mask=None,
                    enabled=True,
                    grpo_step=step,
                    rollout_count=rollout_count,
                    reflex_records=merged_reflex_records,
                    tracker_state=aux_tracker.state_dict(),
                )
                boundary_aux_stats.update(refresh_stats)
                boundary_aux_stats["phase"] = "auxiliary_update"
                refresh_wall_time = time.time() - refresh_start
                boundary_aux_stats["aux_update_wall_time"] = refresh_wall_time
                total_head_update_time += refresh_wall_time
                maybe_empty_cuda_cache(config)
            logger.log(boundary_aux_stats)

            all_attention = [[1] * len(ids) for ids in acc.token_ids]
            loss_mask = [
                [0] * max(prompt_len - 1, 0) + [1] * max(len(ids) - prompt_len + 1, 0)
                for ids, prompt_len in zip(acc.token_ids, acc.prompt_lens)
            ]
            sorted_pairs = sorted(zip(acc.token_ids, all_attention, loss_mask, acc.std_rewards), key=lambda x: len(x[0]))
            all_input_ids, all_attention_mask, all_loss_mask, all_rewards = map(list, zip(*sorted_pairs))
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
                        loss_scale=1.0 / max(len(acc.token_ids), 1),
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
                    force=bool(fg.get("empty_cache_after_target_train", True)),
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
                    "total_head_update_time": total_head_update_time,
                    "mean_reward": float(np.mean(acc.rewards)) if acc.rewards else 0.0,
                    "reward_variance": float(np.var(acc.rewards)) if acc.rewards else 0.0,
                    "ignore_due_correct_cur_epoch": ignored_correct,
                    "ignore_due_incorrect_cur_epoch": ignored_incorrect,
                    "used_time_min": (time.time() - start_time) / 60.0,
                    "baseline_source": "not_available",
                    "gen_speedup": None,
                    "end_to_end_speedup": None,
                    "empty_cache_after_train": cache_cleared,
                }
                logger.log(grpo_log)
                grpo_bar.set_postfix(step=step, train=format_duration(train_elapsed), reward=f"{grpo_log['mean_reward']:.3f}", refresh=False)

            aux_periodic_path = aux_refresher.maybe_save_periodic(
                grpo_step=step,
                rollout_count=rollout_count,
                tracker_state=aux_tracker.state_dict(),
            )
            if aux_periodic_path:
                logger.log({
                    "phase": "auxiliary_checkpoint",
                    "epoch": epoch + 1,
                    "step": step,
                    "rollout_count": rollout_count,
                    "path": aux_periodic_path,
                    "reason": "periodic",
                })

            acc.used_items_at_last_update = acc.used_items
            acc.token_ids.clear()
            acc.prompt_lens.clear()
            acc.rewards.clear()
            acc.std_rewards.clear()
            batch_old_logps.clear()
            batch_ref_logps.clear()
            if save_steps > 0 and step > 0 and step % save_steps == 0:
                target_model.save_pretrained(saved_model_dir / f"step{step}")
                aux_refresher.save_aux_checkpoint(
                    f"step{step}",
                    grpo_step=step,
                    rollout_count=rollout_count,
                    decision=None,
                    tracker_state=aux_tracker.state_dict(),
                )

        epoch_bar.set_postfix(
            used=acc.used_items,
            elapsed=format_duration(time.time() - start_time),
            gen=format_duration(total_generate_time),
            train=format_duration(total_train_time),
            head=format_duration(total_head_update_time),
            refresh=False,
        )

    final_step = max(0, acc.used_items // max(1, batch_size * accumulation_steps))
    target_model.save_pretrained(saved_model_dir / f"step{final_step}")
    aux_refresher.save_aux_checkpoint(
        f"step{final_step}",
        grpo_step=final_step,
        rollout_count=rollout_count,
        decision=None,
        tracker_state=aux_tracker.state_dict(),
    )
    summary = {
        "run_name": run_name,
        "final_step": final_step,
        "used_items": acc.used_items,
        "total_generate_time_s": total_generate_time,
        "total_train_time_s": total_train_time,
        "total_head_update_time_s": total_head_update_time,
        "total_wall_time_s": time.time() - start_time,
        "total_rollout_tokens": int(total_rollout_tokens),
        "generation_tokens_per_s": total_rollout_tokens / max(total_generate_time, 1e-9),
        "average_accept_length": total_accepted_length / max(total_decoded_steps, 1),
        "medusa_acceptance_rate": total_accepted_medusa_tokens / max(total_proposed_medusa_tokens, 1),
        "total_verify_rounds": int(total_verify_rounds),
        "rollout_batches": int(rollout_count - start_rollout_count),
        "metrics_jsonl": str(logger.jsonl_path),
        "metrics_csv": str(logger.csv_path),
        "saved_model_dir": str(saved_model_dir / f"step{final_step}"),
        "saved_medusa_dir": str(saved_medusa_dir / f"step{final_step}"),
    }
    logger.write_summary(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
