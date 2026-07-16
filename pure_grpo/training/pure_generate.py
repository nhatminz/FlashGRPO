from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from pure_grpo.model_utils.qwen_wrapper import (
    autocast_dtype,
    forward_tokens,
    logical_lengths as mask_logical_lengths,
    model_device,
    prefill,
    repeat_interleave_cache,
    select_cache_batch,
    unwrap_causal_lm,
)


@dataclass
class PureGenerateConfig:
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int | None = None


def logits_to_probs(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float | None = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    logits = logits.float()
    if temperature is not None and temperature > 0:
        logits = logits / float(temperature)
    if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
        top_values, top_indices = torch.topk(logits, k=int(top_k), dim=-1)
        filtered = torch.full_like(logits, torch.finfo(logits.dtype).min)
        logits = filtered.scatter(-1, top_indices, top_values)
    probs = F.softmax(logits, dim=-1)
    if top_p is not None and 0 < float(top_p) < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative > float(top_p)
        mask = torch.roll(mask, shifts=1, dims=-1)
        mask[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)
    return probs


def sample_from_logits(
    logits: torch.Tensor,
    *,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float | None = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    if (not do_sample) or temperature == 0:
        return torch.argmax(logits, dim=-1)
    probs = logits_to_probs(logits, temperature=temperature, top_p=top_p, top_k=top_k)
    flat = probs.reshape(-1, probs.shape[-1])
    sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return sampled.view(logits.shape[:-1])


def _last_valid_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_idx = attention_mask.long().sum(dim=-1).clamp_min(1) - 1
    if bool((attention_mask[:, -1] == 1).all().item()):
        return hidden_states[:, -1, :]
    gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, hidden_states.shape[-1])
    return hidden_states.gather(1, gather_idx).squeeze(1)


def pure_target_generate(
    target_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    *,
    repeated_generate_nums: int = 1,
    max_length: int = 2048,
    config: PureGenerateConfig | None = None,
    statistical_time: bool = False,
) -> dict:
    cfg = config or PureGenerateConfig()
    device = model_device(target_model)
    repeats = max(1, int(repeated_generate_nums))
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    base = unwrap_causal_lm(target_model)
    lm_head = base.lm_head
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = pad_token_id

    total_start = time.time()
    if statistical_time and torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_start = time.time()
    prefill_out = prefill(target_model, input_ids, attention_mask)
    if statistical_time and torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_time = time.time() - prefill_start

    past_key_values = prefill_out["past_key_values"]
    current_hidden = _last_valid_hidden(prefill_out["hidden_states"], attention_mask)
    del prefill_out
    with torch.amp.autocast(
        "cuda" if device.type == "cuda" else device.type,
        dtype=autocast_dtype(base),
        enabled=(device.type == "cuda"),
    ):
        current_logits = lm_head(current_hidden.to(dtype=getattr(lm_head.weight, "dtype", current_hidden.dtype)))

    if repeats > 1:
        past_key_values = repeat_interleave_cache(past_key_values, repeats, causal_lm=target_model)
        current_hidden = current_hidden.repeat_interleave(repeats, dim=0).contiguous()
        current_logits = current_logits.repeat_interleave(repeats, dim=0).contiguous()
        attention_mask = attention_mask.repeat_interleave(repeats, dim=0).contiguous()

    total_sequences = attention_mask.shape[0]
    generated: list[list[int]] = [[] for _ in range(total_sequences)]
    active_original_indices = list(range(total_sequences))
    full_attention_mask = attention_mask.long()
    logical_lens = mask_logical_lengths(full_attention_mask)

    active_batch_sum = 0
    cache_update_time = 0.0
    total_acc_length = 0
    total_decoded_steps = 0
    total_verify_rounds = 0
    accept_hist: dict[int, int] = {}

    while active_original_indices:
        active_bsz = len(active_original_indices)
        remaining = max_length - logical_lens
        if not bool((remaining > 0).any().item()):
            break

        next_tokens = sample_from_logits(
            current_logits,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
        )

        total_verify_rounds += 1
        active_batch_sum += active_bsz
        total_acc_length += active_bsz
        total_decoded_steps += active_bsz
        accept_hist[1] = accept_hist.get(1, 0) + active_bsz

        accepted_ids = next_tokens.view(active_bsz, 1)
        valid_ext = torch.ones((active_bsz, 1), dtype=torch.long, device=device)
        position_ids = logical_lens.view(active_bsz, 1)
        finished_flags: list[bool] = []
        for row in range(active_bsz):
            token = int(next_tokens[row].item())
            original_idx = active_original_indices[row]
            generated[original_idx].append(token)
            finished_flags.append(token == eos_token_id or int(logical_lens[row].item()) + 1 >= max_length)

        new_attention_mask = torch.cat([full_attention_mask, valid_ext], dim=1)
        if statistical_time and torch.cuda.is_available():
            torch.cuda.synchronize()
        cache_start = time.time()
        cache_out = forward_tokens(
            target_model,
            accepted_ids,
            new_attention_mask,
            past_key_values,
            position_ids,
        )
        if statistical_time and torch.cuda.is_available():
            torch.cuda.synchronize()
        cache_update_time += time.time() - cache_start

        past_key_values = cache_out["past_key_values"]
        token_hidden = cache_out["hidden_states"]
        current_hidden = token_hidden[:, -1, :]
        with torch.amp.autocast(
            "cuda" if device.type == "cuda" else device.type,
            dtype=autocast_dtype(base),
            enabled=(device.type == "cuda"),
        ):
            current_logits = lm_head(current_hidden.to(dtype=getattr(lm_head.weight, "dtype", current_hidden.dtype)))
        del cache_out, token_hidden

        full_attention_mask = new_attention_mask
        logical_lens = logical_lens + 1
        keep_rows = [idx for idx, done in enumerate(finished_flags) if not done]
        if len(keep_rows) != active_bsz:
            if keep_rows:
                keep = torch.tensor(keep_rows, dtype=torch.long, device=device)
                past_key_values = select_cache_batch(past_key_values, keep, causal_lm=target_model)
                current_hidden = current_hidden.index_select(0, keep)
                current_logits = current_logits.index_select(0, keep)
                full_attention_mask = full_attention_mask.index_select(0, keep)
                logical_lens = logical_lens.index_select(0, keep)
                active_original_indices = [active_original_indices[idx] for idx in keep_rows]
            else:
                active_original_indices = []
                break

    total_time = time.time() - total_start
    avg_accept = total_acc_length / max(total_decoded_steps, 1)
    return {
        "generated_token_ids": generated,
        "max_sequence_length": max((len(seq) for seq in generated), default=0),
        "total_acc_length": int(total_acc_length),
        "average_accept_length": float(avg_accept),
        "accepted_tokens_per_medusa_step": float(avg_accept),
        "total_decoded_token_num": int(total_decoded_steps),
        "total_accepted_draft_tokens": 0,
        "total_proposed_draft_tokens": 0,
        "total_accepted_medusa_tokens": 0,
        "total_proposed_medusa_tokens": 0,
        "draft_acceptance_rate": 0.0,
        "medusa_acceptance_rate": 0.0,
        "total_verify_rounds": int(total_verify_rounds),
        "average_active_batch_size": active_batch_sum / max(total_verify_rounds, 1),
        "average_tree_nodes_per_seq": 1.0,
        "accept_length_histogram": accept_hist,
        "medusa_accept_by_depth": {},
        "medusa_proposed_by_depth": {},
        "last_tree_plan": {
            "proposal_mode": "target_only",
            "active_heads": 0,
            "topk_by_depth": [],
            "actual_nodes": 1,
        },
        "cache_update_mode": "target_only",
        "kv_extraction_success_count": 0,
        "kv_extraction_fallback_count": 0,
        "kv_extraction_time": 0.0,
        "recompute_fallback_time": 0.0,
        "oom_count": 0,
        "oom_split_count": 0,
        "total_time_cost": total_time,
        "prefill_time_cost": prefill_time,
        "target_time_cost": prefill_time + cache_update_time,
        "tree_verify_time_cost": 0.0,
        "cache_update_time_cost": cache_update_time,
        "medusa_head_time_cost": 0.0,
        "draft_time_cost": 0.0,
        "check_time_cost": 0.0,
    }
