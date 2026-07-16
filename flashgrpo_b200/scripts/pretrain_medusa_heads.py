#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flashgrpo_b200.models.medusa_heads import MedusaHeads
from flashgrpo_b200.models.qwen_flashgrpo_wrapper import autocast_dtype, unwrap_causal_lm
from flashgrpo_b200.utils.config import load_config, save_resolved_config
from flashgrpo_b200.utils.seed import seed_everything


def _dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype={name}")


def _resolve_attn_implementation(requested: str | None) -> str:
    requested = str(requested or "sdpa")
    if requested == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        print(
            "Warning: attn_implementation=flash_attention_2 was requested, "
            "but flash_attn is not installed. Falling back to sdpa."
        )
        return "sdpa"
    return requested


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _load_json_dataset(path: str | Path) -> list[Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("train", "data", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        raise ValueError(f"Cannot find list split in JSON object: keys={list(data)[:10]}")
    if not isinstance(data, list):
        raise ValueError(f"Unsupported JSON dataset type: {type(data)}")
    return data


def _module_has_nonfinite(module: torch.nn.Module) -> bool:
    with torch.no_grad():
        for param in module.parameters():
            if not bool(torch.isfinite(param).all().item()):
                return True
    return False


def _normalise_role(role: str) -> str:
    role = str(role).lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "bot"}:
        return "assistant"
    if role == "system":
        return "system"
    return role


def _extract_conversations(example: Any, text_field: str | None = None):
    if isinstance(example, list):
        return example, None
    if isinstance(example, dict):
        if text_field and isinstance(example.get(text_field), str):
            return None, example[text_field]
        if isinstance(example.get("conversations"), list):
            return example["conversations"], None
        if isinstance(example.get("messages"), list):
            return example["messages"], None
        for key in ("text", "prompt", "content"):
            if isinstance(example.get(key), str):
                return None, example[key]
    if isinstance(example, str):
        return None, example
    return None, ""


def _qwen_message_text(role: str, content: str) -> str:
    return f"<|im_start|>{role}\n{content}<|im_end|>\n"


def _crop_to_assistant_window(input_ids: list[int], loss_mask: list[int], max_seq_len: int):
    if len(input_ids) <= max_seq_len:
        return input_ids, loss_mask
    assistant_positions = [idx for idx, value in enumerate(loss_mask) if value]
    if assistant_positions:
        last_assistant = assistant_positions[-1] + 1
        start = max(0, last_assistant - max_seq_len)
        end = min(len(input_ids), start + max_seq_len)
        return input_ids[start:end], loss_mask[start:end]
    return input_ids[:max_seq_len], loss_mask[:max_seq_len]


def encode_sharegpt_example(
    example: Any,
    tokenizer,
    *,
    max_seq_len: int,
    text_field: str | None = None,
    conversation_format: str = "auto",
) -> dict[str, list[int]]:
    conversations, text = _extract_conversations(example, text_field=text_field)
    if text is not None:
        ids = tokenizer.encode(str(text), add_special_tokens=False)[:max_seq_len]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "loss_mask": [1] * len(ids)}

    input_ids: list[int] = []
    loss_mask: list[int] = []
    if conversation_format in {"auto", "sharegpt", "qwen"}:
        system = _qwen_message_text("system", "You are a helpful assistant.")
        system_ids = tokenizer.encode(system, add_special_tokens=False)
        input_ids.extend(system_ids)
        loss_mask.extend([0] * len(system_ids))

    for turn in conversations or []:
        if not isinstance(turn, dict):
            continue
        role = _normalise_role(turn.get("from", turn.get("role", "")))
        content = str(turn.get("value", turn.get("content", "")))
        if not content:
            continue
        if conversation_format in {"auto", "sharegpt", "qwen"}:
            text_piece = _qwen_message_text(role, content)
        else:
            prefix = "Assistant" if role == "assistant" else "Human"
            text_piece = f"{prefix}: {content}\n"
        ids = tokenizer.encode(text_piece, add_special_tokens=False)
        input_ids.extend(ids)
        loss_mask.extend([1 if role == "assistant" else 0] * len(ids))

    input_ids, loss_mask = _crop_to_assistant_window(input_ids, loss_mask, max_seq_len)
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "loss_mask": loss_mask}


class ShareGptMedusaDataset(Dataset):
    def __init__(self, examples, tokenizer, max_seq_len: int, text_field: str | None, conversation_format: str):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.text_field = text_field
        self.conversation_format = conversation_format

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return encode_sharegpt_example(
            self.examples[idx],
            self.tokenizer,
            max_seq_len=self.max_seq_len,
            text_field=self.text_field,
            conversation_format=self.conversation_format,
        )


class PadCollator:
    def __init__(self, pad_token_id: int, max_seq_len: int):
        self.pad_token_id = int(pad_token_id)
        self.max_seq_len = int(max_seq_len)

    def __call__(self, batch):
        batch = [x for x in batch if x["input_ids"]]
        max_len = min(max(len(x["input_ids"]) for x in batch), self.max_seq_len)
        input_ids = []
        attention_mask = []
        loss_mask = []
        for item in batch:
            ids = item["input_ids"][:max_len]
            attn = item["attention_mask"][:max_len]
            mask = item["loss_mask"][:max_len]
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            attention_mask.append(attn + [0] * pad)
            loss_mask.append(mask + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.long),
        }


def medusa_loss_and_metrics(
    medusa_heads,
    hidden_states,
    input_ids,
    attention_mask,
    loss_mask,
    lm_head,
    *,
    embedding_layer=None,
    chunk_size: int,
    topk: int,
    chain_loss_weight: float = 0.0,
    chain_loss_max_depth: int | None = None,
    chain_bootstrap_from_medusa: bool = True,
    reflex_loss_weight: float = 0.0,
    reflex_loss_max_depth: int | None = None,
    reflex_top_m: int = 32,
    reflex_delta_scale: float = 0.05,
    reflex_correction_clip_norm: float = 1.0,
):
    total = hidden_states.new_zeros(())
    stats: dict[str, float] = {}
    valid_heads = 0
    seq_len = input_ids.shape[1]
    for head_idx, head in enumerate(medusa_heads.heads):
        shift = head_idx + 2
        if seq_len <= shift:
            continue
        losses = []
        correct1 = 0
        correctk = 0
        valid_total = 0
        for start in range(0, seq_len - shift, chunk_size):
            end = min(start + chunk_size, seq_len - shift)
            hidden = hidden_states[:, start:end, :]
            labels = input_ids[:, start + shift : end + shift].to(hidden.device)
            valid = attention_mask[:, start + shift : end + shift].bool() & loss_mask[:, start + shift : end + shift].bool()
            labels = labels.masked_fill(~valid, -100)
            if not bool(valid.any().item()):
                continue
            logits = head(hidden, lm_head=lm_head).float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
            losses.append(loss)
            with torch.no_grad():
                pred1 = logits.argmax(dim=-1)
                correct1 += int(((pred1 == labels) & valid).sum().item())
                k = min(int(topk), logits.shape[-1])
                top_idx = torch.topk(logits, k=k, dim=-1).indices
                correctk += int(((top_idx == labels.unsqueeze(-1)) & valid.unsqueeze(-1)).any(dim=-1).sum().item())
                valid_total += int(valid.sum().item())
            del logits
        if losses:
            head_loss = torch.stack(losses).mean()
            weight = medusa_heads.medusa_loss_decay ** head_idx
            total = total + float(weight) * head_loss
            valid_heads += 1
            stats[f"loss_head_{head_idx + 1}"] = float(head_loss.detach().cpu())
            stats[f"top1_head_{head_idx + 1}"] = correct1 / max(valid_total, 1)
            stats[f"top{topk}_head_{head_idx + 1}"] = correctk / max(valid_total, 1)
            stats[f"tokens_head_{head_idx + 1}"] = valid_total
    if valid_heads > 0:
        total = total / valid_heads
        stats["parallel_medusa_loss"] = float(total.detach().cpu())
    else:
        total = hidden_states.sum() * 0.0

    chain_losses = []
    chain_weight = float(chain_loss_weight or 0.0)
    if chain_weight > 0.0 and embedding_layer is not None and lm_head is not None:
        max_depth = min(int(chain_loss_max_depth or medusa_heads.num_heads), medusa_heads.num_heads)
        for depth_idx in range(max_depth):
            shift = depth_idx + 2
            if chain_bootstrap_from_medusa and shift <= 2:
                continue
            if seq_len <= shift:
                continue
            weight = medusa_heads.medusa_loss_decay ** depth_idx
            for start in range(0, seq_len - shift, chunk_size):
                end = min(start + chunk_size, seq_len - shift)
                if chain_bootstrap_from_medusa and len(medusa_heads.heads) > 0:
                    state = medusa_heads.heads[0].project_hidden(hidden_states[:, start:end, :].detach())
                    first_prev_offset = 2
                else:
                    state = hidden_states[:, start:end, :].detach()
                    first_prev_offset = 1
                labels = input_ids[:, start + shift : end + shift].to(state.device)
                valid = attention_mask[:, start + shift : end + shift].bool() & loss_mask[:, start + shift : end + shift].bool()
                valid = valid.to(state.device)
                labels = labels.masked_fill(~valid, -100)
                if not bool(valid.any().item()):
                    continue
                for prev_offset in range(first_prev_offset, shift):
                    prev_tokens = input_ids[:, start + prev_offset : end + prev_offset].to(state.device)
                    state = medusa_heads.chain_next_state(state, prev_tokens, embedding_layer)
                logits = medusa_heads.chain_logits_from_state(state, lm_head).float()
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
                chain_losses.append(float(weight) * loss)
                del logits, state
    if chain_losses:
        chain_loss = torch.stack(chain_losses).mean()
        total = total + chain_weight * chain_loss
        stats["chain_loss"] = float(chain_loss.detach().cpu())
        stats["chain_loss_weight"] = chain_weight

    reflex_losses = []
    reflex_weight = float(reflex_loss_weight or 0.0)
    if (
        reflex_weight > 0.0
        and lm_head is not None
        and getattr(medusa_heads, "reflex_fast_state_dim", 0) > 0
        and getattr(medusa_heads, "reflex_down", None) is not None
    ):
        max_depth = min(int(reflex_loss_max_depth or medusa_heads.num_heads), medusa_heads.num_heads)
        lm_weight = lm_head.weight.detach()
        for head_idx, head in enumerate(medusa_heads.heads[:max_depth]):
            shift = head_idx + 2
            if seq_len <= shift:
                continue
            weight = medusa_heads.medusa_loss_decay ** head_idx
            for start in range(0, seq_len - shift, chunk_size):
                end = min(start + chunk_size, seq_len - shift)
                hidden = hidden_states[:, start:end, :]
                labels = input_ids[:, start + shift : end + shift].to(hidden.device)
                valid = attention_mask[:, start + shift : end + shift].bool() & loss_mask[:, start + shift : end + shift].bool()
                labels_for_loss = labels.masked_fill(~valid, -100)
                if not bool(valid.any().item()):
                    continue
                base_hidden = head.project_hidden(hidden)
                lm_dtype = getattr(lm_head.weight, "dtype", base_hidden.dtype)
                base_logits = lm_head(base_hidden.to(dtype=lm_dtype)).float()
                with torch.no_grad():
                    k = min(max(1, int(reflex_top_m)), base_logits.shape[-1])
                    top_values, top_ids = torch.topk(base_logits.detach(), k=k, dim=-1)
                    top_probs = torch.softmax(top_values, dim=-1)
                    flat_top_ids = top_ids.reshape(-1)
                    top_weight = lm_weight.index_select(0, flat_top_ids).reshape(*top_ids.shape, -1).float()
                    expected_weight = (top_probs.unsqueeze(-1) * top_weight).sum(dim=-2)
                    safe_labels = labels.masked_fill(~valid, 0).reshape(-1)
                    true_weight = lm_weight.index_select(0, safe_labels).reshape(*labels.shape, -1).float()
                    hidden_feedback = (true_weight - expected_weight).masked_fill(~valid.unsqueeze(-1), 0.0)
                fast_feedback = medusa_heads.feedback_to_fast_state(hidden_feedback.reshape(-1, hidden_feedback.shape[-1]))
                delta = medusa_heads.reflex_delta(
                    fast_feedback,
                    head_idx,
                    max_norm=float(reflex_correction_clip_norm),
                    scale=float(reflex_delta_scale),
                    normalize=True,
                )
                if delta is None:
                    continue
                delta = delta.reshape_as(base_hidden).to(device=base_hidden.device, dtype=base_hidden.dtype)
                reflex_hidden = base_hidden + delta
                reflex_logits = lm_head(reflex_hidden.to(dtype=lm_dtype)).float()
                reflex_loss = F.cross_entropy(
                    reflex_logits.reshape(-1, reflex_logits.shape[-1]),
                    labels_for_loss.reshape(-1),
                    ignore_index=-100,
                )
                reflex_losses.append(float(weight) * reflex_loss)
                with torch.no_grad():
                    pred1 = reflex_logits.argmax(dim=-1)
                    valid_total = int(valid.sum().item())
                    stats[f"reflex_top1_head_{head_idx + 1}"] = float(((pred1 == labels) & valid).sum().item()) / max(valid_total, 1)
                del base_logits, reflex_logits, base_hidden, reflex_hidden, delta, fast_feedback
    if reflex_losses:
        reflex_loss = torch.stack(reflex_losses).mean()
        total = total + reflex_weight * reflex_loss
        stats["reflex_loss"] = float(reflex_loss.detach().cpu())
        stats["reflex_loss_weight"] = reflex_weight

    if valid_heads == 0 and not chain_losses and not reflex_losses:
        return hidden_states.sum() * 0.0, {"train_loss": 0.0}
    stats["train_loss"] = float(total.detach().cpu())
    return total, stats


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain FlashGRPO MEDUSA heads on ShareGPT")
    parser.add_argument("--config", default="")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--dataset_split", default=None)
    parser.add_argument("--text_field", default=None)
    parser.add_argument("--conversation_format", default=None)
    parser.add_argument("--model_name_or_path", "--model", dest="model_name_or_path", default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--load_medusa_checkpoint", default=None)
    parser.add_argument("--train_reflex_only", default=None)
    parser.add_argument("--freeze_loaded_base_heads", default=None)
    parser.add_argument("--num_medusa_heads", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default=None)
    parser.add_argument("--head_dtype", choices=["fp16", "bf16", "fp32"], default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--medusa_loss_decay", type=float, default=None)
    parser.add_argument("--loss_chunk_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--topk_metric", type=int, default=None)
    parser.add_argument("--chain_loss_weight", type=float, default=None)
    parser.add_argument("--chain_loss_max_depth", type=int, default=None)
    parser.add_argument("--chain_bootstrap_from_medusa", default=None)
    parser.add_argument("--chain_bottleneck_ratio", type=int, default=None)
    parser.add_argument("--chain_gate_init", type=float, default=None)
    parser.add_argument("--reflex_fast_state_dim", type=int, default=None)
    parser.add_argument("--reflex_init_scale", type=float, default=None)
    parser.add_argument("--reflex_loss_weight", type=float, default=None)
    parser.add_argument("--reflex_loss_max_depth", type=int, default=None)
    parser.add_argument("--reflex_top_m", type=int, default=None)
    parser.add_argument("--reflex_delta_scale", type=float, default=None)
    parser.add_argument("--reflex_correction_clip_norm", type=float, default=None)
    return parser.parse_args()


def merge_args(config: dict[str, Any], args) -> dict[str, Any]:
    cfg = dict(config)
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            cfg[key] = value
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else {}
    cfg = merge_args(cfg, args)
    seed_everything(int(cfg.get("seed", 42)))

    model_name_or_path = cfg.get("model_name_or_path") or cfg.get("model") or "models/Qwen2.5-1.5B-Instruct"
    dataset_path = cfg.get("dataset_path")
    if not dataset_path:
        raise ValueError("--dataset_path is required for the local ShareGPT run")
    output_dir = Path(cfg.get("output_dir", "outputs/flashgrpo_medusa_sharegpt_qwen25_1p5b"))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "pretrain_metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    save_resolved_config(cfg, output_dir / "pretrain_config_resolved.yaml")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = _dtype(cfg.get("dtype", "fp16"))
    head_dtype = _dtype(cfg.get("head_dtype", "fp32"))
    attn_impl = _resolve_attn_implementation(cfg.get("attn_implementation", "sdpa"))
    hf_config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        config=hf_config,
        attn_implementation=attn_impl,
    ).cuda().eval()
    model.requires_grad_(False)
    base = unwrap_causal_lm(model)
    load_medusa_checkpoint = str(cfg.get("load_medusa_checkpoint", "") or "")
    if load_medusa_checkpoint:
        medusa_heads = MedusaHeads.from_pretrained(
            load_medusa_checkpoint,
            map_location="cpu",
            dtype=head_dtype,
            lm_head=base.lm_head,
            chain_bottleneck_ratio=int(cfg.get("chain_bottleneck_ratio", 8)),
            chain_gate_init=float(cfg.get("chain_gate_init", -3.0)),
            reflex_fast_state_dim=int(cfg.get("reflex_fast_state_dim", 0) or 0),
            reflex_init_scale=float(cfg.get("reflex_init_scale", 0.0)),
        )
        print(f"Loaded initial MEDUSA heads from {load_medusa_checkpoint}")
    else:
        medusa_heads = MedusaHeads(
            hf_config.hidden_size,
            hf_config.vocab_size,
            num_heads=int(cfg.get("num_medusa_heads", 3)),
            dtype=head_dtype,
            tie_lm_head=bool(cfg.get("tie_lm_head", True)),
            lm_head=base.lm_head,
            medusa_loss_decay=float(cfg.get("medusa_loss_decay", 0.8)),
            chain_bottleneck_ratio=int(cfg.get("chain_bottleneck_ratio", 8)),
            chain_gate_init=float(cfg.get("chain_gate_init", -3.0)),
            reflex_fast_state_dim=int(cfg.get("reflex_fast_state_dim", 0) or 0),
            reflex_init_scale=float(cfg.get("reflex_init_scale", 0.0)),
        )
    medusa_heads = medusa_heads.cuda().train()
    if _as_bool(cfg.get("train_reflex_only", False)):
        for name, param in medusa_heads.named_parameters():
            param.requires_grad_(name.startswith("reflex_"))
    elif load_medusa_checkpoint and _as_bool(cfg.get("freeze_loaded_base_heads", False)):
        for name, param in medusa_heads.named_parameters():
            if name.startswith("heads.") or name.startswith("chain_cell."):
                param.requires_grad_(False)

    examples = _load_json_dataset(dataset_path)
    random.shuffle(examples)
    num_samples = int(cfg.get("num_samples", 0) or 0)
    if num_samples > 0:
        examples = examples[:num_samples]
    eval_size = int(cfg.get("eval_size", 0) or 0)
    eval_examples = examples[:eval_size] if eval_size > 0 else []
    train_examples = examples[eval_size:] if eval_size > 0 else examples

    train_ds = ShareGptMedusaDataset(
        train_examples,
        tokenizer,
        max_seq_len=int(cfg.get("max_seq_len", 1024)),
        text_field=cfg.get("text_field"),
        conversation_format=cfg.get("conversation_format", "auto"),
    )
    collator = PadCollator(tokenizer.pad_token_id or tokenizer.eos_token_id or 0, int(cfg.get("max_seq_len", 1024)))
    num_workers = int(cfg.get("num_workers", 2))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    trainable_params = [param for param in medusa_heads.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable MEDUSA/reflex parameters. Check train_reflex_only/reflex_fast_state_dim settings.")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.get("learning_rate", 5e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    grad_accum = max(1, int(cfg.get("gradient_accumulation_steps", 8)))
    epochs = int(cfg.get("num_train_epochs", 1))
    total_optim_steps = max(1, math.ceil(len(train_loader) * epochs / grad_accum))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.get("warmup_steps", 100)),
        num_training_steps=total_optim_steps,
    )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    seen_tokens = 0
    peak_memory = 0
    progress = tqdm(range(epochs * len(train_loader)), desc="Pretrain MEDUSA", dynamic_ncols=True)
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(non_blocking=True)
            loss_mask = batch["loss_mask"].cuda(non_blocking=True)
            with torch.no_grad():
                device_type = "cuda" if input_ids.device.type == "cuda" else input_ids.device.type
                with torch.amp.autocast(device_type, dtype=autocast_dtype(base), enabled=(input_ids.device.type == "cuda")):
                    outputs = base.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            loss, stats = medusa_loss_and_metrics(
                medusa_heads,
                hidden_states.detach(),
                input_ids,
                attention_mask,
                loss_mask,
                base.lm_head,
                embedding_layer=base.get_input_embeddings(),
                chunk_size=int(cfg.get("loss_chunk_size", 64)),
                topk=int(cfg.get("topk_metric", 5)),
                chain_loss_weight=float(cfg.get("chain_loss_weight", 0.0)),
                chain_loss_max_depth=int(cfg.get("chain_loss_max_depth", int(cfg.get("num_medusa_heads", 3)))),
                chain_bootstrap_from_medusa=_as_bool(cfg.get("chain_bootstrap_from_medusa", True)),
                reflex_loss_weight=float(cfg.get("reflex_loss_weight", 0.0)),
                reflex_loss_max_depth=int(cfg.get("reflex_loss_max_depth", int(cfg.get("num_medusa_heads", 3)))),
                reflex_top_m=int(cfg.get("reflex_top_m", 32)),
                reflex_delta_scale=float(cfg.get("reflex_delta_scale", 0.05)),
                reflex_correction_clip_norm=float(cfg.get("reflex_correction_clip_norm", 1.0)),
            )
            if (not loss.requires_grad) or (not torch.isfinite(loss)):
                row = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "batch": batch_idx,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "skipped_nonfinite_loss": bool(torch.isfinite(loss).item()) is False,
                    "skipped_no_grad_loss": not loss.requires_grad,
                    **stats,
                }
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=True) + "\n")
                progress.update(1)
                continue
            (loss / grad_accum).backward()
            seen_tokens += int((attention_mask.bool() & loss_mask.bool()).sum().item())
            if torch.cuda.is_available():
                peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())

            if (batch_idx + 1) % grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(medusa_heads.parameters(), float(cfg.get("grad_clip_norm", 1.0)))
                if not bool(torch.isfinite(torch.as_tensor(grad_norm)).item()):
                    row = {
                        "epoch": epoch + 1,
                        "step": global_step,
                        "batch": batch_idx,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "skipped_nonfinite_grad": True,
                        "grad_norm": float("nan"),
                        **stats,
                    }
                    with metrics_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=True) + "\n")
                    optimizer.zero_grad(set_to_none=True)
                    progress.set_postfix(loss="nan_grad", tok_s=f"{seen_tokens / max(time.time() - start_time, 1e-9):.0f}", refresh=False)
                    progress.update(1)
                    continue
                optimizer.step()
                if _module_has_nonfinite(medusa_heads):
                    raise RuntimeError(
                        "Non-finite MEDUSA/reflex parameters after optimizer.step(); "
                        "aborting before saving a corrupted checkpoint. Lower learning_rate or loss weights."
                    )
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                elapsed = time.time() - start_time
                row = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "batch": batch_idx,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "tokens_per_sec": seen_tokens / max(elapsed, 1e-9),
                    "gpu_peak_memory_mb": peak_memory / 1024**2,
                    **stats,
                }
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=True) + "\n")
                progress.set_postfix(loss=f"{row['train_loss']:.3f}", tok_s=f"{row['tokens_per_sec']:.0f}", refresh=False)
                save_steps = int(cfg.get("save_steps", 500))
                if save_steps > 0 and global_step % save_steps == 0:
                    medusa_heads.save_pretrained(output_dir / f"step{global_step}")
            progress.update(1)

    progress.close()
    medusa_heads.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    summary = {
        "base_model_name_or_path": model_name_or_path,
        "dataset_path": str(dataset_path),
        "num_train_examples": len(train_examples),
        "load_medusa_checkpoint": load_medusa_checkpoint,
        "train_reflex_only": _as_bool(cfg.get("train_reflex_only", False)),
        "freeze_loaded_base_heads": _as_bool(cfg.get("freeze_loaded_base_heads", False)),
        "num_medusa_heads": medusa_heads.num_heads,
        "hidden_size": medusa_heads.hidden_size,
        "vocab_size": medusa_heads.vocab_size,
        "medusa_loss_decay": medusa_heads.medusa_loss_decay,
        "chain_bottleneck_ratio": medusa_heads.chain_bottleneck_ratio,
        "chain_gate_init": medusa_heads.chain_gate_init,
        "chain_loss_weight": float(cfg.get("chain_loss_weight", 0.0)),
        "chain_loss_max_depth": int(cfg.get("chain_loss_max_depth", int(cfg.get("num_medusa_heads", 3)))),
        "chain_bootstrap_from_medusa": _as_bool(cfg.get("chain_bootstrap_from_medusa", True)),
        "reflex_fast_state_dim": medusa_heads.reflex_fast_state_dim,
        "reflex_init_scale": medusa_heads.reflex_init_scale,
        "reflex_loss_weight": float(cfg.get("reflex_loss_weight", 0.0)),
        "reflex_loss_max_depth": int(cfg.get("reflex_loss_max_depth", int(cfg.get("num_medusa_heads", 3)))),
        "reflex_top_m": int(cfg.get("reflex_top_m", 32)),
        "reflex_delta_scale": float(cfg.get("reflex_delta_scale", 0.05)),
        "reflex_correction_clip_norm": float(cfg.get("reflex_correction_clip_norm", 1.0)),
        "head_dtype": str(head_dtype).replace("torch.", ""),
        "max_seq_len": int(cfg.get("max_seq_len", 1024)),
        "global_step": global_step,
        "tokens_seen": seen_tokens,
        "peak_memory_mb": peak_memory / 1024**2,
        "output_dir": str(output_dir),
    }
    (output_dir / "pretrain_summary.txt").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "pretrain_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
