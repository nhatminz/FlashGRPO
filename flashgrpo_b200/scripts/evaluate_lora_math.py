#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helper.get_QAs import get_test_QAs, prompt_dict
from helper.rewards import accuracy_reward_func, format_reward_func


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt(tokenizer, question: str) -> str:
    math_prompt = prompt_dict["math"]
    messages = [
        {"role": "system", "content": math_prompt["system_prompt"]},
        {"role": "user", "content": math_prompt["user_prompt"].format_map({"instruction": question})},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model(model_dir: str, adapter_path: str, attn_implementation: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        attn_implementation=attn_implementation,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a LoRA math checkpoint with repository rewards.")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--eval_option", default="simplelr_abel_level3to5")
    parser.add_argument("--name", default="")
    parser.add_argument("--output_dir", default="eval_outputs/lora_math")
    parser.add_argument("--limit", type=int, default=0, help="0 means evaluate the full split")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_input_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--do_sample", type=str2bool, default=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn_implementation", default="sdpa", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--overwrite", type=str2bool, default=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.name or Path(args.adapter_path).name
    jsonl_path = output_dir / f"{run_name}.jsonl"
    summary_path = output_dir / f"{run_name}_summary.json"
    if jsonl_path.exists() and not args.overwrite:
        raise FileExistsError(f"{jsonl_path} exists; pass --overwrite True to replace it")

    qas = get_test_QAs(args.eval_option)
    end = None if args.limit <= 0 else args.start + args.limit
    qas = qas[args.start : end]
    if not qas:
        raise ValueError("No evaluation examples selected")

    model, tokenizer = load_model(args.model_dir, args.adapter_path, args.attn_implementation)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    generation_kwargs = {
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": bool(args.do_sample),
        "pad_token_id": int(pad_token_id),
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if args.do_sample:
        generation_kwargs["temperature"] = float(args.temperature)
        generation_kwargs["top_p"] = float(args.top_p)

    rows = []
    total_new_tokens = 0
    eos_count = 0
    truncated_count = 0
    generation_time = 0.0
    reward_time = 0.0
    wall_start = time.time()

    with jsonl_path.open("w", encoding="utf-8") as out_f:
        for batch_start in tqdm(range(0, len(qas), args.batch_size), desc=f"Eval {run_name}", dynamic_ncols=True):
            batch = qas[batch_start : batch_start + args.batch_size]
            prompts = [build_prompt(tokenizer, item["question"]) for item in batch]
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(args.max_input_length),
                add_special_tokens=False,
            ).to(model.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            gen_start = time.time()
            with torch.inference_mode():
                output_ids = model.generate(**inputs, **generation_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            generation_time += time.time() - gen_start

            prompt_width = inputs["input_ids"].shape[1]
            new_ids = output_ids[:, prompt_width:]
            completions = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
            eos_flags = (new_ids == int(tokenizer.eos_token_id)).any(dim=1).tolist() if tokenizer.eos_token_id is not None else [False] * new_ids.shape[0]
            new_lengths = inputs["attention_mask"].new_tensor(
                [(row != int(pad_token_id)).sum().item() for row in new_ids]
            ).tolist()
            total_new_tokens += int(sum(new_lengths))
            eos_count += int(sum(bool(flag) for flag in eos_flags))
            truncated_count += int(sum((not bool(flag)) and int(length) >= int(args.max_new_tokens) for flag, length in zip(eos_flags, new_lengths)))

            solutions = [item["answer"] for item in batch]
            reward_start = time.time()
            answer_rewards = accuracy_reward_func(completions, solutions)
            format_rewards = format_reward_func(completions)
            reward_time += time.time() - reward_start

            for local_idx, (item, completion, ans_r, fmt_r, new_len, eos_flag) in enumerate(
                zip(batch, completions, answer_rewards, format_rewards, new_lengths, eos_flags)
            ):
                row = {
                    "index": args.start + batch_start + local_idx,
                    "question": item["question"],
                    "solution": item["answer"],
                    "completion": completion,
                    "answer_reward": float(ans_r),
                    "format_reward": float(fmt_r),
                    "combined_reward": float(ans_r) + 0.2 * float(fmt_r),
                    "new_tokens": int(new_len),
                    "ended_with_eos": bool(eos_flag),
                }
                rows.append(row)
                out_f.write(json.dumps(row, ensure_ascii=True) + "\n")
            out_f.flush()

    wall_time = time.time() - wall_start
    answer_acc = float(np.mean([row["answer_reward"] for row in rows])) if rows else 0.0
    format_acc = float(np.mean([row["format_reward"] for row in rows])) if rows else 0.0
    combined = float(np.mean([row["combined_reward"] for row in rows])) if rows else 0.0
    summary = {
        "name": run_name,
        "model_dir": args.model_dir,
        "adapter_path": args.adapter_path,
        "eval_option": args.eval_option,
        "num_examples": len(rows),
        "start": args.start,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "answer_accuracy": answer_acc,
        "format_rate": format_acc,
        "mean_combined_reward": combined,
        "total_new_tokens": int(total_new_tokens),
        "avg_new_tokens": total_new_tokens / max(len(rows), 1),
        "eos_count": int(eos_count),
        "truncated_count": int(truncated_count),
        "truncated_rate": truncated_count / max(len(rows), 1),
        "generation_time_s": generation_time,
        "reward_time_s": reward_time,
        "wall_time_s": wall_time,
        "tokens_per_second": total_new_tokens / max(generation_time, 1e-9),
        "jsonl_path": str(jsonl_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
