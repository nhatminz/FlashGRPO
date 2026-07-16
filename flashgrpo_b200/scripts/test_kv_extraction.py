#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flashgrpo_b200.decoding.kv_extraction import extract_accepted_path_kv
from flashgrpo_b200.decoding.medusa_tree import CandidateTree
from flashgrpo_b200.decoding.tree_attention import build_tree_attention_inputs
from flashgrpo_b200.models.qwen_flashgrpo_wrapper import (
    autocast_dtype,
    forward_tokens,
    forward_tree,
    logical_lengths,
    prefill,
    unwrap_causal_lm,
)


def run_check(model_dir: str, prompt: str, atol: float) -> dict:
    config = AutoConfig.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        config=config,
        attn_implementation="eager",
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = unwrap_causal_lm(model)
    encoded = tokenizer([prompt], return_tensors="pt", padding=True).to("cuda")
    with torch.inference_mode():
        pref = prefill(model, encoded.input_ids, encoded.attention_mask)
        full_mask = encoded.attention_mask.long()
        logical = logical_lengths(full_mask)
        root = int(tokenizer.encode(" A", add_special_tokens=False)[0])
        child = int(tokenizer.encode(" B", add_special_tokens=False)[0])
        tree = CandidateTree(tokens=[root, child], parents=[-1, 0], depths=[1, 2])
        ids, mask4d, pos, _ = build_tree_attention_inputs(
            [tree],
            full_mask,
            logical,
            pad_token_id=tokenizer.pad_token_id or 0,
            dtype=autocast_dtype(base),
        )
        tree_out = forward_tree(model, ids, mask4d, pref["past_key_values"], pos, clone_past=True)
        extracted = extract_accepted_path_kv(
            pref["past_key_values"],
            tree_out["past_key_values"],
            [[0, 1]],
            causal_lm=model,
        ).past_key_values

        accepted_ids = torch.tensor([[root, child]], dtype=torch.long, device="cuda")
        accepted_mask = torch.cat([full_mask, torch.ones((1, 2), dtype=torch.long, device="cuda")], dim=1)
        accepted_pos = logical[:, None] + torch.arange(2, device="cuda").unsqueeze(0)
        recomputed = forward_tokens(model, accepted_ids, accepted_mask, pref["past_key_values"], accepted_pos)["past_key_values"]

        next_token = torch.tensor([[tokenizer.eos_token_id or tokenizer.pad_token_id or 0]], dtype=torch.long, device="cuda")
        next_mask = torch.cat([accepted_mask, torch.ones((1, 1), dtype=torch.long, device="cuda")], dim=1)
        next_pos = logical[:, None] + torch.tensor([[2]], dtype=torch.long, device="cuda")
        out_extract = forward_tokens(model, next_token, next_mask, extracted, next_pos)
        out_recompute = forward_tokens(model, next_token, next_mask, recomputed, next_pos)
        logits_extract = base.lm_head(out_extract["hidden_states"][:, -1:, :]).float()
        logits_recompute = base.lm_head(out_recompute["hidden_states"][:, -1:, :]).float()
        diff = (logits_extract - logits_recompute).abs()
        result = {
            "max_abs_error": float(diff.max().item()),
            "mean_abs_error": float(diff.mean().item()),
            "top1_extract": int(logits_extract.argmax(dim=-1).item()),
            "top1_recompute": int(logits_recompute.argmax(dim=-1).item()),
            "top1_match": int(logits_extract.argmax(dim=-1).item()) == int(logits_recompute.argmax(dim=-1).item()),
            "passed": bool(diff.max().item() <= atol or int(logits_extract.argmax(dim=-1).item()) == int(logits_recompute.argmax(dim=-1).item())),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Check FlashGRPO KV path extraction against recompute")
    parser.add_argument("--model_dir", default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt", default="What is 1+1?")
    parser.add_argument("--atol", type=float, default=5e-3)
    args = parser.parse_args()
    result = run_check(args.model_dir, args.prompt, args.atol)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
