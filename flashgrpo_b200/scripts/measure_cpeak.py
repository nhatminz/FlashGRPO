#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flashgrpo_b200.decoding.medusa_tree import CandidateTree
from flashgrpo_b200.decoding.tree_attention import build_tree_attention_inputs
from flashgrpo_b200.models.qwen_flashgrpo_wrapper import forward_tree, prefill, autocast_dtype, unwrap_causal_lm
from flashgrpo_b200.utils.cpeak_measure import recommended_cpeak


def make_chain_tree(length: int, token_id: int = 1) -> CandidateTree:
    tokens = [token_id] * max(1, int(length))
    parents = [-1] + list(range(max(0, int(length) - 1)))
    depths = list(range(1, max(1, int(length)) + 1))
    return CandidateTree(tokens=tokens, parents=parents, depths=depths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure candidate-node latency for FlashGRPO")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--batch_sizes", default="1,2,4,8")
    parser.add_argument("--node_counts", default="1,2,4,8,16,24,32")
    parser.add_argument("--prompt", default="What is 1+1?")
    parser.add_argument("--out", default="logs/flashgrpo/cpeak_measure.json")
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, attn_implementation="eager").cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]
    node_counts = [int(x) for x in args.node_counts.split(",") if x]
    results = {}
    with torch.inference_mode():
        for bsz in batch_sizes:
            enc = tokenizer([args.prompt] * bsz, return_tensors="pt", padding=True).to("cuda")
            pre = prefill(model, enc.input_ids, enc.attention_mask)
            full_mask = enc.attention_mask.long()
            logical = full_mask.sum(-1)
            per_b = {}
            for nodes in node_counts:
                trees = [make_chain_tree(nodes, token_id=tokenizer.eos_token_id or 1) for _ in range(bsz)]
                ids, mask4d, pos, _ = build_tree_attention_inputs(
                    trees,
                    full_mask,
                    logical,
                    pad_token_id=tokenizer.pad_token_id or 0,
                    dtype=autocast_dtype(unwrap_causal_lm(model)),
                )
                torch.cuda.synchronize()
                start = time.time()
                _ = forward_tree(model, ids, mask4d, pre["past_key_values"], pos, clone_past=True)
                torch.cuda.synchronize()
                per_b[nodes] = time.time() - start
            results[str(bsz)] = per_b
    flat = {nodes: min(results[str(b)].get(nodes, float("inf")) for b in batch_sizes) for nodes in node_counts}
    out = {"latencies": results, "recommended_cpeak_nodes": recommended_cpeak(flat)}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
