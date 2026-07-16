from __future__ import annotations

import torch


def build_tree_attention_inputs(
    trees,
    full_attention_mask: torch.Tensor,
    logical_lengths: torch.Tensor,
    *,
    pad_token_id: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten per-sequence candidate trees and build an ancestor-only mask.

    Each tree node can attend to valid prefix tokens, its ancestors, and itself.
    Siblings/cousins are masked, which makes the batched tree logits equivalent
    to forwarding each root-to-node path independently.
    """

    device = full_attention_mask.device
    batch = len(trees)
    max_nodes = max(max(tree.node_count, 1) for tree in trees)
    past_len = full_attention_mask.shape[1]
    min_dtype = torch.finfo(dtype).min
    input_ids = torch.full((batch, max_nodes), int(pad_token_id), dtype=torch.long, device=device)
    position_ids = torch.zeros((batch, max_nodes), dtype=torch.long, device=device)
    node_mask = torch.zeros((batch, max_nodes), dtype=torch.bool, device=device)
    attn = torch.full((batch, 1, max_nodes, past_len + max_nodes), min_dtype, dtype=dtype, device=device)

    for batch_idx, tree in enumerate(trees):
        valid_prefix = full_attention_mask[batch_idx].bool()
        for node_idx, token in enumerate(tree.tokens):
            input_ids[batch_idx, node_idx] = int(token)
            node_mask[batch_idx, node_idx] = True
            depth = int(tree.depths[node_idx])
            position_ids[batch_idx, node_idx] = int(logical_lengths[batch_idx].item()) + depth - 1
            attn[batch_idx, 0, node_idx, :past_len] = torch.where(
                valid_prefix,
                torch.zeros((), dtype=dtype, device=device),
                torch.full((), min_dtype, dtype=dtype, device=device),
            )
            for ancestor in tree.ancestors_including_self(node_idx):
                attn[batch_idx, 0, node_idx, past_len + ancestor] = 0
        for node_idx in range(tree.node_count, max_nodes):
            # Padded query rows are ignored downstream. Give them a non-empty
            # context to avoid all -inf attention rows on strict kernels.
            attn[batch_idx, 0, node_idx, :past_len] = torch.where(
                valid_prefix,
                torch.zeros((), dtype=dtype, device=device),
                torch.full((), min_dtype, dtype=dtype, device=device),
            )
            attn[batch_idx, 0, node_idx, past_len + node_idx] = 0
    return input_ids, attn, position_ids, node_mask
