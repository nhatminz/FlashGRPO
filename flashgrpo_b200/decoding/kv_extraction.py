from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import DynamicCache

from flashgrpo_b200.models.qwen_flashgrpo_wrapper import (
    _cache_layer_count,
    _get_cache_layer,
    _set_cache_layer,
    cache_seq_length,
    unwrap_causal_lm,
)


@dataclass
class KvExtractionResult:
    past_key_values: object
    max_accepted_length: int
    cache_format: str


def _new_dynamic_cache(causal_lm=None):
    if causal_lm is not None:
        try:
            return DynamicCache(config=unwrap_causal_lm(causal_lm).config)
        except TypeError:
            return DynamicCache()
    return DynamicCache()


def _build_path_index(
    accepted_node_indices: Sequence[Sequence[int]],
    *,
    old_seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    if not accepted_node_indices:
        raise ValueError("accepted_node_indices is empty")
    max_len = max(len(path) for path in accepted_node_indices)
    if max_len <= 0:
        raise ValueError("accepted_node_indices contains an empty path")
    rows = []
    for path in accepted_node_indices:
        if not path:
            raise ValueError("accepted path cannot be empty")
        padded = list(path) + [path[-1]] * (max_len - len(path))
        rows.append([old_seq_len + int(node_idx) for node_idx in padded])
    return torch.tensor(rows, dtype=torch.long, device=device), max_len


def _gather_paths_from_layer(
    tree_tensor: torch.Tensor,
    path_positions: torch.Tensor,
) -> torch.Tensor:
    # Common HF/Qwen cache shape: [batch, num_kv_heads, seq_len, head_dim].
    if tree_tensor.dim() < 4:
        raise ValueError(f"Unsupported cache tensor shape: {tuple(tree_tensor.shape)}")
    batch, heads, _, head_dim = tree_tensor.shape[:4]
    if path_positions.shape[0] != batch:
        raise ValueError(
            f"accepted path batch {path_positions.shape[0]} != cache batch {batch}"
        )
    expanded = path_positions[:, None, :, None].expand(batch, heads, path_positions.shape[1], head_dim)
    return tree_tensor.gather(dim=2, index=expanded).contiguous()


def extract_accepted_path_kv(
    old_past_key_values,
    tree_past_key_values,
    accepted_node_indices: Sequence[Sequence[int]],
    *,
    causal_lm=None,
    cache_format: str = "auto",
) -> KvExtractionResult:
    """Compact tree-forward KV cache to old prefix + accepted path nodes.

    Tree verification forwards all candidate nodes at positions
    ``old_seq_len .. old_seq_len + N_nodes - 1``. For each batch row we gather
    the accepted root-to-leaf path and pad shorter paths by repeating the last
    accepted node. The corresponding attention mask marks those padded cache
    slots invalid, so their key/value contents are never attended.
    """

    if old_past_key_values is None or tree_past_key_values is None:
        raise ValueError("Both old and tree past_key_values are required")
    old_seq_len = cache_seq_length(old_past_key_values)
    first_key, _ = _get_cache_layer(tree_past_key_values, 0)
    path_positions, max_len = _build_path_index(
        accepted_node_indices,
        old_seq_len=old_seq_len,
        device=first_key.device,
    )
    tree_seq_len = int(first_key.shape[2])
    if int(path_positions.max().item()) >= tree_seq_len:
        raise ValueError(
            f"Accepted node points outside tree cache: max_pos={int(path_positions.max().item())}, "
            f"tree_seq_len={tree_seq_len}, old_seq_len={old_seq_len}"
        )

    if hasattr(tree_past_key_values, "key_cache"):
        new_cache = _new_dynamic_cache(causal_lm)
        new_keys = []
        new_values = []
        for layer_idx in range(_cache_layer_count(tree_past_key_values)):
            tree_key, tree_value = _get_cache_layer(tree_past_key_values, layer_idx)
            prefix_key = tree_key[:, :, :old_seq_len, :].contiguous()
            prefix_value = tree_value[:, :, :old_seq_len, :].contiguous()
            path_key = _gather_paths_from_layer(tree_key, path_positions)
            path_value = _gather_paths_from_layer(tree_value, path_positions)
            new_keys.append(torch.cat([prefix_key, path_key], dim=2).contiguous())
            new_values.append(torch.cat([prefix_value, path_value], dim=2).contiguous())
        new_cache.key_cache = new_keys
        new_cache.value_cache = new_values
        return KvExtractionResult(new_cache, max_len, "dynamic_key_cache")

    if isinstance(tree_past_key_values, (tuple, list)):
        layers = []
        for tree_key, tree_value in tree_past_key_values:
            prefix_key = tree_key[:, :, :old_seq_len, :].contiguous()
            prefix_value = tree_value[:, :, :old_seq_len, :].contiguous()
            path_key = _gather_paths_from_layer(tree_key, path_positions)
            path_value = _gather_paths_from_layer(tree_value, path_positions)
            layers.append(
                (
                    torch.cat([prefix_key, path_key], dim=2).contiguous(),
                    torch.cat([prefix_value, path_value], dim=2).contiguous(),
                )
            )
        return KvExtractionResult(tuple(layers), max_len, "legacy_tuple")

    # New Cache classes in transformers may expose layers instead of key_cache.
    if hasattr(tree_past_key_values, "layers"):
        new_cache = _new_dynamic_cache(causal_lm)
        for layer_idx in range(_cache_layer_count(tree_past_key_values)):
            tree_key, tree_value = _get_cache_layer(tree_past_key_values, layer_idx)
            prefix_key = tree_key[:, :, :old_seq_len, :].contiguous()
            prefix_value = tree_value[:, :, :old_seq_len, :].contiguous()
            path_key = _gather_paths_from_layer(tree_key, path_positions)
            path_value = _gather_paths_from_layer(tree_value, path_positions)
            _set_cache_layer(
                new_cache,
                layer_idx,
                torch.cat([prefix_key, path_key], dim=2).contiguous(),
                torch.cat([prefix_value, path_value], dim=2).contiguous(),
            )
        return KvExtractionResult(new_cache, max_len, "dynamic_layers")

    raise TypeError(f"Unsupported cache type for KV path extraction: {type(tree_past_key_values)}")
