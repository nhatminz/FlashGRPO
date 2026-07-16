from __future__ import annotations

import torch
from transformers import DynamicCache


def unwrap_causal_lm(causal_lm):
    if hasattr(causal_lm, "get_base_model"):
        return causal_lm.get_base_model()
    if hasattr(causal_lm, "base_model") and hasattr(causal_lm.base_model, "model"):
        return causal_lm.base_model.model
    return causal_lm


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_dtype(model) -> torch.dtype:
    dtype = getattr(model, "dtype", None)
    if dtype == torch.bfloat16:
        return torch.bfloat16
    return torch.float16


def position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    return position_ids.masked_fill(attention_mask == 0, 0)


def logical_lengths(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.long().sum(dim=-1)


def _cache_layer_count(cache) -> int:
    if hasattr(cache, "key_cache"):
        return len(cache.key_cache)
    if hasattr(cache, "layers"):
        return len(cache.layers)
    return len(cache)


def _get_cache_layer(cache, layer_idx: int):
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    return cache[layer_idx]


def _set_cache_layer(cache, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
    if hasattr(cache, "key_cache"):
        cache.key_cache[layer_idx] = key
        cache.value_cache[layer_idx] = value
    elif hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        layer.keys = key
        layer.values = value
        layer.is_initialized = True
    else:
        cache[layer_idx] = (key, value)


def select_cache_batch(cache, indices: torch.Tensor, causal_lm=None):
    if cache is None:
        return None
    indices = indices.to(dtype=torch.long, device=_get_cache_layer(cache, 0)[0].device)
    if causal_lm is not None:
        try:
            new_cache = DynamicCache(config=unwrap_causal_lm(causal_lm).config)
        except TypeError:
            new_cache = DynamicCache()
    else:
        new_cache = DynamicCache()
    if hasattr(cache, "key_cache"):
        new_cache.key_cache = [key.index_select(0, indices).contiguous() for key in cache.key_cache]
        new_cache.value_cache = [value.index_select(0, indices).contiguous() for value in cache.value_cache]
        return new_cache
    for layer_idx in range(_cache_layer_count(cache)):
        key, value = _get_cache_layer(cache, layer_idx)
        _set_cache_layer(new_cache, layer_idx, key.index_select(0, indices), value.index_select(0, indices))
    return new_cache


def repeat_interleave_cache(cache, repeats: int, causal_lm=None):
    if cache is None or repeats == 1:
        return cache
    if causal_lm is not None:
        try:
            new_cache = DynamicCache(config=unwrap_causal_lm(causal_lm).config)
        except TypeError:
            new_cache = DynamicCache()
    else:
        new_cache = DynamicCache()
    if hasattr(cache, "key_cache"):
        new_cache.key_cache = [key.repeat_interleave(repeats, dim=0).contiguous() for key in cache.key_cache]
        new_cache.value_cache = [value.repeat_interleave(repeats, dim=0).contiguous() for value in cache.value_cache]
        return new_cache
    for layer_idx in range(_cache_layer_count(cache)):
        key, value = _get_cache_layer(cache, layer_idx)
        _set_cache_layer(new_cache, layer_idx, key.repeat_interleave(repeats, dim=0), value.repeat_interleave(repeats, dim=0))
    return new_cache


def _forward_decoder(
    causal_lm,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    past_key_values=None,
    position_ids: torch.Tensor | None = None,
    use_cache: bool = True,
    compute_logits: bool = True,
):
    base = unwrap_causal_lm(causal_lm)
    device = model_device(base)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    if position_ids is None:
        if attention_mask.dim() == 2:
            position_ids = position_ids_from_attention_mask(attention_mask)[:, -input_ids.shape[1] :]
        else:
            raise ValueError("position_ids are required with 4D attention masks")
    position_ids = position_ids.to(device)
    device_type = "cuda" if device.type == "cuda" else device.type
    with torch.amp.autocast(device_type, dtype=autocast_dtype(base), enabled=(device.type == "cuda")):
        outputs = base.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        logits = base.lm_head(hidden_states) if compute_logits else None
    return {
        "hidden_states": hidden_states,
        "logits": logits,
        "past_key_values": outputs.past_key_values if hasattr(outputs, "past_key_values") else None,
    }


def prefill(causal_lm, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    position_ids = position_ids_from_attention_mask(attention_mask)
    return _forward_decoder(
        causal_lm,
        input_ids,
        attention_mask,
        past_key_values=DynamicCache(),
        position_ids=position_ids,
        use_cache=True,
        compute_logits=False,
    )


def forward_tokens(
    causal_lm,
    input_ids: torch.Tensor,
    full_attention_mask: torch.Tensor,
    past_key_values,
    position_ids: torch.Tensor,
):
    return _forward_decoder(
        causal_lm,
        input_ids,
        full_attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
        use_cache=True,
        compute_logits=False,
    )
