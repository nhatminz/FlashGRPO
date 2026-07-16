from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass

import torch

from flashgrpo_b200.decoding.acceptance import exact_accept_paths_batch, sample_from_logits
from flashgrpo_b200.decoding.kv_extraction import extract_accepted_path_kv
from flashgrpo_b200.decoding.medusa_tree import CandidateTree, TreePlan, build_batch_trees, dense_node_count, plan_tree
from flashgrpo_b200.decoding.reflex import (
    LMHeadFeedback,
    PredictionBuffer,
    ReflexAuxiliaryRecordBuffer,
    ReflexBatchStats,
    ReflexStateManager,
)
from flashgrpo_b200.decoding.tree_attention import build_tree_attention_inputs
from flashgrpo_b200.models.qwen_flashgrpo_wrapper import (
    autocast_dtype,
    forward_tokens,
    forward_tree,
    logical_lengths as mask_logical_lengths,
    model_device,
    prefill,
    repeat_interleave_cache,
    select_cache_batch,
    unwrap_causal_lm,
)


@dataclass
class FlashMedusaConfig:
    num_medusa_heads: int = 3
    tree_mode: str = "concurrency_aware"
    tree_layout: str = "dense"
    acceptance: str = "exact_target"
    cache_update_mode: str = "extract_path"
    allow_recompute_fallback: bool = True
    cpeak_nodes: int = 64
    min_tree_nodes_per_seq: int = 1
    max_tree_nodes_per_seq: int = 16
    max_tree_depth: int = 4
    fixed_tree_topk_by_depth: tuple[int, ...] = (4, 3, 2)
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int | None = None
    clone_tree_cache: bool = True
    oom_shrink_factor: float = 0.5
    enable_medusa_spec_after: int = 0
    proposal_mode: str = "medusa"
    chain_enable_after: int = 0
    chain_bootstrap_from_medusa: bool = True
    adaptive_tree_enabled: bool = False
    adaptive_confidence_metric: str = "top1_prob"
    adaptive_confidence_quantile: float = 0.25
    adaptive_confidence_low: float = 0.15
    adaptive_confidence_high: float = 0.45
    adaptive_min_topk_by_depth: tuple[int, ...] = (1, 1, 1)
    reflex_enabled: bool = False
    reflex_state_space: str = "projected"
    reflex_fast_state_dim: int = 128
    reflex_beta: float = 0.95
    reflex_eta: float = 0.1
    reflex_top_m_feedback: int = 64
    reflex_feedback_stride: int = 1
    reflex_feedback_stride_min: int = 1
    reflex_target_topk: int = 32
    reflex_feedback_union_cap: int = 96
    reflex_tv_gate_low: float = 0.05
    reflex_tv_gate_high: float = 0.20
    reflex_horizon_weight_decay: float = 0.85
    reflex_half_life_tokens: float = 48.0
    reflex_feedback_variance_beta: float = 0.99
    reflex_feedback_rms_clip: float = 3.0
    reflex_state_rms_clip: float = 2.0
    reflex_numerical_reset_rms: float = 2.5
    reflex_relative_rms_delta_base: float = 0.01
    reflex_warmup_effective_updates: float = 16.0
    reflex_magnitude_gate_floor: float = 0.25
    reflex_guard_calibration_rollouts: int = 20
    reflex_guard_aal_drop_fraction: float = 0.05
    reflex_guard_patience: int = 2
    reflex_guard_disable_rollouts: int = 50
    reflex_feedback_clip_norm: float = 2.0
    reflex_hidden_feedback_clip_norm: float = 0.0
    reflex_fast_state_clip_norm: float = 8.0
    reflex_correction_clip_norm: float = 1.0
    reflex_normalize_correction: bool = True
    reflex_feedback_ce_gate: bool = True
    reflex_feedback_ce_tau: float = 4.0
    reflex_feedback_ce_threshold: float = 0.4
    reflex_normalize_feedback: bool = True
    reflex_feedback_enabled: bool = False
    reflex_proposal_injection_enabled: bool = False
    reflex_proposal_injection_scale: float = 0.0
    reflex_proposal_injection_after: int = 0
    reflex_proposal_injection_warmup: int = 0
    reflex_aux_cache_enabled: bool = False
    reflex_aux_cache_max_records: int = 8192
    reflex_aux_cache_stride: int = 1
    reflex_aux_store_fast_state: bool = False


class FlashMedusaDecoder:
    def __init__(self, target_model, medusa_heads, tokenizer, config: FlashMedusaConfig):
        self.target_model = target_model
        self.medusa_heads = medusa_heads
        self.tokenizer = tokenizer
        self.config = config
        if config.acceptance != "exact_target":
            raise NotImplementedError("Only exact_target acceptance is implemented in the main FlashGRPO path")
        if config.cache_update_mode not in {"extract_path", "recompute_accepted"}:
            raise ValueError(f"Unsupported cache_update_mode={config.cache_update_mode}")
        if config.proposal_mode not in {"medusa", "chain"}:
            raise ValueError(f"Unsupported proposal_mode={config.proposal_mode}")
        self._reflex_guard_baseline = 0.0
        self._reflex_guard_samples = 0
        self._reflex_guard_bad_windows = 0
        self._reflex_guard_disabled_until = -1

    def _reflex_enabled(self) -> bool:
        if not bool(self.config.reflex_enabled):
            return False
        if self.config.reflex_state_space == "hidden":
            return True
        return (
            int(self.config.reflex_fast_state_dim) > 0
            and getattr(self.medusa_heads, "reflex_fast_state_dim", 0) > 0
        )

    def _reflex_effective_injection_scale(self, generation_step: int) -> float:
        if not (
            self._reflex_enabled()
            and bool(self.config.reflex_proposal_injection_enabled)
            and float(self.config.reflex_proposal_injection_scale) != 0.0
        ):
            return 0.0
        after = int(self.config.reflex_proposal_injection_after)
        step = int(generation_step)
        if step < int(self._reflex_guard_disabled_until):
            return 0.0
        if step < after:
            return 0.0
        scale = float(self.config.reflex_proposal_injection_scale)
        warmup = max(0, int(self.config.reflex_proposal_injection_warmup))
        if warmup > 0:
            scale *= min(1.0, max(0.0, float(step - after) / float(warmup)))
        return scale

    def _update_reflex_degradation_guard(self, average_accept_length: float, generation_step: int) -> None:
        if not self._reflex_enabled() or not math.isfinite(float(average_accept_length)):
            return
        calibration = max(1, int(self.config.reflex_guard_calibration_rollouts))
        value = float(average_accept_length)
        if self._reflex_guard_samples < calibration:
            self._reflex_guard_samples += 1
            self._reflex_guard_baseline += (value - self._reflex_guard_baseline) / self._reflex_guard_samples
            return
        if int(generation_step) < int(self._reflex_guard_disabled_until):
            return
        threshold = self._reflex_guard_baseline * (1.0 - float(self.config.reflex_guard_aal_drop_fraction))
        if value < threshold:
            self._reflex_guard_bad_windows += 1
        else:
            self._reflex_guard_bad_windows = 0
            self._reflex_guard_baseline = 0.98 * self._reflex_guard_baseline + 0.02 * value
        if self._reflex_guard_bad_windows >= max(1, int(self.config.reflex_guard_patience)):
            self._reflex_guard_disabled_until = int(generation_step) + max(
                1,
                int(self.config.reflex_guard_disable_rollouts),
            )
            self._reflex_guard_bad_windows = 0

    def _reflex_injection_enabled(self, generation_step: int) -> bool:
        return self._reflex_effective_injection_scale(generation_step) != 0.0

    def _scaled_fast_state(self, fast_state: torch.Tensor | None, generation_step: int) -> torch.Tensor | None:
        scale = self._reflex_effective_injection_scale(generation_step)
        if fast_state is None or scale == 0.0:
            return None
        return fast_state

    def _apply_reflex_correction(
        self,
        base_hidden: torch.Tensor,
        fast_state: torch.Tensor | None,
        effective_updates: torch.Tensor | None,
        head_idx: int,
        generation_step: int,
    ) -> torch.Tensor:
        scale = self._reflex_effective_injection_scale(generation_step)
        if fast_state is None or scale == 0.0:
            return base_hidden
        cfg = self.config
        if cfg.reflex_state_space != "hidden":
            return self.medusa_heads.add_reflex_delta(
                base_hidden,
                fast_state,
                head_idx,
                max_norm=float(cfg.reflex_correction_clip_norm),
                scale=float(scale),
                normalize=bool(cfg.reflex_normalize_correction),
            )

        state = torch.nan_to_num(fast_state.float(), nan=0.0, posinf=0.0, neginf=0.0)
        state_rms = state.square().mean(dim=-1, keepdim=True).sqrt()
        base_rms = torch.nan_to_num(base_hidden.float(), nan=0.0, posinf=0.0, neginf=0.0).square().mean(
            dim=-1, keepdim=True
        ).sqrt()
        if effective_updates is None:
            effective_updates = state_rms.new_zeros((state_rms.shape[0],))
        warmup = max(float(cfg.reflex_warmup_effective_updates), 1e-6)
        warm_gate = 1.0 - torch.exp(-effective_updates.float().view(-1, 1) / warmup)
        magnitude_gate = state_rms / (state_rms + float(cfg.reflex_magnitude_gate_floor))
        delta_k = float(cfg.reflex_relative_rms_delta_base) / math.sqrt(max(1, int(head_idx) + 1))
        alpha = (
            float(scale)
            * warm_gate
            * magnitude_gate
            * delta_k
            * base_rms
            / state_rms.clamp_min(1e-6)
        )
        correction = alpha * state
        # This clamp is tensor-only and enforces the relative RMS safety cap.
        correction_rms = correction.square().mean(dim=-1, keepdim=True).sqrt()
        cap = 1.01 * float(scale) * delta_k * base_rms
        correction = correction * torch.clamp(cap / correction_rms.clamp_min(1e-6), max=1.0)
        if base_hidden.dim() == 3 and correction.dim() == 2:
            correction = correction.unsqueeze(1)
        return base_hidden + correction.to(device=base_hidden.device, dtype=base_hidden.dtype)

    def _medusa_logits_for_last_hidden(
        self,
        last_hidden: torch.Tensor,
        *,
        lm_head,
        max_heads: int,
        fast_state: torch.Tensor | None,
        effective_updates: torch.Tensor | None,
        generation_step: int,
    ) -> list[torch.Tensor]:
        max_heads = max(0, min(int(max_heads), self.medusa_heads.num_heads))
        logits_by_head: list[torch.Tensor] = []
        for head_idx in range(max_heads):
            medusa_hidden = self.medusa_heads.heads[head_idx].project_hidden(last_hidden)
            medusa_hidden = self._apply_reflex_correction(
                medusa_hidden,
                fast_state,
                effective_updates,
                head_idx,
                generation_step,
            )
            output = self.medusa_heads.heads[head_idx].output
            if output is not None:
                logits = output(medusa_hidden)
            else:
                lm_dtype = getattr(lm_head.weight, "dtype", medusa_hidden.dtype)
                logits = lm_head(medusa_hidden.to(dtype=lm_dtype))
            logits_by_head.append(logits)
        return logits_by_head

    def _normalize_reflex_feedback(self, feedback: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        feedback = torch.nan_to_num(feedback.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if bool(cfg.reflex_normalize_feedback):
            rms = feedback.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            feedback = feedback / rms
        if float(cfg.reflex_feedback_clip_norm) > 0:
            norm = feedback.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            feedback = feedback * torch.clamp(float(cfg.reflex_feedback_clip_norm) / norm, max=1.0)
        return feedback

    def _confidence_from_logits(self, logits: torch.Tensor) -> float:
        cfg = self.config
        logits = torch.nan_to_num(logits.float(), nan=-1.0e9, posinf=1.0e9, neginf=-1.0e9)
        if cfg.adaptive_confidence_metric == "margin":
            values = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1).values
            if values.shape[-1] == 1:
                scores = values[..., 0]
            else:
                scores = values[..., 0] - values[..., 1]
        else:
            top1 = torch.max(logits, dim=-1).values
            scores = torch.exp(top1 - torch.logsumexp(logits, dim=-1))
        scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        q = min(1.0, max(0.0, float(cfg.adaptive_confidence_quantile)))
        return float(torch.quantile(scores.detach(), q).item())

    def _topk_from_confidence(self, confidence: float, base_k: int, depth_idx: int) -> int:
        cfg = self.config
        base_k = max(1, int(base_k))
        min_defaults = list(cfg.adaptive_min_topk_by_depth) or [1]
        min_k = max(1, min(base_k, int(min_defaults[min(depth_idx, len(min_defaults) - 1)])))
        if (not cfg.adaptive_tree_enabled) or base_k <= min_k:
            return base_k
        if not math.isfinite(float(confidence)):
            return min_k
        low = float(cfg.adaptive_confidence_low)
        high = float(cfg.adaptive_confidence_high)
        if high <= low:
            return min_k if confidence >= high else base_k
        if confidence <= low:
            return base_k
        if confidence >= high:
            return min_k
        ratio = (confidence - low) / (high - low)
        k_float = base_k - ratio * (base_k - min_k)
        return max(min_k, min(base_k, int(round(k_float))))

    def _plan_with_topk(self, plan: TreePlan, topk_by_depth: list[int], *, actual_nodes: int | None = None) -> TreePlan:
        topk_by_depth = [max(1, int(k)) for k in topk_by_depth]
        return TreePlan(
            node_budget_per_seq=plan.node_budget_per_seq,
            active_heads=len(topk_by_depth),
            topk_by_depth=topk_by_depth,
            actual_nodes=int(actual_nodes if actual_nodes is not None else dense_node_count(topk_by_depth)),
            mode=plan.mode,
            layout=plan.layout,
        )

    def _adapt_plan_from_logits(self, medusa_logits: list[torch.Tensor], plan: TreePlan) -> tuple[TreePlan, dict]:
        if not self.config.adaptive_tree_enabled or not medusa_logits or not plan.topk_by_depth:
            return plan, {}
        adapted = []
        confidences = []
        for depth_idx, base_k in enumerate(plan.topk_by_depth):
            if depth_idx >= len(medusa_logits):
                break
            confidence = self._confidence_from_logits(medusa_logits[depth_idx])
            confidences.append(confidence)
            adapted.append(self._topk_from_confidence(confidence, int(base_k), depth_idx))
        if not adapted:
            return self._plan_with_topk(plan, []), {"confidence": confidences, "topk": []}
        return self._plan_with_topk(plan, adapted), {"confidence": confidences, "topk": adapted}

    def _build_chain_batch_trees(
        self,
        root_tokens: torch.Tensor,
        current_hidden: torch.Tensor,
        plan: TreePlan,
        *,
        lm_head,
        embedding_layer,
        fast_state: torch.Tensor | None = None,
        effective_updates: torch.Tensor | None = None,
        generation_step: int = 0,
    ) -> tuple[list[CandidateTree], TreePlan, dict]:
        cfg = self.config
        device = current_hidden.device
        batch = int(root_tokens.shape[0])
        row_tokens = [[int(token)] for token in root_tokens.detach().cpu().tolist()]
        row_parents = [[-1] for _ in range(batch)]
        row_depths = [[1] for _ in range(batch)]
        row_scores = [[0.0] for _ in range(batch)]

        if cfg.chain_bootstrap_from_medusa and self.medusa_heads.num_heads > 0:
            parent_states = self.medusa_heads.heads[0].project_hidden(current_hidden.detach())
        else:
            parent_states = self.medusa_heads.chain_next_state(current_hidden.detach(), root_tokens, embedding_layer)
        parent_rows = torch.arange(batch, dtype=torch.long, device=device)
        parent_node_ids = [0 for _ in range(batch)]
        parent_fast_state = self._scaled_fast_state(fast_state, generation_step)
        parent_effective_updates = effective_updates
        adapted_topk: list[int] = []
        confidences: list[float] = []
        record_logits: list[torch.Tensor] = []

        for depth_idx, base_k in enumerate(plan.topk_by_depth):
            if parent_states.numel() == 0:
                break
            logit_states = self._apply_reflex_correction(
                parent_states,
                parent_fast_state,
                parent_effective_updates,
                depth_idx,
                generation_step,
            )
            logits = torch.nan_to_num(
                self.medusa_heads.chain_logits_from_state(logit_states, lm_head).float(),
                nan=-1.0e9,
                posinf=1.0e9,
                neginf=-1.0e9,
            )
            if depth_idx == 0 and parent_rows.numel() == batch and bool((parent_rows == torch.arange(batch, device=device)).all().item()):
                record_logits.append(logits.detach())
            confidence = self._confidence_from_logits(logits)
            confidences.append(confidence)
            k = self._topk_from_confidence(confidence, int(base_k), depth_idx)
            if k <= 0:
                break
            adapted_topk.append(k)
            values, indices = torch.topk(logits, k=min(k, logits.shape[-1]), dim=-1)
            parent_rows_cpu = parent_rows.detach().cpu().tolist()
            token_rows = indices.detach().cpu().tolist()
            score_rows = values.detach().cpu().tolist()
            next_parent_state_indices: list[int] = []
            next_token_ids: list[int] = []
            next_rows: list[int] = []
            next_node_ids: list[int] = []
            for parent_state_idx, row in enumerate(parent_rows_cpu):
                if len(row_tokens[row]) >= plan.node_budget_per_seq:
                    continue
                parent_node = int(parent_node_ids[parent_state_idx])
                for token, score in zip(token_rows[parent_state_idx], score_rows[parent_state_idx]):
                    if len(row_tokens[row]) >= plan.node_budget_per_seq:
                        break
                    row_tokens[row].append(int(token))
                    row_parents[row].append(parent_node)
                    row_depths[row].append(depth_idx + 2)
                    row_scores[row].append(float(score))
                    next_parent_state_indices.append(parent_state_idx)
                    next_token_ids.append(int(token))
                    next_rows.append(int(row))
                    next_node_ids.append(len(row_tokens[row]) - 1)
            if not next_token_ids:
                break
            parent_index = torch.tensor(next_parent_state_indices, dtype=torch.long, device=device)
            token_tensor = torch.tensor(next_token_ids, dtype=torch.long, device=device)
            parent_states = self.medusa_heads.chain_next_state(
                parent_states.index_select(0, parent_index),
                token_tensor,
                embedding_layer,
            )
            parent_fast_state = parent_fast_state.index_select(0, parent_index) if parent_fast_state is not None else None
            parent_effective_updates = (
                parent_effective_updates.index_select(0, parent_index)
                if parent_effective_updates is not None
                else None
            )
            parent_rows = torch.tensor(next_rows, dtype=torch.long, device=device)
            parent_node_ids = next_node_ids
            del logits, values, indices, parent_index, token_tensor

        trees = [
            CandidateTree(tokens=row_tokens[row], parents=row_parents[row], depths=row_depths[row], scores=row_scores[row])
            for row in range(batch)
        ]
        actual_nodes = max((tree.node_count for tree in trees), default=1)
        chain_plan = self._plan_with_topk(plan, adapted_topk, actual_nodes=actual_nodes)
        return trees, chain_plan, {"confidence": confidences, "topk": adapted_topk, "record_logits": record_logits}

    def _last_valid_hidden(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_idx = attention_mask.long().sum(dim=-1).clamp_min(1) - 1
        # The training collator left-pads prompts, so the last non-padding token
        # is at the right edge. This fallback also handles non-left-padded smoke tests.
        if bool((attention_mask[:, -1] == 1).all().item()):
            return hidden_states[:, -1, :]
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, hidden_states.shape[-1])
        return hidden_states.gather(1, gather_idx).squeeze(1)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        repeated_generate_nums: int | None = None,
        max_length: int = 2048,
        statistical_time: bool = False,
        generation_step: int = 0,
        collect_reflex_aux_cache: bool | None = None,
    ) -> dict:
        cfg = self.config
        device = model_device(self.target_model)
        repeats = max(1, int(repeated_generate_nums or 1))
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        base = unwrap_causal_lm(self.target_model)
        lm_head = base.lm_head
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = pad_token_id

        total_start = time.time()
        if statistical_time and torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_start = time.time()
        prefill_out = prefill(self.target_model, input_ids, attention_mask)
        if statistical_time and torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_time = time.time() - prefill_start
        past_key_values = prefill_out["past_key_values"]
        prefill_hidden = prefill_out["hidden_states"]
        current_hidden = self._last_valid_hidden(prefill_hidden, attention_mask)
        del prefill_hidden, prefill_out
        with torch.amp.autocast(
            "cuda" if device.type == "cuda" else device.type,
            dtype=autocast_dtype(base),
            enabled=(device.type == "cuda"),
        ):
            current_logits = lm_head(current_hidden.to(dtype=getattr(lm_head.weight, "dtype", current_hidden.dtype)))
        if repeats > 1:
            past_key_values = repeat_interleave_cache(past_key_values, repeats, causal_lm=self.target_model)
            current_hidden = current_hidden.repeat_interleave(repeats, dim=0).contiguous()
            current_logits = current_logits.repeat_interleave(repeats, dim=0).contiguous()
            attention_mask = attention_mask.repeat_interleave(repeats, dim=0).contiguous()

        total_sequences = attention_mask.shape[0]
        generated: list[list[int]] = [[] for _ in range(total_sequences)]
        active_original_indices = list(range(total_sequences))
        full_attention_mask = attention_mask.long()
        logical_lens = mask_logical_lengths(full_attention_mask)
        initial_logical_lens = logical_lens.clone()
        reflex_enabled = self._reflex_enabled()
        reflex_injection_enabled = self._reflex_injection_enabled(generation_step)
        reflex_feedback_enabled = reflex_enabled and bool(cfg.reflex_feedback_enabled)
        collect_reflex_aux_cache = (
            bool(cfg.reflex_aux_cache_enabled)
            if collect_reflex_aux_cache is None
            else bool(collect_reflex_aux_cache)
        )
        collect_reflex_aux_cache = bool(collect_reflex_aux_cache and reflex_enabled)
        state_dim = int(current_hidden.shape[-1]) if cfg.reflex_state_space == "hidden" else int(cfg.reflex_fast_state_dim)
        reflex_manager = (
            ReflexStateManager(
                total_sequences,
                state_dim,
                device=device,
                half_life_tokens=float(cfg.reflex_half_life_tokens),
                eta=float(cfg.reflex_eta),
                feedback_variance_beta=float(cfg.reflex_feedback_variance_beta),
                feedback_rms_clip=float(cfg.reflex_feedback_rms_clip),
                state_rms_clip=float(cfg.reflex_state_rms_clip),
                numerical_reset_rms=float(cfg.reflex_numerical_reset_rms),
            )
            if reflex_enabled
            else None
        )
        prediction_buffer = PredictionBuffer() if reflex_feedback_enabled else None
        reflex_aux_buffer = (
            ReflexAuxiliaryRecordBuffer(max_records=int(cfg.reflex_aux_cache_max_records))
            if collect_reflex_aux_cache
            else None
        )
        lm_feedback = (
            LMHeadFeedback(
                lm_head,
                target_topk=int(cfg.reflex_target_topk),
                union_cap=int(cfg.reflex_feedback_union_cap),
                tv_gate_low=float(cfg.reflex_tv_gate_low),
                tv_gate_high=float(cfg.reflex_tv_gate_high),
                horizon_weight_decay=float(cfg.reflex_horizon_weight_decay),
            )
            if reflex_feedback_enabled
            else None
        )
        reflex_stats = ReflexBatchStats(num_heads=min(cfg.num_medusa_heads, self.medusa_heads.num_heads))

        total_acc_length = 0
        total_decoded_steps = 0
        total_accepted_medusa_tokens = 0
        total_proposed_medusa_tokens = 0
        total_verify_rounds = 0
        active_batch_sum = 0
        tree_node_sum = 0
        tree_sample_count = 0
        medusa_head_time = 0.0
        tree_verify_time = 0.0
        cache_update_time = 0.0
        kv_extraction_time = 0.0
        recompute_fallback_time = 0.0
        kv_extraction_success_count = 0
        kv_extraction_fallback_count = 0
        oom_count = 0
        accept_hist: dict[int, int] = {}
        accept_by_depth: dict[int, int] = {}
        proposed_by_depth: dict[int, int] = {}
        tree_plan_last = {}
        adaptive_tree_stats_last = {}
        reflex_feedback_collection_rounds = 0

        while active_original_indices:
            active_bsz = len(active_original_indices)
            remaining = max_length - logical_lens
            if not bool((remaining > 0).any().item()):
                break
            old_logical_lens = logical_lens.clone()
            active_fast_state = reflex_manager.get(active_original_indices) if reflex_manager is not None else None
            active_effective_updates = (
                reflex_manager.get_effective_updates(active_original_indices) if reflex_manager is not None else None
            )

            root_tokens = sample_from_logits(
                current_logits,
                do_sample=cfg.do_sample,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
            )

            use_medusa_tree = int(generation_step) >= int(cfg.enable_medusa_spec_after)
            if use_medusa_tree:
                plan = plan_tree(
                    active_batch_size=active_bsz,
                    num_medusa_heads=min(cfg.num_medusa_heads, self.medusa_heads.num_heads),
                    tree_mode=cfg.tree_mode,
                    tree_layout=cfg.tree_layout,
                    cpeak_nodes=cfg.cpeak_nodes,
                    min_tree_nodes_per_seq=cfg.min_tree_nodes_per_seq,
                    max_tree_nodes_per_seq=cfg.max_tree_nodes_per_seq,
                    max_tree_depth=cfg.max_tree_depth,
                    fixed_tree_topk_by_depth=list(cfg.fixed_tree_topk_by_depth),
                )
                use_chain = cfg.proposal_mode == "chain" and int(generation_step) >= int(cfg.chain_enable_after)
                if use_chain:
                    if statistical_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    head_start = time.time()
                    with torch.no_grad():
                        trees, plan, adaptive_tree_stats = self._build_chain_batch_trees(
                            root_tokens,
                            current_hidden.detach(),
                            plan,
                            lm_head=lm_head,
                            embedding_layer=base.get_input_embeddings(),
                            fast_state=active_fast_state,
                            effective_updates=active_effective_updates,
                            generation_step=generation_step,
                        )
                    if statistical_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    medusa_head_time += time.time() - head_start
                    medusa_logits = []
                    record_logits = adaptive_tree_stats.pop("record_logits", [])
                else:
                    if statistical_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    head_start = time.time()
                    with torch.no_grad():
                        medusa_logits = self._medusa_logits_for_last_hidden(
                            current_hidden.detach(),
                            lm_head=lm_head,
                            max_heads=plan.active_heads,
                            fast_state=self._scaled_fast_state(active_fast_state, generation_step),
                            effective_updates=active_effective_updates,
                            generation_step=generation_step,
                        )
                        medusa_logits = [
                            torch.nan_to_num(item.float(), nan=-1.0e9, posinf=1.0e9, neginf=-1.0e9)
                            for item in medusa_logits
                        ]
                    if statistical_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    medusa_head_time += time.time() - head_start
                    plan, adaptive_tree_stats = self._adapt_plan_from_logits(medusa_logits, plan)
                    trees = build_batch_trees(root_tokens, medusa_logits, plan)
                    record_logits = medusa_logits
                proposal_mode_used = "chain" if use_chain else "medusa"
                adaptive_tree_stats_last = adaptive_tree_stats
            else:
                medusa_logits = []
                record_logits = []
                proposal_mode_used = "target_only"
                adaptive_tree_stats_last = {}
                plan = TreePlan(
                    node_budget_per_seq=1,
                    active_heads=0,
                    topk_by_depth=[],
                    actual_nodes=1,
                    mode=cfg.tree_mode,
                    layout=cfg.tree_layout,
                )
                trees = build_batch_trees(root_tokens, medusa_logits, plan)
            active_original_tensor = None
            if (prediction_buffer is not None and record_logits) or (reflex_aux_buffer is not None and plan.active_heads > 0):
                active_original_tensor = torch.as_tensor(active_original_indices, dtype=torch.long, device=device)
            feedback_stride = max(
                int(cfg.reflex_feedback_stride_min),
                int(math.ceil(max(1, int(cfg.reflex_feedback_stride)) * active_bsz / max(total_sequences, 1))),
            )
            collect_feedback_this_round = (
                prediction_buffer is not None
                and bool(record_logits)
                and total_verify_rounds % feedback_stride == 0
            )
            if collect_feedback_this_round:
                reflex_feedback_collection_rounds += 1
                prediction_buffer.add_from_logits(
                    sequence_ids=active_original_indices,
                    anchor_positions=old_logical_lens,
                    logits_by_horizon=record_logits[: plan.active_heads],
                    top_m=int(cfg.reflex_top_m_feedback),
                    initial_lengths=initial_logical_lens.index_select(0, active_original_tensor),
                )
            if (
                reflex_aux_buffer is not None
                and plan.active_heads > 0
                and total_verify_rounds % max(1, int(cfg.reflex_aux_cache_stride)) == 0
            ):
                reflex_aux_buffer.add_anchor_predictions(
                    sequence_ids=active_original_indices,
                    anchor_positions=old_logical_lens,
                    initial_lengths=initial_logical_lens.index_select(0, active_original_tensor),
                    hidden_states=current_hidden.detach(),
                    fast_states=(
                        self._scaled_fast_state(active_fast_state, generation_step)
                        if bool(cfg.reflex_aux_store_fast_state)
                        else None
                    ),
                    max_horizon=min(int(plan.active_heads) + 1, self.medusa_heads.num_heads + 1),
                    reflex_scale=float(self._reflex_effective_injection_scale(generation_step)),
                )
            tree_plan_last = {
                "B_cur": active_bsz,
                "node_budget_per_seq": plan.node_budget_per_seq,
                "active_heads": plan.active_heads,
                "topk_by_depth": plan.topk_by_depth,
                "actual_nodes": plan.actual_nodes,
                "proposal_mode": proposal_mode_used,
                "adaptive_tree": adaptive_tree_stats_last,
            }
            active_batch_sum += active_bsz
            tree_node_sum += sum(tree.node_count for tree in trees) / max(active_bsz, 1)
            tree_sample_count += 1
            for tree in trees:
                for depth in tree.depths:
                    if int(depth) >= 2:
                        proposed_by_depth[int(depth)] = proposed_by_depth.get(int(depth), 0) + 1

            tree_logits = None
            tree_hidden = None
            tree_past_key_values = None
            if plan.active_heads > 0 and max(tree.node_count for tree in trees) > 1:
                try:
                    tree_input_ids, tree_mask, tree_position_ids, _ = build_tree_attention_inputs(
                        trees,
                        full_attention_mask,
                        logical_lens,
                        pad_token_id=pad_token_id,
                        dtype=autocast_dtype(base),
                    )
                    if statistical_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    verify_start = time.time()
                    tree_out = forward_tree(
                        self.target_model,
                        tree_input_ids,
                        tree_mask,
                        past_key_values,
                        tree_position_ids,
                        clone_past=cfg.clone_tree_cache,
                    )
                    if statistical_time and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    tree_verify_time += time.time() - verify_start
                    tree_logits = tree_out["logits"].float()
                    tree_hidden = tree_out["hidden_states"]
                    tree_past_key_values = tree_out["past_key_values"]
                    del tree_out
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    oom_count += 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    tree_logits = None

            if tree_logits is None:
                raw_accepted_per_row = [[int(token)] for token in root_tokens.detach().cpu().tolist()]
                raw_accepted_nodes_per_row = [[0] for _ in trees]
                raw_parent_nodes = [0 for _ in trees]
            else:
                raw_accepted_per_row, raw_accepted_nodes_per_row, raw_parent_nodes = exact_accept_paths_batch(
                    trees,
                    tree_logits,
                    do_sample=cfg.do_sample,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                )

            remaining_cpu = remaining.detach().cpu().tolist()
            old_logical_lens_cpu = old_logical_lens.detach().cpu().tolist()
            accepted_per_row: list[list[int]] = []
            accepted_nodes_per_row: list[list[int]] = []
            finished_flags: list[bool] = []
            parent_nodes: list[int] = []
            for row, (raw_tokens, raw_nodes, raw_parent) in enumerate(
                zip(raw_accepted_per_row, raw_accepted_nodes_per_row, raw_parent_nodes)
            ):
                accepted_tokens = list(raw_tokens)
                accepted_nodes = list(raw_nodes)
                parent = int(raw_parent)
                max_accept = max(1, int(remaining_cpu[row]))
                accepted_tokens = accepted_tokens[:max_accept]
                accepted_nodes = accepted_nodes[: len(accepted_tokens)]
                eos_seen = False
                for eos_pos, token in enumerate(accepted_tokens):
                    if token == eos_token_id:
                        accepted_tokens = accepted_tokens[: eos_pos + 1]
                        accepted_nodes = accepted_nodes[: eos_pos + 1]
                        eos_seen = True
                        break
                accepted_len = len(accepted_tokens)
                accept_hist[accepted_len] = accept_hist.get(accepted_len, 0) + 1
                for depth in range(2, accepted_len + 1):
                    accept_by_depth[depth] = accept_by_depth.get(depth, 0) + 1
                total_acc_length += accepted_len
                total_decoded_steps += 1
                total_accepted_medusa_tokens += max(accepted_len - 1, 0)
                total_proposed_medusa_tokens += max(tree.node_count - 1, 0)
                accepted_per_row.append(accepted_tokens)
                accepted_nodes_per_row.append([int(node_idx) for node_idx in accepted_nodes])
                finished_flags.append(eos_seen or int(old_logical_lens_cpu[row]) + accepted_len >= max_length)
                parent_nodes.append(parent)

            total_verify_rounds += 1
            max_acc = max(len(tokens) for tokens in accepted_per_row)
            accepted_ids_cpu = torch.full((active_bsz, max_acc), int(pad_token_id), dtype=torch.long)
            valid_ext_cpu = torch.zeros((active_bsz, max_acc), dtype=torch.long)
            position_ids_cpu = torch.zeros((active_bsz, max_acc), dtype=torch.long)
            for row, tokens in enumerate(accepted_per_row):
                accepted_ids_cpu[row, : len(tokens)] = torch.as_tensor(tokens, dtype=torch.long)
                valid_ext_cpu[row, : len(tokens)] = 1
                position_ids_cpu[row, : len(tokens)] = int(old_logical_lens_cpu[row]) + torch.arange(len(tokens))
                original_idx = active_original_indices[row]
                generated[original_idx].extend(tokens)
            accepted_ids = accepted_ids_cpu.to(device=device, non_blocking=True)
            valid_ext = valid_ext_cpu.to(device=device, non_blocking=True)
            position_ids = position_ids_cpu.to(device=device, non_blocking=True)

            aux_teachers: dict[tuple[int, int], dict] = {}
            if prediction_buffer is not None and reflex_manager is not None and lm_feedback is not None:
                # States advance once per actual token. Positions accepted in the
                # same verification round are processed in trajectory order.
                for offset in range(1, max_acc + 1):
                    offset_rows = [row for row, tokens in enumerate(accepted_per_row) if len(tokens) >= offset]
                    if not offset_rows:
                        continue
                    offset_seq_ids = [int(active_original_indices[row]) for row in offset_rows]
                    feedback = torch.zeros(
                        (len(offset_rows), reflex_manager.fast_state_dim),
                        device=device,
                        dtype=torch.float32,
                    )
                    has_feedback = torch.zeros((len(offset_rows),), device=device, dtype=torch.bool)
                    effective_mass = torch.zeros((len(offset_rows),), device=device, dtype=torch.float32)
                    record_groups: list[list] = []
                    target_rows: list[torch.Tensor] = []
                    true_tokens: list[int] = []
                    feedback_row_indices: list[int] = []
                    flat_records = []
                    flat_accepted_flags: list[bool] = []

                    for local_row, row in enumerate(offset_rows):
                        seq_id = int(active_original_indices[row])
                        anchor_pos = int(old_logical_lens_cpu[row])
                        token = int(accepted_per_row[row][offset - 1])
                        records = prediction_buffer.pop_mature(seq_id, anchor_pos + offset)
                        if not records:
                            continue
                        if offset == 1:
                            target_row = current_logits[row]
                        else:
                            if tree_logits is None:
                                raise RuntimeError("Accepted tree token is missing target verification logits")
                            parent_node = int(accepted_nodes_per_row[row][offset - 2])
                            target_row = tree_logits[row, parent_node]
                        record_groups.append(records)
                        target_rows.append(target_row)
                        true_tokens.append(token)
                        feedback_row_indices.append(local_row)
                        flat_records.extend(records)
                        flat_accepted_flags.extend(
                            record.anchor_pos == anchor_pos and int(record.horizon) <= len(accepted_per_row[row])
                            for record in records
                        )

                    if record_groups:
                        sparse = lm_feedback.compute_batch(
                            record_groups,
                            torch.stack(target_rows, dim=0),
                            true_tokens,
                        )
                        feedback_index = torch.as_tensor(feedback_row_indices, dtype=torch.long, device=device)
                        feedback.index_copy_(0, feedback_index, sparse.feedback)
                        has_feedback.index_copy_(0, feedback_index, sparse.has_feedback)
                        effective_mass.index_copy_(0, feedback_index, sparse.effective_mass)
                        reflex_stats.add_records(
                            flat_records,
                            sparse.record_true_probs,
                            flat_accepted_flags,
                            sparse.record_tv,
                            sparse.record_gates,
                        )
                        if reflex_aux_buffer is not None:
                            for group_idx, local_row in enumerate(feedback_row_indices):
                                row = offset_rows[local_row]
                                seq_id = int(active_original_indices[row])
                                target_pos = int(old_logical_lens_cpu[row]) + offset
                                aux_teachers[(seq_id, target_pos)] = {
                                    "target_top_ids": sparse.target_top_ids[group_idx],
                                    "target_top_logits": sparse.target_top_logits[group_idx],
                                    "target_logsumexp": float(sparse.target_logsumexp[group_idx]),
                                    "proposal_records": record_groups[group_idx],
                                }

                    feedback_rms = reflex_manager.advance_token(
                        offset_seq_ids,
                        feedback,
                        has_feedback,
                        effective_mass,
                    )
                    reflex_stats.add_feedback_rms(feedback_rms, has_feedback)
            if reflex_aux_buffer is not None:
                for row, tokens in enumerate(accepted_per_row):
                    seq_id = int(active_original_indices[row])
                    anchor_pos = int(old_logical_lens_cpu[row])
                    for offset, token in enumerate(tokens, start=1):
                        reflex_aux_buffer.pop_mature(
                            seq_id,
                            anchor_pos + offset,
                            generated[seq_id],
                            int(token),
                            teacher=aux_teachers.get((seq_id, anchor_pos + offset)),
                        )

            new_attention_mask = torch.cat([full_attention_mask, valid_ext], dim=1)
            use_extract = (
                cfg.cache_update_mode == "extract_path"
                and tree_logits is not None
                and tree_hidden is not None
                and tree_past_key_values is not None
            )
            extracted = False
            if use_extract:
                if statistical_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                extract_start = time.time()
                try:
                    result = extract_accepted_path_kv(
                        past_key_values,
                        tree_past_key_values,
                        accepted_nodes_per_row,
                        causal_lm=self.target_model,
                    )
                    past_key_values = result.past_key_values
                    row_idx = torch.arange(active_bsz, device=device)
                    last_node_idx = torch.tensor(
                        [path[-1] for path in accepted_nodes_per_row],
                        dtype=torch.long,
                        device=device,
                    )
                    current_hidden = tree_hidden[row_idx, last_node_idx, :]
                    current_logits = tree_logits[row_idx, last_node_idx, :]
                    extracted = True
                    kv_extraction_success_count += 1
                    del result, row_idx, last_node_idx
                except Exception as exc:
                    kv_extraction_fallback_count += 1
                    if not cfg.allow_recompute_fallback:
                        raise RuntimeError("KV path extraction failed and allow_recompute_fallback=false") from exc
                    warnings.warn(f"KV path extraction failed; falling back to recompute_accepted: {exc}", RuntimeWarning)
                if statistical_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                kv_extraction_time += time.time() - extract_start
                cache_update_time += time.time() - extract_start
            elif cfg.cache_update_mode == "extract_path" and cfg.allow_recompute_fallback and plan.active_heads > 0:
                kv_extraction_fallback_count += 1

            if not extracted:
                if cfg.cache_update_mode == "extract_path" and not cfg.allow_recompute_fallback and plan.active_heads > 0:
                    raise RuntimeError("KV extraction was requested but no tree cache was available")
                if statistical_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                cache_start = time.time()
                cache_out = forward_tokens(
                    self.target_model,
                    accepted_ids,
                    new_attention_mask,
                    past_key_values,
                    position_ids,
                )
                if statistical_time and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - cache_start
                cache_update_time += elapsed
                if cfg.cache_update_mode == "extract_path" and plan.active_heads > 0:
                    recompute_fallback_time += elapsed
                past_key_values = cache_out["past_key_values"]
                token_hidden = cache_out["hidden_states"]
                last_indices = (valid_ext.sum(dim=-1) - 1).clamp_min(0).view(-1, 1, 1).expand(-1, 1, token_hidden.shape[-1])
                current_hidden = token_hidden.gather(1, last_indices).squeeze(1)
                with torch.amp.autocast(
                    "cuda" if device.type == "cuda" else device.type,
                    dtype=autocast_dtype(base),
                    enabled=(device.type == "cuda"),
                ):
                    current_logits = lm_head(current_hidden.to(dtype=getattr(lm_head.weight, "dtype", current_hidden.dtype)))
                del cache_out, token_hidden, last_indices
            tree_logits = None
            tree_hidden = None
            tree_past_key_values = None
            full_attention_mask = new_attention_mask
            logical_lens = logical_lens + valid_ext.sum(dim=-1)

            keep_rows = [idx for idx, done in enumerate(finished_flags) if not done]
            if len(keep_rows) != active_bsz:
                done_seq_ids = [active_original_indices[idx] for idx, done in enumerate(finished_flags) if done]
                if prediction_buffer is not None:
                    for seq_id in done_seq_ids:
                        prediction_buffer.clear_sequence(seq_id)
                if reflex_aux_buffer is not None:
                    for seq_id in done_seq_ids:
                        reflex_aux_buffer.clear_sequence(seq_id)
                if keep_rows:
                    keep = torch.tensor(keep_rows, dtype=torch.long, device=device)
                    past_key_values = select_cache_batch(past_key_values, keep, causal_lm=self.target_model)
                    current_hidden = current_hidden.index_select(0, keep)
                    current_logits = current_logits.index_select(0, keep)
                    full_attention_mask = full_attention_mask.index_select(0, keep)
                    logical_lens = logical_lens.index_select(0, keep)
                    active_original_indices = [active_original_indices[idx] for idx in keep_rows]
                else:
                    active_original_indices = []
                    break

        max_sequence_length = max((len(seq) for seq in generated), default=0)
        total_time = time.time() - total_start
        accept_rate = total_accepted_medusa_tokens / max(total_proposed_medusa_tokens, 1)
        avg_accept = total_acc_length / max(total_decoded_steps, 1)
        self._update_reflex_degradation_guard(avg_accept, generation_step)
        reflex_metrics = reflex_stats.to_dict()
        reflex_metrics["enabled"] = bool(reflex_enabled)
        reflex_metrics["feedback_enabled"] = bool(reflex_feedback_enabled)
        reflex_metrics["proposal_injection_enabled"] = bool(reflex_injection_enabled)
        reflex_metrics["proposal_injection_scale"] = float(cfg.reflex_proposal_injection_scale)
        reflex_metrics["proposal_injection_effective_scale"] = float(self._reflex_effective_injection_scale(generation_step))
        reflex_metrics["aal_guard_baseline"] = float(self._reflex_guard_baseline)
        reflex_metrics["aal_guard_disabled_until"] = int(self._reflex_guard_disabled_until)
        reflex_metrics["aal_guard_bad_windows"] = int(self._reflex_guard_bad_windows)
        reflex_metrics["pending_prediction_records"] = len(prediction_buffer) if prediction_buffer is not None else 0
        reflex_metrics["feedback_collection_rounds"] = int(reflex_feedback_collection_rounds)
        reflex_metrics["feedback_collection_fraction"] = reflex_feedback_collection_rounds / max(total_verify_rounds, 1)
        if reflex_manager is not None:
            reflex_metrics.update(reflex_manager.norm_stats())
        else:
            reflex_metrics.update({"fast_state_norm_mean": 0.0, "fast_state_norm_p95": 0.0})
        return {
            "generated_token_ids": generated,
            "max_sequence_length": max_sequence_length,
            "total_acc_length": int(total_acc_length),
            "average_accept_length": float(avg_accept),
            "accepted_tokens_per_medusa_step": float(avg_accept),
            "total_decoded_token_num": int(total_decoded_steps),
            "total_accepted_draft_tokens": int(total_accepted_medusa_tokens),
            "total_proposed_draft_tokens": int(total_proposed_medusa_tokens),
            "total_accepted_medusa_tokens": int(total_accepted_medusa_tokens),
            "total_proposed_medusa_tokens": int(total_proposed_medusa_tokens),
            "draft_acceptance_rate": float(accept_rate),
            "medusa_acceptance_rate": float(accept_rate),
            "total_verify_rounds": int(total_verify_rounds),
            "average_active_batch_size": active_batch_sum / max(total_verify_rounds, 1),
            "average_tree_nodes_per_seq": tree_node_sum / max(tree_sample_count, 1),
            "accept_length_histogram": accept_hist,
            "medusa_accept_by_depth": accept_by_depth,
            "medusa_proposed_by_depth": proposed_by_depth,
            "last_tree_plan": tree_plan_last,
            "cache_update_mode": cfg.cache_update_mode,
            "kv_extraction_success_count": int(kv_extraction_success_count),
            "kv_extraction_fallback_count": int(kv_extraction_fallback_count),
            "kv_extraction_time": kv_extraction_time,
            "recompute_fallback_time": recompute_fallback_time,
            "oom_count": int(oom_count),
            "total_time_cost": total_time,
            "prefill_time_cost": prefill_time,
            "target_time_cost": prefill_time + tree_verify_time + cache_update_time,
            "tree_verify_time_cost": tree_verify_time,
            "cache_update_time_cost": cache_update_time,
            "medusa_head_time_cost": medusa_head_time,
            "draft_time_cost": medusa_head_time,
            "check_time_cost": 0.0,
            "reflex_metrics": reflex_metrics,
            "reflex_head_metrics": reflex_metrics.get("per_head", {}),
            "reflex_aux_records": reflex_aux_buffer.to_batch() if reflex_aux_buffer is not None else {},
        }
