from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(slots=True)
class PredictionRecord:
    sequence_id: int
    anchor_pos: int
    target_pos: int
    horizon: int
    top_ids: torch.Tensor
    top_logits: torch.Tensor
    logsumexp: float
    depth: int = 0


class PredictionBuffer:
    """Detached sparse proposal summaries indexed by actual target position."""

    def __init__(self):
        self._records: dict[tuple[int, int], list[PredictionRecord]] = {}

    def add(self, record: PredictionRecord) -> None:
        key = (int(record.sequence_id), int(record.target_pos))
        self._records.setdefault(key, []).append(record)

    @torch.no_grad()
    def add_from_logits(
        self,
        *,
        sequence_ids: list[int],
        anchor_positions: torch.Tensor,
        logits_by_horizon: list[torch.Tensor],
        top_m: int,
        initial_lengths: torch.Tensor | None = None,
    ) -> None:
        if not logits_by_horizon or top_m <= 0 or not sequence_ids:
            return
        anchor_cpu = anchor_positions.detach().to(device="cpu", dtype=torch.long).tolist()
        if initial_lengths is None:
            initial_cpu = anchor_cpu
        else:
            initial_cpu = initial_lengths.detach().to(device="cpu", dtype=torch.long).tolist()
        sequence_cpu = [int(seq_id) for seq_id in sequence_ids]

        for head_idx, logits in enumerate(logits_by_horizon):
            # The target LM head predicts t+1. MEDUSA head 1 predicts t+2.
            horizon = head_idx + 2
            if logits.numel() == 0:
                continue
            safe_logits = torch.nan_to_num(
                logits.detach().float(),
                nan=-1.0e9,
                posinf=1.0e9,
                neginf=-1.0e9,
            )
            k = min(int(top_m), int(safe_logits.shape[-1]))
            top_logits, top_ids = torch.topk(safe_logits, k=k, dim=-1)
            log_z = torch.logsumexp(safe_logits, dim=-1)

            # One bulk device transfer per head. The per-row loop below is CPU-only.
            top_ids_cpu = top_ids.to(device="cpu", dtype=torch.int32)
            top_logits_cpu = top_logits.clamp(min=-65504.0, max=65504.0).to(device="cpu", dtype=torch.float16)
            log_z_cpu = log_z.to(device="cpu", dtype=torch.float32).tolist()
            for row, seq_id in enumerate(sequence_cpu):
                anchor = int(anchor_cpu[row])
                self.add(
                    PredictionRecord(
                        sequence_id=seq_id,
                        anchor_pos=anchor,
                        target_pos=anchor + horizon,
                        horizon=horizon,
                        top_ids=top_ids_cpu[row],
                        top_logits=top_logits_cpu[row],
                        logsumexp=float(log_z_cpu[row]),
                        depth=max(0, anchor - int(initial_cpu[row])),
                    )
                )

    def pop_mature(self, sequence_id: int, target_pos: int) -> list[PredictionRecord]:
        return self._records.pop((int(sequence_id), int(target_pos)), [])

    def clear_sequence(self, sequence_id: int) -> None:
        sequence_id = int(sequence_id)
        for key in [key for key in self._records if key[0] == sequence_id]:
            self._records.pop(key, None)

    def __len__(self) -> int:
        return sum(len(records) for records in self._records.values())


@dataclass(slots=True)
class SparseFeedbackBatch:
    feedback: torch.Tensor
    has_feedback: torch.Tensor
    effective_mass: torch.Tensor
    record_true_probs: torch.Tensor
    record_tv: torch.Tensor
    record_gates: torch.Tensor
    target_top_ids: torch.Tensor
    target_top_logits: torch.Tensor
    target_logsumexp: torch.Tensor


class LMHeadFeedback:
    """Sparse target-distribution feedback in the LM-head hidden space."""

    def __init__(
        self,
        lm_head,
        *,
        target_topk: int = 32,
        union_cap: int = 96,
        tv_gate_low: float = 0.05,
        tv_gate_high: float = 0.20,
        horizon_weight_decay: float = 0.85,
        eps: float = 1e-8,
    ):
        self.lm_head = lm_head
        self.target_topk = max(1, int(target_topk))
        self.union_cap = max(self.target_topk, int(union_cap))
        self.tv_gate_low = float(tv_gate_low)
        self.tv_gate_high = max(float(tv_gate_high), self.tv_gate_low + 1e-6)
        self.horizon_weight_decay = float(horizon_weight_decay)
        self.eps = float(eps)

    @staticmethod
    def _prob_map(ids: torch.Tensor, logits: torch.Tensor, log_z: float) -> dict[int, float]:
        probs = torch.exp(logits.float() - float(log_z)).clamp_(min=0.0, max=1.0)
        return {int(token): float(prob) for token, prob in zip(ids.tolist(), probs.tolist())}

    @torch.no_grad()
    def compute_batch(
        self,
        record_groups: list[list[PredictionRecord]],
        target_logits: torch.Tensor,
        true_tokens: list[int],
    ) -> SparseFeedbackBatch:
        weight = self.lm_head.weight.detach()
        device = weight.device
        group_count = len(record_groups)
        if group_count == 0:
            empty = weight.new_zeros((0,), dtype=torch.float32)
            return SparseFeedbackBatch(
                feedback=weight.new_zeros((0, weight.shape[-1]), dtype=torch.float32),
                has_feedback=empty.bool(),
                effective_mass=empty,
                record_true_probs=empty,
                record_tv=empty,
                record_gates=empty,
                target_top_ids=torch.empty((0, 0), dtype=torch.int32),
                target_top_logits=torch.empty((0, 0), dtype=torch.float16),
                target_logsumexp=torch.empty((0,), dtype=torch.float32),
            )
        if int(target_logits.shape[0]) != group_count or len(true_tokens) != group_count:
            raise ValueError("Sparse feedback groups, target logits, and true tokens must align")

        safe_target = torch.nan_to_num(
            target_logits.detach().float(),
            nan=-1.0e9,
            posinf=1.0e9,
            neginf=-1.0e9,
        )
        k = min(self.target_topk, int(safe_target.shape[-1]))
        target_top_logits, target_top_ids = torch.topk(safe_target, k=k, dim=-1)
        target_log_z = torch.logsumexp(safe_target, dim=-1)
        target_top_ids = target_top_ids.to(device="cpu", dtype=torch.int32)
        target_top_logits = target_top_logits.clamp(min=-65504.0, max=65504.0).to(device="cpu", dtype=torch.float16)
        target_log_z_cpu = target_log_z.to(device="cpu", dtype=torch.float32).tolist()

        support_by_group: list[list[int]] = []
        coeff_by_group: list[list[float]] = []
        has_feedback: list[bool] = []
        effective_mass: list[float] = []
        record_true_probs: list[float] = []
        record_tv: list[float] = []
        record_gates: list[float] = []

        for group_idx, records in enumerate(record_groups):
            p_map = self._prob_map(
                target_top_ids[group_idx],
                target_top_logits[group_idx],
                float(target_log_z_cpu[group_idx]),
            )
            p_tail = max(0.0, 1.0 - sum(p_map.values()))
            q_maps = [self._prob_map(record.top_ids, record.top_logits, record.logsumexp) for record in records]
            support = set(p_map)
            for q_map in q_maps:
                support.update(q_map)

            aggregate = {token: 0.0 for token in support}
            total_weight = 0.0
            gate_sum = 0.0
            actual_token = int(true_tokens[group_idx])
            for record, q_map in zip(records, q_maps):
                q_tail = max(0.0, 1.0 - sum(q_map.values()))
                tv = 0.5 * (
                    sum(abs(p_map.get(token, 0.0) - q_map.get(token, 0.0)) for token in support)
                    + abs(p_tail - q_tail)
                )
                gate = min(1.0, max(0.0, (tv - self.tv_gate_low) / (self.tv_gate_high - self.tv_gate_low)))
                head_idx = max(0, int(record.horizon) - 2)
                reliability = self.horizon_weight_decay**head_idx
                contribution_weight = gate * reliability
                if contribution_weight > 0.0:
                    for token in support:
                        aggregate[token] += contribution_weight * (
                            p_map.get(token, 0.0) - q_map.get(token, 0.0)
                        )
                    total_weight += contribution_weight
                gate_sum += gate
                record_true_probs.append(max(self.eps, q_map.get(actual_token, 0.0)))
                record_tv.append(float(tv))
                record_gates.append(float(gate))

            effective_mass.append(gate_sum / max(len(records), 1))
            if total_weight <= 0.0:
                support_by_group.append([])
                coeff_by_group.append([])
                has_feedback.append(False)
                continue

            ranked = sorted(
                ((token, coeff / (total_weight + self.eps)) for token, coeff in aggregate.items()),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[: self.union_cap]
            support_by_group.append([int(token) for token, _ in ranked])
            coeff_by_group.append([float(coeff) for _, coeff in ranked])
            has_feedback.append(bool(ranked))

        flat_ids: list[int] = []
        flat_coeff: list[float] = []
        flat_groups: list[int] = []
        for group_idx, (support, coeffs) in enumerate(zip(support_by_group, coeff_by_group)):
            flat_ids.extend(support)
            flat_coeff.extend(coeffs)
            flat_groups.extend([group_idx] * len(support))

        feedback = torch.zeros((group_count, weight.shape[-1]), device=device, dtype=torch.float32)
        if flat_ids:
            ids = torch.as_tensor(flat_ids, dtype=torch.long, device=device)
            coeff = torch.as_tensor(flat_coeff, dtype=torch.float32, device=device)
            groups = torch.as_tensor(flat_groups, dtype=torch.long, device=device)
            rows = weight.index_select(0, ids).float()
            feedback.index_add_(0, groups, rows * coeff.unsqueeze(-1))

        return SparseFeedbackBatch(
            feedback=feedback,
            has_feedback=torch.as_tensor(has_feedback, dtype=torch.bool, device=device),
            effective_mass=torch.as_tensor(effective_mass, dtype=torch.float32, device=device),
            record_true_probs=torch.as_tensor(record_true_probs, dtype=torch.float32, device=device),
            record_tv=torch.as_tensor(record_tv, dtype=torch.float32, device=device),
            record_gates=torch.as_tensor(record_gates, dtype=torch.float32, device=device),
            target_top_ids=target_top_ids,
            target_top_logits=target_top_logits,
            target_logsumexp=torch.as_tensor(target_log_z_cpu, dtype=torch.float32),
        )


class ReflexStateManager:
    """Per-rollout hidden-dimensional fast state and running statistics."""

    def __init__(
        self,
        num_sequences: int,
        fast_state_dim: int,
        *,
        device: torch.device,
        half_life_tokens: float = 48.0,
        eta: float = 0.5,
        feedback_variance_beta: float = 0.99,
        feedback_rms_clip: float = 3.0,
        state_rms_clip: float = 2.0,
        numerical_reset_rms: float = 2.5,
        eps: float = 1e-6,
    ):
        self.fast_state_dim = int(fast_state_dim)
        self.rho = float(2.0 ** (-1.0 / max(float(half_life_tokens), 1e-6)))
        self.eta = float(eta)
        self.feedback_variance_beta = float(feedback_variance_beta)
        self.feedback_rms_clip = float(feedback_rms_clip)
        self.state_rms_clip = float(state_rms_clip)
        self.numerical_reset_rms = float(numerical_reset_rms)
        self.eps = float(eps)
        count = int(num_sequences)
        self.states = torch.zeros((count, self.fast_state_dim), device=device, dtype=torch.float32)
        self.feedback_variance = torch.zeros((count,), device=device, dtype=torch.float32)
        self.feedback_variance_initialized = torch.zeros((count,), device=device, dtype=torch.bool)
        self.effective_updates = torch.zeros((count,), device=device, dtype=torch.float32)
        self.numerical_reset_count = torch.zeros((), device=device, dtype=torch.long)

    def _ids(self, sequence_ids: Iterable[int]) -> torch.Tensor:
        return torch.as_tensor(list(sequence_ids), dtype=torch.long, device=self.states.device)

    def get(self, sequence_ids: Iterable[int]) -> torch.Tensor:
        ids = self._ids(sequence_ids)
        if ids.numel() == 0:
            return self.states.new_zeros((0, self.fast_state_dim))
        return self.states.index_select(0, ids)

    def get_effective_updates(self, sequence_ids: Iterable[int]) -> torch.Tensor:
        ids = self._ids(sequence_ids)
        if ids.numel() == 0:
            return self.effective_updates.new_zeros((0,))
        return self.effective_updates.index_select(0, ids)

    @torch.no_grad()
    def advance_token(
        self,
        sequence_ids: list[int],
        feedback: torch.Tensor,
        has_feedback: torch.Tensor,
        effective_mass: torch.Tensor,
    ) -> torch.Tensor:
        """Apply exactly one decay/update for one actual token per sequence."""

        ids = self._ids(sequence_ids)
        if ids.numel() == 0:
            return self.states.new_zeros((0,))
        if feedback.shape != (ids.numel(), self.fast_state_dim):
            raise ValueError("Feedback must be [num_sequences, hidden_size]")

        feedback = torch.nan_to_num(
            feedback.to(device=self.states.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        has_feedback = has_feedback.to(device=self.states.device, dtype=torch.bool)
        effective_mass = torch.nan_to_num(
            effective_mass.to(device=self.states.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min_(0.0)
        raw_ms = feedback.square().mean(dim=-1)
        valid = has_feedback & torch.isfinite(raw_ms) & raw_ms.gt(0.0)

        current = self.states.index_select(0, ids)
        updated = float(self.rho) * current
        feedback_rms = torch.zeros_like(raw_ms)
        valid_ids = ids[valid]
        old_var = self.feedback_variance.index_select(0, valid_ids)
        initialized = self.feedback_variance_initialized.index_select(0, valid_ids)
        new_var = torch.where(
            initialized,
            float(self.feedback_variance_beta) * old_var
            + (1.0 - float(self.feedback_variance_beta)) * raw_ms[valid],
            raw_ms[valid],
        ).clamp_min_(self.eps)
        normalized = feedback[valid] / torch.sqrt(new_var).unsqueeze(-1)
        normalized_rms = normalized.square().mean(dim=-1).sqrt()
        if self.feedback_rms_clip > 0.0:
            normalized = normalized * torch.clamp(
                float(self.feedback_rms_clip) / normalized_rms.clamp_min(self.eps),
                max=1.0,
            ).unsqueeze(-1)
            normalized_rms = normalized.square().mean(dim=-1).sqrt()
        updated[valid] += (1.0 - float(self.rho)) * float(self.eta) * normalized
        feedback_rms[valid] = normalized_rms
        self.feedback_variance.index_copy_(0, valid_ids, new_var)
        self.feedback_variance_initialized.index_fill_(0, valid_ids, True)

        preclip_rms = updated.square().mean(dim=-1).sqrt()
        bad = (~torch.isfinite(preclip_rms)) | preclip_rms.gt(float(self.numerical_reset_rms))
        updated[bad] = 0.0
        bad_ids = ids[bad]
        self.feedback_variance.index_fill_(0, bad_ids, 0.0)
        self.feedback_variance_initialized.index_fill_(0, bad_ids, False)
        self.effective_updates.index_fill_(0, bad_ids, 0.0)
        effective_mass = effective_mass.masked_fill(bad, 0.0)
        self.numerical_reset_count.add_(bad.sum())

        state_rms = updated.square().mean(dim=-1).sqrt()
        if self.state_rms_clip > 0.0:
            updated = updated * torch.clamp(
                float(self.state_rms_clip) / state_rms.clamp_min(self.eps),
                max=1.0,
            ).unsqueeze(-1)
        self.states.index_copy_(0, ids, updated)
        self.effective_updates.index_add_(0, ids, effective_mass)
        return feedback_rms

    def reset(self, sequence_ids: Iterable[int]) -> None:
        ids = self._ids(sequence_ids)
        if ids.numel() == 0:
            return
        self.states.index_fill_(0, ids, 0.0)
        self.feedback_variance.index_fill_(0, ids, 0.0)
        self.feedback_variance_initialized.index_fill_(0, ids, False)
        self.effective_updates.index_fill_(0, ids, 0.0)

    def norm_stats(self) -> dict[str, float]:
        if self.states.numel() == 0:
            return {
                "fast_state_rms_mean": 0.0,
                "fast_state_rms_p95": 0.0,
                "effective_feedback_updates_mean": 0.0,
                "numerical_reset_count": int(self.numerical_reset_count.detach().cpu()),
            }
        state_rms = torch.nan_to_num(self.states, nan=0.0, posinf=0.0, neginf=0.0).square().mean(dim=-1).sqrt()
        return {
            "fast_state_rms_mean": float(state_rms.mean().detach().cpu()),
            "fast_state_rms_p95": float(torch.quantile(state_rms, 0.95).detach().cpu()),
            "fast_state_norm_mean": float(state_rms.mean().detach().cpu()),
            "fast_state_norm_p95": float(torch.quantile(state_rms, 0.95).detach().cpu()),
            "effective_feedback_updates_mean": float(self.effective_updates.mean().detach().cpu()),
            "numerical_reset_count": int(self.numerical_reset_count.detach().cpu()),
        }


@dataclass(slots=True)
class ReflexAuxiliaryAnchor:
    sequence_id: int
    anchor_pos: int
    target_pos: int
    horizon: int
    initial_len: int
    hidden: torch.Tensor
    fast_state: torch.Tensor
    reflex_scale: float


class ReflexAuxiliaryRecordBuffer:
    """Bounded detached hidden-state records used only by rare head refreshes."""

    def __init__(self, *, max_records: int = 8192, hidden_dtype: torch.dtype = torch.float16):
        self.max_records = max(0, int(max_records))
        self.hidden_dtype = hidden_dtype
        self._pending: dict[tuple[int, int], list[ReflexAuxiliaryAnchor]] = {}
        self._records: deque[dict] = deque(maxlen=self.max_records or None)

    def add_anchor_predictions(
        self,
        *,
        sequence_ids: list[int],
        anchor_positions: torch.Tensor,
        initial_lengths: torch.Tensor,
        hidden_states: torch.Tensor,
        fast_states: torch.Tensor | None,
        max_horizon: int,
        reflex_scale: float = 1.0,
    ) -> None:
        if self.max_records <= 0 or max_horizon < 2 or not sequence_ids:
            return
        hidden_cpu = hidden_states.detach().to(device="cpu", dtype=self.hidden_dtype)
        if fast_states is None:
            fast_cpu = torch.zeros((len(sequence_ids), 0), dtype=self.hidden_dtype)
        else:
            fast_cpu = fast_states.detach().to(device="cpu", dtype=self.hidden_dtype)
        anchor_cpu = anchor_positions.detach().to(device="cpu", dtype=torch.long).tolist()
        initial_cpu = initial_lengths.detach().to(device="cpu", dtype=torch.long).tolist()
        for row, raw_seq_id in enumerate(sequence_ids):
            seq_id = int(raw_seq_id)
            anchor = int(anchor_cpu[row])
            for horizon in range(2, int(max_horizon) + 1):
                target_pos = anchor + horizon
                self._pending.setdefault((seq_id, target_pos), []).append(
                    ReflexAuxiliaryAnchor(
                        sequence_id=seq_id,
                        anchor_pos=anchor,
                        target_pos=target_pos,
                        horizon=horizon,
                        initial_len=int(initial_cpu[row]),
                        hidden=hidden_cpu[row],
                        fast_state=fast_cpu[row],
                        reflex_scale=float(reflex_scale),
                    )
                )

    def pop_mature(
        self,
        sequence_id: int,
        target_pos: int,
        generated_tokens: list[int],
        true_token: int,
        teacher: dict | None = None,
    ) -> None:
        if self.max_records <= 0:
            return
        anchors = self._pending.pop((int(sequence_id), int(target_pos)), [])
        for anchor in anchors:
            prev_tokens: list[int] = []
            valid = True
            for abs_pos in range(anchor.anchor_pos + 1, anchor.target_pos):
                rel = abs_pos - anchor.initial_len
                if rel < 0 or rel >= len(generated_tokens):
                    valid = False
                    break
                prev_tokens.append(int(generated_tokens[rel]))
            if not valid:
                continue
            proposal = None
            if teacher:
                proposal = next(
                    (
                        record
                        for record in teacher.get("proposal_records", [])
                        if int(record.anchor_pos) == int(anchor.anchor_pos)
                        and int(record.horizon) == int(anchor.horizon)
                    ),
                    None,
                )
            self._records.append(
                {
                    "hidden": anchor.hidden,
                    "fast_state": anchor.fast_state,
                    "label": int(true_token),
                    "horizon": int(anchor.horizon),
                    "prev_tokens": prev_tokens,
                    "reflex_scale": float(anchor.reflex_scale),
                    "target_top_ids": (
                        teacher["target_top_ids"].clone()
                        if teacher and teacher.get("target_top_ids") is not None
                        else torch.empty((0,), dtype=torch.int32)
                    ),
                    "target_top_logits": (
                        teacher["target_top_logits"].clone()
                        if teacher and teacher.get("target_top_logits") is not None
                        else torch.empty((0,), dtype=torch.float16)
                    ),
                    "target_logsumexp": float(teacher.get("target_logsumexp", 0.0)) if teacher else 0.0,
                    "old_top_ids": proposal.top_ids.clone() if proposal is not None else torch.empty((0,), dtype=torch.int32),
                    "old_top_logits": (
                        proposal.top_logits.clone() if proposal is not None else torch.empty((0,), dtype=torch.float16)
                    ),
                    "old_logsumexp": float(proposal.logsumexp) if proposal is not None else 0.0,
                    "has_sparse_teacher": bool(teacher is not None and proposal is not None),
                }
            )

    def clear_sequence(self, sequence_id: int) -> None:
        sequence_id = int(sequence_id)
        for key in [key for key in self._pending if key[0] == sequence_id]:
            self._pending.pop(key, None)

    def to_batch(self) -> dict[str, torch.Tensor]:
        if not self._records:
            return {}
        max_prev = max((len(item["prev_tokens"]) for item in self._records), default=0)
        max_target_topk = max((int(item["target_top_ids"].numel()) for item in self._records), default=0)
        max_old_topk = max((int(item["old_top_ids"].numel()) for item in self._records), default=0)
        hidden = torch.stack([item["hidden"] for item in self._records], dim=0).contiguous()
        fast_state = torch.stack([item["fast_state"] for item in self._records], dim=0).contiguous()
        labels = torch.tensor([item["label"] for item in self._records], dtype=torch.long)
        horizons = torch.tensor([item["horizon"] for item in self._records], dtype=torch.long)
        reflex_scale = torch.tensor([item["reflex_scale"] for item in self._records], dtype=torch.float32)
        prev_tokens = torch.full((len(self._records), max_prev), -1, dtype=torch.long)
        prev_lens = torch.zeros((len(self._records),), dtype=torch.long)
        target_top_ids = torch.full((len(self._records), max_target_topk), -1, dtype=torch.int32)
        target_top_logits = torch.zeros((len(self._records), max_target_topk), dtype=torch.float16)
        old_top_ids = torch.full((len(self._records), max_old_topk), -1, dtype=torch.int32)
        old_top_logits = torch.zeros((len(self._records), max_old_topk), dtype=torch.float16)
        target_logsumexp = torch.tensor([item["target_logsumexp"] for item in self._records], dtype=torch.float32)
        old_logsumexp = torch.tensor([item["old_logsumexp"] for item in self._records], dtype=torch.float32)
        has_sparse_teacher = torch.tensor([item["has_sparse_teacher"] for item in self._records], dtype=torch.bool)
        for idx, item in enumerate(self._records):
            tokens = item["prev_tokens"]
            prev_lens[idx] = len(tokens)
            if tokens:
                prev_tokens[idx, : len(tokens)] = torch.as_tensor(tokens, dtype=torch.long)
            target_count = int(item["target_top_ids"].numel())
            if target_count:
                target_top_ids[idx, :target_count] = item["target_top_ids"]
                target_top_logits[idx, :target_count] = item["target_top_logits"]
            old_count = int(item["old_top_ids"].numel())
            if old_count:
                old_top_ids[idx, :old_count] = item["old_top_ids"]
                old_top_logits[idx, :old_count] = item["old_top_logits"]
        return {
            "hidden": hidden,
            "fast_state": fast_state,
            "labels": labels,
            "horizons": horizons,
            "reflex_scale": reflex_scale,
            "prev_tokens": prev_tokens,
            "prev_lens": prev_lens,
            "target_top_ids": target_top_ids,
            "target_top_logits": target_top_logits,
            "target_logsumexp": target_logsumexp,
            "old_top_ids": old_top_ids,
            "old_top_logits": old_top_logits,
            "old_logsumexp": old_logsumexp,
            "has_sparse_teacher": has_sparse_teacher,
        }

    def __len__(self) -> int:
        return len(self._records)


class ReflexBatchStats:
    def __init__(self, num_heads: int):
        self.num_heads = int(num_heads)
        self.mature = [0 for _ in range(self.num_heads)]
        self.accepted = [0 for _ in range(self.num_heads)]
        self.ce_sum = [0.0 for _ in range(self.num_heads)]
        self.tv_sum = [0.0 for _ in range(self.num_heads)]
        self.gated = [0 for _ in range(self.num_heads)]
        self.depth_buckets: list[dict[str, dict[str, float]]] = [dict() for _ in range(self.num_heads)]
        self.feedback_rms: list[float] = []

    @staticmethod
    def _depth_bucket(depth: int) -> str:
        if depth < 128:
            return "0-128"
        if depth < 256:
            return "128-256"
        if depth < 512:
            return "256-512"
        if depth < 1024:
            return "512-1024"
        return "1024+"

    def add_records(
        self,
        records: list[PredictionRecord],
        true_probs: torch.Tensor,
        accepted_flags: list[bool],
        tv: torch.Tensor,
        gates: torch.Tensor,
    ) -> None:
        if not records:
            return
        true_probs_cpu = true_probs.detach().float().clamp_min(1e-8).cpu().tolist()
        tv_cpu = tv.detach().float().cpu().tolist()
        gates_cpu = gates.detach().float().cpu().tolist()
        for idx, record in enumerate(records):
            head_idx = int(record.horizon) - 2
            if head_idx < 0 or head_idx >= self.num_heads:
                continue
            accepted = int(bool(accepted_flags[idx]))
            ce = -math.log(max(float(true_probs_cpu[idx]), 1e-8))
            cur_tv = float(tv_cpu[idx])
            gate = float(gates_cpu[idx])
            self.mature[head_idx] += 1
            self.accepted[head_idx] += accepted
            self.ce_sum[head_idx] += ce
            self.tv_sum[head_idx] += cur_tv
            self.gated[head_idx] += int(gate > 0.0)
            bucket = self._depth_bucket(int(record.depth))
            stats = self.depth_buckets[head_idx].setdefault(
                bucket,
                {"mature": 0.0, "accepted": 0.0, "tv_sum": 0.0},
            )
            stats["mature"] += 1.0
            stats["accepted"] += float(accepted)
            stats["tv_sum"] += cur_tv

    def add_feedback_rms(self, values: torch.Tensor, has_feedback: torch.Tensor) -> None:
        if values.numel() == 0 or not bool(has_feedback.any().item()):
            return
        self.feedback_rms.extend(values[has_feedback].detach().float().cpu().tolist())

    def to_dict(self) -> dict:
        per_head: dict[str, dict] = {}
        total_mature = 0
        total_gated = 0
        for head_idx in range(self.num_heads):
            mature = self.mature[head_idx]
            accepted = self.accepted[head_idx]
            total_mature += mature
            total_gated += self.gated[head_idx]
            buckets = {}
            for name, raw in self.depth_buckets[head_idx].items():
                count = int(raw["mature"])
                buckets[name] = {
                    "mature": count,
                    "acceptance_rate": float(raw["accepted"] / max(count, 1)),
                    "sparse_tv": float(raw["tv_sum"] / max(count, 1)),
                }
            per_head[str(head_idx + 1)] = {
                "mature": mature,
                "accepted": accepted,
                "acceptance_rate": accepted / max(mature, 1),
                "rejection_rate": 1.0 - accepted / max(mature, 1) if mature else 0.0,
                "mature_ce": self.ce_sum[head_idx] / max(mature, 1),
                "sparse_tv": self.tv_sum[head_idx] / max(mature, 1),
                "nonzero_gate_fraction": self.gated[head_idx] / max(mature, 1),
                "depth_buckets": buckets,
            }
        feedback = torch.tensor(self.feedback_rms, dtype=torch.float32) if self.feedback_rms else None
        return {
            "num_reflex_updates": int(total_gated),
            "feedback_rms_mean": float(feedback.mean()) if feedback is not None else 0.0,
            "feedback_rms_p95": float(torch.quantile(feedback, 0.95)) if feedback is not None else 0.0,
            "feedback_norm_mean": float(feedback.mean()) if feedback is not None else 0.0,
            "feedback_norm_p95": float(torch.quantile(feedback, 0.95)) if feedback is not None else 0.0,
            "nonzero_gate_fraction": total_gated / max(total_mature, 1),
            "per_head": per_head,
        }
