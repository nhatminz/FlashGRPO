from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class PredictionRecord:
    sequence_id: int
    anchor_pos: int
    target_pos: int
    horizon: int
    top_ids: torch.Tensor
    top_probs: torch.Tensor
    entropy: float = 0.0


class PredictionBuffer:
    def __init__(self):
        self._records: dict[tuple[int, int], list[PredictionRecord]] = {}

    def add(self, record: PredictionRecord) -> None:
        key = (int(record.sequence_id), int(record.target_pos))
        self._records.setdefault(key, []).append(record)

    def add_from_logits(
        self,
        *,
        sequence_ids: list[int],
        anchor_positions: torch.Tensor,
        logits_by_horizon: list[torch.Tensor],
        top_m: int,
    ) -> None:
        if not logits_by_horizon or top_m <= 0:
            return
        anchor_cpu = anchor_positions.detach().cpu().tolist()
        for head_idx, logits in enumerate(logits_by_horizon):
            horizon = head_idx + 2
            if logits.numel() == 0:
                continue
            k = min(int(top_m), logits.shape[-1])
            values, indices = torch.topk(logits.detach().float(), k=k, dim=-1)
            probs = torch.softmax(values, dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            for row, seq_id in enumerate(sequence_ids):
                self.add(
                    PredictionRecord(
                        sequence_id=int(seq_id),
                        anchor_pos=int(anchor_cpu[row]),
                        target_pos=int(anchor_cpu[row]) + horizon,
                        horizon=horizon,
                        top_ids=indices[row].detach(),
                        top_probs=probs[row].detach(),
                        entropy=float(entropy[row].detach().cpu()),
                    )
                )

    def pop_mature(self, sequence_id: int, target_pos: int) -> list[PredictionRecord]:
        return self._records.pop((int(sequence_id), int(target_pos)), [])

    def clear_sequence(self, sequence_id: int) -> None:
        sequence_id = int(sequence_id)
        for key in [key for key in self._records if key[0] == sequence_id]:
            self._records.pop(key, None)

    def __len__(self) -> int:
        return sum(len(v) for v in self._records.values())


class ReflexStateManager:
    def __init__(
        self,
        num_sequences: int,
        fast_state_dim: int,
        *,
        device: torch.device,
        beta: float = 0.95,
        eta: float = 0.1,
        max_norm: float = 8.0,
    ):
        self.fast_state_dim = int(fast_state_dim)
        self.beta = float(beta)
        self.eta = float(eta)
        self.max_norm = float(max_norm)
        self.states = torch.zeros((int(num_sequences), self.fast_state_dim), device=device, dtype=torch.float32)

    def get(self, sequence_ids: Iterable[int]) -> torch.Tensor:
        ids = torch.tensor(list(sequence_ids), dtype=torch.long, device=self.states.device)
        if ids.numel() == 0:
            return self.states.new_zeros((0, self.fast_state_dim))
        return self.states.index_select(0, ids)

    def update(self, sequence_ids: list[int], feedback: torch.Tensor) -> torch.Tensor:
        if not sequence_ids or feedback.numel() == 0:
            return feedback.new_zeros((0,))
        feedback = feedback.to(device=self.states.device, dtype=torch.float32)
        feedback_norm = feedback.norm(dim=-1)
        for idx, seq_id in enumerate(sequence_ids):
            current = self.states[int(seq_id)]
            updated = self.beta * current + self.eta * feedback[idx]
            if self.max_norm > 0:
                norm = updated.norm().clamp_min(1e-6)
                updated = updated * torch.clamp(torch.tensor(float(self.max_norm), device=updated.device) / norm, max=1.0)
            self.states[int(seq_id)] = updated
        return feedback_norm

    def reset(self, sequence_ids: Iterable[int]) -> None:
        ids_list = list(sequence_ids)
        if not ids_list:
            return
        ids = torch.tensor(ids_list, dtype=torch.long, device=self.states.device)
        self.states.index_fill_(0, ids, 0.0)

    def norm_stats(self) -> dict[str, float]:
        if self.states.numel() == 0:
            return {"fast_state_norm_mean": 0.0, "fast_state_norm_p95": 0.0}
        norms = self.states.norm(dim=-1)
        return {
            "fast_state_norm_mean": float(norms.mean().detach().cpu()),
            "fast_state_norm_p95": float(torch.quantile(norms.detach().float(), 0.95).cpu()),
        }


class LMHeadFeedback:
    def __init__(self, lm_head, *, eps: float = 1e-8, max_hidden_norm: float = 0.0):
        self.lm_head = lm_head
        self.eps = float(eps)
        self.max_hidden_norm = float(max_hidden_norm)

    def compute_batch(self, records: list[PredictionRecord], true_tokens: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        if not records:
            weight = self.lm_head.weight
            return weight.new_zeros((0, weight.shape[-1])), weight.new_zeros((0,))
        weight = self.lm_head.weight.detach()
        device = weight.device
        top_ids = torch.stack([record.top_ids.to(device=device) for record in records], dim=0).long()
        top_probs = torch.stack([record.top_probs.to(device=device) for record in records], dim=0).to(dtype=torch.float32)
        true = torch.tensor(true_tokens, dtype=torch.long, device=device)
        top_weight = weight.index_select(0, top_ids.reshape(-1)).reshape(top_ids.shape[0], top_ids.shape[1], -1).float()
        expected = (top_probs.unsqueeze(-1) * top_weight).sum(dim=1)
        true_weight = weight.index_select(0, true).float()
        hidden_feedback = true_weight - expected
        if self.max_hidden_norm > 0:
            norm = hidden_feedback.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            hidden_feedback = hidden_feedback * torch.clamp(float(self.max_hidden_norm) / norm, max=1.0)
        true_match = top_ids.eq(true.unsqueeze(-1))
        true_probs = torch.where(true_match, top_probs, torch.zeros_like(top_probs)).sum(dim=-1).clamp_min(self.eps)
        return hidden_feedback, true_probs


class ReflexBatchStats:
    def __init__(self, num_heads: int):
        self.num_heads = int(num_heads)
        self.mature = [0 for _ in range(self.num_heads)]
        self.accepted = [0 for _ in range(self.num_heads)]
        self.ce_sum = [0.0 for _ in range(self.num_heads)]
        self.feedback_norms: list[float] = []
        self.update_count = 0

    def add_records(
        self,
        records: list[PredictionRecord],
        true_probs: torch.Tensor,
        accepted_flags: list[bool],
        feedback_norms: torch.Tensor,
    ) -> None:
        if not records:
            return
        ce = (-true_probs.detach().float().clamp_min(1e-8).log()).cpu().tolist()
        norms = feedback_norms.detach().float().cpu().tolist()
        for idx, record in enumerate(records):
            head_idx = int(record.horizon) - 2
            if head_idx < 0 or head_idx >= self.num_heads:
                continue
            self.mature[head_idx] += 1
            self.accepted[head_idx] += int(bool(accepted_flags[idx]))
            self.ce_sum[head_idx] += float(ce[idx])
            self.feedback_norms.append(float(norms[idx]))
            self.update_count += 1

    def to_dict(self) -> dict:
        per_head = {}
        for head_idx in range(self.num_heads):
            mature = self.mature[head_idx]
            accepted = self.accepted[head_idx]
            ce = self.ce_sum[head_idx] / max(mature, 1)
            acc = accepted / max(mature, 1)
            per_head[str(head_idx + 1)] = {
                "mature": mature,
                "accepted": accepted,
                "acceptance_rate": acc,
                "rejection_rate": 1.0 - acc if mature else 0.0,
                "mature_ce": ce,
            }
        if self.feedback_norms:
            feedback = torch.tensor(self.feedback_norms, dtype=torch.float32)
            feedback_mean = float(feedback.mean())
            feedback_p95 = float(torch.quantile(feedback, 0.95))
        else:
            feedback_mean = 0.0
            feedback_p95 = 0.0
        return {
            "num_reflex_updates": int(self.update_count),
            "feedback_norm_mean": feedback_mean,
            "feedback_norm_p95": feedback_p95,
            "per_head": per_head,
        }
