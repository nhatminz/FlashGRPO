from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from flashgrpo_b200.models.qwen_flashgrpo_wrapper import autocast_dtype, unwrap_causal_lm


@dataclass
class OnlineMedusaConfig:
    medusa_lr: float = 5e-4
    medusa_weight_decay: float = 0.0
    medusa_train_every: int = 1
    medusa_update_steps_per_iter: int = 1
    medusa_microbatch_size: int = 1
    medusa_max_tokens_per_update: int = 8192
    medusa_loss_decay: float = 0.8
    medusa_loss_chunk_size: int = 128
    chain_loss_weight: float = 0.0
    chain_loss_max_depth: int = 3
    chain_bootstrap_from_medusa: bool = True
    grad_clip_norm: float = 1.0
    reflex_record_microbatch_size: int = 256
    reflex_correction_clip_norm: float = 1.0
    reflex_normalize_correction: bool = True
    rollback_nonfinite_update: bool = True
    refresh_distill_weight: float = 0.7
    refresh_hard_token_weight: float = 0.3
    refresh_proximal_weight: float = 0.1


class OnlineMedusaTrainer:
    def __init__(self, target_model, medusa_heads, optimizer, config: OnlineMedusaConfig):
        self.target_model = target_model
        self.medusa_heads = medusa_heads
        self.optimizer = optimizer
        self.config = config

    def _trainable_param_backup(self) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
        if not bool(self.config.rollback_nonfinite_update):
            return []
        return [(param, param.detach().clone()) for param in self.medusa_heads.parameters() if param.requires_grad]

    @staticmethod
    def _params_are_finite(module: torch.nn.Module) -> bool:
        with torch.no_grad():
            for param in module.parameters():
                if param.requires_grad and not bool(torch.isfinite(param.detach()).all().item()):
                    return False
        return True

    def _restore_backup(self, backup: list[tuple[torch.nn.Parameter, torch.Tensor]]) -> None:
        with torch.no_grad():
            for param, saved in backup:
                param.copy_(saved)
        self.optimizer.zero_grad(set_to_none=True)

    def update(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        head_weights: dict[int | str, float] | list[float] | tuple[float, ...] | None = None,
    ) -> dict:
        cfg = self.config
        if input_ids.numel() == 0:
            return {"medusa_loss": 0.0, "head_update_tokens": 0, "head_update_time": 0.0}
        start_time = time.time()
        device = next(self.medusa_heads.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        loss_mask = loss_mask.to(device) if loss_mask is not None else None
        max_total_tokens = int(cfg.medusa_max_tokens_per_update or 0)
        if max_total_tokens > 0 and int(attention_mask.sum().item()) > max_total_tokens:
            lengths = attention_mask.sum(dim=-1).long()
            # Randomized row subsampling keeps online head learning diverse
            # while bounding the expensive backbone hidden-state pass.
            perm = torch.randperm(input_ids.shape[0], device=device)
            selected = []
            running_tokens = 0
            for idx in perm.tolist():
                row_tokens = int(lengths[idx].item())
                if selected and running_tokens + row_tokens > max_total_tokens:
                    continue
                selected.append(idx)
                running_tokens += row_tokens
                if running_tokens >= max_total_tokens:
                    break
            if selected:
                keep = torch.tensor(sorted(selected), dtype=torch.long, device=device)
                input_ids = input_ids.index_select(0, keep)
                attention_mask = attention_mask.index_select(0, keep)
                loss_mask = loss_mask.index_select(0, keep) if loss_mask is not None else None
        base = unwrap_causal_lm(self.target_model)
        lm_head = base.lm_head
        embedding_layer = base.get_input_embeddings()
        self.medusa_heads.train()

        total_loss = 0.0
        total_tokens = 0
        per_head_sums: dict[str, float] = {}
        updates = 0
        max_rows = max(1, int(cfg.medusa_microbatch_size))
        rows = input_ids.shape[0]
        grad_denom = max(1, (rows + max_rows - 1) // max_rows)
        self.optimizer.zero_grad(set_to_none=True)

        for start in range(0, rows, max_rows):
            end = min(start + max_rows, rows)
            mb_input = input_ids[start:end]
            mb_mask = attention_mask[start:end]
            mb_loss_mask = loss_mask[start:end] if loss_mask is not None else None
            valid_tokens = int(mb_mask.sum().item())
            if cfg.medusa_max_tokens_per_update and valid_tokens > cfg.medusa_max_tokens_per_update:
                keep_len = max(2, int(cfg.medusa_max_tokens_per_update // max(1, end - start)))
                mb_input = mb_input[:, -keep_len:]
                mb_mask = mb_mask[:, -keep_len:]
                mb_loss_mask = mb_loss_mask[:, -keep_len:] if mb_loss_mask is not None else None
            with torch.no_grad():
                device_type = "cuda" if device.type == "cuda" else device.type
                with torch.amp.autocast(device_type, dtype=autocast_dtype(base), enabled=(device.type == "cuda")):
                    outputs = base.model(
                        input_ids=mb_input,
                        attention_mask=mb_mask,
                        use_cache=False,
                        return_dict=True,
                    )
                    hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            loss, stats = self.medusa_heads.compute_loss(
                hidden_states.detach(),
                mb_input,
                mb_mask,
                lm_head=lm_head,
                loss_mask=mb_loss_mask,
                chunk_size=cfg.medusa_loss_chunk_size,
                chain_loss_weight=cfg.chain_loss_weight,
                chain_max_depth=cfg.chain_loss_max_depth,
                chain_bootstrap_from_medusa=cfg.chain_bootstrap_from_medusa,
                embedding_layer=embedding_layer,
                head_weights=head_weights,
            )
            if not torch.isfinite(loss):
                continue
            if not loss.requires_grad:
                continue
            (loss / grad_denom).backward()
            total_loss += float(loss.detach().cpu())
            total_tokens += int(mb_mask.sum().item())
            updates += 1
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    per_head_sums[key] = per_head_sums.get(key, 0.0) + float(value)

        reverted_nonfinite = False
        if updates:
            backup = self._trainable_param_backup()
            torch.nn.utils.clip_grad_norm_(self.medusa_heads.parameters(), cfg.grad_clip_norm)
            self.optimizer.step()
            if backup and not self._params_are_finite(self.medusa_heads):
                self._restore_backup(backup)
                reverted_nonfinite = True
        self.optimizer.zero_grad(set_to_none=True)
        elapsed = time.time() - start_time
        out = {
            "medusa_loss": total_loss / max(updates, 1),
            "head_update_tokens": int(total_tokens),
            "head_update_time": elapsed,
            "head_update_tokens_per_sec": total_tokens / max(elapsed, 1e-9),
            "head_update_steps": int(updates),
            "head_update_reverted_nonfinite": bool(reverted_nonfinite),
        }
        if head_weights is not None:
            for idx in range(len(self.medusa_heads.heads)):
                if isinstance(head_weights, (list, tuple)):
                    value = float(head_weights[idx]) if idx < len(head_weights) else 0.0
                else:
                    value = float(
                        head_weights.get(str(idx + 1))
                        or head_weights.get(idx + 1)
                        or head_weights.get(str(idx))
                        or head_weights.get(idx)
                        or 0.0
                    )
                out[f"aux_weight_head_{idx + 1}"] = value
        for key, value in per_head_sums.items():
            out[key] = value / max(updates, 1)
        return out

    @staticmethod
    def _head_weight(head_weights, head_idx: int) -> float:
        if head_weights is None:
            return 1.0
        if isinstance(head_weights, (list, tuple)):
            return float(head_weights[head_idx]) if head_idx < len(head_weights) else 0.0
        return float(
            head_weights.get(str(head_idx + 1))
            or head_weights.get(head_idx + 1)
            or head_weights.get(str(head_idx))
            or head_weights.get(head_idx)
            or 0.0
        )

    @staticmethod
    def _lm_head_logits(hidden: torch.Tensor, lm_head) -> torch.Tensor:
        weight = lm_head.weight.detach()
        bias = getattr(lm_head, "bias", None)
        bias = bias.detach() if bias is not None else None
        return F.linear(hidden.to(dtype=weight.dtype), weight, bias)

    @staticmethod
    def _sparse_cross_entropy_with_tail(
        new_logits: torch.Tensor,
        support_ids: torch.Tensor,
        support_logits: torch.Tensor,
        support_logsumexp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = support_ids.ge(0)
        safe_ids = support_ids.clamp_min(0)
        new_log_z = torch.logsumexp(new_logits.float(), dim=-1, keepdim=True)
        selected_new_logp = torch.gather(new_logits.float(), -1, safe_ids) - new_log_z
        selected_new_prob = torch.exp(selected_new_logp).masked_fill(~valid, 0.0)
        teacher_prob = torch.exp(
            support_logits.float() - support_logsumexp.float().unsqueeze(-1)
        ).masked_fill(~valid, 0.0)
        teacher_tail = (1.0 - teacher_prob.sum(dim=-1)).clamp(min=1e-8, max=1.0)
        new_tail = (1.0 - selected_new_prob.sum(dim=-1)).clamp(min=1e-8, max=1.0)
        distill = -(
            (teacher_prob * selected_new_logp.masked_fill(~valid, 0.0)).sum(dim=-1)
            + teacher_tail * torch.log(new_tail)
        )
        tv = 0.5 * (
            (teacher_prob - selected_new_prob).abs().sum(dim=-1)
            + (teacher_tail - new_tail).abs()
        )
        return distill, tv

    @staticmethod
    def _sparse_proximal_kl_with_tail(
        new_logits: torch.Tensor,
        old_ids: torch.Tensor,
        old_logits: torch.Tensor,
        old_logsumexp: torch.Tensor,
    ) -> torch.Tensor:
        valid = old_ids.ge(0)
        safe_ids = old_ids.clamp_min(0)
        new_log_z = torch.logsumexp(new_logits.float(), dim=-1, keepdim=True)
        new_logp = torch.gather(new_logits.float(), -1, safe_ids) - new_log_z
        new_prob = torch.exp(new_logp).masked_fill(~valid, 0.0)
        old_logp = old_logits.float() - old_logsumexp.float().unsqueeze(-1)
        old_prob = torch.exp(old_logp).masked_fill(~valid, 0.0)
        old_tail = (1.0 - old_prob.sum(dim=-1)).clamp(min=1e-8, max=1.0)
        new_tail = (1.0 - new_prob.sum(dim=-1)).clamp(min=1e-8, max=1.0)
        return (
            old_prob * (old_logp.masked_fill(~valid, 0.0) - new_logp.masked_fill(~valid, 0.0))
        ).sum(dim=-1) + old_tail * (torch.log(old_tail) - torch.log(new_tail))

    def _project_with_reflex(
        self,
        hidden: torch.Tensor,
        fast_state: torch.Tensor,
        head_idx: int,
        *,
        update_fast_state_injections: bool,
        scale: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        head = self.medusa_heads.heads[head_idx]
        projected = head.project_hidden(hidden)
        if getattr(self.medusa_heads, "reflex_fast_state_dim", 0) <= 0 or fast_state.numel() == 0:
            return projected
        up = self.medusa_heads.reflex_up[head_idx]
        fast = fast_state.to(device=up.weight.device, dtype=up.weight.dtype)
        if update_fast_state_injections:
            delta = up(fast)
        else:
            delta = F.linear(fast, up.weight.detach(), None)
        if bool(self.config.reflex_normalize_correction):
            delta_float = torch.nan_to_num(delta.float(), nan=0.0, posinf=0.0, neginf=0.0)
            rms = delta_float.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
            delta = (delta_float / rms).to(dtype=delta.dtype)
        if float(self.config.reflex_correction_clip_norm) > 0:
            norm = torch.nan_to_num(delta.float(), nan=0.0, posinf=0.0, neginf=0.0).norm(dim=-1, keepdim=True).clamp_min(1e-6)
            delta = delta * torch.clamp(float(self.config.reflex_correction_clip_norm) / norm, max=1.0).to(dtype=delta.dtype)
        if torch.is_tensor(scale):
            delta = delta * scale.to(device=delta.device, dtype=delta.dtype).view(-1, 1)
        elif float(scale) != 1.0:
            delta = delta * float(scale)
        return projected + delta.to(device=projected.device, dtype=projected.dtype)

    def _logits_from_hidden(self, hidden: torch.Tensor, head_idx: int, lm_head) -> torch.Tensor:
        output = self.medusa_heads.heads[head_idx].output
        if output is not None:
            return output(hidden)
        return self._lm_head_logits(hidden, lm_head)

    def _chain_logits_from_state(self, state: torch.Tensor, lm_head) -> torch.Tensor:
        return self._lm_head_logits(state, lm_head)

    def update_reflex_records(
        self,
        records: dict,
        *,
        head_weights: dict[int | str, float] | list[float] | tuple[float, ...] | None = None,
        update_fast_state_injections: bool = False,
    ) -> dict:
        cfg = self.config
        hidden_cpu = records.get("hidden") if records else None
        if hidden_cpu is None or int(hidden_cpu.shape[0]) == 0:
            return {"medusa_loss": 0.0, "head_update_tokens": 0, "head_update_time": 0.0, "head_update_steps": 0}
        start_time = time.time()
        device = next(self.medusa_heads.parameters()).device
        hidden_cpu = hidden_cpu.detach()
        total_records = int(hidden_cpu.shape[0])
        max_records = int(cfg.medusa_max_tokens_per_update or 0)
        if max_records > 0 and total_records > max_records:
            keep = torch.randperm(total_records)[:max_records]
            records = {
                key: value.index_select(0, keep) if torch.is_tensor(value) and value.shape[:1] == (total_records,) else value
                for key, value in records.items()
            }
            total_records = max_records

        base = unwrap_causal_lm(self.target_model)
        lm_head = base.lm_head
        embedding_layer = base.get_input_embeddings()
        self.medusa_heads.train()
        self.optimizer.zero_grad(set_to_none=True)

        micro = max(1, int(cfg.reflex_record_microbatch_size or 256))
        grad_denom = max(1, (total_records + micro - 1) // micro)
        total_loss = 0.0
        total_tokens = 0
        updates = 0
        stat_sums: dict[str, float] = {}

        for start in range(0, total_records, micro):
            end = min(start + micro, total_records)
            hidden = records["hidden"][start:end].to(device=device, dtype=next(self.medusa_heads.parameters()).dtype)
            fast = records["fast_state"][start:end].to(device=device, dtype=next(self.medusa_heads.parameters()).dtype)
            labels = records["labels"][start:end].to(device=device).long()
            horizons = records["horizons"][start:end].to(device=device).long()
            scales = records.get("reflex_scale")
            scales = scales[start:end].to(device=device, dtype=torch.float32) if torch.is_tensor(scales) else torch.ones((end - start,), device=device)
            prev_tokens = records["prev_tokens"][start:end].to(device=device).long()
            has_sparse_teacher = records.get("has_sparse_teacher")
            has_sparse_teacher = (
                has_sparse_teacher[start:end].to(device=device).bool()
                if torch.is_tensor(has_sparse_teacher)
                else torch.zeros((end - start,), device=device, dtype=torch.bool)
            )
            target_top_ids = records.get("target_top_ids")
            target_top_logits = records.get("target_top_logits")
            target_logsumexp = records.get("target_logsumexp")
            old_top_ids = records.get("old_top_ids")
            old_top_logits = records.get("old_top_logits")
            old_logsumexp = records.get("old_logsumexp")
            losses = []
            parallel_losses = []
            chain_losses = []
            weight_sum = 0.0

            for head_idx in range(len(self.medusa_heads.heads)):
                horizon = head_idx + 2
                mask = horizons.eq(horizon)
                if not bool(mask.any().item()):
                    continue
                aux_weight = self._head_weight(head_weights, head_idx)
                if aux_weight <= 0.0:
                    continue
                decay = float(cfg.medusa_loss_decay ** head_idx)
                h = hidden[mask]
                f = fast[mask]
                s = scales[mask]
                y = labels[mask]
                projected = self._project_with_reflex(
                    h,
                    f,
                    head_idx,
                    update_fast_state_injections=update_fast_state_injections,
                    scale=s,
                )
                logits = self._logits_from_hidden(projected, head_idx, lm_head).float()
                hard_loss = F.cross_entropy(logits, y, reduction="none")
                selected_teacher = has_sparse_teacher[mask]
                per_record_loss = hard_loss
                if (
                    bool(selected_teacher.any().item())
                    and torch.is_tensor(target_top_ids)
                    and torch.is_tensor(target_top_logits)
                    and torch.is_tensor(target_logsumexp)
                ):
                    target_ids = target_top_ids[start:end].to(device=device).long()[mask]
                    target_values = target_top_logits[start:end].to(device=device).float()[mask]
                    target_lse = target_logsumexp[start:end].to(device=device).float()[mask]
                    distill, _ = self._sparse_cross_entropy_with_tail(
                        logits,
                        target_ids,
                        target_values,
                        target_lse,
                    )
                    proximal = torch.zeros_like(distill)
                    if (
                        float(cfg.refresh_proximal_weight) > 0.0
                        and torch.is_tensor(old_top_ids)
                        and torch.is_tensor(old_top_logits)
                        and torch.is_tensor(old_logsumexp)
                    ):
                        proximal = self._sparse_proximal_kl_with_tail(
                            logits,
                            old_top_ids[start:end].to(device=device).long()[mask],
                            old_top_logits[start:end].to(device=device).float()[mask],
                            old_logsumexp[start:end].to(device=device).float()[mask],
                        )
                    sparse_loss = (
                        float(cfg.refresh_distill_weight) * distill
                        + float(cfg.refresh_hard_token_weight) * hard_loss
                        + float(cfg.refresh_proximal_weight) * proximal
                    )
                    per_record_loss = torch.where(selected_teacher, sparse_loss, hard_loss)
                    stat_sums[f"head_{head_idx + 1}_distill"] = stat_sums.get(
                        f"head_{head_idx + 1}_distill", 0.0
                    ) + float(distill[selected_teacher].mean().detach().cpu())
                loss = per_record_loss.mean()
                weighted = float(aux_weight * decay) * loss
                losses.append(weighted)
                parallel_losses.append(weighted)
                weight_sum += float(aux_weight)
                stat_sums[f"head_{head_idx + 1}"] = stat_sums.get(f"head_{head_idx + 1}", 0.0) + float(loss.detach().cpu())
                stat_sums[f"head_{head_idx + 1}_tokens"] = stat_sums.get(f"head_{head_idx + 1}_tokens", 0.0) + int(mask.sum().item())
                stat_sums[f"head_{head_idx + 1}_weight"] = float(aux_weight)

            chain_weight = float(cfg.chain_loss_weight or 0.0)
            if chain_weight > 0.0 and prev_tokens.numel() > 0:
                max_depth = min(int(cfg.chain_loss_max_depth or len(self.medusa_heads.heads)), len(self.medusa_heads.heads))
                for depth_idx in range(max_depth):
                    horizon = depth_idx + 2
                    mask = horizons.eq(horizon)
                    if not bool(mask.any().item()):
                        continue
                    aux_weight = self._head_weight(head_weights, depth_idx)
                    if aux_weight <= 0.0:
                        continue
                    h = hidden[mask]
                    f = fast[mask]
                    s = scales[mask]
                    y = labels[mask]
                    prev = prev_tokens[mask]
                    if cfg.chain_bootstrap_from_medusa and len(self.medusa_heads.heads) > 0:
                        state = self.medusa_heads.heads[0].project_hidden(h)
                        first_prev_index = 1
                    else:
                        state = h
                        first_prev_index = 0
                    for prev_index in range(first_prev_index, max(horizon - 1, first_prev_index)):
                        if prev_index >= prev.shape[1]:
                            break
                        tokens = prev[:, prev_index]
                        valid = tokens.ge(0)
                        if not bool(valid.all().item()):
                            state = state[valid]
                            f = f[valid]
                            s = s[valid]
                            y = y[valid]
                            tokens = tokens[valid]
                            if state.numel() == 0:
                                break
                        state = self.medusa_heads.chain_next_state(state, tokens, embedding_layer)
                    if state.numel() == 0:
                        continue
                    if update_fast_state_injections:
                        delta = self.medusa_heads.reflex_up[depth_idx](
                            f.to(dtype=self.medusa_heads.reflex_up[depth_idx].weight.dtype)
                        )
                    else:
                        delta = F.linear(
                            f.to(dtype=self.medusa_heads.reflex_up[depth_idx].weight.dtype),
                            self.medusa_heads.reflex_up[depth_idx].weight.detach(),
                            None,
                        )
                    if bool(cfg.reflex_normalize_correction):
                        delta_float = torch.nan_to_num(delta.float(), nan=0.0, posinf=0.0, neginf=0.0)
                        rms = delta_float.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
                        delta = (delta_float / rms).to(dtype=delta.dtype)
                    if float(cfg.reflex_correction_clip_norm) > 0:
                        norm = torch.nan_to_num(delta.float(), nan=0.0, posinf=0.0, neginf=0.0).norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        delta = delta * torch.clamp(float(cfg.reflex_correction_clip_norm) / norm, max=1.0).to(dtype=delta.dtype)
                    delta = delta * s.to(device=delta.device, dtype=delta.dtype).view(-1, 1)
                    state = state + delta.to(dtype=state.dtype)
                    logits = self._chain_logits_from_state(state, lm_head).float()
                    loss = F.cross_entropy(logits, y)
                    weighted = float(aux_weight * (cfg.medusa_loss_decay ** depth_idx)) * loss
                    chain_losses.append(weighted)
                    losses.append(chain_weight * weighted)

            if not losses:
                continue
            loss = torch.stack(losses).mean() if weight_sum <= 0 else torch.stack(losses).sum() / max(weight_sum, 1e-6)
            if not torch.isfinite(loss) or not loss.requires_grad:
                continue
            (loss / grad_denom).backward()
            total_loss += float(loss.detach().cpu())
            total_tokens += int(end - start)
            updates += 1
            if parallel_losses:
                stat_sums["parallel_medusa_loss"] = stat_sums.get("parallel_medusa_loss", 0.0) + float(torch.stack(parallel_losses).mean().detach().cpu())
            if chain_losses:
                stat_sums["chain_loss"] = stat_sums.get("chain_loss", 0.0) + float(torch.stack(chain_losses).mean().detach().cpu())
                stat_sums["chain_loss_weight"] = chain_weight

        reverted_nonfinite = False
        if updates:
            backup = self._trainable_param_backup()
            torch.nn.utils.clip_grad_norm_(self.medusa_heads.parameters(), cfg.grad_clip_norm)
            self.optimizer.step()
            if backup and not self._params_are_finite(self.medusa_heads):
                self._restore_backup(backup)
                reverted_nonfinite = True
        self.optimizer.zero_grad(set_to_none=True)
        elapsed = time.time() - start_time
        out = {
            "medusa_loss": total_loss / max(updates, 1),
            "head_update_tokens": int(total_tokens),
            "head_update_time": elapsed,
            "head_update_tokens_per_sec": total_tokens / max(elapsed, 1e-9),
            "head_update_steps": int(updates),
            "reflex_cached_records": int(total_records),
            "reflex_cached_update": True,
            "head_update_reverted_nonfinite": bool(reverted_nonfinite),
        }
        for key, value in stat_sums.items():
            if key.endswith("_tokens") or key.endswith("_weight") or key == "chain_loss_weight":
                out[key] = value
            else:
                out[key] = value / max(updates, 1)
        return out

    @torch.no_grad()
    def evaluate_reflex_records(
        self,
        records: dict,
        *,
        head_weights: dict[int | str, float] | list[float] | tuple[float, ...] | None = None,
    ) -> dict[str, float]:
        hidden_cpu = records.get("hidden") if records else None
        if hidden_cpu is None or int(hidden_cpu.shape[0]) == 0:
            return {
                "validation_ce": float("inf"),
                "validation_records": 0,
                "validation_sparse_tv": float("inf"),
                "validation_sparse_records": 0,
            }
        device = next(self.medusa_heads.parameters()).device
        base = unwrap_causal_lm(self.target_model)
        lm_head = base.lm_head
        micro = max(1, int(self.config.reflex_record_microbatch_size or 64))
        loss_sum = 0.0
        count = 0
        tv_sum = 0.0
        tv_count = 0
        was_training = self.medusa_heads.training
        self.medusa_heads.eval()
        for start in range(0, int(hidden_cpu.shape[0]), micro):
            end = min(start + micro, int(hidden_cpu.shape[0]))
            hidden = records["hidden"][start:end].to(
                device=device,
                dtype=next(self.medusa_heads.parameters()).dtype,
            )
            labels = records["labels"][start:end].to(device=device).long()
            horizons = records["horizons"][start:end].to(device=device).long()
            has_sparse_teacher = records.get("has_sparse_teacher")
            has_sparse_teacher = (
                has_sparse_teacher[start:end].to(device=device).bool()
                if torch.is_tensor(has_sparse_teacher)
                else torch.zeros((end - start,), device=device, dtype=torch.bool)
            )
            for head_idx, head in enumerate(self.medusa_heads.heads):
                mask = horizons.eq(head_idx + 2)
                if not bool(mask.any().item()) or self._head_weight(head_weights, head_idx) <= 0.0:
                    continue
                logits = self._logits_from_hidden(head.project_hidden(hidden[mask]), head_idx, lm_head).float()
                loss_sum += float(F.cross_entropy(logits, labels[mask], reduction="sum").cpu())
                count += int(mask.sum().item())
                selected_teacher = has_sparse_teacher[mask]
                if bool(selected_teacher.any().item()) and torch.is_tensor(records.get("target_top_ids")):
                    _, tv = self._sparse_cross_entropy_with_tail(
                        logits,
                        records["target_top_ids"][start:end].to(device=device).long()[mask],
                        records["target_top_logits"][start:end].to(device=device).float()[mask],
                        records["target_logsumexp"][start:end].to(device=device).float()[mask],
                    )
                    tv_sum += float(tv[selected_teacher].sum().cpu())
                    tv_count += int(selected_teacher.sum().item())
        self.medusa_heads.train(was_training)
        return {
            "validation_ce": loss_sum / max(count, 1),
            "validation_records": int(count),
            "validation_sparse_tv": tv_sum / max(tv_count, 1) if tv_count else float("inf"),
            "validation_sparse_records": int(tv_count),
        }
