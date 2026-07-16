from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import torch


@dataclass
class ReliabilityDecision:
    evaluated: bool = False
    triggered: bool = False
    triggered_heads: list[int] = field(default_factory=list)
    drift_scores: dict[str, float] = field(default_factory=dict)
    head_metrics: dict[str, dict] = field(default_factory=dict)
    reason: str = ""


class ReliabilityTracker:
    def __init__(
        self,
        *,
        mode: str = "reliability_triggered",
        calibration_iterations: int = 20,
        check_interval: int = 20,
        min_records_per_head: int = 1024,
        trigger_z_high: float = 2.5,
        acceptance_drop_min: float = 0.05,
        patience: int = 2,
        cooldown_iterations: int = 50,
        max_heads_per_event: int = 2,
        mad_floor: float = 0.01,
    ):
        self.mode = str(mode)
        self.calibration_iterations = max(0, int(calibration_iterations))
        self.check_interval = max(1, int(check_interval))
        self.min_records_per_head = max(1, int(min_records_per_head))
        self.trigger_z_high = float(trigger_z_high)
        self.acceptance_drop_min = float(acceptance_drop_min)
        self.patience = max(1, int(patience))
        self.cooldown_iterations = max(0, int(cooldown_iterations))
        self.max_heads_per_event = max(1, int(max_heads_per_event))
        self.mad_floor = max(float(mad_floor), 1e-6)
        self.current: dict[str, dict] = {}
        self.interval_stats: dict[str, dict] = {}
        self.baseline_tv_samples: dict[str, list[float]] = {}
        self.baseline_accept_samples: dict[str, list[float]] = {}
        self.patience_counters: dict[str, int] = {}
        self.cooldown_until: dict[str, int] = {}
        self.last_metrics: dict[str, dict] = {}
        self.last_decision = ReliabilityDecision()

    @staticmethod
    def _empty_head() -> dict:
        return {
            "mature": 0,
            "accepted": 0,
            "tv_sum": 0.0,
            "ce_sum": 0.0,
            "depth_buckets": {},
        }

    @classmethod
    def _merge_head(cls, dst: dict, metrics: dict) -> None:
        mature = int(metrics.get("mature", 0) or 0)
        if mature <= 0:
            return
        accepted = int(metrics.get("accepted", round(float(metrics.get("acceptance_rate", 0.0)) * mature)) or 0)
        dst["mature"] += mature
        dst["accepted"] += accepted
        dst["tv_sum"] += float(metrics.get("sparse_tv", 0.0) or 0.0) * mature
        dst["ce_sum"] += float(metrics.get("mature_ce", 0.0) or 0.0) * mature
        for bucket, raw in (metrics.get("depth_buckets") or {}).items():
            count = int(raw.get("mature", 0) or 0)
            if count <= 0:
                continue
            bucket_dst = dst["depth_buckets"].setdefault(
                str(bucket),
                {"mature": 0, "accepted": 0.0, "tv_sum": 0.0},
            )
            bucket_dst["mature"] += count
            bucket_dst["accepted"] += float(raw.get("acceptance_rate", 0.0) or 0.0) * count
            bucket_dst["tv_sum"] += float(raw.get("sparse_tv", 0.0) or 0.0) * count

    @classmethod
    def _merge_stats(cls, dst: dict[str, dict], src: dict[str, dict]) -> None:
        for head, metrics in src.items():
            head_dst = dst.setdefault(str(head), cls._empty_head())
            cls._merge_head(head_dst, metrics)

    @staticmethod
    def _summarize(raw: dict[str, dict]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for head, stats in raw.items():
            mature = int(stats.get("mature", 0) or 0)
            if mature <= 0:
                continue
            buckets = {}
            for bucket, values in (stats.get("depth_buckets") or {}).items():
                count = int(values.get("mature", 0) or 0)
                if count <= 0:
                    continue
                buckets[str(bucket)] = {
                    "mature": count,
                    "acceptance_rate": float(values.get("accepted", 0.0)) / count,
                    "sparse_tv": float(values.get("tv_sum", 0.0)) / count,
                }
            out[str(head)] = {
                "mature": mature,
                "accepted": int(stats.get("accepted", 0) or 0),
                "acceptance_rate": float(stats.get("accepted", 0.0)) / mature,
                "sparse_tv": float(stats.get("tv_sum", 0.0)) / mature,
                "mature_ce": float(stats.get("ce_sum", 0.0)) / mature,
                "depth_buckets": buckets,
            }
        return out

    def observe(self, reflex_metrics: dict | None) -> None:
        per_head = ((reflex_metrics or {}).get("per_head") or {})
        self.last_metrics = per_head
        self._merge_stats(self.current, per_head)

    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "calibration_iterations": self.calibration_iterations,
            "check_interval": self.check_interval,
            "min_records_per_head": self.min_records_per_head,
            "trigger_z_high": self.trigger_z_high,
            "acceptance_drop_min": self.acceptance_drop_min,
            "patience": self.patience,
            "cooldown_iterations": self.cooldown_iterations,
            "max_heads_per_event": self.max_heads_per_event,
            "mad_floor": self.mad_floor,
            "current": self.current,
            "interval_stats": self.interval_stats,
            "baseline_tv_samples": self.baseline_tv_samples,
            "baseline_accept_samples": self.baseline_accept_samples,
            "patience_counters": self.patience_counters,
            "cooldown_until": self.cooldown_until,
            "last_metrics": dict(self.last_metrics),
            "last_decision": self.last_decision.__dict__,
        }

    def load_state_dict(self, state: dict) -> None:
        self.current = dict(state.get("current") or {})
        self.interval_stats = dict(state.get("interval_stats") or {})
        self.baseline_tv_samples = {
            str(k): [float(item) for item in values]
            for k, values in (state.get("baseline_tv_samples") or {}).items()
        }
        self.baseline_accept_samples = {
            str(k): [float(item) for item in values]
            for k, values in (state.get("baseline_accept_samples") or {}).items()
        }
        self.patience_counters = {str(k): int(v) for k, v in (state.get("patience_counters") or {}).items()}
        self.cooldown_until = {str(k): int(v) for k, v in (state.get("cooldown_until") or {}).items()}
        self.last_metrics = dict(state.get("last_metrics") or {})

    def should_evaluate(self, grpo_iteration: int) -> bool:
        if self.mode == "none":
            return False
        step = int(grpo_iteration)
        return step > self.calibration_iterations and step % self.check_interval == 0

    def update(self, reflex_metrics: dict | None, grpo_iteration: int) -> ReliabilityDecision:
        if reflex_metrics:
            self.observe(reflex_metrics)
        step = int(grpo_iteration)
        current_summary = self._summarize(self.current)
        self.current = {}
        if self.mode == "none":
            self.last_decision = ReliabilityDecision(reason="mode_none", head_metrics=current_summary)
            return self.last_decision

        if step <= self.calibration_iterations:
            for head, metrics in current_summary.items():
                self.baseline_accept_samples.setdefault(head, []).append(float(metrics["acceptance_rate"]))
                buckets = metrics.get("depth_buckets") or {"all": metrics}
                for bucket, values in buckets.items():
                    self.baseline_tv_samples.setdefault(f"{head}|{bucket}", []).append(float(values["sparse_tv"]))
            self.last_decision = ReliabilityDecision(
                evaluated=False,
                reason=f"calibration_{step}/{self.calibration_iterations}",
                head_metrics=current_summary,
            )
            return self.last_decision

        self._merge_stats(self.interval_stats, current_summary)
        if not self.should_evaluate(step):
            self.last_decision = ReliabilityDecision(
                evaluated=False,
                reason="check_interval_wait",
                head_metrics=current_summary,
            )
            return self.last_decision

        interval_summary = self._summarize(self.interval_stats)
        self.interval_stats = {}
        drift_scores: dict[str, float] = {}
        candidates: list[tuple[float, int]] = []
        for head, metrics in interval_summary.items():
            mature = int(metrics.get("mature", 0) or 0)
            baseline_accept = self.baseline_accept_samples.get(head, [])
            if mature < self.min_records_per_head or not baseline_accept:
                self.patience_counters[head] = 0
                continue
            weighted_z = 0.0
            z_weight = 0
            buckets = metrics.get("depth_buckets") or {"all": metrics}
            for bucket, values in buckets.items():
                samples = self.baseline_tv_samples.get(f"{head}|{bucket}")
                if not samples:
                    samples = self.baseline_tv_samples.get(f"{head}|all")
                if not samples:
                    continue
                center = float(median(samples))
                deviations = [abs(value - center) for value in samples]
                scale = max(1.4826 * float(median(deviations)), self.mad_floor)
                count = int(values.get("mature", 0) or 0)
                weighted_z += count * (float(values.get("sparse_tv", 0.0)) - center) / scale
                z_weight += count
            z_score = weighted_z / max(z_weight, 1)
            drift_scores[head] = float(z_score)
            acc_drop = float(median(baseline_accept)) - float(metrics.get("acceptance_rate", 0.0))
            in_cooldown = step < int(self.cooldown_until.get(head, -1))
            degraded = z_score > self.trigger_z_high and acc_drop >= self.acceptance_drop_min and not in_cooldown
            self.patience_counters[head] = self.patience_counters.get(head, 0) + 1 if degraded else 0
            if self.patience_counters[head] >= self.patience:
                candidates.append((float(z_score), int(head)))

        candidates.sort(reverse=True)
        triggered_heads = [head for _, head in candidates[: self.max_heads_per_event]]
        if self.mode == "always":
            triggered_heads = [int(head) for head in interval_summary][: self.max_heads_per_event]
            triggered = bool(triggered_heads)
            reason = "always"
        else:
            triggered = bool(triggered_heads)
            reason = "persistent_tv_and_acceptance_drift" if triggered else "reliability_stable"
        for head in triggered_heads:
            key = str(head)
            self.cooldown_until[key] = step + self.cooldown_iterations
            self.patience_counters[key] = 0
        self.last_decision = ReliabilityDecision(
            evaluated=True,
            triggered=triggered,
            triggered_heads=triggered_heads,
            drift_scores=drift_scores,
            head_metrics=interval_summary,
            reason=reason,
        )
        return self.last_decision


@dataclass
class AuxiliaryHeadRefreshConfig:
    steps: int = 1
    drift_threshold: float = 0.5
    head_weight_max: float = 2.0
    only_triggered_heads: bool = True
    update_future_heads: bool = True
    update_fast_state_injections: bool = False
    update_feedback_projection: bool = False
    update_backbone: bool = False
    update_lm_head: bool = False
    reflex_cache_enabled: bool = False
    require_reflex_cache: bool = False
    max_cached_records: int = 8192
    validation_fraction: float = 0.10
    max_validation_ce_regression: float = 0.01
    rollback_on_validation_regression: bool = True
    save_aux_every_grpo_iters: int = 0
    save_aux_on_triggered_update: bool = False
    min_iters_between_aux_trigger_saves: int = 50
    keep_last_aux_checkpoints: int = 0


class AuxiliaryHeadRefresher:
    """Small coordinator for reliability-triggered auxiliary-head updates."""

    def __init__(
        self,
        *,
        medusa_heads,
        trainer,
        optimizer: torch.optim.Optimizer,
        save_dir: str | Path,
        config: AuxiliaryHeadRefreshConfig,
    ):
        self.medusa_heads = medusa_heads
        self.trainer = trainer
        self.optimizer = optimizer
        self.save_dir = Path(save_dir)
        self.config = config
        self.last_trigger_save_step = -10**9
        self._saved_aux_dirs: list[Path] = []

    @classmethod
    def from_config(
        cls,
        *,
        medusa_heads,
        trainer,
        optimizer: torch.optim.Optimizer,
        save_dir: str | Path,
        aux_cfg: dict[str, Any],
        checkpoint_cfg: dict[str, Any],
        fallback_steps: int = 1,
    ) -> "AuxiliaryHeadRefresher":
        cfg = AuxiliaryHeadRefreshConfig(
            steps=max(1, int(aux_cfg.get("steps", fallback_steps))),
            drift_threshold=float(aux_cfg.get("trigger_z_high", aux_cfg.get("drift_threshold", 2.5))),
            head_weight_max=float(aux_cfg.get("head_weight_max", 2.0)),
            only_triggered_heads=bool(aux_cfg.get("only_triggered_heads", True)),
            update_future_heads=bool(aux_cfg.get("update_future_heads", True)),
            update_fast_state_injections=bool(aux_cfg.get("update_fast_state_injections", False)),
            update_feedback_projection=bool(aux_cfg.get("update_feedback_projection", False)),
            update_backbone=bool(aux_cfg.get("update_backbone", False)),
            update_lm_head=bool(aux_cfg.get("update_lm_head", False)),
            reflex_cache_enabled=bool(aux_cfg.get("reflex_cache_enabled", False)),
            require_reflex_cache=bool(aux_cfg.get("require_reflex_cache", False)),
            max_cached_records=int(aux_cfg.get("max_cached_records", 8192)),
            validation_fraction=float(aux_cfg.get("validation_fraction", 0.10)),
            max_validation_ce_regression=float(aux_cfg.get("max_validation_ce_regression", 0.01)),
            rollback_on_validation_regression=bool(aux_cfg.get("rollback_on_validation_regression", True)),
            save_aux_every_grpo_iters=int(checkpoint_cfg.get("save_aux_every_grpo_iters", 0)),
            save_aux_on_triggered_update=bool(checkpoint_cfg.get("save_aux_on_triggered_update", False)),
            min_iters_between_aux_trigger_saves=int(checkpoint_cfg.get("min_iters_between_aux_trigger_saves", 50)),
            keep_last_aux_checkpoints=int(checkpoint_cfg.get("keep_last_aux_checkpoints", 0)),
        )
        return cls(medusa_heads=medusa_heads, trainer=trainer, optimizer=optimizer, save_dir=save_dir, config=cfg)

    def head_weights_from_decision(self, decision: ReliabilityDecision) -> dict[int, float] | None:
        if not decision.triggered:
            return {}
        if decision.reason == "always":
            return None
        threshold = max(float(self.config.drift_threshold), 1e-6)
        max_weight = max(float(self.config.head_weight_max), 0.0)
        if not self.config.only_triggered_heads and decision.drift_scores:
            raw_heads = [int(k) for k in decision.drift_scores.keys() if str(k).isdigit()]
        else:
            raw_heads = list(decision.triggered_heads)
        weights: dict[int, float] = {}
        for head in raw_heads:
            drift = float(decision.drift_scores.get(str(head), threshold))
            weights[int(head)] = min(max(drift / threshold, 0.0), max_weight)
        return weights

    def maybe_update(
        self,
        *,
        decision: ReliabilityDecision,
        head_ids: torch.Tensor | None,
        head_mask: torch.Tensor | None,
        head_loss_mask: torch.Tensor | None,
        enabled: bool,
        grpo_step: int,
        rollout_count: int,
        reflex_records: dict | None = None,
        tracker_state: dict | None = None,
    ) -> dict[str, Any]:
        base = {
            "aux_update_evaluated": bool(decision.evaluated),
            "aux_update_triggered": bool(decision.triggered),
            "aux_update_reason": decision.reason,
            "aux_triggered_heads": list(decision.triggered_heads),
            "aux_drift_scores": dict(decision.drift_scores),
            "aux_checkpoint_path": "",
        }
        if not enabled:
            base["aux_update_reason"] = "online_medusa_disabled"
            return base
        if not decision.triggered:
            return base
        if not self.config.update_future_heads:
            base["aux_update_reason"] = "update_future_heads_disabled"
            return base
        use_reflex_cache = bool(
            self.config.reflex_cache_enabled
            and reflex_records
            and reflex_records.get("hidden") is not None
            and int(reflex_records["hidden"].shape[0]) > 0
        )
        if self.config.require_reflex_cache and not use_reflex_cache:
            base["aux_update_reason"] = "missing_reflex_aux_cache"
            return base
        if (not use_reflex_cache) and (head_ids is None or head_mask is None or head_ids.numel() == 0):
            base["aux_update_reason"] = "empty_aux_update_batch"
            return base

        head_weights = self.head_weights_from_decision(decision)
        if head_weights == {}:
            base["aux_update_reason"] = "no_selected_heads"
            return base

        train_records = reflex_records
        validation_records: dict = {}
        parameter_backup = None
        optimizer_backup = None
        validation_before = {"validation_ce": float("inf"), "validation_records": 0}
        if use_reflex_cache and bool(self.config.rollback_on_validation_regression):
            total_records = int(reflex_records["hidden"].shape[0])
            validation_count = int(total_records * float(self.config.validation_fraction))
            if validation_count > 0 and total_records - validation_count > 0:
                permutation = torch.randperm(total_records)
                val_index = permutation[:validation_count]
                train_index = permutation[validation_count:]

                def select_records(index: torch.Tensor) -> dict:
                    return {
                        key: value.index_select(0, index)
                        for key, value in reflex_records.items()
                        if torch.is_tensor(value) and value.shape[:1] == (total_records,)
                    }

                validation_records = select_records(val_index)
                train_records = select_records(train_index)
                validation_before = self.trainer.evaluate_reflex_records(
                    validation_records,
                    head_weights=head_weights,
                )
                parameter_backup = {
                    key: value.detach().clone()
                    for key, value in self.medusa_heads.state_dict().items()
                }
                optimizer_backup = copy.deepcopy(self.optimizer.state_dict())

        merged: dict[str, float] = {}
        steps = max(1, int(self.config.steps))
        for _ in range(steps):
            if use_reflex_cache:
                stats = self.trainer.update_reflex_records(
                    train_records,
                    head_weights=head_weights,
                    update_fast_state_injections=bool(self.config.update_fast_state_injections),
                )
            else:
                stats = self.trainer.update(head_ids, head_mask, head_loss_mask, head_weights=head_weights)
            for key, value in stats.items():
                if isinstance(value, bool):
                    base[key] = value
                elif isinstance(value, (int, float)):
                    merged[key] = merged.get(key, 0.0) + float(value)
        for key in list(merged.keys()):
            if key not in {"head_update_time", "head_update_tokens"}:
                merged[key] /= steps
        merged["head_update_time"] = merged.get("head_update_time", 0.0)
        merged["head_update_tokens"] = merged.get("head_update_tokens", 0.0)
        if merged["head_update_time"] > 0:
            merged["head_update_tokens_per_sec"] = merged["head_update_tokens"] / merged["head_update_time"]
        base.update(merged)
        base["aux_update_steps_requested"] = steps
        base["aux_head_weights"] = head_weights if head_weights is not None else "all"
        refresh_committed = True
        if validation_records:
            validation_after = self.trainer.evaluate_reflex_records(
                validation_records,
                head_weights=head_weights,
            )
            before_ce = float(validation_before.get("validation_ce", float("inf")))
            after_ce = float(validation_after.get("validation_ce", float("inf")))
            before_tv = float(validation_before.get("validation_sparse_tv", float("inf")))
            after_tv = float(validation_after.get("validation_sparse_tv", float("inf")))
            max_ce = before_ce * (1.0 + float(self.config.max_validation_ce_regression))
            has_sparse_validation = math.isfinite(before_tv) and math.isfinite(after_tv)
            primary_improved = after_tv < before_tv if has_sparse_validation else after_ce < before_ce
            refresh_committed = bool(
                math.isfinite(before_ce)
                and math.isfinite(after_ce)
                and after_ce <= max_ce
                and primary_improved
            )
            base["refresh_validation_ce_before"] = before_ce
            base["refresh_validation_ce_after"] = after_ce
            base["refresh_validation_tv_before"] = before_tv
            base["refresh_validation_tv_after"] = after_tv
            base["refresh_validation_records"] = int(validation_after.get("validation_records", 0))
            if not refresh_committed and parameter_backup is not None:
                self.medusa_heads.load_state_dict(parameter_backup)
                if optimizer_backup is not None:
                    self.optimizer.load_state_dict(optimizer_backup)
                self.optimizer.zero_grad(set_to_none=True)
        base["refresh_committed"] = bool(refresh_committed)
        base["refresh_rolled_back"] = not bool(refresh_committed)

        if self.config.save_aux_on_triggered_update and refresh_committed:
            min_gap = max(0, int(self.config.min_iters_between_aux_trigger_saves))
            if int(grpo_step) - self.last_trigger_save_step >= min_gap:
                tag = f"aux_trigger_step{int(grpo_step)}_rollout{int(rollout_count)}"
                path = self.save_aux_checkpoint(tag, grpo_step=grpo_step, rollout_count=rollout_count, decision=decision, tracker_state=tracker_state)
                self.last_trigger_save_step = int(grpo_step)
                base["aux_checkpoint_path"] = str(path)
        return base

    def maybe_save_periodic(
        self,
        *,
        grpo_step: int,
        rollout_count: int,
        tracker_state: dict | None = None,
    ) -> str:
        every = int(self.config.save_aux_every_grpo_iters or 0)
        if every <= 0 or int(grpo_step) <= 0 or int(grpo_step) % every != 0:
            return ""
        path = self.save_aux_checkpoint(
            f"aux_periodic_step{int(grpo_step)}",
            grpo_step=grpo_step,
            rollout_count=rollout_count,
            decision=None,
            tracker_state=tracker_state,
        )
        return str(path)

    def save_aux_checkpoint(
        self,
        tag: str,
        *,
        grpo_step: int,
        rollout_count: int,
        decision: ReliabilityDecision | None,
        tracker_state: dict | None = None,
    ) -> Path:
        path = self.save_dir / tag
        self.medusa_heads.save_pretrained(path)
        state = {
            "grpo_step": int(grpo_step),
            "rollout_count": int(rollout_count),
            "optimizer": self.optimizer.state_dict(),
            "reliability_tracker": tracker_state or {},
            "decision": decision.__dict__ if decision is not None else {},
            "refresh_config": self.config.__dict__,
        }
        torch.save(state, path / "aux_refresh_state.pt")
        if str(tag).startswith("aux_"):
            self._remember_checkpoint(path)
        return path

    def load_aux_checkpoint(self, load_dir: str | Path, tracker: ReliabilityTracker | None = None) -> dict:
        load_dir = Path(load_dir)
        state_path = load_dir / "aux_refresh_state.pt"
        if not state_path.exists():
            return {}
        state = torch.load(state_path, map_location="cpu")
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if tracker is not None and state.get("reliability_tracker"):
            tracker.load_state_dict(state["reliability_tracker"])
        return state

    def _remember_checkpoint(self, path: Path) -> None:
        keep = int(self.config.keep_last_aux_checkpoints or 0)
        if keep <= 0:
            return
        self._saved_aux_dirs.append(path)
        while len(self._saved_aux_dirs) > keep:
            old = self._saved_aux_dirs.pop(0)
            if old.exists() and old.is_dir():
                for child in old.iterdir():
                    child.unlink()
                old.rmdir()
