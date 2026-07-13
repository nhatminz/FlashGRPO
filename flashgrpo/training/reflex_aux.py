from __future__ import annotations

from dataclasses import dataclass, field


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
        interval: int = 10,
        reject_weight: float = 1.0,
        drift_threshold: float = 8.0,
        ema_beta: float = 0.9,
        min_mature_records: int = 64,
    ):
        self.mode = str(mode)
        self.interval = max(1, int(interval))
        self.reject_weight = float(reject_weight)
        self.drift_threshold = float(drift_threshold)
        self.ema_beta = float(ema_beta)
        self.min_mature_records = int(min_mature_records)
        self.ema_ce: dict[str, float] = {}
        self.ema_accept: dict[str, float] = {}
        self.last_metrics: dict[str, dict] = {}
        self.last_decision = ReliabilityDecision()

    def update(self, reflex_metrics: dict | None, rollout_count: int) -> ReliabilityDecision:
        per_head = ((reflex_metrics or {}).get("per_head") or {})
        self.last_metrics = per_head
        if self.mode == "none":
            self.last_decision = ReliabilityDecision(reason="mode_none", head_metrics=per_head)
            return self.last_decision
        if int(rollout_count) % self.interval != 0:
            self.last_decision = ReliabilityDecision(reason="interval_skip", head_metrics=per_head)
            return self.last_decision

        drift_scores: dict[str, float] = {}
        triggered_heads: list[int] = []
        total_mature = 0
        for raw_head, metrics in per_head.items():
            mature = int(metrics.get("mature", 0) or 0)
            total_mature += mature
            if mature <= 0:
                continue
            ce = float(metrics.get("mature_ce", 0.0) or 0.0)
            acc = float(metrics.get("acceptance_rate", 0.0) or 0.0)
            old_ce = self.ema_ce.get(raw_head, ce)
            old_acc = self.ema_accept.get(raw_head, acc)
            ce_ema = self.ema_beta * old_ce + (1.0 - self.ema_beta) * ce
            acc_ema = self.ema_beta * old_acc + (1.0 - self.ema_beta) * acc
            self.ema_ce[raw_head] = ce_ema
            self.ema_accept[raw_head] = acc_ema
            drift = ce_ema + self.reject_weight * (1.0 - acc_ema)
            drift_scores[raw_head] = drift
            if drift > self.drift_threshold:
                try:
                    triggered_heads.append(int(raw_head))
                except ValueError:
                    pass

        if self.mode == "always":
            triggered = total_mature >= self.min_mature_records or not per_head
            reason = "always" if triggered else "not_enough_mature_records"
        elif total_mature < self.min_mature_records:
            triggered = False
            reason = "not_enough_mature_records"
        else:
            triggered = bool(triggered_heads)
            reason = "drift_threshold" if triggered else "below_threshold"
        self.last_decision = ReliabilityDecision(
            evaluated=True,
            triggered=triggered,
            triggered_heads=triggered_heads,
            drift_scores=drift_scores,
            head_metrics=per_head,
            reason=reason,
        )
        return self.last_decision
