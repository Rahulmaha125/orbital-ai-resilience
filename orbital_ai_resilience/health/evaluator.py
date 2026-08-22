"""Deterministic, rule-based Health/Trust score evaluator."""

from typing import Dict, Optional, Tuple
from orbital_ai_resilience.core.types import HealthState
from orbital_ai_resilience.health.config import HealthConfig
from orbital_ai_resilience.telemetry.snapshot import TelemetryHistory, TelemetrySnapshot


class HealthEvaluator:
    """Computes transparent, explainable Health/Trust Scores and HealthStates.

    Formula:
        Score = clamp(0.0, 100.0, BaseScore - TrendPenalty)
        BaseScore = w_power * S_power + w_temp * S_temp + w_latency * S_latency + w_error * S_error
    """

    def __init__(self, config: Optional[HealthConfig] = None) -> None:
        self.config: HealthConfig = config or HealthConfig()

    def compute_sub_scores(self, snapshot: TelemetrySnapshot) -> Dict[str, float]:
        """Compute individual sub-scores [0.0, 100.0] for each telemetry metric."""
        cfg = self.config

        # 1. Power sub-score
        if snapshot.power_level >= cfg.power_nominal_min:
            s_power = 100.0
        elif snapshot.power_level <= cfg.power_critical_min:
            s_power = 0.0
        else:
            s_power = (snapshot.power_level - cfg.power_critical_min) / (
                cfg.power_nominal_min - cfg.power_critical_min
            ) * 100.0

        # 2. Temperature sub-score
        if snapshot.temperature <= cfg.temp_nominal_max:
            s_temp = 100.0
        elif snapshot.temperature >= cfg.temp_critical_max:
            s_temp = 0.0
        else:
            s_temp = (cfg.temp_critical_max - snapshot.temperature) / (
                cfg.temp_critical_max - cfg.temp_nominal_max
            ) * 100.0

        # 3. Latency sub-score
        if snapshot.latency <= cfg.latency_nominal_max:
            s_latency = 100.0
        elif snapshot.latency >= cfg.latency_critical_max:
            s_latency = 0.0
        else:
            s_latency = (cfg.latency_critical_max - snapshot.latency) / (
                cfg.latency_critical_max - cfg.latency_nominal_max
            ) * 100.0

        # 4. Error rate sub-score
        if snapshot.error_rate <= cfg.error_nominal_max:
            s_error = 100.0
        elif snapshot.error_rate >= cfg.error_critical_max:
            s_error = 0.0
        else:
            s_error = (cfg.error_critical_max - snapshot.error_rate) / (
                cfg.error_critical_max - cfg.error_nominal_max
            ) * 100.0

        return {
            "power": max(0.0, min(100.0, s_power)),
            "temp": max(0.0, min(100.0, s_temp)),
            "latency": max(0.0, min(100.0, s_latency)),
            "error": max(0.0, min(100.0, s_error)),
        }

    def compute_trend_penalty(self, history: TelemetryHistory) -> float:
        """Compute penalty points derived from recent telemetry worsening trends."""
        cfg = self.config
        recent = history.get_recent(cfg.trend_window_size)
        if len(recent) < 2:
            return 0.0

        first = recent[0]
        last = recent[-1]

        penalty = 0.0

        # Temperature trend penalty (if rising)
        temp_delta = last.temperature - first.temperature
        if temp_delta > 0.0:
            penalty += temp_delta * cfg.trend_penalty_weight_temp

        # Error rate trend penalty (if rising)
        error_delta = last.error_rate - first.error_rate
        if error_delta > 0.0:
            penalty += error_delta * cfg.trend_penalty_weight_error

        return min(cfg.trend_penalty_max, max(0.0, penalty))

    def evaluate_health(
        self,
        snapshot: TelemetrySnapshot,
        history: Optional[TelemetryHistory] = None,
    ) -> Tuple[float, HealthState, Dict[str, float]]:
        """Calculate composite Health Score, HealthState, and detailed breakdown.

        Args:
            snapshot: Current telemetry snapshot.
            history: Optional telemetry history window for trend penalty evaluation.

        Returns:
            Tuple of (composite_score, health_state, detailed_breakdown_dict).
        """
        cfg = self.config
        sub_scores = self.compute_sub_scores(snapshot)

        base_score = (
            cfg.weight_power * sub_scores["power"]
            + cfg.weight_temp * sub_scores["temp"]
            + cfg.weight_latency * sub_scores["latency"]
            + cfg.weight_error * sub_scores["error"]
        )

        trend_penalty = self.compute_trend_penalty(history) if history else 0.0
        final_score = max(0.0, min(100.0, base_score - trend_penalty))

        # Classify HealthState based on configurable thresholds
        if final_score >= cfg.threshold_healthy:
            state = HealthState.HEALTHY
        elif final_score >= cfg.threshold_warning:
            state = HealthState.WARNING
        elif final_score >= cfg.threshold_degraded:
            state = HealthState.DEGRADED
        else:
            state = HealthState.CRITICAL

        breakdown = {
            "base_score": round(base_score, 2),
            "trend_penalty": round(trend_penalty, 2),
            "final_score": round(final_score, 2),
            "sub_scores": {k: round(v, 2) for k, v in sub_scores.items()},
        }

        return final_score, state, breakdown
