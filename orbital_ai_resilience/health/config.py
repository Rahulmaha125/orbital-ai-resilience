"""Configurable parameters and thresholds for baseline Health/Trust scoring."""

from dataclasses import dataclass


@dataclass
class HealthConfig:
    """Configurable weights, operating limits, and thresholds for node health evaluation.

    Attributes:
        weight_power: Weight assigned to power sub-score (default 0.20).
        weight_temp: Weight assigned to temperature sub-score (default 0.25).
        weight_latency: Weight assigned to latency sub-score (default 0.20).
        weight_error: Weight assigned to error rate sub-score (default 0.35).
        
        power_nominal_min: Power level below which degradation begins (default 80.0).
        power_critical_min: Power level at or below which sub-score is 0 (default 20.0).
        
        temp_nominal_max: Operating temp up to which sub-score is 100 (default 60.0).
        temp_critical_max: Operating temp at or above which sub-score is 0 (default 95.0).
        
        latency_nominal_max: Latency up to which sub-score is 100 (default 15.0 ms).
        latency_critical_max: Latency at or above which sub-score is 0 (default 100.0 ms).
        
        error_nominal_max: Error rate up to which sub-score is 100 (default 0.0).
        error_critical_max: Error rate at or above which sub-score is 0 (default 0.10).
        
        trend_window_size: Number of recent snapshots evaluated for trend penalty (default 5).
        trend_penalty_weight_temp: Penalty points per degree Celsius of sustained rise (default 0.3).
        trend_penalty_weight_error: Penalty points per 0.01 rise in error rate (default 100.0).
        trend_penalty_max: Maximum total trend penalty points (default 20.0).
        
        threshold_healthy: Score threshold for HEALTHY state (default 90.0).
        threshold_warning: Score threshold for WARNING state (default 75.0).
        threshold_degraded: Score threshold for DEGRADED state (default 60.0).
    """

    weight_power: float = 0.20
    weight_temp: float = 0.25
    weight_latency: float = 0.20
    weight_error: float = 0.35

    power_nominal_min: float = 80.0
    power_critical_min: float = 20.0

    temp_nominal_max: float = 60.0
    temp_critical_max: float = 95.0

    latency_nominal_max: float = 15.0
    latency_critical_max: float = 100.0

    error_nominal_max: float = 0.0
    error_critical_max: float = 0.10

    trend_window_size: int = 5
    trend_penalty_weight_temp: float = 0.3
    trend_penalty_weight_error: float = 100.0  # 1.0 pt per 0.01 error rate rise
    trend_penalty_max: float = 20.0

    threshold_healthy: float = 90.0
    threshold_warning: float = 75.0
    threshold_degraded: float = 60.0

    def __post_init__(self) -> None:
        """Validate that sub-score weights sum approximately to 1.0."""
        total_weight = self.weight_power + self.weight_temp + self.weight_latency + self.weight_error
        if abs(total_weight - 1.0) > 1e-4:
            raise ValueError(f"Sub-score weights must sum to 1.0, got {total_weight}")
