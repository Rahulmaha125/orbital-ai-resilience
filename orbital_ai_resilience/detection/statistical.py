"""Deterministic statistical baseline anomaly detector."""

import math
from typing import Any, Dict, List, Optional
from orbital_ai_resilience.core.types import HealthState
from orbital_ai_resilience.detection.base import BaseDetector
from orbital_ai_resilience.detection.behavior_score import BehavioralScoreEvaluator
from orbital_ai_resilience.detection.types import BehavioralState, DetectionResult


class StatisticalDetector(BaseDetector):
    """Deterministic, transparent statistical detector monitoring AI output deviation.

    Uses rolling z-scores, absolute MSE thresholds, and Behavioral Integrity Scoring.
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        mse_absolute_threshold: float = 0.002,
        window_size: int = 10,
        behavior_evaluator: Optional[BehavioralScoreEvaluator] = None,
    ) -> None:
        super().__init__(name="Statistical_ZScore_Detector")
        self.z_threshold: float = z_threshold
        self.mse_absolute_threshold: float = mse_absolute_threshold
        self.window_size: int = window_size
        self.behavior_evaluator: BehavioralScoreEvaluator = behavior_evaluator or BehavioralScoreEvaluator()

    def evaluate(
        self,
        exec_log: Dict[str, Any],
        history_logs: Optional[List[Dict[str, Any]]] = None,
    ) -> DetectionResult:
        """Evaluate an execution payload log against moving baseline statistical thresholds."""
        deviation = exec_log.get("deviation", {})
        mse = deviation.get("mse", 0.0)

        # 1. Compute AI Behavioral Integrity Score & State
        b_score, b_state, sub_scores = self.behavior_evaluator.compute_score(deviation)

        # 2. Extract Phase 2 physical health metrics
        phys_score = exec_log.get("health_score", 100.0)
        phys_state_str = exec_log.get("health_state", "HEALTHY")
        phys_state = HealthState(phys_state_str)
        phys_telemetry = exec_log.get("physical_telemetry", {})
        node_status = phys_telemetry.get("status", "ONLINE")

        # 3. Rolling window z-score calculation
        z_score = 0.0
        mse_mean = mse
        mse_std = 0.0

        if history_logs and len(history_logs) >= 2:
            recent = history_logs[-self.window_size :]
            mses = [h.get("deviation", {}).get("mse", 0.0) for h in recent]
            mse_mean = sum(mses) / len(mses)
            variance = sum((x - mse_mean) ** 2 for x in mses) / len(mses)
            mse_std = math.sqrt(variance)
            if mse_std > 1e-7:
                z_score = (mse - mse_mean) / mse_std

        # 4. Determine anomaly flag
        is_anomaly = (
            z_score >= self.z_threshold
            or mse >= self.mse_absolute_threshold
            or b_state in (BehavioralState.DEGRADED, BehavioralState.CRITICAL)
        )

        # 5. Determine SILENT DEGRADATION flag
        # Node remains ONLINE and physical health is HEALTHY, but AI output is abnormal
        is_silent_degradation = (
            is_anomaly
            and phys_score >= 90.0
            and node_status == "ONLINE"
        )

        # Confidence calculation
        confidence = min(1.0, max(0.0, (mse / (self.mse_absolute_threshold * 2.0)))) if is_anomaly else 1.0

        details = {
            "mse": round(mse, 6),
            "mae": round(deviation.get("mae", 0.0), 6),
            "cosine_sim": round(deviation.get("cosine_sim", 1.0), 4),
            "z_score": round(z_score, 2),
            "rolling_mse_mean": round(mse_mean, 6),
            "rolling_mse_std": round(mse_std, 6),
            "sub_scores": sub_scores,
        }

        return DetectionResult(
            timestamp=exec_log.get("timestamp", 0.0),
            tick=exec_log.get("tick", 0),
            node_id=exec_log.get("target_node_id", "unknown"),
            is_anomaly=is_anomaly,
            is_silent_degradation=is_silent_degradation,
            confidence=round(confidence, 2),
            detector_name=self.name,
            behavioral_score=b_score,
            behavioral_state=b_state,
            physical_health_score=phys_score,
            physical_health_state=phys_state,
            details=details,
        )
