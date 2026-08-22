"""AI Behavioral Integrity Score evaluator for monitoring model execution quality."""

from typing import Dict, Tuple
from orbital_ai_resilience.detection.types import BehavioralState


class BehavioralScoreEvaluator:
    """Computes transparent, explainable AI Behavioral Integrity Scores [0.0, 100.0].

    Distinct from physical telemetry health scoring. Evaluates AI computational
    integrity using MSE, MAE, and Cosine Similarity outputs.
    """

    def __init__(
        self,
        mse_nominal_max: float = 0.0005,
        mse_critical_max: float = 0.02,
        mae_nominal_max: float = 0.001,
        mae_critical_max: float = 0.08,
        cos_nominal_min: float = 0.999,
        cos_critical_min: float = 0.95,
    ) -> None:
        self.mse_nominal_max: float = mse_nominal_max
        self.mse_critical_max: float = mse_critical_max
        self.mae_nominal_max: float = mae_nominal_max
        self.mae_critical_max: float = mae_critical_max
        self.cos_nominal_min: float = cos_nominal_min
        self.cos_critical_min: float = cos_critical_min

    def compute_score(self, deviation_metrics: Dict[str, float]) -> Tuple[float, BehavioralState, Dict[str, float]]:
        """Calculate composite AI Behavioral Integrity Score and classification state.

        Args:
            deviation_metrics: Dict containing 'mse', 'mae', and 'cosine_sim'.

        Returns:
            Tuple of (composite_score, behavioral_state, sub_scores_dict).
        """
        mse = deviation_metrics.get("mse", 0.0)
        mae = deviation_metrics.get("mae", 0.0)
        cos_sim = deviation_metrics.get("cosine_sim", 1.0)

        # 1. MSE Sub-score
        if mse <= self.mse_nominal_max:
            s_mse = 100.0
        elif mse >= self.mse_critical_max:
            s_mse = 0.0
        else:
            s_mse = (self.mse_critical_max - mse) / (self.mse_critical_max - self.mse_nominal_max) * 100.0

        # 2. MAE Sub-score
        if mae <= self.mae_nominal_max:
            s_mae = 100.0
        elif mae >= self.mae_critical_max:
            s_mae = 0.0
        else:
            s_mae = (self.mae_critical_max - mae) / (self.mae_critical_max - self.mae_nominal_max) * 100.0

        # 3. Cosine Similarity Sub-score
        if cos_sim >= self.cos_nominal_min:
            s_cos = 100.0
        elif cos_sim <= self.cos_critical_min:
            s_cos = 0.0
        else:
            s_cos = (cos_sim - self.cos_critical_min) / (self.cos_nominal_min - self.cos_critical_min) * 100.0

        s_mse = max(0.0, min(100.0, s_mse))
        s_mae = max(0.0, min(100.0, s_mae))
        s_cos = max(0.0, min(100.0, s_cos))

        # Weighted combination
        final_score = 0.40 * s_mse + 0.30 * s_mae + 0.30 * s_cos
        final_score = max(0.0, min(100.0, final_score))

        # State classification
        if final_score >= 90.0:
            state = BehavioralState.NORMAL
        elif final_score >= 75.0:
            state = BehavioralState.WARNING
        elif final_score >= 60.0:
            state = BehavioralState.DEGRADED
        else:
            state = BehavioralState.CRITICAL

        sub_scores = {
            "mse_sub_score": round(s_mse, 2),
            "mae_sub_score": round(s_mae, 2),
            "cos_sub_score": round(s_cos, 2),
        }

        return round(final_score, 2), state, sub_scores
