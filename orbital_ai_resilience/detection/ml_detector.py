"""Machine learning anomaly detector using Scikit-learn Isolation Forest."""

import numpy as np
from typing import Any, Dict, List, Optional
from sklearn.ensemble import IsolationForest
from orbital_ai_resilience.core.types import HealthState
from orbital_ai_resilience.detection.base import BaseDetector
from orbital_ai_resilience.detection.behavior_score import BehavioralScoreEvaluator
from orbital_ai_resilience.detection.types import BehavioralState, DetectionResult


class IsolationForestDetector(BaseDetector):
    """Scikit-learn Isolation Forest ML detector for AI compute anomaly identification.

    Consumes physical telemetry and AI behavioral feature vectors.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
        behavior_evaluator: Optional[BehavioralScoreEvaluator] = None,
    ) -> None:
        super().__init__(name="ML_IsolationForest_Detector")
        self.random_state: int = random_state
        self.model: IsolationForest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self.is_fitted: bool = False
        self.behavior_evaluator: BehavioralScoreEvaluator = behavior_evaluator or BehavioralScoreEvaluator()

    def extract_feature_vector(
        self,
        exec_log: Dict[str, Any],
        history_logs: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """Extract physical and behavioral feature vector from execution log."""
        phys = exec_log.get("physical_telemetry", {})
        dev = exec_log.get("deviation", {})

        power = float(phys.get("power_level", 100.0))
        temp = float(phys.get("temperature", 45.0))
        lat = float(phys.get("latency", 10.0))
        err = float(phys.get("error_rate", 0.0))
        phys_score = float(exec_log.get("health_score", 100.0))

        mse = float(dev.get("mse", 0.0))
        mae = float(dev.get("mae", 0.0))
        cos_sim = float(dev.get("cosine_sim", 1.0))

        if history_logs and len(history_logs) > 0:
            recent = history_logs[-10:]
            mses = [h.get("deviation", {}).get("mse", 0.0) for h in recent]
            rolling_mean = float(np.mean(mses))
        else:
            rolling_mean = mse

        return np.array(
            [power, temp, lat, err, phys_score, mse, mae, cos_sim, rolling_mean],
            dtype=np.float64,
        )

    def fit(self, baseline_logs: List[Dict[str, Any]]) -> None:
        """Train the Isolation Forest model on clean baseline execution logs."""
        if not baseline_logs:
            raise ValueError("baseline_logs cannot be empty for fitting ML model")

        raw_features = np.array(
            [self.extract_feature_vector(log) for log in baseline_logs],
            dtype=np.float64,
        )
        # Add slight natural jitter to baseline features to ensure proper tree splitting
        rng = np.random.default_rng(self.random_state)
        jitter = rng.normal(0.0, 1e-5, size=raw_features.shape)
        feature_matrix = raw_features + jitter

        self.model.fit(feature_matrix)
        self.is_fitted = True

    def evaluate(
        self,
        exec_log: Dict[str, Any],
        history_logs: Optional[List[Dict[str, Any]]] = None,
    ) -> DetectionResult:
        """Evaluate execution log using the trained Isolation Forest model."""
        feature_vector = self.extract_feature_vector(exec_log, history_logs).reshape(1, -1)

        # Fallback if fit() was not called explicitly
        if not self.is_fitted:
            dummy_normal = np.tile(feature_vector, (30, 1))
            dummy_normal += np.random.default_rng(self.random_state).normal(0, 1e-5, dummy_normal.shape)
            self.model.fit(dummy_normal)
            self.is_fitted = True

        # Predict: 1 = normal, -1 = anomaly
        prediction = self.model.predict(feature_vector)[0]
        anomaly_score = float(self.model.decision_function(feature_vector)[0])
        
        # Also check if MSE is distinctly non-zero to supplement ML decision
        mse = float(exec_log.get("deviation", {}).get("mse", 0.0))
        is_anomaly = bool(prediction == -1 or mse >= 0.001)

        # Compute AI Behavioral Integrity Score & State
        b_score, b_state, sub_scores = self.behavior_evaluator.compute_score(exec_log.get("deviation", {}))

        phys_score = exec_log.get("health_score", 100.0)
        phys_state = HealthState(exec_log.get("health_state", "HEALTHY"))
        status = exec_log.get("physical_telemetry", {}).get("status", "ONLINE")

        # Silent degradation condition
        is_silent_degradation = is_anomaly and phys_score >= 90.0 and status == "ONLINE"

        confidence = float(np.clip(0.5 - anomaly_score, 0.0, 1.0)) if is_anomaly else float(np.clip(0.5 + anomaly_score, 0.0, 1.0))

        details = {
            "prediction": int(prediction),
            "decision_function_score": round(anomaly_score, 4),
            "mse": round(mse, 6),
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
