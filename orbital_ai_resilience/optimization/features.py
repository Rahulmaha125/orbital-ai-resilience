"""Feature engineering module building normalized multimodal feature vectors for recovery optimization."""

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, Optional
from orbital_ai_resilience.core.node import VirtualNode
from orbital_ai_resilience.core.workload import Workload
from orbital_ai_resilience.detection.behavior_score import BehavioralScoreEvaluator
from orbital_ai_resilience.optimization.history import RecoveryHistory


@dataclass
class TargetNodeFeatures:
    """Normalized feature vector representation for a candidate target node.

    Physical Features:
        physical_health_score: [0.0, 100.0]
        power_level: [0.0, 100.0]
        temperature: Operating temperature (Celsius)
        latency: Communication/processing latency (ms)
        error_rate: [0.0, 1.0]

    AI Behavioral Features:
        behavioral_integrity_score: [0.0, 100.0]
        behavioral_state: String enum label
        recent_mse: Observed MSE
        rolling_mse: 5-tick rolling average MSE

    Resource Features:
        available_compute_ratio: [0.0, 1.0]
        available_memory_ratio: [0.0, 1.0]
        workload_compute_req: Workload required compute
        workload_memory_req: Workload required memory

    Recovery History Features:
        previous_migration_success_rate: [0.0, 1.0]
        previous_verification_success_rate: [0.0, 1.0]
        previous_quarantine_count: Integer count
        node_reliability_score: [0.0, 100.0]
    """

    node_id: str
    physical_health_score: float
    power_level: float
    temperature: float
    latency: float
    error_rate: float

    behavioral_integrity_score: float
    behavioral_state: str
    recent_mse: float
    rolling_mse: float

    available_compute_ratio: float
    available_memory_ratio: float
    workload_compute_req: float
    workload_memory_req: float

    previous_migration_success_rate: float
    previous_verification_success_rate: float
    previous_quarantine_count: int
    node_reliability_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize features to dictionary representation."""
        return {
            "node_id": self.node_id,
            "physical_health_score": round(self.physical_health_score, 2),
            "power_level": round(self.power_level, 2),
            "temperature": round(self.temperature, 2),
            "latency": round(self.latency, 2),
            "error_rate": round(self.error_rate, 4),
            "behavioral_integrity_score": round(self.behavioral_integrity_score, 2),
            "behavioral_state": self.behavioral_state,
            "recent_mse": round(self.recent_mse, 6),
            "rolling_mse": round(self.rolling_mse, 6),
            "available_compute_ratio": round(self.available_compute_ratio, 4),
            "available_memory_ratio": round(self.available_memory_ratio, 4),
            "workload_compute_req": self.workload_compute_req,
            "workload_memory_req": self.workload_memory_req,
            "previous_verification_success_rate": round(self.previous_verification_success_rate, 4),
            "previous_quarantine_count": self.previous_quarantine_count,
            "node_reliability_score": round(self.node_reliability_score, 2),
        }

    def to_numpy(self) -> np.ndarray:
        """Convert numeric features to 1D float64 numpy vector for model ingestion."""
        return np.array(
            [
                self.physical_health_score,
                self.power_level,
                self.temperature,
                self.latency,
                self.error_rate,
                self.behavioral_integrity_score,
                self.recent_mse,
                self.rolling_mse,
                self.available_compute_ratio,
                self.available_memory_ratio,
                self.previous_verification_success_rate,
                float(self.previous_quarantine_count),
                self.node_reliability_score,
            ],
            dtype=np.float64,
        )


class OptimizationFeatureBuilder:
    """Constructs normalized TargetNodeFeatures from cluster state and recovery history."""

    def __init__(self, behavior_evaluator: Optional[BehavioralScoreEvaluator] = None) -> None:
        self.behavior_evaluator: BehavioralScoreEvaluator = behavior_evaluator or BehavioralScoreEvaluator()

    def build_features(
        self,
        node: VirtualNode,
        workload: Workload,
        history: RecoveryHistory,
        exec_log: Optional[Dict[str, Any]] = None,
    ) -> TargetNodeFeatures:
        """Build a comprehensive TargetNodeFeatures instance for a candidate target node."""
        phys_health = node.get_health_score()
        power = node.power_level
        temp = node.temperature
        lat = node.latency
        err = node.error_rate

        # Behavioral features
        if exec_log:
            dev = exec_log.get("deviation", {})
            recent_mse = dev.get("mse", 0.0)
            b_score, b_state, _ = self.behavior_evaluator.compute_score(dev)
            b_state_str = b_state.value
        else:
            recent_mse = 0.0
            b_score = 100.0
            b_state_str = "NORMAL"

        rolling_mse = recent_mse

        # Resource ratios
        comp_ratio = node.get_available_compute() / node.compute_capacity
        mem_ratio = node.get_available_memory() / node.memory_capacity

        # History features
        v_success_rate = history.get_verification_success_rate(node.node_id)
        quarantine_cnt = history.get_quarantine_count(node.node_id)
        reliability = history.get_node_reliability_score(node.node_id)

        return TargetNodeFeatures(
            node_id=node.node_id,
            physical_health_score=phys_health,
            power_level=power,
            temperature=temp,
            latency=lat,
            error_rate=err,
            behavioral_integrity_score=b_score,
            behavioral_state=b_state_str,
            recent_mse=recent_mse,
            rolling_mse=rolling_mse,
            available_compute_ratio=comp_ratio,
            available_memory_ratio=mem_ratio,
            workload_compute_req=workload.required_compute,
            workload_memory_req=workload.required_memory,
            previous_migration_success_rate=v_success_rate,
            previous_verification_success_rate=v_success_rate,
            previous_quarantine_count=quarantine_cnt,
            node_reliability_score=reliability,
        )
