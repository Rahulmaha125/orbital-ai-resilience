"""Type definitions and dataclasses for anomaly detection and behavioral scoring."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from orbital_ai_resilience.core.types import HealthState


class BehavioralState(str, Enum):
    """Classification state for AI output behavioral integrity."""
    NORMAL = "NORMAL"        # Score 90.0 - 100.0
    WARNING = "WARNING"      # Score 75.0 - 89.9
    DEGRADED = "DEGRADED"    # Score 60.0 - 74.9
    CRITICAL = "CRITICAL"    # Score 0.0 - 59.9


@dataclass
class DetectionResult:
    """Standardized output payload from an anomaly detector evaluation.

    Attributes:
        timestamp: Epoch timestamp of evaluation.
        tick: Simulation tick index.
        node_id: Target node ID.
        is_anomaly: True if AI output anomaly detected.
        is_silent_degradation: True if AI degradation detected while physical telemetry is HEALTHY.
        confidence: Detector confidence score [0.0, 1.0].
        detector_name: Identifier of the detector algorithm.
        behavioral_score: AI Behavioral Integrity Score [0.0, 100.0].
        behavioral_state: BehavioralState enum classification.
        physical_health_score: Phase 2 Physical Health Score [0.0, 100.0].
        physical_health_state: Phase 2 HealthState classification.
        details: Diagnostic dictionary containing metrics and z-scores.
    """

    timestamp: float
    tick: int
    node_id: str
    is_anomaly: bool
    is_silent_degradation: bool
    confidence: float
    detector_name: str
    behavioral_score: float
    behavioral_state: BehavioralState
    physical_health_score: float
    physical_health_state: HealthState
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize detection result to dictionary."""
        return {
            "timestamp": self.timestamp,
            "tick": self.tick,
            "node_id": self.node_id,
            "is_anomaly": self.is_anomaly,
            "is_silent_degradation": self.is_silent_degradation,
            "confidence": self.confidence,
            "detector_name": self.detector_name,
            "behavioral_score": self.behavioral_score,
            "behavioral_state": self.behavioral_state.value,
            "physical_health_score": self.physical_health_score,
            "physical_health_state": self.physical_health_state.value,
            "details": self.details,
        }
