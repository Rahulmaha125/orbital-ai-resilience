"""Verification evidence data structure for auditing output verification results."""

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np
from orbital_ai_resilience.verification.types import VerificationResultState


def compute_tensor_hash(tensor: np.ndarray) -> str:
    """Compute deterministic SHA-256 hash of a numerical tensor."""
    tensor_bytes = np.ascontiguousarray(tensor).tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()[:16]


@dataclass
class VerificationEvidence:
    """Immutable evidence record documenting a target node verification attempt.

    Attributes:
        verification_id: Unique string identifier for this verification audit.
        workload_id: ID of the workload evaluated.
        source_node_id: Original source node ID.
        target_node_id: Target compute node ID evaluated.
        timestamp: Epoch timestamp of verification.
        reference_output_hash: SHA-256 hash of trusted reference output tensor.
        target_output_hash: SHA-256 hash of target node output tensor.
        mse: Observed Mean Squared Error.
        mae: Observed Mean Absolute Error.
        cosine_sim: Observed Cosine Similarity.
        thresholds: Dictionary of threshold criteria used for decision.
        verification_result: VerificationResultState enum (VERIFIED, VERIFICATION_FAILED).
        attempt_number: Migration/recovery attempt number (1, 2, 3...).
        fault_context: Optional active fault or diagnostic metadata dictionary.
    """

    verification_id: str
    workload_id: str
    source_node_id: str
    target_node_id: str
    timestamp: float
    reference_output_hash: str
    target_output_hash: str
    mse: float
    mae: float
    cosine_sim: float
    thresholds: Dict[str, float]
    verification_result: VerificationResultState
    attempt_number: int
    fault_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize verification evidence to dictionary."""
        return {
            "verification_id": self.verification_id,
            "workload_id": self.workload_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "timestamp": self.timestamp,
            "reference_output_hash": self.reference_output_hash,
            "target_output_hash": self.target_output_hash,
            "mse": round(self.mse, 6),
            "mae": round(self.mae, 6),
            "cosine_sim": round(self.cosine_sim, 4),
            "thresholds": self.thresholds,
            "verification_result": self.verification_result.value,
            "attempt_number": self.attempt_number,
            "fault_context": self.fault_context,
        }
