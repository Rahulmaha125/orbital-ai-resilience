"""Structured event logging for migration lifecycle and audit trails."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from orbital_ai_resilience.recovery.types import MigrationState, VerificationStatus


@dataclass
class MigrationEvent:
    """Structured record of a workload migration attempt and verification result.

    Attributes:
        event_id: Unique string event identifier.
        timestamp: Epoch timestamp of event generation.
        migration_id: Associated migration snapshot ID.
        workload_id: Target workload ID.
        source_node_id: Source compute node ID.
        target_node_id: Target compute node ID (None if no target found).
        source_health_score: Physical health score of source node.
        source_behavior_score: AI behavioral score of source node.
        target_health_score: Physical health score of target node.
        target_behavior_score: AI behavioral score of target node.
        migration_reason: Reason string for initiating migration.
        migration_attempt: Attempt counter for this workload.
        migration_status: Current MigrationState enum.
        verification_status: VerificationStatus enum.
        failure_reason: Diagnostic string if migration/verification failed.
        details: Additional contextual dictionary.
    """

    migration_id: str
    workload_id: str
    source_node_id: str
    target_node_id: Optional[str]
    source_health_score: float
    source_behavior_score: float
    target_health_score: float
    target_behavior_score: float
    migration_reason: str
    migration_attempt: int
    migration_status: MigrationState
    verification_status: VerificationStatus
    event_id: str = field(default_factory=lambda: f"ev-{uuid.uuid4().hex[:8]}")
    timestamp: float = field(default_factory=time.time)
    failure_reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize migration event to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "migration_id": self.migration_id,
            "workload_id": self.workload_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "source_health_score": self.source_health_score,
            "source_behavior_score": self.source_behavior_score,
            "target_health_score": self.target_health_score,
            "target_behavior_score": self.target_behavior_score,
            "migration_reason": self.migration_reason,
            "migration_attempt": self.migration_attempt,
            "migration_status": self.migration_status.value,
            "verification_status": self.verification_status.value,
            "failure_reason": self.failure_reason,
            "details": self.details,
        }
