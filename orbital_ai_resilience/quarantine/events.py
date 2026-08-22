"""Structured event data structure for node quarantine actions."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from orbital_ai_resilience.quarantine.state import TrustState


@dataclass
class QuarantineEvent:
    """Structured record of a node quarantine or isolation action.

    Attributes:
        event_id: Unique event string identifier.
        timestamp: Epoch timestamp of action.
        node_id: Target node ID affected.
        previous_trust_state: TrustState before transition.
        new_trust_state: TrustState after transition.
        quarantine_reason: Human-readable diagnostic reason for quarantine.
        evidence_id: Associated verification evidence ID if triggered by output failure.
        details: Additional contextual dictionary.
    """

    node_id: str
    previous_trust_state: TrustState
    new_trust_state: TrustState
    quarantine_reason: str
    event_id: str = field(default_factory=lambda: f"qev-{uuid.uuid4().hex[:8]}")
    timestamp: float = field(default_factory=time.time)
    evidence_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize quarantine event to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "node_id": self.node_id,
            "previous_trust_state": self.previous_trust_state.value,
            "new_trust_state": self.new_trust_state.value,
            "quarantine_reason": self.quarantine_reason,
            "evidence_id": self.evidence_id,
            "details": self.details,
        }
