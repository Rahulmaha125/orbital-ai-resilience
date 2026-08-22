"""Historical node recovery tracking and empirical reliability scoring."""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional
from orbital_ai_resilience.verification.types import VerificationResultState


@dataclass
class NodeHistoryRecord:
    """Single historical outcome record for a target compute node."""

    node_id: str
    timestamp: float
    verification_result: VerificationResultState
    was_quarantined: bool
    duration_sec: float
    workload_id: str
    details: Dict[str, Any] = field(default_factory=dict)


class RecoveryHistory:
    """Tracks historical target node verification outcomes and calculates empirical node reliability."""

    def __init__(self) -> None:
        self.records: List[NodeHistoryRecord] = []

    def record_outcome(
        self,
        node_id: str,
        verification_result: VerificationResultState,
        was_quarantined: bool,
        duration_sec: float = 0.0,
        workload_id: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a target node verification outcome into historical memory."""
        rec = NodeHistoryRecord(
            node_id=node_id,
            timestamp=time.time(),
            verification_result=verification_result,
            was_quarantined=was_quarantined,
            duration_sec=duration_sec,
            workload_id=workload_id,
            details=details or {},
        )
        self.records.append(rec)

    def get_node_records(self, node_id: str) -> List[NodeHistoryRecord]:
        """Return all historical records for a specific target node."""
        return [r for r in self.records if r.node_id == node_id]

    def get_verification_success_rate(self, node_id: str) -> float:
        """Calculate ratio of verified outcomes over total verification attempts for a node."""
        node_recs = self.get_node_records(node_id)
        if not node_recs:
            return 1.0  # Default 100% for pristine nodes without prior failures
        successes = sum(1 for r in node_recs if r.verification_result == VerificationResultState.VERIFIED)
        return round(successes / len(node_recs), 4)

    def get_quarantine_count(self, node_id: str) -> int:
        """Return total number of times node has been quarantined."""
        return sum(1 for r in self.records if r.node_id == node_id and r.was_quarantined)

    def get_node_reliability_score(self, node_id: str) -> float:
        """Calculate composite empirical node reliability score in range [0.0, 100.0].

        Score = max(0.0, 100.0 * success_rate - 25.0 * quarantine_count)
        """
        success_rate = self.get_verification_success_rate(node_id)
        quarantine_cnt = self.get_quarantine_count(node_id)
        score = (100.0 * success_rate) - (25.0 * quarantine_cnt)
        return round(max(0.0, min(100.0, score)), 2)

    def clear(self) -> None:
        """Clear recorded recovery history."""
        self.records.clear()
