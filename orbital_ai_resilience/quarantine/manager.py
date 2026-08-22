"""QuarantineManager preserving computational trust integrity across cluster nodes."""

from typing import Dict, List, Optional
from orbital_ai_resilience.core.node import VirtualNode
from orbital_ai_resilience.core.types import NodeStatus
from orbital_ai_resilience.quarantine.events import QuarantineEvent
from orbital_ai_resilience.quarantine.state import TrustState
from orbital_ai_resilience.utils.logger import StateLogger


class QuarantineManager:
    """Manages node computational trust states, quarantine lists, and evidence preservation.

    Preserves separation between physical operational status (NodeStatus) and
    computational trust (TrustState).
    """

    def __init__(self, logger: Optional[StateLogger] = None) -> None:
        self.node_trust_states: Dict[str, TrustState] = {}
        self.quarantine_history: List[QuarantineEvent] = []
        self.logger: StateLogger = logger or StateLogger()

    def get_trust_state(self, node_id: str) -> TrustState:
        """Retrieve current TrustState for a given node ID."""
        return self.node_trust_states.get(node_id, TrustState.TRUSTED)

    def is_node_trusted(self, node_id: str) -> bool:
        """Check if node is currently classified as TRUSTED."""
        return self.get_trust_state(node_id) == TrustState.TRUSTED

    def quarantine_node(
        self,
        node: VirtualNode,
        reason: str,
        evidence_id: Optional[str] = None,
    ) -> QuarantineEvent:
        """Place a node into QUARANTINED state following a failed output verification.

        Node physical status remains unchanged (e.g. ONLINE), but its TrustState
        becomes QUARANTINED, preventing its selection for future workload assignments.
        """
        prev_state = self.get_trust_state(node.node_id)
        new_state = TrustState.QUARANTINED
        self.node_trust_states[node.node_id] = new_state

        event = QuarantineEvent(
            node_id=node.node_id,
            previous_trust_state=prev_state,
            new_trust_state=new_state,
            quarantine_reason=reason,
            evidence_id=evidence_id,
            details={"physical_status": node.status.value, "health_score": node.get_health_score()},
        )
        self.quarantine_history.append(event)
        self.logger.log_event("NODE_QUARANTINED", event.to_dict())
        return event

    def isolate_node(
        self,
        node: VirtualNode,
        reason: str,
    ) -> QuarantineEvent:
        """Place a source node into ISOLATED state following confirmed silent degradation recovery.

        Updates both TrustState to ISOLATED and NodeStatus to ISOLATED.
        """
        prev_state = self.get_trust_state(node.node_id)
        new_state = TrustState.ISOLATED
        self.node_trust_states[node.node_id] = new_state
        node.set_status(NodeStatus.ISOLATED)

        event = QuarantineEvent(
            node_id=node.node_id,
            previous_trust_state=prev_state,
            new_trust_state=new_state,
            quarantine_reason=reason,
            details={"physical_status": node.status.value, "health_score": node.get_health_score()},
        )
        self.quarantine_history.append(event)
        self.logger.log_event("NODE_ISOLATED", event.to_dict())
        return event

    def get_quarantined_node_ids(self) -> List[str]:
        """Return list of all currently quarantined node IDs."""
        return [
            nid for nid, state in self.node_trust_states.items() if state == TrustState.QUARANTINED
        ]

    def get_isolated_node_ids(self) -> List[str]:
        """Return list of all currently isolated node IDs."""
        return [
            nid for nid, state in self.node_trust_states.items() if state == TrustState.ISOLATED
        ]
