"""Deterministic target node selection algorithm for autonomous workload migration."""

from typing import Dict, List, Optional, Tuple
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.core.node import VirtualNode
from orbital_ai_resilience.core.types import NodeStatus
from orbital_ai_resilience.core.workload import Workload
from orbital_ai_resilience.detection.behavior_score import BehavioralScoreEvaluator
from orbital_ai_resilience.quarantine.manager import QuarantineManager
from orbital_ai_resilience.quarantine.state import TrustState
from orbital_ai_resilience.recovery.policy import MigrationPolicy


class TargetSelector:
    """Evaluates candidate cluster nodes and selects the optimal target destination."""

    def __init__(
        self,
        policy: Optional[MigrationPolicy] = None,
        behavior_evaluator: Optional[BehavioralScoreEvaluator] = None,
        quarantine_manager: Optional[QuarantineManager] = None,
    ) -> None:
        self.policy: MigrationPolicy = policy or MigrationPolicy()
        self.behavior_evaluator: BehavioralScoreEvaluator = behavior_evaluator or BehavioralScoreEvaluator()
        self.quarantine_manager: Optional[QuarantineManager] = quarantine_manager

    def calculate_target_score(self, node: VirtualNode, workload: Workload) -> float:
        """Calculate transparent weighted suitability score for a candidate target node."""
        p = self.policy

        s_phys = node.get_health_score()
        s_behav = 100.0  # Target candidate nodes are checked for clean baseline AI status
        s_cap = (node.get_available_compute() / node.compute_capacity) * 100.0
        s_power = max(0.0, min(100.0, node.power_level))
        penalty_latency = node.latency

        score = (
            p.weight_health * s_phys
            + p.weight_behavior * s_behav
            + p.weight_capacity * s_cap
            + p.weight_power * s_power
            - p.weight_latency * penalty_latency
        )

        return round(max(0.0, score), 2)

    def is_eligible_target(
        self,
        node: VirtualNode,
        source_node_id: str,
        workload: Workload,
        quarantine_manager: Optional[QuarantineManager] = None,
    ) -> bool:
        """Check if candidate node satisfies all hard eligibility criteria.

        Excludes source node, OFFLINE nodes, DEGRADED nodes, ISOLATED nodes,
        QUARANTINED nodes, and nodes without sufficient resources.
        """
        p = self.policy
        qm = quarantine_manager or self.quarantine_manager

        # 1. Must not be source node
        if node.node_id == source_node_id:
            return False

        # 2. Must be ONLINE
        if node.status != NodeStatus.ONLINE:
            return False

        # 3. Must not be QUARANTINED or ISOLATED by QuarantineManager
        if qm is not None:
            trust = qm.get_trust_state(node.node_id)
            if trust in (TrustState.QUARANTINED, TrustState.ISOLATED):
                return False

        # 4. Must satisfy minimum physical health score and allowed physical state
        if node.get_health_score() < p.min_target_physical_health:
            return False

        if node.get_health_state() not in p.allowed_target_physical_states:
            return False

        # 5. Must have sufficient compute and memory capacity
        if workload.required_compute > node.get_available_compute():
            return False

        if workload.required_memory > node.get_available_memory():
            return False

        return True

    def select_best_target(
        self,
        cluster: VirtualCluster,
        source_node_id: str,
        workload: Workload,
        quarantine_manager: Optional[QuarantineManager] = None,
    ) -> Tuple[Optional[str], float, Dict[str, float]]:
        """Identify and return the highest-scoring eligible target node.

        Args:
            cluster: VirtualCluster instance.
            source_node_id: Source node ID initiating migration.
            workload: Workload to be migrated.
            quarantine_manager: Optional QuarantineManager instance.

        Returns:
            Tuple of (selected_node_id, best_score, candidate_scores_dict).
        """
        candidate_scores: Dict[str, float] = {}
        best_node_id: Optional[str] = None
        best_score: float = -1.0
        qm = quarantine_manager or self.quarantine_manager

        for node_id, node in cluster.nodes.items():
            if self.is_eligible_target(node, source_node_id, workload, quarantine_manager=qm):
                score = self.calculate_target_score(node, workload)
                candidate_scores[node_id] = score
                if score > best_score:
                    best_score = score
                    best_node_id = node_id

        return best_node_id, max(0.0, best_score), candidate_scores
