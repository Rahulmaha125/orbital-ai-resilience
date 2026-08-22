"""Recovery decision policies including deterministic baseline, adaptive policy, and experimental RL stub."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.core.node import VirtualNode
from orbital_ai_resilience.core.workload import Workload
from orbital_ai_resilience.optimization.cost_model import RecoveryCostModel
from orbital_ai_resilience.optimization.features import OptimizationFeatureBuilder, TargetNodeFeatures
from orbital_ai_resilience.optimization.history import RecoveryHistory
from orbital_ai_resilience.quarantine.manager import QuarantineManager
from orbital_ai_resilience.recovery.policy import MigrationPolicy
from orbital_ai_resilience.recovery.selector import TargetSelector


class BaseRecoveryPolicy(ABC):
    """Abstract base class interface for target node recovery policies."""

    def __init__(self, name: str) -> None:
        self.name: str = name

    @abstractmethod
    def select_target(
        self,
        cluster: VirtualCluster,
        source_node_id: str,
        workload: Workload,
        history: RecoveryHistory,
        quarantine_manager: Optional[QuarantineManager] = None,
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """Select optimal destination target node for workload recovery.

        Returns:
            Tuple of (selected_node_id, score, decision_explanation_dict).
        """
        pass


class DeterministicBaselinePolicy(BaseRecoveryPolicy):
    """Phase 5 Deterministic TargetSelector policy acting as benchmark baseline."""

    def __init__(self, selector: Optional[TargetSelector] = None) -> None:
        super().__init__(name="DeterministicBaselinePolicy")
        self.selector: TargetSelector = selector or TargetSelector()

    def select_target(
        self,
        cluster: VirtualCluster,
        source_node_id: str,
        workload: Workload,
        history: RecoveryHistory,
        quarantine_manager: Optional[QuarantineManager] = None,
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        best_id, best_score, candidate_scores = self.selector.select_best_target(
            cluster=cluster,
            source_node_id=source_node_id,
            workload=workload,
            quarantine_manager=quarantine_manager,
        )
        explanation = {
            "policy_type": "Deterministic Baseline",
            "candidate_scores": candidate_scores,
            "selected_node": best_id,
            "score": best_score,
            "reason": "Highest deterministic weighted score under Phase 5 TargetSelector",
        }
        return best_id, best_score, explanation


class AdaptiveRecoveryPolicy(BaseRecoveryPolicy):
    """Phase 8 Adaptive Target Policy adjusting node scoring using recovery history, cost model & risk."""

    def __init__(
        self,
        policy: Optional[MigrationPolicy] = None,
        cost_model: Optional[RecoveryCostModel] = None,
        feature_builder: Optional[OptimizationFeatureBuilder] = None,
    ) -> None:
        super().__init__(name="AdaptiveRecoveryPolicy")
        self.policy: MigrationPolicy = policy or MigrationPolicy()
        self.cost_model: RecoveryCostModel = cost_model or RecoveryCostModel()
        self.feature_builder: OptimizationFeatureBuilder = feature_builder or OptimizationFeatureBuilder()
        self.selector: TargetSelector = TargetSelector(policy=self.policy)

    def select_target(
        self,
        cluster: VirtualCluster,
        source_node_id: str,
        workload: Workload,
        history: RecoveryHistory,
        quarantine_manager: Optional[QuarantineManager] = None,
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """Select safest and most efficient target node using adaptive feature scoring."""
        best_node_id: Optional[str] = None
        best_score: float = -1e9
        breakdowns: Dict[str, Any] = {}

        for node_id, node in cluster.nodes.items():
            if self.selector.is_eligible_target(node, source_node_id, workload, quarantine_manager=quarantine_manager):
                # 1. Build features & cost model
                feats = self.feature_builder.build_features(node, workload, history)
                cost = self.cost_model.calculate_cost(feats)

                # 2. Adaptive Score Calculation:
                # Base score + Reliability bonus + Verification bonus - Cost penalty - Quarantine penalty
                base_score = self.selector.calculate_target_score(node, workload)
                rel_bonus = 0.30 * feats.node_reliability_score
                v_bonus = 15.0 * feats.previous_verification_success_rate
                cost_penalty = 0.50 * cost.total_cost
                q_penalty = 20.0 * feats.previous_quarantine_count

                adaptive_score = base_score + rel_bonus + v_bonus - cost_penalty - q_penalty
                adaptive_score = round(max(0.0, adaptive_score), 2)

                # 3. Expected Risk Assessment
                if feats.physical_health_score >= 95.0 and feats.node_reliability_score >= 90.0 and feats.previous_quarantine_count == 0:
                    risk_level = "LOW"
                elif feats.physical_health_score >= 90.0 and feats.node_reliability_score >= 70.0:
                    risk_level = "MED"
                else:
                    risk_level = "HIGH"

                breakdowns[node_id] = {
                    "adaptive_score": adaptive_score,
                    "base_score": base_score,
                    "reliability_score": feats.node_reliability_score,
                    "verification_success_rate": feats.previous_verification_success_rate,
                    "total_cost": cost.total_cost,
                    "risk_level": risk_level,
                    "features": feats.to_dict(),
                    "cost_breakdown": cost.to_dict(),
                }

                if adaptive_score > best_score:
                    best_score = adaptive_score
                    best_node_id = node_id

        selected_explanation = breakdowns.get(best_node_id, {}) if best_node_id else {}
        explanation = {
            "policy_type": "Adaptive Optimization Policy",
            "selected_node": best_node_id,
            "adaptive_score": best_score,
            "risk_level": selected_explanation.get("risk_level", "UNKNOWN"),
            "candidate_breakdowns": breakdowns,
            "why_selected": (
                f"Selected {best_node_id} (Adaptive Score: {best_score}) due to high physical health "
                f"({selected_explanation.get('features', {}).get('physical_health_score', 0)}), "
                f"reliability ({selected_explanation.get('features', {}).get('node_reliability_score', 0)}), "
                f"and low recovery risk ({selected_explanation.get('risk_level', 'UNKNOWN')})."
            )
            if best_node_id
            else "No eligible target node found",
        }

        return best_node_id, max(0.0, best_score), explanation


class RLRecoveryPolicy(BaseRecoveryPolicy):
    """EXPERIMENTAL ONLY: Reinforcement Learning policy stub interface.

    Explicitly marked EXPERIMENTAL. Does not replace production baseline or adaptive policies.
    """

    def __init__(self) -> None:
        super().__init__(name="RLRecoveryPolicy_EXPERIMENTAL")
        self.is_experimental: bool = True

    def select_target(
        self,
        cluster: VirtualCluster,
        source_node_id: str,
        workload: Workload,
        history: RecoveryHistory,
        quarantine_manager: Optional[QuarantineManager] = None,
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        # Experimental fallback delegates to Adaptive policy
        adaptive = AdaptiveRecoveryPolicy()
        target_id, score, exp = adaptive.select_target(cluster, source_node_id, workload, history, quarantine_manager)
        exp["policy_type"] = "RL Policy (EXPERIMENTAL STUB)"
        return target_id, score, exp
