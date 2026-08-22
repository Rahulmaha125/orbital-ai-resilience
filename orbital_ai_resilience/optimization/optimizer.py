"""AdaptiveRecoveryOptimizer coordinating intelligent policy selection, target ranking & reward evaluation."""

from typing import Any, Dict, Optional, Tuple
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.core.workload import Workload
from orbital_ai_resilience.optimization.cost_model import RecoveryCostModel
from orbital_ai_resilience.optimization.features import OptimizationFeatureBuilder
from orbital_ai_resilience.optimization.history import RecoveryHistory
from orbital_ai_resilience.optimization.policy import (
    AdaptiveRecoveryPolicy,
    BaseRecoveryPolicy,
    DeterministicBaselinePolicy,
)
from orbital_ai_resilience.optimization.reward import RewardBreakdown, RewardCalculator
from orbital_ai_resilience.quarantine.manager import QuarantineManager
from orbital_ai_resilience.verification.types import VerificationResultState


class AdaptiveRecoveryOptimizer:
    """Orchestrates adaptive recovery optimization, cost modeling, and empirical decision evaluation."""

    def __init__(
        self,
        baseline_policy: Optional[DeterministicBaselinePolicy] = None,
        adaptive_policy: Optional[AdaptiveRecoveryPolicy] = None,
        history: Optional[RecoveryHistory] = None,
        cost_model: Optional[RecoveryCostModel] = None,
        reward_calculator: Optional[RewardCalculator] = None,
    ) -> None:
        self.history: RecoveryHistory = history or RecoveryHistory()
        self.baseline_policy: DeterministicBaselinePolicy = baseline_policy or DeterministicBaselinePolicy()
        self.cost_model: RecoveryCostModel = cost_model or RecoveryCostModel()
        self.adaptive_policy: AdaptiveRecoveryPolicy = adaptive_policy or AdaptiveRecoveryPolicy(cost_model=self.cost_model)
        self.reward_calculator: RewardCalculator = reward_calculator or RewardCalculator()
        self.feature_builder: OptimizationFeatureBuilder = OptimizationFeatureBuilder()

    def select_target_node(
        self,
        cluster: VirtualCluster,
        source_node_id: str,
        workload: Workload,
        policy_name: str = "adaptive",
        quarantine_manager: Optional[QuarantineManager] = None,
    ) -> Tuple[Optional[str], float, Dict[str, Any]]:
        """Select destination node using requested policy ('baseline' or 'adaptive').

        Returns:
            Tuple of (selected_node_id, score, explanation_dict).
        """
        policy: BaseRecoveryPolicy = self.baseline_policy if policy_name.lower() == "baseline" else self.adaptive_policy
        return policy.select_target(
            cluster=cluster,
            source_node_id=source_node_id,
            workload=workload,
            history=self.history,
            quarantine_manager=quarantine_manager,
        )

    def record_outcome(
        self,
        node_id: str,
        verification_result: VerificationResultState,
        was_quarantined: bool,
        duration_sec: float = 0.0,
        workload_id: str = "unknown",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record target node recovery outcome into history."""
        self.history.record_outcome(
            node_id=node_id,
            verification_result=verification_result,
            was_quarantined=was_quarantined,
            duration_sec=duration_sec,
            workload_id=workload_id,
            details=details,
        )

    def compute_reward(
        self,
        is_success: bool,
        verification_state: VerificationResultState,
        was_quarantined: bool,
        recovery_cost: float,
        is_workload_lost: bool = False,
    ) -> RewardBreakdown:
        """Calculate reward breakdown for an evaluated recovery outcome."""
        return self.reward_calculator.calculate_reward(
            is_success=is_success,
            verification_state=verification_state,
            was_quarantined=was_quarantined,
            recovery_cost=recovery_cost,
            is_workload_lost=is_workload_lost,
        )
