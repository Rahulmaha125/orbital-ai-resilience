"""Unit and integration tests for Phase 8: Intelligent Decision-Making & Adaptive Recovery Optimization."""

import numpy as np
import unittest
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.core.node import VirtualNode
from orbital_ai_resilience.core.types import NodeStatus
from orbital_ai_resilience.core.workload import Workload
from orbital_ai_resilience.optimization.benchmark import OptimizationBenchmark
from orbital_ai_resilience.optimization.cost_model import RecoveryCostModel
from orbital_ai_resilience.optimization.features import OptimizationFeatureBuilder, TargetNodeFeatures
from orbital_ai_resilience.optimization.history import RecoveryHistory
from orbital_ai_resilience.optimization.optimizer import AdaptiveRecoveryOptimizer
from orbital_ai_resilience.optimization.policy import (
    AdaptiveRecoveryPolicy,
    DeterministicBaselinePolicy,
    RLRecoveryPolicy,
)
from orbital_ai_resilience.optimization.reward import RewardCalculator
from orbital_ai_resilience.quarantine.manager import QuarantineManager
from orbital_ai_resilience.verification.types import VerificationResultState
from orbital_ai_resilience.workloads.synthetic import SyntheticAIWorkload


class TestPhase8AdaptiveOptimization(unittest.TestCase):
    """Test suite for Phase 8 feature engineering, cost model, reward calculation, and adaptive policies."""

    def setUp(self) -> None:
        self.cluster = VirtualCluster.create_default_cluster(num_nodes=5)
        self.workload = SyntheticAIWorkload(name="opt_test_task", seed=42)
        self.history = RecoveryHistory()
        self.cost_model = RecoveryCostModel()
        self.feature_builder = OptimizationFeatureBuilder()
        self.reward_calculator = RewardCalculator()
        self.adaptive_policy = AdaptiveRecoveryPolicy(cost_model=self.cost_model, feature_builder=self.feature_builder)
        self.baseline_policy = DeterministicBaselinePolicy()
        self.quarantine_manager = QuarantineManager()

    def test_feature_builder_normalization(self) -> None:
        """Verify OptimizationFeatureBuilder constructs normalized features from node & workload."""
        n1 = self.cluster.get_node("node-1")
        feats = self.feature_builder.build_features(n1, self.workload, self.history)

        self.assertEqual(feats.node_id, "node-1")
        self.assertEqual(feats.physical_health_score, 100.0)
        self.assertEqual(feats.previous_verification_success_rate, 1.0)
        self.assertEqual(feats.node_reliability_score, 100.0)

        vec = feats.to_numpy()
        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(len(vec), 13)

    def test_cost_model_breakdown(self) -> None:
        """Verify RecoveryCostModel calculates total cost and individual component breakdowns."""
        n1 = self.cluster.get_node("node-1")
        feats = self.feature_builder.build_features(n1, self.workload, self.history)
        cost_breakdown = self.cost_model.calculate_cost(feats)

        self.assertGreater(cost_breakdown.total_cost, 0.0)
        self.assertGreater(cost_breakdown.migration_cost, 0.0)
        self.assertGreater(cost_breakdown.communication_cost, 0.0)
        self.assertGreater(cost_breakdown.verification_cost, 0.0)
        self.assertEqual(
            round(cost_breakdown.total_cost, 2),
            round(
                cost_breakdown.migration_cost
                + cost_breakdown.communication_cost
                + cost_breakdown.compute_cost
                + cost_breakdown.power_cost
                + cost_breakdown.risk_cost
                + cost_breakdown.verification_cost,
                2,
            ),
        )

    def test_reward_calculator(self) -> None:
        """Verify RewardCalculator evaluates positive rewards for success and penalties for loss/quarantine."""
        # 1. Success case
        reward_success = self.reward_calculator.calculate_reward(
            is_success=True,
            verification_state=VerificationResultState.VERIFIED,
            was_quarantined=False,
            recovery_cost=10.0,
        )
        self.assertGreater(reward_success.total_reward, 100.0)

        # 2. Failure case with quarantine & workload loss
        reward_fail = self.reward_calculator.calculate_reward(
            is_success=False,
            verification_state=VerificationResultState.VERIFICATION_FAILED,
            was_quarantined=True,
            recovery_cost=10.0,
            is_workload_lost=True,
        )
        self.assertLess(reward_fail.total_reward, -200.0)

    def test_recovery_history_reliability(self) -> None:
        """Verify RecoveryHistory tracks node verification outcomes and calculates reliability scores."""
        # Node-1 passes 2x
        self.history.record_outcome("node-1", VerificationResultState.VERIFIED, was_quarantined=False)
        self.history.record_outcome("node-1", VerificationResultState.VERIFIED, was_quarantined=False)
        self.assertEqual(self.history.get_verification_success_rate("node-1"), 1.0)
        self.assertEqual(self.history.get_node_reliability_score("node-1"), 100.0)

        # Node-2 fails 2x and was quarantined 2x
        self.history.record_outcome("node-2", VerificationResultState.VERIFICATION_FAILED, was_quarantined=True)
        self.history.record_outcome("node-2", VerificationResultState.VERIFICATION_FAILED, was_quarantined=True)
        self.assertEqual(self.history.get_verification_success_rate("node-2"), 0.0)
        self.assertEqual(self.history.get_node_reliability_score("node-2"), 0.0)

    def test_adaptive_policy_selection(self) -> None:
        """Verify AdaptiveRecoveryPolicy selects optimal candidate node and produces explainable breakdowns."""
        best_id, score, explanation = self.adaptive_policy.select_target(
            cluster=self.cluster,
            source_node_id="node-3",
            workload=self.workload,
            history=self.history,
            quarantine_manager=self.quarantine_manager,
        )
        self.assertIsNotNone(best_id)
        self.assertNotEqual(best_id, "node-3")
        self.assertIn("candidate_breakdowns", explanation)
        self.assertIn("why_selected", explanation)

    def test_unreliable_target_rejection(self) -> None:
        """Verify AdaptiveRecoveryPolicy rejects historically unreliable target nodes in favor of clean targets."""
        # Record 3 failures for Node-1 in history
        for _ in range(3):
            self.history.record_outcome("node-1", VerificationResultState.VERIFICATION_FAILED, was_quarantined=True)

        best_id, score, explanation = self.adaptive_policy.select_target(
            cluster=self.cluster,
            source_node_id="node-3",
            workload=self.workload,
            history=self.history,
            quarantine_manager=self.quarantine_manager,
        )
        # Adaptive policy should avoid Node-1 and select a clean node (e.g. Node-2 or Node-4)
        self.assertNotEqual(best_id, "node-1")

    def test_unsafe_node_rejection(self) -> None:
        """Verify AdaptiveRecoveryPolicy never selects OFFLINE, DEGRADED, ISOLATED, or QUARANTINED nodes."""
        n1 = self.cluster.get_node("node-1")
        self.quarantine_manager.quarantine_node(n1, reason="Test Quarantine")

        n2 = self.cluster.get_node("node-2")
        n2.set_status(NodeStatus.DEGRADED)

        best_id, score, exp = self.adaptive_policy.select_target(
            cluster=self.cluster,
            source_node_id="node-3",
            workload=self.workload,
            history=self.history,
            quarantine_manager=self.quarantine_manager,
        )
        self.assertNotIn(best_id, ["node-1", "node-2", "node-3"])

    def test_resource_constraint_rejection(self) -> None:
        """Verify candidate targets without sufficient compute/memory capacity are rejected."""
        w_huge = SyntheticAIWorkload(name="huge_task", required_compute=500.0, required_memory=65536.0)
        best_id, score, exp = self.adaptive_policy.select_target(
            cluster=self.cluster,
            source_node_id="node-3",
            workload=w_huge,
            history=self.history,
            quarantine_manager=self.quarantine_manager,
        )
        self.assertIsNone(best_id)

    def test_deterministic_reproducibility(self) -> None:
        """Verify AdaptiveRecoveryPolicy decision ranking is 100% deterministic and reproducible."""
        res1 = self.adaptive_policy.select_target(self.cluster, "node-3", self.workload, self.history, self.quarantine_manager)
        res2 = self.adaptive_policy.select_target(self.cluster, "node-3", self.workload, self.history, self.quarantine_manager)
        self.assertEqual(res1[0], res2[0])
        self.assertEqual(res1[1], res2[1])

    def test_rl_policy_experimental_stub(self) -> None:
        """Verify RLRecoveryPolicy is explicitly marked EXPERIMENTAL and functions safely."""
        rl_policy = RLRecoveryPolicy()
        self.assertTrue(rl_policy.is_experimental)
        best_id, score, exp = rl_policy.select_target(self.cluster, "node-3", self.workload, self.history, self.quarantine_manager)
        self.assertIsNotNone(best_id)
        self.assertIn("EXPERIMENTAL", exp["policy_type"])

    def test_optimization_benchmark_suite(self) -> None:
        """Verify OptimizationBenchmark runs full 10-scenario 2-policy comparison suite."""
        bm = OptimizationBenchmark(seed=42)
        results, improvements = bm.run_full_benchmark_suite()
        self.assertEqual(len(results), 20)  # 10 scenarios * 2 policies
        self.assertIn("recovery_cost_reduction_pct", improvements)
        self.assertIn("communication_cost_reduction_pct", improvements)
        self.assertIn("total_reward_improvement_pct", improvements)

    def test_optimizer_integration(self) -> None:
        """Verify AdaptiveRecoveryOptimizer coordinates baseline vs. adaptive selection and history updates."""
        optimizer = AdaptiveRecoveryOptimizer()
        b_id, b_sc, b_exp = optimizer.select_target_node(self.cluster, "node-3", self.workload, policy_name="baseline")
        a_id, a_sc, a_exp = optimizer.select_target_node(self.cluster, "node-3", self.workload, policy_name="adaptive")

        self.assertIsNotNone(b_id)
        self.assertIsNotNone(a_id)
        self.assertEqual(b_exp["policy_type"], "Deterministic Baseline")
        self.assertEqual(a_exp["policy_type"], "Adaptive Optimization Policy")


if __name__ == "__main__":
    unittest.main()
