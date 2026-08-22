"""Unit and integration tests for Phase 6: Output Verification, Quarantine & Multi-Stage Recovery."""

import numpy as np
import unittest
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.core.node import VirtualNode
from orbital_ai_resilience.core.types import NodeStatus
from orbital_ai_resilience.core.workload import Workload
from orbital_ai_resilience.detection.statistical import StatisticalDetector
from orbital_ai_resilience.faults.injector import FaultInjector
from orbital_ai_resilience.faults.profile import FaultProfile
from orbital_ai_resilience.faults.types import FaultType
from orbital_ai_resilience.quarantine.manager import QuarantineManager
from orbital_ai_resilience.quarantine.state import TrustState
from orbital_ai_resilience.recovery.migration import MigrationManager
from orbital_ai_resilience.recovery.policy import MigrationPolicy
from orbital_ai_resilience.recovery.selector import TargetSelector
from orbital_ai_resilience.recovery.types import MigrationState, VerificationStatus
from orbital_ai_resilience.verification.evidence import VerificationEvidence
from orbital_ai_resilience.verification.policy import VerificationPolicy
from orbital_ai_resilience.verification.reference import ReferenceProvider
from orbital_ai_resilience.verification.types import VerificationResultState
from orbital_ai_resilience.verification.verifier import OutputVerifier
from orbital_ai_resilience.workloads.synthetic import SyntheticAIWorkload


class TestPhase6VerificationAndQuarantine(unittest.TestCase):
    """Test suite for Phase 6 output verification, node quarantine, and multi-stage recovery."""

    def setUp(self) -> None:
        self.cluster = VirtualCluster.create_default_cluster(num_nodes=5)
        self.policy = MigrationPolicy()
        self.v_policy = VerificationPolicy(mse_max=0.001, mae_max=0.01, cosine_min=0.999)
        self.verifier = OutputVerifier(policy=self.v_policy)
        self.quarantine_mgr = QuarantineManager()
        self.selector = TargetSelector(policy=self.policy, quarantine_manager=self.quarantine_mgr)
        self.manager = MigrationManager(
            policy=self.policy,
            selector=self.selector,
            verifier=self.verifier,
            quarantine_manager=self.quarantine_mgr,
        )
        self.detector = StatisticalDetector()

    def test_output_verifier_normal_and_failure(self) -> None:
        """Verify OutputVerifier evaluates MSE, MAE, Cosine Similarity and generates SHA-256 evidence."""
        w = SyntheticAIWorkload(name="task_verify", seed=42)
        ref_out = w.compute_reference_output()

        # 1. Normal clean output
        v_state, evidence = self.verifier.verify_target_output(w, ref_out, "node-3", "node-1")
        self.assertEqual(v_state, VerificationResultState.VERIFIED)
        self.assertEqual(evidence.mse, 0.0)
        self.assertEqual(evidence.cosine_sim, 1.0)
        self.assertIsNotNone(evidence.reference_output_hash)
        self.assertEqual(evidence.reference_output_hash, evidence.target_output_hash)

        # 2. Corrupted target output
        corrupted_out = ref_out + 0.10
        v_state_fail, evidence_fail = self.verifier.verify_target_output(w, corrupted_out, "node-3", "node-1")
        self.assertEqual(v_state_fail, VerificationResultState.VERIFICATION_FAILED)
        self.assertGreater(evidence_fail.mse, 0.001)
        self.assertNotEqual(evidence_fail.reference_output_hash, evidence_fail.target_output_hash)

    def test_reference_provider_trusted_node_rule(self) -> None:
        """Verify ReferenceProvider rejects reference calculation from DEGRADED or ISOLATED nodes."""
        ref_provider = ReferenceProvider()
        w = SyntheticAIWorkload(name="task_ref", seed=42)

        node_degraded = self.cluster.get_node("node-5")
        node_degraded.set_status(NodeStatus.DEGRADED)

        with self.assertRaises(ValueError):
            ref_provider.get_reference_output(w, reference_node=node_degraded)

    def test_quarantine_manager_isolation_and_trust_state(self) -> None:
        """Verify QuarantineManager manages node TrustState separate from NodeStatus."""
        n1 = self.cluster.get_node("node-1")
        self.assertTrue(self.quarantine_mgr.is_node_trusted("node-1"))

        # Quarantine node-1
        ev = self.quarantine_mgr.quarantine_node(n1, reason="Verification FAILED")
        self.assertEqual(n1.status, NodeStatus.ONLINE)  # Physical status remains ONLINE!
        self.assertEqual(self.quarantine_mgr.get_trust_state("node-1"), TrustState.QUARANTINED)  # TrustState is QUARANTINED!
        self.assertFalse(self.quarantine_mgr.is_node_trusted("node-1"))

    def test_target_selector_excludes_quarantined_nodes(self) -> None:
        """Verify TargetSelector excludes QUARANTINED nodes from candidate selection."""
        n1 = self.cluster.get_node("node-1")
        w = Workload(name="job", required_compute=10.0, required_memory=1024.0)

        # Quarantine node-1
        self.quarantine_mgr.quarantine_node(n1, reason="Test Quarantine")

        best_id, score, candidate_scores = self.selector.select_best_target(self.cluster, source_node_id="node-3", workload=w)
        self.assertNotEqual(best_id, "node-1")
        self.assertNotIn("node-1", candidate_scores)

    def test_cascading_failure_and_multi_stage_recovery(self) -> None:
        """CRITICAL PHASE 6 DEMO TEST: Cascading Failure Recovery (Scenario C).

        Scenario:
        - Node-3: Source node with silent AI degradation.
        - Node-1: Initial target selected, but produces corrupted output (Attempt 1 fails).
        - TargetSelector quarantines Node-1, re-evaluates remaining trusted nodes, selects Node-2.
        - Node-2: Target executes cleanly (Attempt 2 succeeds).
        - Node-3 isolated, Node-1 quarantined, Node-2 trusted. Recovery COMPLETED!
        """
        # Node-5 degraded physically
        n5 = self.cluster.get_node("node-5")
        n5.set_status(NodeStatus.DEGRADED)

        n3 = self.cluster.get_node("node-3")
        w_task = SyntheticAIWorkload(name="cascading_task", seed=42)
        n3.assign_workload(w_task)

        # Setup FaultInjector:
        # 1. Silent degradation on source node-3
        # 2. Memory bit-flip/corruption fault on target node-1!
        injector = FaultInjector()
        injector.add_profile(
            FaultProfile(
                fault_type=FaultType.SILENT_MODEL_DEGRADATION,
                target_node_id="node-3",
                intensity=0.15,
                seed=42,
            )
        )
        injector.add_profile(
            FaultProfile(
                fault_type=FaultType.MEMORY_BIT_FLIP,
                target_node_id="node-1",  # Target node-1 produces corrupted output!
                intensity=0.20,
                seed=99,
            )
        )

        # Detect silent degradation on Node-3
        exec_log = w_task.execute_on_node(n3, fault_injector=injector, tick=0)
        det_result = self.detector.evaluate(exec_log)
        self.assertTrue(det_result.is_silent_degradation)

        # Execute Multi-Stage Recovery
        event = self.manager.execute_autonomous_recovery(
            cluster=self.cluster,
            source_node_id="node-3",
            workload=w_task,
            detection_result=det_result,
            fault_injector=injector,
        )

        # Assertions for Multi-Stage Cascading Recovery:
        self.assertIsNotNone(event)
        self.assertEqual(event.migration_status, MigrationState.COMPLETED)
        self.assertEqual(event.verification_status, VerificationStatus.VERIFIED)
        
        # Verify Node-1 was quarantined after Attempt 1 failure!
        self.assertEqual(self.quarantine_mgr.get_trust_state("node-1"), TrustState.QUARANTINED)

        # Verify Node-2 was selected on Attempt 2 and succeeded!
        self.assertEqual(event.target_node_id, "node-2")
        self.assertEqual(event.migration_attempt, 2)

        # Verify Source Node-3 was isolated!
        self.assertEqual(n3.status, NodeStatus.ISOLATED)

        # Verify Recovery Metrics
        m = self.manager.metrics
        self.assertEqual(m.successful_migrations, 1)
        self.assertEqual(m.verification_failures, 1)
        self.assertEqual(m.quarantined_nodes, 1)
        self.assertEqual(m.recovery_retries, 1)
        self.assertEqual(m.successful_retry_recoveries, 1)
        self.assertEqual(m.cascading_failures_recovered, 1)

    def test_all_targets_failing_aborts_safely(self) -> None:
        """Verify recovery aborts safely after max_attempts if all available targets fail output verification."""
        n3 = self.cluster.get_node("node-3")
        w_task = SyntheticAIWorkload(name="all_fail_task", seed=42)
        n3.assign_workload(w_task)

        # Inject corruption on source node-3 AND ALL potential target nodes (node-1, node-2, node-4)
        injector = FaultInjector()
        injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.15))
        injector.add_profile(FaultProfile(fault_type=FaultType.OUTPUT_DRIFT, target_node_id="node-1", intensity=0.50))
        injector.add_profile(FaultProfile(fault_type=FaultType.OUTPUT_DRIFT, target_node_id="node-2", intensity=0.50))
        injector.add_profile(FaultProfile(fault_type=FaultType.OUTPUT_DRIFT, target_node_id="node-4", intensity=0.50))

        # Node 5 degraded
        self.cluster.get_node("node-5").set_status(NodeStatus.DEGRADED)

        exec_log = w_task.execute_on_node(n3, fault_injector=injector, tick=0)
        det_result = self.detector.evaluate(exec_log)

        event = self.manager.execute_autonomous_recovery(self.cluster, "node-3", w_task, det_result, fault_injector=injector)

        self.assertIsNotNone(event)
        self.assertEqual(event.migration_status, MigrationState.FAILED)
        self.assertEqual(self.manager.metrics.aborted_recoveries, 1)
        self.assertGreaterEqual(self.manager.metrics.quarantined_nodes, 3)


if __name__ == "__main__":
    unittest.main()
