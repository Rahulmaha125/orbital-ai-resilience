"""Unit and integration tests for Phase 5: Autonomous Workload Migration & Recovery."""

import unittest
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.core.node import VirtualNode
from orbital_ai_resilience.core.types import NodeStatus
from orbital_ai_resilience.core.workload import Workload
from orbital_ai_resilience.detection.statistical import StatisticalDetector
from orbital_ai_resilience.faults.injector import FaultInjector
from orbital_ai_resilience.faults.profile import FaultProfile
from orbital_ai_resilience.faults.types import FaultType
from orbital_ai_resilience.recovery.migration import MigrationManager
from orbital_ai_resilience.recovery.policy import MigrationPolicy
from orbital_ai_resilience.recovery.selector import TargetSelector
from orbital_ai_resilience.recovery.state import WorkloadSnapshot
from orbital_ai_resilience.recovery.types import MigrationState, VerificationStatus
from orbital_ai_resilience.workloads.synthetic import SyntheticAIWorkload


class TestPhase5Recovery(unittest.TestCase):
    """Comprehensive test suite for Phase 5 autonomous recovery and workload migration."""

    def setUp(self) -> None:
        self.cluster = VirtualCluster.create_default_cluster(num_nodes=5)
        self.policy = MigrationPolicy()
        self.selector = TargetSelector(policy=self.policy)
        self.manager = MigrationManager(policy=self.policy, selector=self.selector)
        self.detector = StatisticalDetector()

    def test_migration_trigger(self) -> None:
        """Verify migration policy triggers on silent degradation or degraded behavioral state."""
        node = self.cluster.get_node("node-3")
        injector = FaultInjector()
        injector.add_profile(
            FaultProfile(
                fault_type=FaultType.SILENT_MODEL_DEGRADATION,
                target_node_id="node-3",
                intensity=0.15,
                seed=42,
            )
        )
        w = SyntheticAIWorkload(name="task1", seed=42)
        exec_log = w.execute_on_node(node, fault_injector=injector, tick=0)
        det_result = self.detector.evaluate(exec_log)

        self.assertTrue(self.policy.should_migrate(det_result))

    def test_target_eligibility_and_rejection(self) -> None:
        """Verify candidate target node eligibility rules."""
        w = Workload(name="job", required_compute=30.0, required_memory=2048.0)
        
        node_healthy = self.cluster.get_node("node-1")
        self.assertTrue(self.selector.is_eligible_target(node_healthy, source_node_id="node-3", workload=w))

        # Rejection: Source node itself
        self.assertFalse(self.selector.is_eligible_target(node_healthy, source_node_id="node-1", workload=w))

        # Rejection: Unhealthy / degraded physical status
        node_healthy.set_status(NodeStatus.DEGRADED)
        self.assertFalse(self.selector.is_eligible_target(node_healthy, source_node_id="node-3", workload=w))

        # Rejection: Insufficient compute capacity
        w_huge = Workload(name="huge", required_compute=500.0, required_memory=2048.0)
        node_online = self.cluster.get_node("node-2")
        self.assertFalse(self.selector.is_eligible_target(node_online, source_node_id="node-3", workload=w_huge))

    def test_target_scoring_and_best_selection(self) -> None:
        """Verify target scoring ranks higher-capacity, lower-latency healthy nodes highest."""
        n1 = self.cluster.get_node("node-1")  # temp 42, latency 6
        n2 = self.cluster.get_node("node-2")  # temp 44, latency 7
        w = Workload(name="job", required_compute=10.0, required_memory=1024.0)

        best_id, best_score, scores = self.selector.select_best_target(self.cluster, source_node_id="node-3", workload=w)
        self.assertEqual(best_id, "node-1")  # node-1 has lower latency and temp
        self.assertGreater(scores["node-1"], scores["node-2"])

    def test_workload_state_snapshot_transfer(self) -> None:
        """Verify WorkloadSnapshot creates valid transferable snapshot."""
        w = SyntheticAIWorkload(name="task_snap", seed=42)
        snap = WorkloadSnapshot.create_from_workload(w, source_node_id="node-3", reason="Test_Snapshot")

        self.assertEqual(snap.workload_id, w.workload_id)
        self.assertEqual(snap.source_node_id, "node-3")
        self.assertEqual(snap.migration_count, 1)

    def test_no_target_available_scenario(self) -> None:
        """Verify graceful failure handling when no eligible target node is available."""
        # Mark all other nodes OFFLINE
        for nid, n in self.cluster.nodes.items():
            if nid != "node-3":
                n.set_status(NodeStatus.OFFLINE)

        w = SyntheticAIWorkload(name="task_fail", seed=42)
        n3 = self.cluster.get_node("node-3")
        n3.assign_workload(w)

        # Trigger detection
        injector = FaultInjector()
        injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.15))
        exec_log = w.execute_on_node(n3, fault_injector=injector, tick=0)
        det_result = self.detector.evaluate(exec_log)

        event = self.manager.execute_autonomous_recovery(self.cluster, "node-3", w, det_result)

        self.assertIsNotNone(event)
        self.assertEqual(event.migration_status, MigrationState.FAILED)
        self.assertEqual(self.manager.metrics.failed_migrations, 1)

    def test_duplicate_ownership_prevention(self) -> None:
        """Verify duplicate migration attempts for the same workload are prevented."""
        w = SyntheticAIWorkload(name="task_dup", seed=42)
        n3 = self.cluster.get_node("node-3")
        n3.assign_workload(w)

        injector = FaultInjector()
        injector.add_profile(FaultProfile(fault_type=FaultType.SILENT_MODEL_DEGRADATION, target_node_id="node-3", intensity=0.15))
        exec_log = w.execute_on_node(n3, fault_injector=injector, tick=0)
        det_result = self.detector.evaluate(exec_log)

        # First migration attempt succeeds
        ev1 = self.manager.execute_autonomous_recovery(self.cluster, "node-3", w, det_result)
        self.assertIsNotNone(ev1)
        self.assertEqual(ev1.migration_status, MigrationState.COMPLETED)

        # Second migration attempt for same workload returns None
        ev2 = self.manager.execute_autonomous_recovery(self.cluster, "node-3", w, det_result)
        self.assertIsNone(ev2)

    def test_full_5node_multi_node_recovery_scenario(self) -> None:
        """CRITICAL END-TO-END DEMO TEST: 5-Node Cluster Autonomous Recovery.

        Scenario Setup:
        - Node-1: Healthy (ONLINE, low temp, low latency)
        - Node-2: Healthy (ONLINE)
        - Node-3: Silently Degraded AI Node (ONLINE, physical health 100, AI output corrupted)
        - Node-4: Healthy (ONLINE)
        - Node-5: Physically Degraded Node (DEGRADED, high temp 75C)

        Workflow:
        1. Workload assigned to Node-3.
        2. Silent degradation detected by Phase 4.
        3. Target selection evaluates nodes, rejects Node-5 and Node-3, selects best target (Node-1).
        4. Workload migrated to Node-1.
        5. Workload executed on Node-1 cleanly.
        6. Output verified (VERIFIED).
        7. Node-3 marked ISOLATED.
        """
        # Set Node-5 as physically DEGRADED
        n5 = self.cluster.get_node("node-5")
        n5.update_telemetry(temperature=78.0)
        n5.set_status(NodeStatus.DEGRADED)

        n3 = self.cluster.get_node("node-3")
        w_ai = SyntheticAIWorkload(name="orbital_earth_observation_task", seed=42)
        n3.assign_workload(w_ai)

        # Inject silent model degradation on Node-3
        injector = FaultInjector()
        injector.add_profile(
            FaultProfile(
                fault_type=FaultType.SILENT_MODEL_DEGRADATION,
                target_node_id="node-3",
                start_tick=0,
                duration=10,
                intensity=0.15,
                seed=42,
            )
        )

        # Phase 3: Execute on Node-3
        exec_log = w_ai.execute_on_node(n3, fault_injector=injector, tick=0)

        # Phase 4: Detect silent degradation
        det_result = self.detector.evaluate(exec_log)
        self.assertTrue(det_result.is_silent_degradation)

        # Phase 5: Execute Autonomous Recovery
        event = self.manager.execute_autonomous_recovery(
            cluster=self.cluster,
            source_node_id="node-3",
            workload=w_ai,
            detection_result=det_result,
            fault_injector=injector,
        )

        # Assertions
        self.assertIsNotNone(event)
        self.assertEqual(event.migration_status, MigrationState.COMPLETED)
        self.assertEqual(event.verification_status, VerificationStatus.VERIFIED)
        self.assertEqual(event.target_node_id, "node-1")
        self.assertEqual(n3.status, NodeStatus.ISOLATED)

        # Verify target node-1 is executing workload
        n1 = self.cluster.get_node("node-1")
        self.assertEqual(len(n1.workload_queue), 1)

        # Verify metrics
        m = self.manager.metrics
        self.assertEqual(m.successful_migrations, 1)
        self.assertEqual(m.workloads_recovered, 1)
        self.assertEqual(m.source_nodes_isolated, 1)


if __name__ == "__main__":
    unittest.main()
