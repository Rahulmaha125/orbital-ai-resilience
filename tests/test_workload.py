"""Unit tests for Workload objects and workload assignment logic."""

import unittest
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.core.node import VirtualNode
from orbital_ai_resilience.core.types import NodeStatus, WorkloadStatus
from orbital_ai_resilience.core.workload import Workload
from orbital_ai_resilience.utils.logger import StateLogger


class TestWorkloadAssignment(unittest.TestCase):
    """Test suite for workload creation and node assignment logic."""

    def test_workload_creation(self) -> None:
        """Verify workload initialization and parameter validation."""
        w = Workload(name="test_inference", required_compute=20.0, required_memory=2048.0)
        self.assertEqual(w.name, "test_inference")
        self.assertEqual(w.required_compute, 20.0)
        self.assertEqual(w.required_memory, 2048.0)
        self.assertEqual(w.status, WorkloadStatus.PENDING)
        self.assertIsNotNone(w.workload_id)

    def test_workload_invalid_parameters(self) -> None:
        """Verify invalid resource requirements raise errors."""
        with self.assertRaises(ValueError):
            Workload(name="bad_w1", required_compute=0.0, required_memory=1024.0)
        with self.assertRaises(ValueError):
            Workload(name="bad_w2", required_compute=10.0, required_memory=-512.0)

    def test_workload_assignment_to_node(self) -> None:
        """Verify successful assignment of workload to node with sufficient capacity."""
        node = VirtualNode(node_id="node-1", compute_capacity=100.0, memory_capacity=10000.0)
        w = Workload(name="job-1", required_compute=30.0, required_memory=3000.0)

        success = node.assign_workload(w)
        self.assertTrue(success)
        self.assertEqual(len(node.workload_queue), 1)
        self.assertEqual(w.status, WorkloadStatus.RUNNING)
        self.assertEqual(node.get_used_compute(), 30.0)
        self.assertEqual(node.get_used_memory(), 3000.0)
        self.assertEqual(node.get_available_compute(), 70.0)
        self.assertEqual(node.get_available_memory(), 7000.0)

    def test_workload_rejection_exceeds_compute(self) -> None:
        """Verify workload rejection when node compute capacity is exceeded."""
        node = VirtualNode(node_id="node-small", compute_capacity=50.0, memory_capacity=10000.0)
        w = Workload(name="big-job", required_compute=60.0, required_memory=1000.0)

        success = node.assign_workload(w)
        self.assertFalse(success)
        self.assertEqual(len(node.workload_queue), 0)
        self.assertEqual(w.status, WorkloadStatus.PENDING)

    def test_workload_rejection_exceeds_memory(self) -> None:
        """Verify workload rejection when node memory capacity is exceeded."""
        node = VirtualNode(node_id="node-mem", compute_capacity=100.0, memory_capacity=2000.0)
        w = Workload(name="heavy-mem", required_compute=10.0, required_memory=4000.0)

        success = node.assign_workload(w)
        self.assertFalse(success)

    def test_workload_rejection_non_online_node(self) -> None:
        """Verify workload rejection when target node is OFFLINE or ISOLATED."""
        node = VirtualNode(node_id="node-offline", status=NodeStatus.OFFLINE)
        w = Workload(name="job-2", required_compute=10.0, required_memory=512.0)

        self.assertFalse(node.assign_workload(w))

        node.set_status(NodeStatus.ISOLATED)
        self.assertFalse(node.assign_workload(w))

    def test_cluster_auto_assignment(self) -> None:
        """Verify automatic cluster workload assignment across multiple nodes."""
        cluster = VirtualCluster.create_default_cluster(num_nodes=3, default_compute=50.0, default_memory=4000.0)
        
        w1 = Workload(name="job-1", required_compute=40.0, required_memory=2000.0)
        w2 = Workload(name="job-2", required_compute=40.0, required_memory=2000.0)

        assigned_node_1 = cluster.assign_workload_auto(w1)
        self.assertEqual(assigned_node_1, "node-1")

        # Second job won't fit on node-1 (since 40+40 > 50 compute), so it should auto-assign to node-2
        assigned_node_2 = cluster.assign_workload_auto(w2)
        self.assertEqual(assigned_node_2, "node-2")

    def test_state_logger(self) -> None:
        """Verify state logger records cluster and node events."""
        logger = StateLogger()
        cluster = VirtualCluster.create_default_cluster(num_nodes=2)
        
        rec = logger.log_cluster_snapshot(cluster.get_cluster_status())
        self.assertEqual(rec["event_type"], "CLUSTER_SNAPSHOT")
        self.assertEqual(len(logger.logs_history), 1)


if __name__ == "__main__":
    unittest.main()
