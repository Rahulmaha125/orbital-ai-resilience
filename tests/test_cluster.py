"""Unit tests for VirtualCluster management."""

import unittest
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.core.node import VirtualNode
from orbital_ai_resilience.core.types import NodeStatus


class TestVirtualCluster(unittest.TestCase):
    """Test suite for VirtualCluster creation and management."""

    def test_empty_cluster_creation(self) -> None:
        """Verify initialization of an empty cluster."""
        cluster = VirtualCluster()
        self.assertEqual(len(cluster.nodes), 0)
        self.assertEqual(cluster.list_nodes(), [])

    def test_default_5_node_cluster_creation(self) -> None:
        """Verify default cluster factory creates exactly 5 virtual nodes."""
        cluster = VirtualCluster.create_default_cluster(num_nodes=5)
        self.assertEqual(len(cluster.nodes), 5)
        
        # Verify node IDs node-1 through node-5
        for i in range(1, 6):
            node_id = f"node-{i}"
            node = cluster.get_node(node_id)
            self.assertIsNotNone(node)
            self.assertEqual(node.node_id, node_id)
            self.assertEqual(node.status, NodeStatus.ONLINE)

    def test_add_and_remove_node(self) -> None:
        """Verify adding and removing nodes from cluster."""
        cluster = VirtualCluster()
        node = VirtualNode(node_id="custom-node-1")
        cluster.add_node(node)
        self.assertIn("custom-node-1", cluster.nodes)

        # Adding duplicate ID raises error
        with self.assertRaises(ValueError):
            cluster.add_node(VirtualNode(node_id="custom-node-1"))

        removed = cluster.remove_node("custom-node-1")
        self.assertEqual(removed, node)
        self.assertNotIn("custom-node-1", cluster.nodes)

    def test_cluster_status_summary(self) -> None:
        """Verify aggregate cluster status calculation."""
        cluster = VirtualCluster.create_default_cluster(num_nodes=3, default_compute=50.0, default_memory=8192.0)
        status = cluster.get_cluster_status()

        self.assertEqual(status["total_nodes"], 3)
        self.assertEqual(status["online_nodes"], 3)
        self.assertEqual(status["total_compute_capacity"], 150.0)
        self.assertEqual(status["total_memory_capacity"], 24576.0)


if __name__ == "__main__":
    unittest.main()
