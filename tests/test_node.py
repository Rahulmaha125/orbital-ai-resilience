"""Unit tests for VirtualNode abstraction."""

import unittest
from orbital_ai_resilience.core.node import VirtualNode
from orbital_ai_resilience.core.types import NodeStatus
from orbital_ai_resilience.core.workload import Workload


class TestVirtualNode(unittest.TestCase):
    """Test suite for VirtualNode creation, state updates, and telemetry."""

    def test_node_creation_default_values(self) -> None:
        """Verify node creation with default telemetry values."""
        node = VirtualNode(node_id="node-test-1")
        self.assertEqual(node.node_id, "node-test-1")
        self.assertEqual(node.compute_capacity, 100.0)
        self.assertEqual(node.memory_capacity, 16384.0)
        self.assertEqual(node.power_level, 100.0)
        self.assertEqual(node.temperature, 45.0)
        self.assertEqual(node.latency, 10.0)
        self.assertEqual(node.error_rate, 0.0)
        self.assertEqual(node.status, NodeStatus.ONLINE)
        self.assertEqual(len(node.workload_queue), 0)

    def test_node_creation_custom_values(self) -> None:
        """Verify node creation with explicit custom parameter values."""
        node = VirtualNode(
            node_id="node-custom",
            compute_capacity=200.0,
            memory_capacity=32768.0,
            power_level=85.5,
            temperature=55.0,
            latency=12.5,
            error_rate=0.01,
            status=NodeStatus.DEGRADED,
        )
        self.assertEqual(node.node_id, "node-custom")
        self.assertEqual(node.compute_capacity, 200.0)
        self.assertEqual(node.memory_capacity, 32768.0)
        self.assertEqual(node.power_level, 85.5)
        self.assertEqual(node.temperature, 55.0)
        self.assertEqual(node.latency, 12.5)
        self.assertEqual(node.error_rate, 0.01)
        self.assertEqual(node.status, NodeStatus.DEGRADED)

    def test_node_invalid_capacity_raises_error(self) -> None:
        """Verify ValueError is raised when initializing with zero or negative capacity."""
        with self.assertRaises(ValueError):
            VirtualNode(node_id="bad-node-1", compute_capacity=0.0)
        with self.assertRaises(ValueError):
            VirtualNode(node_id="bad-node-2", memory_capacity=-100.0)

    def test_telemetry_updates(self) -> None:
        """Verify telemetry attributes can be updated selectively."""
        node = VirtualNode(node_id="node-telem")
        node.update_telemetry(power_level=75.0, temperature=62.0)
        self.assertEqual(node.power_level, 75.0)
        self.assertEqual(node.temperature, 62.0)
        self.assertEqual(node.latency, 10.0)  # unchanged
        self.assertEqual(node.error_rate, 0.0)  # unchanged

        node.update_telemetry(latency=25.0, error_rate=0.05)
        self.assertEqual(node.latency, 25.0)
        self.assertEqual(node.error_rate, 0.05)

    def test_set_status(self) -> None:
        """Verify node status transition updates."""
        node = VirtualNode(node_id="node-status")
        self.assertEqual(node.status, NodeStatus.ONLINE)

        node.set_status(NodeStatus.DEGRADED)
        self.assertEqual(node.status, NodeStatus.DEGRADED)

        node.set_status(NodeStatus.ISOLATED)
        self.assertEqual(node.status, NodeStatus.ISOLATED)

    def test_node_to_dict_serialization(self) -> None:
        """Verify node state dictionary representation."""
        node = VirtualNode(node_id="node-dict")
        d = node.to_dict()
        self.assertEqual(d["node_id"], "node-dict")
        self.assertEqual(d["status"], "ONLINE")
        self.assertIn("available_compute", d)
        self.assertIn("available_memory", d)


if __name__ == "__main__":
    unittest.main()
