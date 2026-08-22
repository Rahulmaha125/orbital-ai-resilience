"""Virtual Cluster management with time-step orchestration and health aggregation."""

from typing import Any, Dict, List, Optional
from orbital_ai_resilience.core.node import VirtualNode
from orbital_ai_resilience.core.types import NodeStatus
from orbital_ai_resilience.core.workload import Workload
from orbital_ai_resilience.telemetry.snapshot import TelemetrySnapshot


class VirtualCluster:
    """Manages a collection of VirtualNode compute nodes.

    Attributes:
        nodes: Dictionary mapping node IDs to VirtualNode instances.
    """

    def __init__(self, nodes: Optional[List[VirtualNode]] = None) -> None:
        self.nodes: Dict[str, VirtualNode] = {}
        if nodes:
            for node in nodes:
                self.add_node(node)

    @classmethod
    def create_default_cluster(
        cls,
        num_nodes: int = 5,
        default_compute: float = 100.0,
        default_memory: float = 16384.0,
    ) -> "VirtualCluster":
        """Factory method creating a cluster populated with specified number of default VirtualNodes.

        Args:
            num_nodes: Number of virtual nodes to generate (default 5).
            default_compute: Compute capacity for each node (TFLOPS).
            default_memory: Memory capacity for each node (MB).

        Returns:
            A VirtualCluster populated with initial VirtualNode instances.
        """
        cluster = cls()
        for i in range(1, num_nodes + 1):
            node_id = f"node-{i}"
            node = VirtualNode(
                node_id=node_id,
                compute_capacity=default_compute,
                memory_capacity=default_memory,
                power_level=100.0,
                temperature=40.0 + (i * 2.0),  # baseline variation
                latency=5.0 + i,                # baseline variation
                error_rate=0.0,
                status=NodeStatus.ONLINE,
            )
            cluster.add_node(node)
        return cluster

    def add_node(self, node: VirtualNode) -> None:
        """Add a VirtualNode to the cluster.

        Args:
            node: VirtualNode instance to add.
        """
        if node.node_id in self.nodes:
            raise ValueError(f"Node with ID {node.node_id!r} already exists in cluster.")
        self.nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> Optional[VirtualNode]:
        """Remove a node from the cluster by ID.

        Args:
            node_id: ID of the node to remove.

        Returns:
            The removed VirtualNode, or None if not found.
        """
        return self.nodes.pop(node_id, None)

    def get_node(self, node_id: str) -> Optional[VirtualNode]:
        """Retrieve a VirtualNode by ID."""
        return self.nodes.get(node_id)

    def list_nodes(self) -> List[VirtualNode]:
        """Return list of all VirtualNodes in the cluster."""
        return list(self.nodes.values())

    def step_all(self, timestamp: Optional[float] = None) -> Dict[str, TelemetrySnapshot]:
        """Advance simulation time-step across all cluster nodes and record telemetry.

        Args:
            timestamp: Optional epoch timestamp override.

        Returns:
            Dictionary mapping node IDs to their recorded TelemetrySnapshots.
        """
        return {node_id: node.tick(timestamp=timestamp) for node_id, node in self.nodes.items()}

    def assign_workload_to_node(self, node_id: str, workload: Workload) -> bool:
        """Assign a workload directly to a specific target node.

        Args:
            node_id: Target node ID.
            workload: Workload instance.

        Returns:
            True if assignment succeeded, False if node not found or assignment failed.
        """
        node = self.get_node(node_id)
        if not node:
            return False
        return node.assign_workload(workload)

    def assign_workload_auto(self, workload: Workload) -> Optional[str]:
        """Assign workload to the first available healthy node with sufficient capacity.

        Args:
            workload: Workload to assign.

        Returns:
            Node ID assigned to, or None if no node had sufficient capacity.
        """
        for node in self.nodes.values():
            if node.status == NodeStatus.ONLINE:
                if node.assign_workload(workload):
                    return node.node_id
        return None

    def get_cluster_status(self) -> Dict[str, Any]:
        """Generate summary statistics for the entire cluster."""
        total_compute = sum(n.compute_capacity for n in self.nodes.values())
        used_compute = sum(n.get_used_compute() for n in self.nodes.values())
        total_memory = sum(n.memory_capacity for n in self.nodes.values())
        used_memory = sum(n.get_used_memory() for n in self.nodes.values())
        online_nodes = sum(1 for n in self.nodes.values() if n.status == NodeStatus.ONLINE)

        health_scores = [n.get_health_score() for n in self.nodes.values()]
        avg_health = sum(health_scores) / len(health_scores) if health_scores else 0.0

        return {
            "total_nodes": len(self.nodes),
            "online_nodes": online_nodes,
            "avg_health_score": round(avg_health, 2),
            "total_compute_capacity": total_compute,
            "used_compute": used_compute,
            "available_compute": total_compute - used_compute,
            "total_memory_capacity": total_memory,
            "used_memory": used_memory,
            "available_memory": total_memory - used_memory,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
        }

    def __repr__(self) -> str:
        return f"VirtualCluster(nodes={len(self.nodes)}, online={sum(1 for n in self.nodes.values() if n.status == NodeStatus.ONLINE)})"
