"""Trusted Reference Provider for output verification."""

import numpy as np
from typing import Optional
from orbital_ai_resilience.core.node import VirtualNode
from orbital_ai_resilience.core.types import NodeStatus
from orbital_ai_resilience.workloads.synthetic import SyntheticAIWorkload


class ReferenceProvider:
    """Generates trusted reference outputs for post-migration verification.

    Enforces strict security rule: Never generates reference computation from
    a node marked DEGRADED, ISOLATED, or QUARANTINED.
    """

    def get_reference_output(
        self,
        workload: SyntheticAIWorkload,
        reference_node: Optional[VirtualNode] = None,
    ) -> np.ndarray:
        """Retrieve trusted reference output tensor for a workload.

        Args:
            workload: SyntheticAIWorkload to evaluate.
            reference_node: Optional node instance to compute reference on.

        Returns:
            Clean reference output numpy array.

        Raises:
            ValueError: If reference_node is non-ONLINE or untrusted.
        """
        if reference_node is not None:
            if reference_node.status != NodeStatus.ONLINE:
                raise ValueError(
                    f"Security Error: Reference node {reference_node.node_id!r} has untrusted status {reference_node.status.value!r}."
                )

        # Primary deterministic ground-truth computation
        return workload.compute_reference_output()
