"""Workload snapshot and migration state representation."""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from orbital_ai_resilience.core.workload import Workload


@dataclass
class WorkloadSnapshot:
    """Immutable snapshot of a workload state captured prior to migration.

    Attributes:
        migration_id: Unique identifier for this migration transaction.
        workload_id: ID of the workload being migrated.
        workload_name: Name of the workload.
        required_compute: Compute units required.
        required_memory: Memory capacity required.
        payload: Input data payload / model tensors.
        source_node_id: ID of the source compute node.
        migration_reason: Explanatory text / fault trigger reason.
        created_at: Epoch timestamp of snapshot creation.
        migration_count: Incrementing count of migrations this workload has undergone.
        additional_metadata: Arbitrary state dictionary.
    """

    migration_id: str
    workload_id: str
    workload_name: str
    required_compute: float
    required_memory: float
    payload: Any
    source_node_id: str
    migration_reason: str
    created_at: float = field(default_factory=time.time)
    migration_count: int = 1
    additional_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create_from_workload(
        cls,
        workload: Workload,
        source_node_id: str,
        reason: str,
        migration_count: int = 1,
    ) -> "WorkloadSnapshot":
        """Factory method snapshotting an active Workload instance."""
        migration_id = f"mig-{uuid.uuid4().hex[:8]}"
        return cls(
            migration_id=migration_id,
            workload_id=workload.workload_id,
            workload_name=workload.name,
            required_compute=workload.required_compute,
            required_memory=workload.required_memory,
            payload=workload.payload,
            source_node_id=source_node_id,
            migration_reason=reason,
            migration_count=migration_count,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize workload snapshot to dictionary."""
        return {
            "migration_id": self.migration_id,
            "workload_id": self.workload_id,
            "workload_name": self.workload_name,
            "required_compute": self.required_compute,
            "required_memory": self.required_memory,
            "source_node_id": self.source_node_id,
            "migration_reason": self.migration_reason,
            "created_at": self.created_at,
            "migration_count": self.migration_count,
        }
