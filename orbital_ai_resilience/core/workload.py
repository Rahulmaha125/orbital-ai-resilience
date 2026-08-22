"""Workload representation for distributed AI compute nodes."""

import time
import uuid
from typing import Any, Dict, Optional
from orbital_ai_resilience.core.types import WorkloadStatus


class Workload:
    """Represents a computational job/task assigned to a VirtualNode.

    Attributes:
        workload_id: Unique string identifier for the workload.
        name: Human-readable name or label for the workload.
        required_compute: Compute units (TFLOPS) needed for execution.
        required_memory: Memory capacity (MB) needed for execution.
        status: Current lifecycle status of the workload.
        payload: Arbitrary input data or model payload for execution.
        result: Execution output or model predictions.
        created_at: Epoch timestamp of workload instantiation.
        started_at: Epoch timestamp when workload execution started.
        completed_at: Epoch timestamp when workload finished or failed.
    """

    def __init__(
        self,
        name: str,
        required_compute: float,
        required_memory: float,
        workload_id: Optional[str] = None,
        payload: Optional[Any] = None,
    ) -> None:
        if required_compute <= 0:
            raise ValueError("required_compute must be positive")
        if required_memory <= 0:
            raise ValueError("required_memory must be positive")

        self.workload_id: str = workload_id or f"wkl-{uuid.uuid4().hex[:8]}"
        self.name: str = name
        self.required_compute: float = float(required_compute)
        self.required_memory: float = float(required_memory)
        self.status: WorkloadStatus = WorkloadStatus.PENDING
        self.payload: Optional[Any] = payload
        self.result: Optional[Any] = None
        self.created_at: float = time.time()
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None

    def start(self) -> None:
        """Mark workload execution as started."""
        self.status = WorkloadStatus.RUNNING
        self.started_at = time.time()

    def complete(self, result: Optional[Any] = None) -> None:
        """Mark workload execution as successfully completed."""
        self.status = WorkloadStatus.COMPLETED
        self.result = result
        self.completed_at = time.time()

    def fail(self, error: Optional[Any] = None) -> None:
        """Mark workload execution as failed."""
        self.status = WorkloadStatus.FAILED
        self.result = error
        self.completed_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize workload state to a dictionary representation."""
        return {
            "workload_id": self.workload_id,
            "name": self.name,
            "required_compute": self.required_compute,
            "required_memory": self.required_memory,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    def __repr__(self) -> str:
        return (
            f"Workload(id={self.workload_id!r}, name={self.name!r}, "
            f"compute={self.required_compute}, memory={self.required_memory}, "
            f"status={self.status.value!r})"
        )
