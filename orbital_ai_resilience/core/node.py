"""Virtual Node abstraction for distributed AI compute nodes with telemetry history & health scoring."""

import time
from typing import Any, Dict, List, Optional
from orbital_ai_resilience.core.types import HealthState, NodeStatus, WorkloadStatus
from orbital_ai_resilience.core.workload import Workload
from orbital_ai_resilience.health.config import HealthConfig
from orbital_ai_resilience.health.evaluator import HealthEvaluator
from orbital_ai_resilience.telemetry.snapshot import TelemetryHistory, TelemetrySnapshot


class VirtualNode:
    """Represents a virtual AI compute node in an orbital/distributed environment.

    Attributes:
        node_id: Unique string identifier for the node (e.g. "node-1").
        compute_capacity: Maximum compute capacity (e.g., TFLOPS).
        memory_capacity: Maximum memory capacity (e.g., MB).
        power_level: Current available power level (Watts or percentage).
        temperature: Current operating temperature (Celsius).
        latency: Current communication/processing latency (milliseconds).
        error_rate: Observed error rate ratio (0.0 = fault-free, 1.0 = total corruption).
        workload_queue: List of active or queued workloads assigned to this node.
        status: Current operational status of the node.
        telemetry_history: Time-series history of telemetry snapshots.
        health_evaluator: Evaluator for calculating baseline Health/Trust score.
    """

    def __init__(
        self,
        node_id: str,
        compute_capacity: float = 100.0,
        memory_capacity: float = 16384.0,
        power_level: float = 100.0,
        temperature: float = 45.0,
        latency: float = 10.0,
        error_rate: float = 0.0,
        status: NodeStatus = NodeStatus.ONLINE,
        max_history_len: int = 100,
        health_config: Optional[HealthConfig] = None,
    ) -> None:
        if compute_capacity <= 0:
            raise ValueError("compute_capacity must be positive")
        if memory_capacity <= 0:
            raise ValueError("memory_capacity must be positive")

        self.node_id: str = node_id
        self.compute_capacity: float = float(compute_capacity)
        self.memory_capacity: float = float(memory_capacity)
        self.power_level: float = float(power_level)
        self.temperature: float = float(temperature)
        self.latency: float = float(latency)
        self.error_rate: float = float(error_rate)
        self.status: NodeStatus = status
        self.workload_queue: List[Workload] = []

        # Telemetry History & Health Scoring
        self.telemetry_history: TelemetryHistory = TelemetryHistory(max_length=max_history_len)
        self.health_evaluator: HealthEvaluator = HealthEvaluator(config=health_config)

        # Record initial baseline telemetry snapshot
        self.tick()

    def get_used_compute(self) -> float:
        """Calculate total compute resources requested by non-completed workloads."""
        return sum(
            w.required_compute
            for w in self.workload_queue
            if w.status in (WorkloadStatus.PENDING, WorkloadStatus.RUNNING)
        )

    def get_used_memory(self) -> float:
        """Calculate total memory resources requested by non-completed workloads."""
        return sum(
            w.required_memory
            for w in self.workload_queue
            if w.status in (WorkloadStatus.PENDING, WorkloadStatus.RUNNING)
        )

    def get_available_compute(self) -> float:
        """Calculate remaining unallocated compute capacity."""
        return max(0.0, self.compute_capacity - self.get_used_compute())

    def get_available_memory(self) -> float:
        """Calculate remaining unallocated memory capacity."""
        return max(0.0, self.memory_capacity - self.get_used_memory())

    def tick(self, timestamp: Optional[float] = None) -> TelemetrySnapshot:
        """Record a time-step telemetry snapshot into historical memory.

        Args:
            timestamp: Optional epoch timestamp (defaults to current time.time()).

        Returns:
            The recorded TelemetrySnapshot instance.
        """
        ts = timestamp if timestamp is not None else time.time()
        snapshot = TelemetrySnapshot(
            timestamp=ts,
            power_level=self.power_level,
            temperature=self.temperature,
            latency=self.latency,
            error_rate=self.error_rate,
            used_compute=self.get_used_compute(),
            used_memory=self.get_used_memory(),
        )
        self.telemetry_history.add_snapshot(snapshot)
        return snapshot

    def get_health_score(self, config: Optional[HealthConfig] = None) -> float:
        """Calculate node's current Health/Trust score [0.0, 100.0].

        Args:
            config: Optional override HealthConfig parameters.

        Returns:
            Composite health score float.
        """
        latest = self.telemetry_history.get_latest()
        if not latest:
            latest = self.tick()

        evaluator = self.health_evaluator if config is None else HealthEvaluator(config=config)
        score, _, _ = evaluator.evaluate_health(latest, self.telemetry_history)
        return score

    def get_health_state(self, config: Optional[HealthConfig] = None) -> HealthState:
        """Calculate node's current HealthState classification.

        Args:
            config: Optional override HealthConfig parameters.

        Returns:
            HealthState enum (HEALTHY, WARNING, DEGRADED, CRITICAL).
        """
        latest = self.telemetry_history.get_latest()
        if not latest:
            latest = self.tick()

        evaluator = self.health_evaluator if config is None else HealthEvaluator(config=config)
        _, state, _ = evaluator.evaluate_health(latest, self.telemetry_history)
        return state

    def get_health_breakdown(self, config: Optional[HealthConfig] = None) -> Dict[str, Any]:
        """Return detailed breakdown of health score evaluation."""
        latest = self.telemetry_history.get_latest()
        if not latest:
            latest = self.tick()

        evaluator = self.health_evaluator if config is None else HealthEvaluator(config=config)
        _, state, breakdown = evaluator.evaluate_health(latest, self.telemetry_history)
        breakdown["state"] = state.value
        return breakdown

    def assign_workload(self, workload: Workload) -> bool:
        """Assign a workload to this node if capacity allows and node is ONLINE.

        Args:
            workload: The Workload instance to assign.

        Returns:
            True if assignment succeeded, False if rejected due to status or capacity.
        """
        if self.status != NodeStatus.ONLINE:
            return False

        if workload.required_compute > self.get_available_compute():
            return False

        if workload.required_memory > self.get_available_memory():
            return False

        self.workload_queue.append(workload)
        if workload.status == WorkloadStatus.PENDING:
            workload.start()
        return True

    def remove_workload(self, workload_id: str) -> Optional[Workload]:
        """Remove a workload from the queue by ID.

        Args:
            workload_id: ID of the workload to remove.

        Returns:
            The removed Workload instance, or None if not found.
        """
        for index, w in enumerate(self.workload_queue):
            if w.workload_id == workload_id:
                return self.workload_queue.pop(index)
        return None

    def update_telemetry(
        self,
        power_level: Optional[float] = None,
        temperature: Optional[float] = None,
        latency: Optional[float] = None,
        error_rate: Optional[float] = None,
        auto_tick: bool = True,
    ) -> None:
        """Update node physical and operational telemetry readings.

        Args:
            power_level: Updated power level.
            temperature: Updated operating temperature.
            latency: Updated communication/processing latency.
            error_rate: Updated error rate ratio.
            auto_tick: If True, automatically records a telemetry snapshot.
        """
        if power_level is not None:
            self.power_level = float(power_level)
        if temperature is not None:
            self.temperature = float(temperature)
        if latency is not None:
            self.latency = float(latency)
        if error_rate is not None:
            self.error_rate = float(error_rate)

        if auto_tick:
            self.tick()

    def set_status(self, new_status: NodeStatus) -> None:
        """Update the operational status of the node."""
        self.status = new_status

    def to_dict(self) -> Dict[str, Any]:
        """Serialize node state and telemetry into a dictionary."""
        return {
            "node_id": self.node_id,
            "compute_capacity": self.compute_capacity,
            "memory_capacity": self.memory_capacity,
            "used_compute": self.get_used_compute(),
            "used_memory": self.get_used_memory(),
            "available_compute": self.get_available_compute(),
            "available_memory": self.get_available_memory(),
            "power_level": self.power_level,
            "temperature": self.temperature,
            "latency": self.latency,
            "error_rate": self.error_rate,
            "health_score": self.get_health_score(),
            "health_state": self.get_health_state().value,
            "status": self.status.value,
            "telemetry_history_len": len(self.telemetry_history),
            "workload_count": len(self.workload_queue),
            "workloads": [w.to_dict() for w in self.workload_queue],
        }

    def __repr__(self) -> str:
        return (
            f"VirtualNode(id={self.node_id!r}, status={self.status.value!r}, "
            f"health={self.get_health_score():.1f} ({self.get_health_state().value}), "
            f"compute={self.get_used_compute()}/{self.compute_capacity}, "
            f"memory={self.get_used_memory()}/{self.memory_capacity}, "
            f"power={self.power_level}, temp={self.temperature}, "
            f"latency={self.latency}, error_rate={self.error_rate})"
        )
