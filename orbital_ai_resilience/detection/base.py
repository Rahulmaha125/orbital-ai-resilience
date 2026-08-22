"""Abstract base class interface for anomaly detectors."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from orbital_ai_resilience.detection.types import DetectionResult


class BaseDetector(ABC):
    """Interface for AI behavioral and node anomaly detection algorithms."""

    def __init__(self, name: str) -> None:
        self.name: str = name

    @abstractmethod
    def evaluate(
        self,
        exec_log: Dict[str, Any],
        history_logs: Optional[List[Dict[str, Any]]] = None,
    ) -> DetectionResult:
        """Evaluate an execution payload log and return a DetectionResult.

        Args:
            exec_log: Workload execution log dictionary from Phase 3.
            history_logs: Optional historical execution logs for trend analysis.

        Returns:
            DetectionResult payload.
        """
        pass
