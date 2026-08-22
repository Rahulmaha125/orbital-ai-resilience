"""Structured state logging utilities for Orbital AI Resilience system telemetry."""

import json
import logging
import time
from typing import Any, Dict, Optional


def setup_logger(name: str = "orbital_ai_resilience", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a standard logger for the system."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


class StateLogger:
    """Utility for logging structured telemetry snapshots and cluster events.

    Attributes:
        logger: Underlying python logger instance.
        logs_history: In-memory list storing logged event dictionaries.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger: logging.Logger = logger or setup_logger()
        self.logs_history: list[Dict[str, Any]] = []

    def log_event(self, event_type: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """Record a structured system event with timestamp.

        Args:
            event_type: Category label for the event (e.g., 'NODE_STATE_CHANGE', 'WORKLOAD_ASSIGNED').
            details: Contextual payload dictionary.

        Returns:
            The structured log record dictionary.
        """
        record = {
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details,
        }
        self.logs_history.append(record)
        self.logger.info(f"EVENT:{event_type} | {json.dumps(details)}")
        return record

    def log_cluster_snapshot(self, cluster_status: Dict[str, Any]) -> Dict[str, Any]:
        """Record a cluster-wide telemetry snapshot.

        Args:
            cluster_status: Serialized cluster status dictionary.

        Returns:
            The structured log record dictionary.
        """
        return self.log_event("CLUSTER_SNAPSHOT", cluster_status)

    def clear(self) -> None:
        """Clear recorded log history."""
        self.logs_history.clear()
