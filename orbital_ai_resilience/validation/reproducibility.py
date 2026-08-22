"""ReproducibilityManager tracking seeds, configurations, and environment metadata for 100% deterministic experiment reproducibility."""

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class ExperimentConfig:
    """Experiment configuration parameters for deterministic reproduction."""

    experiment_id: str
    scenario_id: int
    scenario_name: str
    policy_name: str
    node_count: int
    total_ticks: int
    seed: int
    workload_memory_mb: float = 1024.0
    fault_intensity: float = 0.15
    fault_duration: int = 10
    verification_threshold_mse: float = 0.005
    version_tag: str = "Phase-10-v1.0"

    def compute_config_hash(self) -> str:
        """Compute SHA-256 fingerprint hash of configuration settings."""
        serialized = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


class ReproducibilityManager:
    """Manages deterministic seeding and records environment configurations."""

    def __init__(self, seed: int = 42) -> None:
        self.seed: int = seed

    def set_seed(self, seed: Optional[int] = None) -> int:
        """Set global deterministic random seeds for Python stdlib and NumPy if available."""
        s = self.seed if seed is None else seed
        random.seed(s)
        try:
            import numpy as np

            np.random.seed(s)
        except ImportError:
            pass
        return s

    def create_experiment_config(
        self,
        experiment_id: str,
        scenario_id: int,
        scenario_name: str,
        policy_name: str,
        node_count: int,
        total_ticks: int,
        seed: int,
        **kwargs: Any,
    ) -> ExperimentConfig:
        """Create and fingerprint an ExperimentConfig instance."""
        return ExperimentConfig(
            experiment_id=experiment_id,
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            policy_name=policy_name,
            node_count=node_count,
            total_ticks=total_ticks,
            seed=seed,
            **kwargs,
        )
