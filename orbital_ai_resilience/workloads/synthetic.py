"""Synthetic AI Workload for controlled evaluation and output deviation measurement."""

import numpy as np
import time
from typing import Any, Dict, Optional, Tuple
from orbital_ai_resilience.core.node import VirtualNode
from orbital_ai_resilience.core.workload import Workload
from orbital_ai_resilience.faults.injector import FaultInjector


class SyntheticAIWorkload(Workload):
    """Synthetic AI model inference workload performing matrix tensor transformations.

    Attributes:
        input_data: Synthetic feature matrix X (shape: num_samples x in_features).
        reference_weights: Ground truth model weight tensor W (shape: in_features x out_features).
        bias: Model bias vector B (shape: out_features).
    """

    def __init__(
        self,
        name: str = "synthetic_ai_inference",
        required_compute: float = 10.0,
        required_memory: float = 512.0,
        num_samples: int = 32,
        in_features: int = 16,
        out_features: int = 8,
        seed: int = 42,
        workload_id: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name,
            required_compute=required_compute,
            required_memory=required_memory,
            workload_id=workload_id,
        )

        rng = np.random.default_rng(seed)
        self.num_samples: int = num_samples
        self.in_features: int = in_features
        self.out_features: int = out_features

        # Synthesize ground truth tensors
        self.input_data: np.ndarray = rng.standard_normal((num_samples, in_features))
        self.reference_weights: np.ndarray = rng.standard_normal((in_features, out_features)) / np.sqrt(in_features)
        self.bias: np.ndarray = np.zeros(out_features)

        self.payload = {
            "num_samples": num_samples,
            "in_features": in_features,
            "out_features": out_features,
        }

    def compute_reference_output(self) -> np.ndarray:
        """Compute the ground-truth reference output tensor: Y_ref = ReLU(X * W + B)."""
        logits = np.dot(self.input_data, self.reference_weights) + self.bias
        return np.maximum(0.0, logits)  # ReLU activation

    def execute_on_node(
        self,
        node: VirtualNode,
        fault_injector: Optional[FaultInjector] = None,
        tick: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute the synthetic AI workload on a target VirtualNode.

        If a FaultInjector is provided, applies active fault perturbations (e.g. silent model degradation).

        Args:
            node: VirtualNode executing the computation.
            fault_injector: Optional FaultInjector instance.
            tick: Optional simulation tick index.

        Returns:
            Dictionary containing reference output, affected output, deviation metrics, and telemetry log.
        """
        ref_output = self.compute_reference_output()

        if fault_injector is not None:
            affected_weights, affected_output = fault_injector.transform_weights_or_output(
                node_id=node.node_id,
                weights=self.reference_weights,
                reference_output=ref_output,
                tick=tick,
            )
            # If weights were perturbed, compute output using affected weights
            if not np.array_equal(affected_weights, self.reference_weights):
                logits = np.dot(self.input_data, affected_weights) + self.bias
                exec_output = np.maximum(0.0, logits)
            else:
                exec_output = affected_output
        else:
            exec_output = ref_output

        # Calculate deviation metrics
        mse = float(np.mean((ref_output - exec_output) ** 2))
        mae = float(np.mean(np.abs(ref_output - exec_output)))
        max_ae = float(np.max(np.abs(ref_output - exec_output)))

        # Cosine similarity calculation
        ref_flat = ref_output.reshape(-1)
        exec_flat = exec_output.reshape(-1)
        denom = (np.linalg.norm(ref_flat) * np.linalg.norm(exec_flat))
        cosine_sim = float(np.dot(ref_flat, exec_flat) / denom) if denom > 1e-9 else 1.0

        # Mark workload complete
        self.complete(result={"mse": mse, "mae": mae, "cosine_sim": cosine_sim})

        # Compile comprehensive experiment log
        active_faults = (
            [p.to_dict() for p in fault_injector.get_active_profiles_for_node(node.node_id, tick)]
            if fault_injector
            else []
        )

        return {
            "tick": tick if tick is not None else (fault_injector.current_tick if fault_injector else 0),
            "timestamp": time.time(),
            "target_node_id": node.node_id,
            "workload_id": self.workload_id,
            "active_faults": active_faults,
            "reference_summary": {
                "mean": float(np.mean(ref_output)),
                "std": float(np.std(ref_output)),
                "norm": float(np.linalg.norm(ref_output)),
            },
            "affected_summary": {
                "mean": float(np.mean(exec_output)),
                "std": float(np.std(exec_output)),
                "norm": float(np.linalg.norm(exec_output)),
            },
            "deviation": {
                "mse": mse,
                "mae": mae,
                "max_ae": max_ae,
                "cosine_sim": cosine_sim,
            },
            "physical_telemetry": {
                "power_level": node.power_level,
                "temperature": node.temperature,
                "latency": node.latency,
                "error_rate": node.error_rate,
                "status": node.status.value,
            },
            "health_score": node.get_health_score(),
            "health_state": node.get_health_state().value,
        }
