"""Fault Injection engine for simulating silent AI model degradation and hardware stress."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.core.node import VirtualNode
from orbital_ai_resilience.faults.profile import FaultProfile
from orbital_ai_resilience.faults.types import FaultType


class FaultInjector:
    """Orchestrates deterministic fault injection across cluster nodes.

    Attributes:
        profiles: List of active or scheduled FaultProfile instances.
        current_tick: Current simulation tick index.
    """

    def __init__(self, profiles: Optional[List[FaultProfile]] = None) -> None:
        self.profiles: List[FaultProfile] = profiles or []
        self.current_tick: int = 0

    def add_profile(self, profile: FaultProfile) -> None:
        """Register a new fault profile."""
        self.profiles.append(profile)

    def remove_profile(self, fault_id: str) -> Optional[FaultProfile]:
        """Remove a fault profile by ID."""
        for idx, p in enumerate(self.profiles):
            if p.fault_id == fault_id:
                return self.profiles.pop(idx)
        return None

    def get_active_profiles_for_node(self, node_id: str, tick: Optional[int] = None) -> List[FaultProfile]:
        """Return all fault profiles currently active for a target node."""
        t = self.current_tick if tick is None else tick
        return [p for p in self.profiles if p.target_node_id == node_id and p.is_active_at(t)]

    def advance_tick(self, tick: Optional[int] = None) -> int:
        """Advance the simulation tick counter."""
        if tick is not None:
            self.current_tick = tick
        else:
            self.current_tick += 1
        return self.current_tick

    def apply_physical_telemetry_faults(self, cluster: VirtualCluster, tick: Optional[int] = None) -> None:
        """Apply active environmental stress faults to physical node telemetry.

        Note: SILENT_MODEL_DEGRADATION explicitly bypasses physical telemetry updates
              to preserve nominal physical health while corrupting AI compute.
        """
        t = self.current_tick if tick is None else tick
        for p in self.profiles:
            if p.is_active_at(t) and p.fault_type == FaultType.ENVIRONMENTAL_STRESS:
                node = cluster.get_node(p.target_node_id)
                if node:
                    # Apply controlled physical stress proportional to intensity
                    new_temp = node.temperature + (30.0 * p.intensity)
                    new_error_rate = min(1.0, node.error_rate + (0.08 * p.intensity))
                    new_power = max(0.0, node.power_level - (20.0 * p.intensity))
                    new_latency = node.latency + (50.0 * p.intensity)
                    node.update_telemetry(
                        power_level=new_power,
                        temperature=new_temp,
                        latency=new_latency,
                        error_rate=new_error_rate,
                        auto_tick=True,
                    )

    def inject_memory_bit_flip(
        self, array: np.ndarray, intensity: float, seed: int
    ) -> np.ndarray:
        """Simulate single-event bit flips in a float64/float32 numpy array.

        Args:
            array: Input clean numerical array.
            intensity: Ratio determining how many elements suffer bit flips.
            seed: Seed for reproducible bit-flip selection.

        Returns:
            New array with inverted bits in selected elements.
        """
        rng = np.random.default_rng(seed)
        corrupted = np.copy(array)
        total_elements = corrupted.size

        # Determine number of elements to corrupt
        num_flips = max(1, int(total_elements * intensity))
        indices = rng.choice(total_elements, size=num_flips, replace=False)

        # Work with uint view for bitwise operations
        if corrupted.dtype == np.float64:
            uint_view = corrupted.reshape(-1).view(np.uint64)
            bit_width = 64
        elif corrupted.dtype == np.float32:
            uint_view = corrupted.reshape(-1).view(np.uint32)
            bit_width = 32
        else:
            flat = corrupted.reshape(-1)
            for idx in indices:
                flat[idx] += rng.normal(0, intensity * 10.0)
            return corrupted

        for idx in indices:
            bit_pos = rng.integers(0, bit_width)
            uint_view[idx] ^= (np.uint64(1) << np.uint64(bit_pos))

        return corrupted

    def inject_parameter_perturbation(
        self, weights: np.ndarray, intensity: float, seed: int, elapsed_ticks: int = 0
    ) -> np.ndarray:
        """Inject numerical noise into model parameters/weights.

        Uses a base random direction matrix scaled deterministically by intensity
        and elapsed ticks to model continuous parameter degradation.
        """
        rng = np.random.default_rng(seed)
        base_noise = rng.normal(loc=0.0, scale=1.0, size=weights.shape)
        std_dev = float(np.std(weights)) if float(np.std(weights)) > 0 else 1.0
        
        # Scaling increases deterministically with elapsed_ticks
        scale = intensity * std_dev * (1.0 + 0.5 * elapsed_ticks)
        return weights + (base_noise * scale)

    def inject_output_drift(
        self, output: np.ndarray, intensity: float, seed: int, elapsed_ticks: int = 0
    ) -> np.ndarray:
        """Inject systematic directional baseline drift into compute outputs."""
        rng = np.random.default_rng(seed)
        direction = rng.uniform(-1.0, 1.0, size=output.shape)
        drift_magnitude = intensity * (1.0 + 0.5 * elapsed_ticks)
        return output + (direction * drift_magnitude)

    def transform_weights_or_output(
        self,
        node_id: str,
        weights: np.ndarray,
        reference_output: np.ndarray,
        tick: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply active compute/silent degradation faults to weights or output."""
        t = self.current_tick if tick is None else tick
        active_profiles = self.get_active_profiles_for_node(node_id, t)

        current_weights = np.copy(weights)
        current_output = np.copy(reference_output)

        for p in active_profiles:
            elapsed = t - p.start_tick

            if p.fault_type == FaultType.MEMORY_BIT_FLIP:
                current_weights = self.inject_memory_bit_flip(current_weights, p.intensity, p.seed + elapsed)

            elif p.fault_type in (FaultType.PARAMETER_PERTURBATION, FaultType.SILENT_MODEL_DEGRADATION):
                current_weights = self.inject_parameter_perturbation(
                    current_weights, p.intensity, p.seed, elapsed_ticks=elapsed
                )

            elif p.fault_type == FaultType.OUTPUT_DRIFT:
                current_output = self.inject_output_drift(
                    current_output, p.intensity, p.seed, elapsed_ticks=elapsed
                )

            elif p.fault_type == FaultType.INTERMITTENT_COMPUTATION:
                rng = np.random.default_rng(p.seed + elapsed)
                if rng.random() < p.intensity:
                    current_output = current_output + rng.normal(0, 10.0, size=current_output.shape)

        return current_weights, current_output
