"""Fault types and enumerations for software fault injection."""

from enum import Enum


class FaultType(str, Enum):
    """Supported fault injection categories for orbital AI node resilience testing."""
    MEMORY_BIT_FLIP = "MEMORY_BIT_FLIP"
    PARAMETER_PERTURBATION = "PARAMETER_PERTURBATION"
    OUTPUT_DRIFT = "OUTPUT_DRIFT"
    INTERMITTENT_COMPUTATION = "INTERMITTENT_COMPUTATION"
    ENVIRONMENTAL_STRESS = "ENVIRONMENTAL_STRESS"
    SILENT_MODEL_DEGRADATION = "SILENT_MODEL_DEGRADATION"
