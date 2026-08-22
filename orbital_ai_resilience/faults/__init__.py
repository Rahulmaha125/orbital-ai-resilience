"""Fault injection and silent degradation package."""

from orbital_ai_resilience.faults.injector import FaultInjector
from orbital_ai_resilience.faults.profile import FaultProfile
from orbital_ai_resilience.faults.types import FaultType

__all__ = ["FaultType", "FaultProfile", "FaultInjector"]
