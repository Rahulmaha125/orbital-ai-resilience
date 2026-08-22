"""OrbitalPredictionModel predicting future eclipse risk, power reserves, and link availability over N ticks."""

from dataclasses import dataclass
from typing import Any, Dict, List
from orbital_ai_resilience.orbital.eclipse import EclipseModel, EclipseStatus
from orbital_ai_resilience.orbital.models import OrbitalState
from orbital_ai_resilience.orbital.propagation import OrbitalPropagator


@dataclass
class FutureOrbitalPrediction:
    """Predictive forecast of a satellite's future orbital & power conditions over N ticks."""

    satellite_id: str
    lookahead_ticks: int
    future_eclipse_risk: float
    future_power_reserve: float
    is_eclipse_imminent: bool
    predicted_min_battery_level: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize prediction to dictionary."""
        return {
            "satellite_id": self.satellite_id,
            "lookahead_ticks": self.lookahead_ticks,
            "future_eclipse_risk": round(self.future_eclipse_risk, 4),
            "future_power_reserve": round(self.future_power_reserve, 2),
            "is_eclipse_imminent": self.is_eclipse_imminent,
            "predicted_min_battery_level": round(self.predicted_min_battery_level, 2),
        }


class OrbitalPredictionModel:
    """Predicts future orbital conditions, eclipse risks, and power reserves for candidate target nodes."""

    def __init__(
        self,
        propagator: Optional[OrbitalPropagator] = None,
        eclipse_model: Optional[EclipseModel] = None,
    ) -> None:
        self.propagator: OrbitalPropagator = propagator or OrbitalPropagator()
        self.eclipse_model: EclipseModel = eclipse_model or EclipseModel()

    def predict_future_state(
        self,
        satellite_id: str,
        initial_phase_deg: float,
        current_tick: float,
        current_battery_level: float,
        lookahead_ticks: int = 6,
        workload_power_drain: float = 5.0,
    ) -> FutureOrbitalPrediction:
        """Forecast satellite illumination and battery trajectory across future ticks.

        Args:
            satellite_id: Node ID evaluated.
            initial_phase_deg: Initial orbital phase angle.
            current_tick: Current simulation tick.
            current_battery_level: Current battery percentage [0, 100].
            lookahead_ticks: Forecast window size (default 6 ticks).
            workload_power_drain: Power drain per tick during workload execution.

        Returns:
            FutureOrbitalPrediction instance.
        """
        eclipse_ticks = 0
        min_battery = current_battery_level
        batt = current_battery_level

        for step in range(1, lookahead_ticks + 1):
            future_tick = current_tick + step
            state = self.propagator.compute_state(satellite_id, initial_phase_deg, future_tick)
            status = self.eclipse_model.evaluate_illumination(state)

            if status.is_eclipse:
                eclipse_ticks += 1
                # In eclipse: battery drains due to zero solar generation
                batt = max(0.0, batt - (workload_power_drain + 2.0))
            else:
                # In sunlight: solar panels charge battery
                batt = min(100.0, batt + 4.0 - workload_power_drain)

            min_battery = min(min_battery, batt)

        eclipse_risk = round(eclipse_ticks / max(1, lookahead_ticks), 4)
        is_imminent = eclipse_risk > 0.30 or (current_battery_level < 40.0 and eclipse_ticks > 0)

        return FutureOrbitalPrediction(
            satellite_id=satellite_id,
            lookahead_ticks=lookahead_ticks,
            future_eclipse_risk=eclipse_risk,
            future_power_reserve=batt,
            is_eclipse_imminent=is_imminent,
            predicted_min_battery_level=min_battery,
        )
