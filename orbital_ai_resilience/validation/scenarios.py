"""Reusable Failure Scenario definitions for Phase 10 validation experiments."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.faults.injector import FaultInjector
from orbital_ai_resilience.faults.profile import FaultProfile, FaultType
from orbital_ai_resilience.orbital.models import OrbitalState
from orbital_ai_resilience.orbital.propagation import OrbitalPropagator
from orbital_ai_resilience.workloads.synthetic import SyntheticAIWorkload


@dataclass
class ScenarioDefinition:
    """Definition of a validation scenario including cluster size, fault profiles, and orbital positions."""

    scenario_id: int
    name: str
    description: str
    node_count: int
    fault_profiles: List[FaultProfile]
    orbital_states: Dict[str, OrbitalState]
    workload_template: SyntheticAIWorkload


class ScenarioEngine:
    """Constructs deterministic validation scenarios across 10 controlled failure conditions."""

    def __init__(self, seed: int = 42) -> None:
        self.seed: int = seed

    def create_scenario(
        self,
        scenario_id: int,
        node_count: int = 5,
    ) -> ScenarioDefinition:
        """Construct deterministic ScenarioDefinition by scenario_id."""
        cluster = VirtualCluster.create_default_cluster(num_nodes=node_count)
        propagator = OrbitalPropagator()
        states = propagator.generate_constellation_states(num_satellites=node_count, tick=0.0)

        fault_profiles: List[FaultProfile] = []
        w_task = SyntheticAIWorkload(name=f"task_sc_{scenario_id}", seed=self.seed)

        if scenario_id == 1:
            name = "Scenario 1 — Stable Operation"
            desc = "Normal orbital operation without injected failures."

        elif scenario_id == 2:
            name = "Scenario 2 — Silent AI Degradation"
            desc = "Inject behavioral degradation on node-3 while physical health remains 100%."
            fault_profiles.append(
                FaultProfile(
                    fault_type=FaultType.SILENT_MODEL_DEGRADATION,
                    target_node_id="node-3",
                    start_tick=0,
                    duration=10,
                    intensity=0.15,
                    seed=self.seed,
                )
            )

        elif scenario_id == 3:
            name = "Scenario 3 — Sudden Severe Degradation"
            desc = "Inject severe silent AI degradation (intensity 0.35) requiring fast detection & recovery."
            fault_profiles.append(
                FaultProfile(
                    fault_type=FaultType.SILENT_MODEL_DEGRADATION,
                    target_node_id="node-3",
                    start_tick=0,
                    duration=10,
                    intensity=0.35,
                    seed=self.seed,
                )
            )

        elif scenario_id == 4:
            name = "Scenario 4 — Cascading Target Failure"
            desc = "Target node-1 fails verification, triggering quarantine and retry migration to node-2."
            fault_profiles.append(
                FaultProfile(
                    fault_type=FaultType.SILENT_MODEL_DEGRADATION,
                    target_node_id="node-3",
                    start_tick=0,
                    duration=10,
                    intensity=0.15,
                    seed=self.seed,
                )
            )
            fault_profiles.append(
                FaultProfile(
                    fault_type=FaultType.MEMORY_BIT_FLIP,
                    target_node_id="node-1",
                    start_tick=0,
                    duration=10,
                    intensity=0.25,
                    seed=99,
                )
            )

        elif scenario_id == 5:
            name = "Scenario 5 — Communication Failure"
            desc = "Direct link between node-3 and target obstructed by Earth; multi-hop Dijkstra route required."
            states["node-3"] = propagator.compute_state("node-3", 0.0, 0.0)
            states["node-2"] = propagator.compute_state("node-2", 30.0, 0.0)
            states["node-4"] = propagator.compute_state("node-4", 60.0, 0.0)
            fault_profiles.append(
                FaultProfile(
                    fault_type=FaultType.SILENT_MODEL_DEGRADATION,
                    target_node_id="node-3",
                    start_tick=0,
                    duration=10,
                    intensity=0.15,
                    seed=self.seed,
                )
            )

        elif scenario_id == 6:
            name = "Scenario 6 — Eclipse Event"
            desc = "Candidate target node-1 enters Earth shadow; predictive solar model avoids shadow target."
            states["node-1"] = propagator.compute_state("node-1", 105.0, 0.0)  # Entering shadow
            fault_profiles.append(
                FaultProfile(
                    fault_type=FaultType.SILENT_MODEL_DEGRADATION,
                    target_node_id="node-3",
                    start_tick=0,
                    duration=10,
                    intensity=0.15,
                    seed=self.seed,
                )
            )

        elif scenario_id == 7:
            name = "Scenario 7 — Low Bandwidth"
            desc = "Communication bandwidth bottleneck on link requiring transfer time & bandwidth validation."
            fault_profiles.append(
                FaultProfile(
                    fault_type=FaultType.SILENT_MODEL_DEGRADATION,
                    target_node_id="node-3",
                    start_tick=0,
                    duration=10,
                    intensity=0.15,
                    seed=self.seed,
                )
            )

        elif scenario_id == 8:
            name = "Scenario 8 — Low Power"
            desc = "Target node-1 battery depleted (15%); power-aware policy selects healthy battery target node-2."
            fault_profiles.append(
                FaultProfile(
                    fault_type=FaultType.SILENT_MODEL_DEGRADATION,
                    target_node_id="node-3",
                    start_tick=0,
                    duration=10,
                    intensity=0.15,
                    seed=self.seed,
                )
            )

        elif scenario_id == 9:
            name = "Scenario 9 — Multiple Simultaneous Failures"
            desc = "Simultaneous silent degradation on node-3 and physical telemetry fault on node-5."
            fault_profiles.append(
                FaultProfile(
                    fault_type=FaultType.SILENT_MODEL_DEGRADATION,
                    target_node_id="node-3",
                    start_tick=0,
                    duration=10,
                    intensity=0.20,
                    seed=self.seed,
                )
            )
            fault_profiles.append(
                FaultProfile(
                    fault_type=FaultType.LATENCY_SPIKE,
                    target_node_id="node-5",
                    start_tick=0,
                    duration=10,
                    intensity=0.40,
                    seed=77,
                )
            )

        elif scenario_id == 10:
            name = "Scenario 10 — Large Constellation Stress"
            desc = f"Large constellation stress test across {node_count} nodes with continuous propagation & routing."
            fault_profiles.append(
                FaultProfile(
                    fault_type=FaultType.SILENT_MODEL_DEGRADATION,
                    target_node_id="node-3",
                    start_tick=0,
                    duration=10,
                    intensity=0.20,
                    seed=self.seed,
                )
            )
        else:
            raise ValueError(f"Invalid scenario_id: {scenario_id}")

        return ScenarioDefinition(
            scenario_id=scenario_id,
            name=name,
            description=desc,
            node_count=node_count,
            fault_profiles=fault_profiles,
            orbital_states=states,
            workload_template=w_task,
        )
