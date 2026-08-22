"""ExperimentRunner executing 3-policy comparisons and exporting ExperimentResult to JSON & CSV."""

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from orbital_ai_resilience.validation.metrics import ValidationMetrics
from orbital_ai_resilience.validation.reproducibility import ExperimentConfig, ReproducibilityManager
from orbital_ai_resilience.validation.scenarios import ScenarioDefinition, ScenarioEngine
from orbital_ai_resilience.validation.simulation import SimulationConfig, SimulationEngine


@dataclass
class ExperimentResult:
    """Complete experimental result container capturing metrics and configuration metadata."""

    experiment_id: str
    scenario_id: int
    scenario_name: str
    policy_name: str
    node_count: int
    total_ticks: int
    seed: int
    metrics: ValidationMetrics
    config: ExperimentConfig

    def to_dict(self) -> Dict[str, Any]:
        """Serialize experiment result to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "policy_name": self.policy_name,
            "node_count": self.node_count,
            "total_ticks": self.total_ticks,
            "seed": self.seed,
            "metrics": self.metrics.to_dict(),
            "config_hash": self.config.compute_config_hash(),
        }


class ExperimentRunner:
    """Executes scientific multi-policy comparison experiments across scenarios and exports results."""

    def __init__(self, seed: int = 42) -> None:
        self.seed: int = seed
        self.scenario_engine = ScenarioEngine(seed=seed)
        self.reproducibility_manager = ReproducibilityManager(seed=seed)

    def run_policy_scenario_experiment(
        self,
        scenario_id: int,
        policy_name: str,
        node_count: int = 5,
        ticks: int = 50,
    ) -> ExperimentResult:
        """Run single policy scenario experiment with scientific fairness."""
        exp_id = f"exp_sc{scenario_id}_{policy_name.replace(' ', '_').lower()}_n{node_count}_t{ticks}"
        sc_def = self.scenario_engine.create_scenario(scenario_id=scenario_id, node_count=node_count)

        exp_cfg = self.reproducibility_manager.create_experiment_config(
            experiment_id=exp_id,
            scenario_id=scenario_id,
            scenario_name=sc_def.name,
            policy_name=policy_name,
            node_count=node_count,
            total_ticks=ticks,
            seed=self.seed,
        )

        sim_cfg = SimulationConfig(
            node_count=node_count,
            ticks=ticks,
            seed=self.seed,
            policy_name=policy_name,
        )

        engine = SimulationEngine(config=sim_cfg)
        for profile in sc_def.fault_profiles:
            engine.controller.fault_injector.add_profile(profile)

        summaries, metrics = engine.run()
        metrics.scenario_name = sc_def.name

        return ExperimentResult(
            experiment_id=exp_id,
            scenario_id=scenario_id,
            scenario_name=sc_def.name,
            policy_name=policy_name,
            node_count=node_count,
            total_ticks=ticks,
            seed=self.seed,
            metrics=metrics,
            config=exp_cfg,
        )

    def run_3policy_comparison(
        self,
        scenario_id: int,
        node_count: int = 5,
        ticks: int = 50,
    ) -> List[ExperimentResult]:
        """Run identical scenario across Baseline, Adaptive, and Orbital-Aware policies under strict scientific fairness."""
        policies = ["Baseline Deterministic", "Adaptive Recovery", "Phase 9 Orbital Policy"]
        results = []
        for p in policies:
            res = self.run_policy_scenario_experiment(
                scenario_id=scenario_id,
                policy_name=p,
                node_count=node_count,
                ticks=ticks,
            )
            results.append(res)
        return results

    def export_to_json(self, results: List[ExperimentResult], file_path: str) -> str:
        """Export experiment results to JSON file."""
        data = [r.to_dict() for r in results]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return file_path

    def export_to_csv(self, results: List[ExperimentResult], file_path: str) -> str:
        """Export experiment results to CSV file using pandas."""
        data = []
        for r in results:
            flat = {
                "experiment_id": r.experiment_id,
                "scenario_id": r.scenario_id,
                "scenario_name": r.scenario_name,
                "policy_name": r.policy_name,
                "node_count": r.node_count,
                "total_ticks": r.total_ticks,
                "seed": r.seed,
                **r.metrics.to_dict(),
            }
            data.append(flat)

        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        return file_path
