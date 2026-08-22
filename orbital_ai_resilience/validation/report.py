"""ResearchReportGenerator generating comprehensive scientific reports and % Delta policy comparison metrics."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from orbital_ai_resilience.validation.experiment import ExperimentResult
from orbital_ai_resilience.validation.scalability import ScalabilityResult


@dataclass
class PolicyComparisonDelta:
    """Scientific delta improvement metric comparing a new policy against baseline."""

    metric_name: str
    baseline_value: float
    new_policy_value: float
    delta_pct: float
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize delta metric to dictionary."""
        return {
            "metric_name": self.metric_name,
            "baseline_value": round(self.baseline_value, 4),
            "new_policy_value": round(self.new_policy_value, 4),
            "delta_pct": round(self.delta_pct, 2),
            "interpretation": self.interpretation,
        }


class ResearchReportGenerator:
    """Generates scientific research reports and calculates honest improvement percentages."""

    def __init__(self) -> None:
        pass

    def calculate_policy_deltas(
        self,
        baseline_result: ExperimentResult,
        new_policy_result: ExperimentResult,
    ) -> List[PolicyComparisonDelta]:
        """Calculate honest improvement percentages between Baseline and New Policy.

        Formula:
            Higher is better: ((New - Baseline) / Baseline) * 100
            Lower is better: ((Baseline - New) / Baseline) * 100
        """
        base_m = baseline_result.metrics
        new_m = new_policy_result.metrics

        deltas = []

        # 1. Recovery Success Rate (Higher is better)
        d_rec = self._calc_delta(base_m.recovery_success_rate, new_m.recovery_success_rate, higher_is_better=True)
        deltas.append(
            PolicyComparisonDelta(
                metric_name="Recovery Success Rate",
                baseline_value=base_m.recovery_success_rate,
                new_policy_value=new_m.recovery_success_rate,
                delta_pct=d_rec,
                interpretation=f"Improved by {d_rec:.2f}%" if d_rec >= 0 else f"Worse by {abs(d_rec):.2f}%",
            )
        )

        # 2. Verification Success Rate (Higher is better)
        d_ver = self._calc_delta(base_m.verification_success_rate, new_m.verification_success_rate, higher_is_better=True)
        deltas.append(
            PolicyComparisonDelta(
                metric_name="Verification Success Rate",
                baseline_value=base_m.verification_success_rate,
                new_policy_value=new_m.verification_success_rate,
                delta_pct=d_ver,
                interpretation=f"Improved by {d_ver:.2f}%" if d_ver >= 0 else f"Worse by {abs(d_ver):.2f}%",
            )
        )

        # 3. Workload Survival Rate (Higher is better)
        d_surv = self._calc_delta(base_m.workload_recovery_rate, new_m.workload_recovery_rate, higher_is_better=True)
        deltas.append(
            PolicyComparisonDelta(
                metric_name="Workload Survival Rate",
                baseline_value=base_m.workload_recovery_rate,
                new_policy_value=new_m.workload_recovery_rate,
                delta_pct=d_surv,
                interpretation=f"Improved by {d_surv:.2f}%" if d_surv >= 0 else f"Worse by {abs(d_surv):.2f}%",
            )
        )

        # 4. Total Recovery Cost (Lower is better)
        d_cost = self._calc_delta(base_m.average_recovery_cost, new_m.average_recovery_cost, higher_is_better=False)
        deltas.append(
            PolicyComparisonDelta(
                metric_name="Total Recovery Cost",
                baseline_value=base_m.average_recovery_cost,
                new_policy_value=new_m.average_recovery_cost,
                delta_pct=d_cost,
                interpretation=f"Reduced by {d_cost:.2f}%" if d_cost >= 0 else f"Increased by {abs(d_cost):.2f}% (Worse)",
            )
        )

        return deltas

    def _calc_delta(self, base_val: float, new_val: float, higher_is_better: bool = True) -> float:
        """Calculate percentage delta with zero division handling."""
        if base_val == 0.0:
            return 0.0 if new_val == 0.0 else 100.0 if higher_is_better else -100.0

        if higher_is_better:
            return ((new_val - base_val) / abs(base_val)) * 100.0
        else:
            return ((base_val - new_val) / abs(base_val)) * 100.0

    def generate_full_markdown_report(
        self,
        results: List[ExperimentResult],
        scalability_results: List[ScalabilityResult],
    ) -> str:
        """Generate scientific markdown research report summarizing experimental findings."""
        report = []
        report.append("# Continuous Autonomous Orbital AI Resilience Research Report\n")
        report.append("## 1. Executive Summary\n")
        report.append(f"Total Experiments Executed: **{len(results)}**")
        report.append(f"Constellation Scalability Evaluated: **{len(scalability_results)} sizes**\n")

        report.append("## 2. Policy Performance Comparison\n")
        report.append("| Experiment ID | Scenario | Policy | Recovery Success | Verification Success | Workload Survival | Recovery Cost |")
        report.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: |")

        for r in results:
            m = r.metrics
            report.append(
                f"| `{r.experiment_id}` | {r.scenario_name} | {r.policy_name} | {m.recovery_success_rate * 100:.1f}% | {m.verification_success_rate * 100:.1f}% | {m.workload_recovery_rate * 100:.1f}% | {m.average_recovery_cost:.2f} |"
            )

        report.append("\n## 3. Scalability Analysis\n")
        report.append("| Constellation Size | Total Ticks | Total Duration (s) | Avg Tick Duration (s) | Active Crosslinks | Recoveries | Workload Survival |")
        report.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

        for s in scalability_results:
            report.append(
                f"| **{s.node_count} Nodes** | {s.total_ticks} | {s.total_duration_sec:.3f}s | {s.avg_tick_duration_sec:.5f}s | {s.active_links_count} | {s.recovery_count} | {s.workload_survival_rate * 100:.1f}% |"
            )

        return "\n".join(report)
