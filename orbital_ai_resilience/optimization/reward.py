"""Reward and objective function calculator for recovery optimization evaluation."""

from dataclasses import dataclass
from typing import Any, Dict
from orbital_ai_resilience.verification.types import VerificationResultState


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components."""

    recovery_success_reward: float
    verification_success_reward: float
    cost_penalty: float
    verification_failure_penalty: float
    quarantine_penalty: float
    workload_loss_penalty: float
    total_reward: float

    def to_dict(self) -> Dict[str, float]:
        """Serialize reward breakdown to dictionary."""
        return {
            "recovery_success_reward": round(self.recovery_success_reward, 2),
            "verification_success_reward": round(self.verification_success_reward, 2),
            "cost_penalty": round(self.cost_penalty, 2),
            "verification_failure_penalty": round(self.verification_failure_penalty, 2),
            "quarantine_penalty": round(self.quarantine_penalty, 2),
            "workload_loss_penalty": round(self.workload_loss_penalty, 2),
            "total_reward": round(self.total_reward, 2),
        }


class RewardCalculator:
    """Calculates quantitative scalar rewards for evaluated recovery decisions.

    Reward = + R_recovery + R_verification - RecoveryCost - P_verif_fail - P_quarantine - P_loss
    """

    def __init__(
        self,
        reward_recovery_success: float = 100.0,
        reward_verification_success: float = 50.0,
        penalty_verification_failure: float = 75.0,
        penalty_quarantine: float = 100.0,
        penalty_workload_loss: float = 200.0,
    ) -> None:
        self.reward_recovery_success: float = reward_recovery_success
        self.reward_verification_success: float = reward_verification_success
        self.penalty_verification_failure: float = penalty_verification_failure
        self.penalty_quarantine: float = penalty_quarantine
        self.penalty_workload_loss: float = penalty_workload_loss

    def calculate_reward(
        self,
        is_success: bool,
        verification_state: VerificationResultState,
        was_quarantined: bool,
        recovery_cost: float,
        is_workload_lost: bool = False,
    ) -> RewardBreakdown:
        """Calculate total scalar reward and component breakdown.

        Args:
            is_success: True if recovery completed successfully.
            verification_state: VerificationResultState enum.
            was_quarantined: True if target node was quarantined during attempt.
            recovery_cost: Total recovery cost float.
            is_workload_lost: True if recovery aborted and workload was lost.

        Returns:
            RewardBreakdown instance.
        """
        r_rec = self.reward_recovery_success if is_success else 0.0
        r_verif = self.reward_verification_success if verification_state == VerificationResultState.VERIFIED else 0.0

        p_cost = max(0.0, recovery_cost)
        p_verif_fail = self.penalty_verification_failure if verification_state == VerificationResultState.VERIFICATION_FAILED else 0.0
        p_quarantine = self.penalty_quarantine if was_quarantined else 0.0
        p_loss = self.penalty_workload_loss if is_workload_lost else 0.0

        total = r_rec + r_verif - p_cost - p_verif_fail - p_quarantine - p_loss

        return RewardBreakdown(
            recovery_success_reward=r_rec,
            verification_success_reward=r_verif,
            cost_penalty=p_cost,
            verification_failure_penalty=p_verif_fail,
            quarantine_penalty=p_quarantine,
            workload_loss_penalty=p_loss,
            total_reward=round(total, 2),
        )
