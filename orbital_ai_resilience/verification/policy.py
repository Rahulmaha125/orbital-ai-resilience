"""Verification policy defining threshold constraints for output integrity."""

from dataclasses import dataclass


@dataclass
class VerificationPolicy:
    """Configurable thresholds for post-migration output verification.

    Attributes:
        mse_max: Maximum allowable Mean Squared Error vs. reference output (default 0.001).
        mae_max: Maximum allowable Mean Absolute Error vs. reference output (default 0.01).
        cosine_min: Minimum allowable Cosine Similarity vs. reference output (default 0.999).
    """

    mse_max: float = 0.001
    mae_max: float = 0.01
    cosine_min: float = 0.999

    def is_verified(self, mse: float, mae: float, cosine_sim: float) -> bool:
        """Check if metrics satisfy all verification thresholds."""
        return (
            mse <= self.mse_max
            and mae <= self.mae_max
            and cosine_sim >= self.cosine_min
        )
