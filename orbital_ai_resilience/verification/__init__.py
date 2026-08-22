"""Output verification package."""

from orbital_ai_resilience.verification.evidence import VerificationEvidence, compute_tensor_hash
from orbital_ai_resilience.verification.policy import VerificationPolicy
from orbital_ai_resilience.verification.reference import ReferenceProvider
from orbital_ai_resilience.verification.types import VerificationResultState
from orbital_ai_resilience.verification.verifier import OutputVerifier

__all__ = [
    "VerificationResultState",
    "VerificationEvidence",
    "compute_tensor_hash",
    "VerificationPolicy",
    "ReferenceProvider",
    "OutputVerifier",
]
