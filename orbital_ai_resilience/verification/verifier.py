"""OutputVerifier module comparing target execution outputs against reference ground-truth."""

import time
import uuid
from typing import Any, Dict, Optional, Tuple
import numpy as np
from orbital_ai_resilience.verification.evidence import VerificationEvidence, compute_tensor_hash
from orbital_ai_resilience.verification.policy import VerificationPolicy
from orbital_ai_resilience.verification.reference import ReferenceProvider
from orbital_ai_resilience.verification.types import VerificationResultState
from orbital_ai_resilience.workloads.synthetic import SyntheticAIWorkload


class OutputVerifier:
    """Independent output verification engine evaluating computational target outputs."""

    def __init__(
        self,
        policy: Optional[VerificationPolicy] = None,
        reference_provider: Optional[ReferenceProvider] = None,
    ) -> None:
        self.policy: VerificationPolicy = policy or VerificationPolicy()
        self.reference_provider: ReferenceProvider = reference_provider or ReferenceProvider()

    def verify_target_output(
        self,
        workload: SyntheticAIWorkload,
        target_output: np.ndarray,
        source_node_id: str,
        target_node_id: str,
        attempt_number: int = 1,
        fault_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[VerificationResultState, VerificationEvidence]:
        """Compare target node output tensor against trusted reference output.

        Args:
            workload: SyntheticAIWorkload instance evaluated.
            target_output: Output tensor produced by target node execution.
            source_node_id: Original source node ID.
            target_node_id: Target compute node ID being verified.
            attempt_number: Current migration attempt index.
            fault_context: Optional metadata describing fault context.

        Returns:
            Tuple of (VerificationResultState, VerificationEvidence).
        """
        ref_output = self.reference_provider.get_reference_output(workload)

        # Compute numerical comparison metrics
        mse = float(np.mean((ref_output - target_output) ** 2))
        mae = float(np.mean(np.abs(ref_output - target_output)))

        ref_flat = ref_output.reshape(-1)
        target_flat = target_output.reshape(-1)
        denom = np.linalg.norm(ref_flat) * np.linalg.norm(target_flat)
        cosine_sim = float(np.dot(ref_flat, target_flat) / denom) if denom > 1e-9 else 1.0

        # Evaluate against verification policy thresholds
        is_ok = self.policy.is_verified(mse=mse, mae=mae, cosine_sim=cosine_sim)
        state = VerificationResultState.VERIFIED if is_ok else VerificationResultState.VERIFICATION_FAILED

        # Create cryptographic hashes of output tensors
        ref_hash = compute_tensor_hash(ref_output)
        target_hash = compute_tensor_hash(target_output)

        verification_id = f"ver-{uuid.uuid4().hex[:8]}"

        evidence = VerificationEvidence(
            verification_id=verification_id,
            workload_id=workload.workload_id,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            timestamp=time.time(),
            reference_output_hash=ref_hash,
            target_output_hash=target_hash,
            mse=mse,
            mae=mae,
            cosine_sim=cosine_sim,
            thresholds={
                "mse_max": self.policy.mse_max,
                "mae_max": self.policy.mae_max,
                "cosine_min": self.policy.cosine_min,
            },
            verification_result=state,
            attempt_number=attempt_number,
            fault_context=fault_context or {},
        )

        return state, evidence
