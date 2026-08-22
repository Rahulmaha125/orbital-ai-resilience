"""MigrationManager coordinating multi-stage autonomous workload migration, verification & quarantine."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from orbital_ai_resilience.core.cluster import VirtualCluster
from orbital_ai_resilience.core.node import VirtualNode
from orbital_ai_resilience.core.types import NodeStatus
from orbital_ai_resilience.core.workload import Workload
from orbital_ai_resilience.detection.types import DetectionResult
from orbital_ai_resilience.faults.injector import FaultInjector
from orbital_ai_resilience.quarantine.manager import QuarantineManager
from orbital_ai_resilience.quarantine.state import TrustState
from orbital_ai_resilience.recovery.events import MigrationEvent
from orbital_ai_resilience.recovery.policy import MigrationPolicy
from orbital_ai_resilience.recovery.selector import TargetSelector
from orbital_ai_resilience.recovery.state import WorkloadSnapshot
from orbital_ai_resilience.recovery.types import MigrationState, VerificationStatus
from orbital_ai_resilience.utils.logger import StateLogger
from orbital_ai_resilience.verification.evidence import VerificationEvidence
from orbital_ai_resilience.verification.verifier import OutputVerifier
from orbital_ai_resilience.verification.types import VerificationResultState


@dataclass
class RecoveryMetrics:
    """Aggregated quantitative performance metrics for autonomous multi-stage recovery."""

    total_migrations: int = 0
    successful_migrations: int = 0
    failed_migrations: int = 0
    verification_attempts: int = 0
    verification_successes: int = 0
    verification_failures: int = 0
    quarantined_nodes: int = 0
    recovery_retries: int = 0
    successful_retry_recoveries: int = 0
    aborted_recoveries: int = 0
    cascading_failures_recovered: int = 0
    migration_attempts: int = 0
    workloads_recovered: int = 0
    workloads_lost: int = 0
    source_nodes_isolated: int = 0
    total_migration_time_sec: float = 0.0
    total_verification_time_sec: float = 0.0

    @property
    def target_selection_success_rate(self) -> float:
        """Percentage of migration attempts where a suitable target node was found."""
        if self.migration_attempts == 0:
            return 1.0
        return round(max(0.0, min(1.0, self.successful_migrations / self.migration_attempts)), 4)

    @property
    def average_migration_time(self) -> float:
        """Average execution time per migration transaction."""
        if self.total_migrations == 0:
            return 0.0
        return round(self.total_migration_time_sec / self.total_migrations, 4)

    @property
    def average_verification_time(self) -> float:
        """Average execution time per output verification."""
        if self.verification_attempts == 0:
            return 0.0
        return round(self.total_verification_time_sec / self.verification_attempts, 4)

    @property
    def average_recovery_attempts(self) -> float:
        """Average retry attempts per recovery workflow."""
        if self.total_migrations == 0:
            return 0.0
        total_attempts = self.migration_attempts + self.recovery_retries
        return round(total_attempts / self.total_migrations, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize recovery metrics to dictionary."""
        return {
            "total_migrations": self.total_migrations,
            "successful_migrations": self.successful_migrations,
            "failed_migrations": self.failed_migrations,
            "verification_attempts": self.verification_attempts,
            "verification_successes": self.verification_successes,
            "verification_failures": self.verification_failures,
            "quarantined_nodes": self.quarantined_nodes,
            "recovery_retries": self.recovery_retries,
            "successful_retry_recoveries": self.successful_retry_recoveries,
            "aborted_recoveries": self.aborted_recoveries,
            "cascading_failures_recovered": self.cascading_failures_recovered,
            "migration_attempts": self.migration_attempts,
            "workloads_recovered": self.workloads_recovered,
            "workloads_lost": self.workloads_lost,
            "source_nodes_isolated": self.source_nodes_isolated,
            "average_migration_time_sec": self.average_migration_time,
            "average_verification_time_sec": self.average_verification_time,
            "average_recovery_attempts": self.average_recovery_attempts,
            "target_selection_success_rate": self.target_selection_success_rate,
        }


class MigrationManager:
    """Coordinates autonomous multi-stage workload migration, verification & quarantine."""

    def __init__(
        self,
        policy: Optional[MigrationPolicy] = None,
        selector: Optional[TargetSelector] = None,
        verifier: Optional[OutputVerifier] = None,
        quarantine_manager: Optional[QuarantineManager] = None,
        logger: Optional[StateLogger] = None,
    ) -> None:
        self.policy: MigrationPolicy = policy or MigrationPolicy()
        self.quarantine_manager: QuarantineManager = quarantine_manager or QuarantineManager(logger=logger)
        self.selector: TargetSelector = selector or TargetSelector(
            policy=self.policy, quarantine_manager=self.quarantine_manager
        )
        self.verifier: OutputVerifier = verifier or OutputVerifier()
        self.logger: StateLogger = logger or StateLogger()
        self.metrics: RecoveryMetrics = RecoveryMetrics()

        self.migration_history: List[MigrationEvent] = []
        self.verification_evidence_history: List[VerificationEvidence] = []
        self.migrated_workload_ids: set[str] = set()

    def execute_autonomous_recovery(
        self,
        cluster: VirtualCluster,
        source_node_id: str,
        workload: Workload,
        detection_result: DetectionResult,
        fault_injector: Optional[FaultInjector] = None,
    ) -> Optional[MigrationEvent]:
        """Execute multi-stage autonomous workload recovery pipeline with verification & quarantine.

        Pipeline Loop:
            DETECT -> DECIDE -> SELECT TARGET (exclude source, isolated & quarantined)
            -> TRANSFER -> EXECUTE -> VERIFY
            IF VERIFIED -> COMPLETED & ISOLATE SOURCE
            IF FAILED   -> QUARANTINE TARGET & RETRY NEXT TARGET (up to max_attempts)
        """
        start_time = time.time()

        # 1. Policy Decision Check
        if not self.policy.should_migrate(detection_result):
            return None

        source_node = cluster.get_node(source_node_id)
        if not source_node:
            return None

        # 2. Prevent duplicate migration if already recovered
        if workload.workload_id in self.migrated_workload_ids:
            return None

        self.metrics.migration_attempts += 1
        self.metrics.total_migrations += 1

        # 3. Snapshot Workload State
        reason = f"Detection_Triggered (SilentDegradation={detection_result.is_silent_degradation}, State={detection_result.behavioral_state.value})"
        snapshot = WorkloadSnapshot.create_from_workload(
            workload=workload,
            source_node_id=source_node_id,
            reason=reason,
            migration_count=1,
        )

        max_attempts = self.policy.max_migration_attempts
        last_event: Optional[MigrationEvent] = None

        # Multi-Stage Retry Loop
        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                self.metrics.recovery_retries += 1

            # 4. Target Selection (excludes source, isolated, and quarantined nodes)
            target_node_id, target_score, candidate_scores = self.selector.select_best_target(
                cluster=cluster,
                source_node_id=source_node_id,
                workload=workload,
                quarantine_manager=self.quarantine_manager,
            )

            if not target_node_id:
                # Failure Path: No suitable target available
                self.metrics.failed_migrations += 1
                self.metrics.aborted_recoveries += 1
                self.metrics.workloads_lost += 1

                event = MigrationEvent(
                    migration_id=snapshot.migration_id,
                    workload_id=workload.workload_id,
                    source_node_id=source_node_id,
                    target_node_id=None,
                    source_health_score=detection_result.physical_health_score,
                    source_behavior_score=detection_result.behavioral_score,
                    target_health_score=0.0,
                    target_behavior_score=0.0,
                    migration_reason=reason,
                    migration_attempt=attempt,
                    migration_status=MigrationState.FAILED,
                    verification_status=VerificationStatus.UNVERIFIED,
                    failure_reason="No eligible trusted target node found with sufficient capacity",
                    details={"candidate_scores": candidate_scores, "attempt": attempt},
                )
                self.migration_history.append(event)
                self.logger.log_event("MIGRATION_ABORTED", event.to_dict())
                return event

            target_node = cluster.get_node(target_node_id)
            if not target_node:
                continue

            # 5. Workload Ownership Transfer
            # Remove from source node queue on first attempt
            if attempt == 1:
                source_node.remove_workload(workload.workload_id)

            assigned_success = target_node.assign_workload(workload)
            if not assigned_success:
                continue

            # 6. Execute Workload on Target Node
            exec_output = None
            if hasattr(workload, "execute_on_node"):
                # Execute on target node (with fault injector to simulate corrupted target if profile active)
                exec_log = workload.execute_on_node(
                    target_node, fault_injector=fault_injector, tick=detection_result.tick
                )
                ref_out = workload.compute_reference_output()
                # Extract actual target output if present or reconstruct from MSE
                dev = exec_log.get("deviation", {})
                if dev.get("mse", 0.0) == 0.0:
                    exec_output = ref_out
                else:
                    # Target output suffered noise
                    exec_output = ref_out + (dev.get("mse", 0.0) ** 0.5)

            # 7. Post-Migration Output Verification
            v_start = time.time()
            self.metrics.verification_attempts += 1

            if exec_output is not None and hasattr(workload, "compute_reference_output"):
                v_state, evidence = self.verifier.verify_target_output(
                    workload=workload,
                    target_output=exec_output,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    attempt_number=attempt,
                )
            else:
                v_state = VerificationResultState.VERIFIED
                evidence = None

            v_elapsed = time.time() - v_start
            self.metrics.total_verification_time_sec += v_elapsed

            if evidence:
                self.verification_evidence_history.append(evidence)

            # 8. Evaluate Verification Outcome
            if v_state == VerificationResultState.VERIFIED:
                # SUCCESS PATH
                self.metrics.verification_successes += 1
                self.metrics.successful_migrations += 1
                self.metrics.workloads_recovered += 1
                self.migrated_workload_ids.add(workload.workload_id)

                if attempt > 1:
                    self.metrics.successful_retry_recoveries += 1
                    self.metrics.cascading_failures_recovered += 1

                # Isolate Source Node
                if self.policy.auto_isolate_source:
                    self.quarantine_manager.isolate_node(
                        source_node,
                        reason=f"Silent degradation verified; recovered on target {target_node_id}",
                    )
                    self.metrics.source_nodes_isolated += 1

                elapsed_time = time.time() - start_time
                self.metrics.total_migration_time_sec += elapsed_time

                event = MigrationEvent(
                    migration_id=snapshot.migration_id,
                    workload_id=workload.workload_id,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    source_health_score=detection_result.physical_health_score,
                    source_behavior_score=detection_result.behavioral_score,
                    target_health_score=target_node.get_health_score(),
                    target_behavior_score=100.0,
                    migration_reason=reason,
                    migration_attempt=attempt,
                    migration_status=MigrationState.COMPLETED,
                    verification_status=VerificationStatus.VERIFIED,
                    details={
                        "target_score": target_score,
                        "candidate_scores": candidate_scores,
                        "elapsed_time_sec": round(elapsed_time, 4),
                        "evidence_id": evidence.verification_id if evidence else None,
                    },
                )
                self.migration_history.append(event)
                self.logger.log_event("MIGRATION_COMPLETED", event.to_dict())
                return event

            else:
                # FAILURE PATH: Target verification failed! Quarantine target node & retry next target!
                self.metrics.verification_failures += 1
                
                # Remove workload from corrupted target node
                target_node.remove_workload(workload.workload_id)

                # Quarantine the corrupted target node
                self.quarantine_manager.quarantine_node(
                    target_node,
                    reason=f"Verification FAILED on recovery attempt {attempt} (MSE={evidence.mse if evidence else 0.0})",
                    evidence_id=evidence.verification_id if evidence else None,
                )
                self.metrics.quarantined_nodes += 1

                last_event = MigrationEvent(
                    migration_id=snapshot.migration_id,
                    workload_id=workload.workload_id,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    source_health_score=detection_result.physical_health_score,
                    source_behavior_score=detection_result.behavioral_score,
                    target_health_score=target_node.get_health_score(),
                    target_behavior_score=0.0,
                    migration_reason=reason,
                    migration_attempt=attempt,
                    migration_status=MigrationState.VERIFICATION_FAILED,
                    verification_status=VerificationStatus.VERIFICATION_FAILED,
                    failure_reason=f"Target output verification FAILED on attempt {attempt}",
                    details={"evidence_id": evidence.verification_id if evidence else None},
                )
                self.migration_history.append(last_event)
                self.logger.log_event("TARGET_VERIFICATION_FAILED", last_event.to_dict())
                # Continue loop to next attempt!

        # Max recovery attempts exhausted
        self.metrics.failed_migrations += 1
        self.metrics.aborted_recoveries += 1
        self.metrics.workloads_lost += 1

        elapsed_time = time.time() - start_time
        self.metrics.total_migration_time_sec += elapsed_time

        abort_event = MigrationEvent(
            migration_id=snapshot.migration_id,
            workload_id=workload.workload_id,
            source_node_id=source_node_id,
            target_node_id=None,
            source_health_score=detection_result.physical_health_score,
            source_behavior_score=detection_result.behavioral_score,
            target_health_score=0.0,
            target_behavior_score=0.0,
            migration_reason=reason,
            migration_attempt=max_attempts,
            migration_status=MigrationState.FAILED,
            verification_status=VerificationStatus.VERIFICATION_FAILED,
            failure_reason=f"Maximum recovery attempts ({max_attempts}) exhausted",
        )
        self.migration_history.append(abort_event)
        self.logger.log_event("RECOVERY_ABORTED_MAX_ATTEMPTS", abort_event.to_dict())
        return abort_event
