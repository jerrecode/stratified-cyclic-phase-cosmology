"""Persistent records for successful, rejected, and failed scan runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from scpc.models.phase import SCPCSolution
from scpc.scans.identity import RunIdentity
from scpc.scans.outcomes import OutcomeAssessment


class RunStatus(StrEnum):
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


class FailureClass(StrEnum):
    INVALID_INITIAL_CONSTRAINT = "invalid_initial_constraint"
    CONFIGURATION_ERROR = "configuration_error"
    PHYSICAL_DOMAIN_FAILURE = "physical_domain_failure"
    SOLVER_FAILURE = "solver_failure"
    UNEXPECTED_ERROR = "unexpected_error"


@dataclass(frozen=True)
class RunRecord:
    """Serializable index record for one immutable experiment specification."""

    run_id: str
    run_sha256: str
    status: RunStatus
    specification: dict[str, Any]
    outcome: str | None = None
    reason: str | None = None
    numerically_valid: bool | None = None
    failure_class: FailureClass | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    bounce_count: int = 0
    turnaround_count: int = 0
    degenerate_count: int = 0
    event_sequence: tuple[str, ...] = ()
    max_abs_constraint_residual: float | None = None
    return_sequence_classifications: dict[str, str] | None = None
    solver_metadata: dict[str, Any] | None = None
    trajectory_path: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        mapping = asdict(self)
        mapping["status"] = self.status.value
        mapping["failure_class"] = self.failure_class.value if self.failure_class else None
        mapping["event_sequence"] = list(self.event_sequence)
        return mapping

    def to_flat_row(self) -> dict[str, Any]:
        """Return a CSV-safe row while preserving nested values as JSON."""

        mapping = self.to_mapping()
        for key in (
            "specification",
            "event_sequence",
            "return_sequence_classifications",
            "solver_metadata",
        ):
            mapping[key] = json.dumps(
                mapping[key],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        return mapping


def completed_run_record(
    identity: RunIdentity,
    specification: dict[str, Any],
    assessment: OutcomeAssessment,
    solution: SCPCSolution,
    *,
    trajectory_path: str | Path | None = None,
) -> RunRecord:
    status = RunStatus.COMPLETED if assessment.numerically_valid else RunStatus.REJECTED
    return RunRecord(
        run_id=identity.run_id,
        run_sha256=identity.sha256,
        status=status,
        specification=specification,
        outcome=assessment.outcome.value,
        reason=assessment.reason,
        numerically_valid=assessment.numerically_valid,
        bounce_count=assessment.bounce_count,
        turnaround_count=assessment.turnaround_count,
        degenerate_count=assessment.degenerate_count,
        event_sequence=assessment.event_sequence,
        max_abs_constraint_residual=assessment.max_abs_constraint_residual,
        return_sequence_classifications=assessment.return_sequence_classifications,
        solver_metadata=dict(solution.solver_metadata),
        trajectory_path=str(trajectory_path) if trajectory_path is not None else None,
    )


def classify_exception(error: Exception) -> FailureClass:
    message = str(error).lower()
    if isinstance(error, ValueError) and "initial state violates the friedmann constraint" in message:
        return FailureClass.INVALID_INITIAL_CONSTRAINT
    if isinstance(error, FloatingPointError):
        return FailureClass.PHYSICAL_DOMAIN_FAILURE
    if isinstance(error, RuntimeError) and "integration failed" in message:
        return FailureClass.SOLVER_FAILURE
    if isinstance(error, (KeyError, TypeError, ValueError)):
        return FailureClass.CONFIGURATION_ERROR
    return FailureClass.UNEXPECTED_ERROR


def failed_run_record(
    identity: RunIdentity,
    specification: dict[str, Any],
    error: Exception,
) -> RunRecord:
    """Record an exception without promoting it to a physical singularity."""

    failure_class = classify_exception(error)
    return RunRecord(
        run_id=identity.run_id,
        run_sha256=identity.sha256,
        status=RunStatus.FAILED,
        specification=specification,
        reason="The numerical experiment did not return a completed trajectory.",
        numerically_valid=False,
        failure_class=failure_class,
        exception_type=type(error).__name__,
        exception_message=str(error),
    )
