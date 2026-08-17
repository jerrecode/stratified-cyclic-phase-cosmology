"""Stage-1 recurrence-candidate verification for homogeneous SCPC backgrounds.

This module deliberately stops before Floquet/Poincare stability analysis.  Its
purpose is to decide whether a *background candidate* is numerically reproduced
well enough to be admissible input to Stage 2.  A positive result is not a
stability, perturbative-viability, or observational-support claim.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from scpc.models.phase import SCPCParameters, SCPCSolution, integrate_scpc
from scpc.numerics.convergence import compare_solutions
from scpc.numerics.cycles import CycleReturnMetric, cycle_return_metrics
from scpc.numerics.turning_audit import audit_turning_points


@dataclass(frozen=True)
class EventTopologyDifference:
    """Difference between two exact root-localized event sequences."""

    event_count_reference: int
    event_count_candidate: int
    event_count_match: bool
    event_sequence_match: bool
    termination_kind_match: bool
    completion_match: bool
    max_normalized_event_time_error: float | None
    max_normalized_event_state_error: float | None
    maximum: float


@dataclass(frozen=True)
class CandidateRunSummary:
    """Scientific diagnostics for one member of the verification ensemble."""

    label: str
    method: str
    rtol: float
    atol: float
    completed_to_requested_end: bool
    termination_kind: str | None
    max_abs_constraint_residual: float
    turning_event_count: int
    turning_sequence: tuple[str, ...]
    return_metrics: tuple[dict[str, object], ...]
    close_return_kinds: tuple[str, ...]
    turning_audit_consistent: bool


def _exact_event_states(solution: SCPCSolution) -> np.ndarray:
    count = len(solution.turning_times)
    if solution.turning_state_vectors is None:
        if count:
            raise ValueError("exact turning_state_vectors are required for candidate verification")
        return np.empty((0, 4), dtype=float)
    states = np.asarray(solution.turning_state_vectors, dtype=float)
    if states.shape != (count, 4) or np.any(~np.isfinite(states)):
        raise ValueError("turning_state_vectors must be finite with shape (events, 4)")
    return states


def _relative_error(reference: float, candidate: float, floor: float) -> float:
    return abs(candidate - reference) / max(abs(reference), abs(candidate), floor)


def compare_event_topology(
    reference: SCPCSolution,
    candidate: SCPCSolution,
    *,
    relative_floor: float = 1.0e-12,
) -> EventTopologyDifference:
    """Compare exact event sequence, times, states, completion, and termination.

    The scalar-field coordinate is compared on its unwrapped lift, normalized by
    the potential period.  This intentionally detects hidden winding/phase slips
    even for a circular target space.
    """

    if relative_floor <= 0.0:
        raise ValueError("relative_floor must be positive")
    ref_times = np.asarray(reference.turning_times, dtype=float)
    cand_times = np.asarray(candidate.turning_times, dtype=float)
    ref_sequence = tuple(reference.turning_kinds)
    cand_sequence = tuple(candidate.turning_kinds)
    if ref_times.ndim != 1 or cand_times.ndim != 1:
        raise ValueError("turning times must be one-dimensional")
    if ref_times.size != len(ref_sequence) or cand_times.size != len(cand_sequence):
        raise ValueError("turning times and kinds must have equal lengths")

    ref_states = _exact_event_states(reference)
    cand_states = _exact_event_states(candidate)
    count_match = ref_times.size == cand_times.size
    sequence_match = count_match and ref_sequence == cand_sequence
    termination_match = reference.termination_kind == candidate.termination_kind
    completion_match = reference.completed_to_requested_end == candidate.completed_to_requested_end

    time_error: float | None = None
    state_error: float | None = None
    if count_match and ref_times.size:
        duration = max(
            abs(float(reference.t[-1] - reference.t[0])),
            abs(float(candidate.t[-1] - candidate.t[0])),
            relative_floor,
        )
        time_error = float(np.max(np.abs(cand_times - ref_times)) / duration)

        period = max(reference.parameters.potential.potential_period, relative_floor)
        event_errors: list[float] = []
        for ref_state, cand_state in zip(ref_states, cand_states, strict=True):
            event_errors.extend(
                [
                    _relative_error(float(ref_state[0]), float(cand_state[0]), relative_floor),
                    abs(float(cand_state[1] - ref_state[1])),
                    abs(float(cand_state[2] - ref_state[2])) / period,
                    _relative_error(float(ref_state[3]), float(cand_state[3]), relative_floor),
                ]
            )
        state_error = float(max(event_errors, default=0.0))
    elif count_match:
        time_error = 0.0
        state_error = 0.0

    numeric_errors = [value for value in (time_error, state_error) if value is not None]
    if not (count_match and sequence_match and termination_match and completion_match):
        maximum = float("inf")
    else:
        maximum = float(max(numeric_errors, default=0.0))
    return EventTopologyDifference(
        event_count_reference=int(ref_times.size),
        event_count_candidate=int(cand_times.size),
        event_count_match=bool(count_match),
        event_sequence_match=bool(sequence_match),
        termination_kind_match=bool(termination_match),
        completion_match=bool(completion_match),
        max_normalized_event_time_error=time_error,
        max_normalized_event_state_error=state_error,
        maximum=maximum,
    )


def _close_return_kinds(
    metrics: Iterable[CycleReturnMetric],
    *,
    max_return_error: float,
    minimum_metrics: int,
) -> tuple[str, ...]:
    grouped: dict[str, list[CycleReturnMetric]] = {}
    for metric in metrics:
        grouped.setdefault(metric.kind, []).append(metric)
    return tuple(
        sorted(
            kind
            for kind, values in grouped.items()
            if len(values) >= minimum_metrics
            and all(np.isfinite(item.maximum_error) and item.maximum_error <= max_return_error for item in values)
        )
    )


def _run_summary(
    label: str,
    method: str,
    rtol: float,
    atol: float,
    solution: SCPCSolution,
    *,
    max_return_error: float,
    minimum_metrics: int,
) -> CandidateRunSummary:
    metrics = cycle_return_metrics(solution)
    audits = audit_turning_points(solution)
    return CandidateRunSummary(
        label=label,
        method=method,
        rtol=rtol,
        atol=atol,
        completed_to_requested_end=bool(solution.completed_to_requested_end),
        termination_kind=solution.termination_kind,
        max_abs_constraint_residual=float(np.max(np.abs(solution.constraint_residual))),
        turning_event_count=len(solution.turning_times),
        turning_sequence=tuple(solution.turning_kinds),
        return_metrics=tuple(asdict(metric) for metric in metrics),
        close_return_kinds=_close_return_kinds(
            metrics,
            max_return_error=max_return_error,
            minimum_metrics=minimum_metrics,
        ),
        turning_audit_consistent=all(item.kind_consistent for item in audits),
    )


def verify_recurrence_candidate(
    parameters: SCPCParameters,
    *,
    integration_options: dict[str, Any],
    tolerance_levels: tuple[tuple[str, float, float], ...],
    primary_method: str,
    independent_method: str,
    independent_rtol: float,
    independent_atol: float,
    max_constraint_residual: float,
    max_solution_error: float,
    max_event_time_error: float,
    max_event_state_error: float,
    max_return_error: float,
    minimum_same_kind_return_metrics: int = 2,
) -> dict[str, object]:
    """Execute the dependency-gated Stage-1 recurrence-candidate audit.

    The tightest primary-method run is the reference.  A candidate is promoted
    only if repeated close same-kind returns occur in *every* run, exact event
    topology agrees, the global trajectory converges, constraints pass, and an
    algorithmically distinct solver reproduces the result.
    """

    if len(tolerance_levels) < 2:
        raise ValueError("at least two primary-method tolerance levels are required")
    if primary_method == independent_method:
        raise ValueError("independent_method must differ from primary_method")
    if minimum_same_kind_return_metrics < 2:
        raise ValueError("candidate verification requires at least two same-kind return metrics")
    thresholds = (
        max_constraint_residual,
        max_solution_error,
        max_event_time_error,
        max_event_state_error,
        max_return_error,
        independent_rtol,
        independent_atol,
    )
    if any(not np.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError("verification tolerances and thresholds must be finite and positive")

    base_options = dict(integration_options)
    for reserved in ("method", "rtol", "atol"):
        base_options.pop(reserved, None)

    solutions: dict[str, SCPCSolution] = {}
    run_specs: dict[str, tuple[str, float, float]] = {}
    for label, rtol, atol in tolerance_levels:
        if label in solutions:
            raise ValueError("tolerance-level labels must be unique")
        if not np.isfinite(rtol) or not np.isfinite(atol) or rtol <= 0.0 or atol <= 0.0:
            raise ValueError("tolerance levels must be finite and positive")
        solutions[label] = integrate_scpc(
            parameters,
            method=primary_method,
            rtol=rtol,
            atol=atol,
            **base_options,
        )
        run_specs[label] = (primary_method, rtol, atol)

    independent_label = f"independent:{independent_method}"
    if independent_label in solutions:
        raise ValueError("independent solver label collides with a tolerance-level label")
    solutions[independent_label] = integrate_scpc(
        parameters,
        method=independent_method,
        rtol=independent_rtol,
        atol=independent_atol,
        **base_options,
    )
    run_specs[independent_label] = (independent_method, independent_rtol, independent_atol)

    reference_label, _, _ = min(tolerance_levels, key=lambda item: (item[1], item[2]))
    reference = solutions[reference_label]
    summaries = {
        label: _run_summary(
            label,
            *run_specs[label],
            solution,
            max_return_error=max_return_error,
            minimum_metrics=minimum_same_kind_return_metrics,
        )
        for label, solution in solutions.items()
    }

    comparisons: dict[str, dict[str, object]] = {}
    numerical_reasons: list[str] = []
    for label, solution in solutions.items():
        if label == reference_label:
            continue
        topology = compare_event_topology(reference, solution)
        try:
            global_difference = compare_solutions(reference, solution)
            global_error = float(global_difference.maximum)
            global_payload: dict[str, object] | None = asdict(global_difference)
        except ValueError as exc:
            global_error = float("inf")
            global_payload = {"error": str(exc)}
        comparisons[label] = {
            "global_solution_difference": global_payload,
            "event_topology_difference": asdict(topology),
        }
        if global_error > max_solution_error:
            numerical_reasons.append(f"{label}:global_solution_disagreement")
        if not topology.event_count_match or not topology.event_sequence_match:
            numerical_reasons.append(f"{label}:event_sequence_disagreement")
        if not topology.termination_kind_match or not topology.completion_match:
            numerical_reasons.append(f"{label}:completion_or_termination_disagreement")
        if (
            topology.max_normalized_event_time_error is None
            or topology.max_normalized_event_time_error > max_event_time_error
        ):
            numerical_reasons.append(f"{label}:event_time_disagreement")
        if (
            topology.max_normalized_event_state_error is None
            or topology.max_normalized_event_state_error > max_event_state_error
        ):
            numerical_reasons.append(f"{label}:event_state_disagreement")

    for label, summary in summaries.items():
        if not summary.completed_to_requested_end:
            numerical_reasons.append(f"{label}:incomplete_trajectory")
        if summary.max_abs_constraint_residual > max_constraint_residual:
            numerical_reasons.append(f"{label}:constraint_residual")
        if not summary.turning_audit_consistent:
            numerical_reasons.append(f"{label}:turning_identity_inconsistent")

    reference_close = set(summaries[reference_label].close_return_kinds)
    common_close = set(reference_close)
    for summary in summaries.values():
        common_close.intersection_update(summary.close_return_kinds)

    if not reference_close:
        status = "not_candidate_insufficient_or_nonclosing_returns"
        candidate_gate_passed = False
        reasons = ["reference_has_no_repeated_close_same_kind_returns", *sorted(set(numerical_reasons))]
    elif numerical_reasons:
        status = "unresolved_candidate_numerical_disagreement"
        candidate_gate_passed = False
        reasons = sorted(set(numerical_reasons))
    elif not common_close:
        status = "unresolved_candidate_returns_not_reproduced"
        candidate_gate_passed = False
        reasons = ["repeated_close_returns_not_common_to_all_verification_runs"]
    else:
        status = "stage1_candidate_reproduced"
        candidate_gate_passed = True
        reasons = []

    return {
        "schema_version": 1,
        "status": status,
        "candidate_gate_passed": candidate_gate_passed,
        "stage2_input_eligible": candidate_gate_passed,
        "scientific_scope": (
            "homogeneous numerical recurrence-candidate gate only; not Floquet stability, "
            "perturbative viability, geodesic completeness, or observational support"
        ),
        "reference_label": reference_label,
        "reference_close_return_kinds": sorted(reference_close),
        "common_reproduced_close_return_kinds": sorted(common_close),
        "reasons": reasons,
        "thresholds": {
            "max_constraint_residual": max_constraint_residual,
            "max_solution_error": max_solution_error,
            "max_event_time_error": max_event_time_error,
            "max_event_state_error": max_event_state_error,
            "max_return_error": max_return_error,
            "minimum_same_kind_return_metrics": minimum_same_kind_return_metrics,
        },
        "runs": {label: asdict(summary) for label, summary in summaries.items()},
        "comparisons_to_reference": comparisons,
    }
