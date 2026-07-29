"""Conservative outcome classification for homogeneous background trajectories."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from scpc.models.phase import (
    DOMAIN_TERMINATION_KINDS,
    SCPCIntegrationDomain,
    SCPCSolution,
    evaluate_domain_boundary,
)
from scpc.numerics.cycles import classify_return_sequences, cycle_return_metrics
from scpc.scans.errors import ResultIntegrityError


class OutcomeClass(StrEnum):
    """Mutually exclusive high-level classes for returned integrations."""

    NONFINITE_STATE = "nonfinite_state"
    NONPOSITIVE_SCALE_FACTOR = "nonpositive_scale_factor"
    CONSTRAINT_VIOLATION = "constraint_violation"
    DEGENERATE_TURNING_EVENT = "degenerate_turning_event"
    UNRESOLVED_EVENT_DETECTION = "unresolved_event_detection"
    PHYSICAL_DOMAIN_TERMINATION = "physical_domain_termination"
    MONOTONIC_EXPANSION = "monotonic_expansion"
    MONOTONIC_CONTRACTION = "monotonic_contraction"
    QUASI_STATIC_OR_AMBIGUOUS = "quasi_static_or_ambiguous"
    RECOLLAPSE_WITHOUT_BOUNCE = "recollapse_without_bounce"
    ONE_OFF_BOUNCE = "one_off_bounce"
    SINGLE_BOUNCE_TURNAROUND_PAIR = "single_bounce_turnaround_pair"
    REPEATED_TURNING_POINTS = "repeated_turning_points"


RETAINABLE_OUTCOMES = (
    OutcomeClass.MONOTONIC_EXPANSION,
    OutcomeClass.MONOTONIC_CONTRACTION,
    OutcomeClass.QUASI_STATIC_OR_AMBIGUOUS,
    OutcomeClass.RECOLLAPSE_WITHOUT_BOUNCE,
    OutcomeClass.ONE_OFF_BOUNCE,
    OutcomeClass.SINGLE_BOUNCE_TURNAROUND_PAIR,
    OutcomeClass.REPEATED_TURNING_POINTS,
)

_DOMAIN_FIELD_TO_KIND = {
    "min_scale_factor": "minimum_scale_factor",
    "max_scale_factor": "maximum_scale_factor",
    "max_total_density": "maximum_total_density",
    "max_abs_hubble": "maximum_absolute_hubble",
    "max_abs_ricci_scalar": "maximum_absolute_ricci_scalar",
    "max_abs_field": "maximum_absolute_field",
    "max_abs_field_velocity": "maximum_absolute_field_velocity",
}


@dataclass(frozen=True)
class OutcomeAssessment:
    outcome: OutcomeClass
    reason: str
    numerically_valid: bool
    bounce_count: int
    turnaround_count: int
    degenerate_count: int
    event_sequence: tuple[str, ...]
    max_abs_constraint_residual: float
    return_sequence_classifications: dict[str, str]


@dataclass(frozen=True)
class _SampledCrossing:
    kind: str
    start_time: float
    end_time: float


def _state_arrays(solution: SCPCSolution) -> tuple[np.ndarray, ...]:
    return (
        np.asarray(solution.t, dtype=float),
        np.asarray(solution.a, dtype=float),
        np.asarray(solution.H, dtype=float),
        np.asarray(solution.phi, dtype=float),
        np.asarray(solution.phi_dot, dtype=float),
        np.asarray(solution.rho_m, dtype=float),
        np.asarray(solution.rho_r, dtype=float),
        np.asarray(solution.rho_phi, dtype=float),
        np.asarray(solution.p_phi, dtype=float),
        np.asarray(solution.constraint_residual, dtype=float),
    )


def _positive_finite(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _sampled_hubble_crossings(
    time: np.ndarray,
    hubble: np.ndarray,
    tolerance: float,
) -> tuple[_SampledCrossing, ...]:
    signs = np.where(hubble > tolerance, 1, np.where(hubble < -tolerance, -1, 0))
    nonzero = np.flatnonzero(signs)
    crossings: list[_SampledCrossing] = []
    for left, right in zip(nonzero[:-1], nonzero[1:], strict=True):
        left_sign = int(signs[left])
        right_sign = int(signs[right])
        if left_sign == right_sign:
            continue
        crossings.append(
            _SampledCrossing(
                kind="bounce" if left_sign < right_sign else "turnaround",
                start_time=float(time[left]),
                end_time=float(time[right]),
            )
        )
    return tuple(crossings)


def _unmatched_sampled_crossings(
    solution: SCPCSolution,
    tolerance: float,
) -> tuple[_SampledCrossing, ...]:
    time = np.asarray(solution.t, dtype=float)
    hubble = np.asarray(solution.H, dtype=float)
    expected = _sampled_hubble_crossings(time, hubble, tolerance)
    recorded = [
        (float(event_time), kind)
        for event_time, kind in zip(solution.turning_times, solution.turning_kinds, strict=True)
        if kind in {"bounce", "turnaround"}
    ]
    used: set[int] = set()
    unmatched: list[_SampledCrossing] = []
    scale = max(1.0, float(np.max(np.abs(time))))
    slack = 32.0 * np.finfo(float).eps * scale
    for crossing in expected:
        match = next(
            (
                index
                for index, (event_time, kind) in enumerate(recorded)
                if index not in used
                and kind == crossing.kind
                and crossing.start_time - slack <= event_time <= crossing.end_time + slack
            ),
            None,
        )
        if match is None:
            unmatched.append(crossing)
        else:
            used.add(match)
    return tuple(unmatched)


def _event_tolerance(solution: SCPCSolution, scale: float) -> float:
    rtol = abs(float(solution.solver_metadata.get("solver_rtol", 1.0e-9)))
    atol = abs(float(solution.solver_metadata.get("solver_atol", 1.0e-11)))
    magnitude = abs(float(scale))
    roundoff = 64.0 * np.finfo(float).eps * max(magnitude, np.finfo(float).tiny)
    return roundoff + 128.0 * (atol + rtol * magnitude)


def _validate_boundary_record(
    boundary: dict[str, Any],
    state: np.ndarray,
    solution: SCPCSolution,
) -> dict[str, Any]:
    try:
        kind = str(boundary["kind"])
        threshold = float(boundary["threshold"])
        recorded_observed = float(boundary["observed"])
        units = str(boundary["units"])
    except (KeyError, TypeError, ValueError) as error:
        raise ResultIntegrityError("Termination boundary record is malformed") from error
    if kind not in DOMAIN_TERMINATION_KINDS:
        raise ResultIntegrityError(f"Unknown physical-domain termination kind: {kind}")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ResultIntegrityError("Termination threshold must be finite and positive")
    if not np.isfinite(recorded_observed) or recorded_observed <= 0.0:
        raise ResultIntegrityError("Termination observed value must be finite and positive")
    recomputed, expected_units = evaluate_domain_boundary(kind, state, solution.parameters)
    tolerance = _event_tolerance(solution, max(abs(recomputed), abs(threshold)))
    if units != expected_units:
        raise ResultIntegrityError(
            f"Termination units for {kind} are {units!r}, expected {expected_units!r}"
        )
    if abs(recorded_observed - recomputed) > tolerance:
        raise ResultIntegrityError(
            f"Recorded termination observable for {kind} disagrees with the final state"
        )
    if abs(recomputed - threshold) > tolerance:
        raise ResultIntegrityError(
            f"Termination state does not lie on the recorded {kind} threshold"
        )
    return {
        "kind": kind,
        "threshold": threshold,
        "observed": recorded_observed,
        "units": units,
    }


def _configured_coincident_boundaries(
    solution: SCPCSolution,
    state: np.ndarray,
) -> tuple[dict[str, Any], ...]:
    raw_domain = solution.solver_metadata.get("integration_domain")
    if not isinstance(raw_domain, str):
        raise ResultIntegrityError(
            "Terminated solution is missing serialized integration-domain metadata"
        )
    try:
        payload = json.loads(raw_domain)
    except (TypeError, json.JSONDecodeError) as error:
        raise ResultIntegrityError("Integration-domain metadata is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ResultIntegrityError("Integration-domain metadata must decode to a mapping")
    if any(value is not None and isinstance(value, bool) for value in payload.values()):
        raise ResultIntegrityError("Integration-domain thresholds may not be boolean")
    try:
        domain = SCPCIntegrationDomain(**payload)
        domain.validate_for(solution.parameters)
    except (TypeError, ValueError) as error:
        raise ResultIntegrityError("Integration-domain metadata is invalid") from error
    if not domain.configured:
        raise ResultIntegrityError(
            "Terminated solution has no configured integration-domain surfaces"
        )

    expected: list[dict[str, Any]] = []
    for field, value in asdict(domain).items():
        if value is None:
            continue
        kind = _DOMAIN_FIELD_TO_KIND[field]
        threshold = float(value)
        observed, units = evaluate_domain_boundary(kind, state, solution.parameters)
        tolerance = _event_tolerance(solution, max(abs(observed), abs(threshold)))
        if abs(observed - threshold) <= tolerance:
            expected.append(
                {
                    "kind": kind,
                    "threshold": threshold,
                    "observed": observed,
                    "units": units,
                }
            )
    if not expected:
        raise ResultIntegrityError(
            "No configured integration-domain surface matches the exact terminal state"
        )
    return tuple(sorted(expected, key=lambda boundary: str(boundary["kind"])))


def _validate_termination_metadata(
    solution: SCPCSolution,
    arrays: tuple[np.ndarray, ...],
) -> tuple[dict[str, Any], ...]:
    kind = solution.termination_kind
    scalar_metadata = (
        solution.termination_time,
        solution.termination_state_vector,
        solution.termination_threshold,
        solution.termination_observed,
        solution.termination_units,
    )
    if kind is None:
        if any(value is not None for value in scalar_metadata) or solution.termination_boundaries:
            raise ResultIntegrityError(
                "Nonterminated solution contains partial termination metadata"
            )
        if not solution.completed_to_requested_end:
            raise ResultIntegrityError(
                "Integration ended before its requested endpoint without a declared termination"
            )
        return ()

    if (
        solution.termination_time is None
        or solution.termination_state_vector is None
        or solution.termination_threshold is None
        or solution.termination_observed is None
        or not solution.termination_units
        or solution.requested_end_time is None
        or not solution.termination_boundaries
    ):
        raise ResultIntegrityError(
            "Physical-domain termination requires complete scalar and boundary-set metadata"
        )

    termination_time = float(solution.termination_time)
    requested_end_time = float(solution.requested_end_time)
    state = np.asarray(solution.termination_state_vector, dtype=float)
    if not np.isfinite(termination_time):
        raise ResultIntegrityError("Termination time must be finite")
    if not np.isfinite(requested_end_time):
        raise ResultIntegrityError("Requested end time must be finite")
    if state.shape != (4,) or np.any(~np.isfinite(state)):
        raise ResultIntegrityError("Termination state must be a finite vector with shape (4,)")

    time = arrays[0]
    interval_slack = 64.0 * np.finfo(float).eps * max(
        1.0,
        abs(float(time[0])),
        abs(requested_end_time),
        abs(termination_time),
    )
    lower = min(float(time[0]), requested_end_time) - interval_slack
    upper = max(float(time[0]), requested_end_time) + interval_slack
    if termination_time < lower or termination_time > upper:
        raise ResultIntegrityError(
            "Termination time lies outside the requested integration interval"
        )

    time_tolerance = _event_tolerance(solution, max(abs(termination_time), abs(float(time[-1]))))
    if not np.isclose(time[-1], termination_time, rtol=0.0, atol=time_tolerance):
        raise ResultIntegrityError("Exact termination time must be the final stored time")
    final_state = np.asarray([arrays[1][-1], arrays[2][-1], arrays[3][-1], arrays[4][-1]])
    if not np.allclose(final_state, state, rtol=1.0e-10, atol=1.0e-12):
        raise ResultIntegrityError("Exact termination state must equal the final stored state")
    if solution.completed_to_requested_end:
        raise ResultIntegrityError(
            "Domain-terminated solution cannot be marked complete to the requested endpoint"
        )

    validated = tuple(
        _validate_boundary_record(dict(boundary), state, solution)
        for boundary in solution.termination_boundaries
    )
    kinds = tuple(boundary["kind"] for boundary in validated)
    if kinds != tuple(sorted(set(kinds))):
        raise ResultIntegrityError(
            "Termination boundary records must be unique and lexically ordered"
        )

    expected = _configured_coincident_boundaries(solution, state)
    expected_kinds = tuple(boundary["kind"] for boundary in expected)
    if kinds != expected_kinds:
        raise ResultIntegrityError(
            "Termination boundary set does not match configured coincident surfaces"
        )
    for supplied, configured in zip(validated, expected, strict=True):
        tolerance = _event_tolerance(
            solution,
            max(abs(configured["observed"]), abs(configured["threshold"])),
        )
        if supplied["units"] != configured["units"]:
            raise ResultIntegrityError(
                f"Termination units for {supplied['kind']} do not match configured domain"
            )
        if abs(supplied["threshold"] - configured["threshold"]) > tolerance:
            raise ResultIntegrityError(
                f"Termination threshold for {supplied['kind']} does not match configured domain"
            )

    primary = validated[0]
    primary_threshold = float(solution.termination_threshold)
    primary_observed = float(solution.termination_observed)
    if (
        not np.isfinite(primary_threshold)
        or primary_threshold <= 0.0
        or not np.isfinite(primary_observed)
        or primary_observed <= 0.0
    ):
        raise ResultIntegrityError(
            "Primary termination threshold and observation must be finite and positive"
        )
    primary_tolerance = _event_tolerance(
        solution,
        max(abs(primary["observed"]), abs(primary["threshold"])),
    )
    if kind != primary["kind"] or solution.termination_units != primary["units"]:
        raise ResultIntegrityError("Primary termination labels do not match the boundary set")
    if abs(primary_threshold - primary["threshold"]) > primary_tolerance:
        raise ResultIntegrityError("Primary termination threshold does not match the boundary set")
    if abs(primary_observed - primary["observed"]) > primary_tolerance:
        raise ResultIntegrityError("Primary termination observation does not match the boundary set")
    return validated


def _assessment(
    *,
    outcome: OutcomeClass,
    reason: str,
    numerically_valid: bool,
    event_sequence: tuple[str, ...],
    max_constraint: float,
    return_classifications: dict[str, str] | None = None,
) -> OutcomeAssessment:
    return OutcomeAssessment(
        outcome=outcome,
        reason=reason,
        numerically_valid=numerically_valid,
        bounce_count=event_sequence.count("bounce"),
        turnaround_count=event_sequence.count("turnaround"),
        degenerate_count=event_sequence.count("degenerate"),
        event_sequence=event_sequence,
        max_abs_constraint_residual=max_constraint,
        return_sequence_classifications=dict(return_classifications or {}),
    )


def assess_solution(
    solution: SCPCSolution,
    *,
    constraint_threshold: float = 1.0e-7,
    hubble_zero_tolerance: float = 1.0e-10,
    return_tolerance: float = 1.0e-3,
) -> OutcomeAssessment:
    """Classify one returned integration without making a cyclicity claim."""

    constraint_threshold = _positive_finite("constraint_threshold", constraint_threshold)
    hubble_zero_tolerance = _positive_finite(
        "hubble_zero_tolerance",
        hubble_zero_tolerance,
    )
    return_tolerance = _positive_finite("return_tolerance", return_tolerance)

    arrays = _state_arrays(solution)
    if any(array.ndim != 1 for array in arrays):
        raise ResultIntegrityError("All stored trajectory arrays must be one-dimensional")
    lengths = {array.size for array in arrays}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ResultIntegrityError(
            "All stored trajectory arrays must be nonempty and have equal lengths"
        )

    time = arrays[0]
    if np.any(np.diff(time) <= 0.0):
        raise ResultIntegrityError("Stored trajectory times must be strictly increasing")

    event_sequence = tuple(solution.turning_kinds)
    turning_times = np.asarray(solution.turning_times, dtype=float)
    if turning_times.ndim != 1 or turning_times.size != len(event_sequence):
        raise ResultIntegrityError(
            "Turning times and event kinds must be one-dimensional and have equal lengths"
        )
    unknown_kinds = set(event_sequence) - {"bounce", "turnaround", "degenerate"}
    if unknown_kinds:
        raise ResultIntegrityError(f"Unknown turning-event kinds: {sorted(unknown_kinds)}")
    if np.any(~np.isfinite(turning_times)):
        raise ResultIntegrityError("Turning-event times must be finite")
    if turning_times.size and (
        np.any(turning_times < time[0]) or np.any(turning_times > time[-1])
    ):
        raise ResultIntegrityError(
            "Turning-event times must lie inside the integrated interval"
        )

    constraint = arrays[-1]
    finite_constraint = constraint[np.isfinite(constraint)]
    max_constraint = (
        float(np.max(np.abs(finite_constraint))) if finite_constraint.size else float("inf")
    )

    if any(np.any(~np.isfinite(array)) for array in arrays):
        return _assessment(
            outcome=OutcomeClass.NONFINITE_STATE,
            reason="At least one stored state, density, time, or constraint value is nonfinite.",
            numerically_valid=False,
            event_sequence=event_sequence,
            max_constraint=max_constraint,
        )

    termination_boundaries = _validate_termination_metadata(solution, arrays)

    if np.any(arrays[1] <= 0.0):
        return _assessment(
            outcome=OutcomeClass.NONPOSITIVE_SCALE_FACTOR,
            reason="The stored trajectory contains a nonpositive scale factor.",
            numerically_valid=False,
            event_sequence=event_sequence,
            max_constraint=max_constraint,
        )

    if max_constraint > constraint_threshold:
        return _assessment(
            outcome=OutcomeClass.CONSTRAINT_VIOLATION,
            reason=(
                f"Maximum absolute Friedmann residual {max_constraint:.6g} exceeds "
                f"the declared threshold {constraint_threshold:.6g}."
            ),
            numerically_valid=False,
            event_sequence=event_sequence,
            max_constraint=max_constraint,
        )

    if "degenerate" in event_sequence:
        return _assessment(
            outcome=OutcomeClass.DEGENERATE_TURNING_EVENT,
            reason="At least one H=0 event has unresolved sign in dH/dt.",
            numerically_valid=False,
            event_sequence=event_sequence,
            max_constraint=max_constraint,
        )

    unmatched = _unmatched_sampled_crossings(solution, hubble_zero_tolerance)
    if unmatched:
        descriptions = ", ".join(
            f"{crossing.kind}@[{crossing.start_time:.6g},{crossing.end_time:.6g}]"
            for crossing in unmatched
        )
        return _assessment(
            outcome=OutcomeClass.UNRESOLVED_EVENT_DETECTION,
            reason=(
                "Sampled Hubble sign transitions lack matching root-localized events: "
                f"{descriptions}."
            ),
            numerically_valid=False,
            event_sequence=event_sequence,
            max_constraint=max_constraint,
        )

    return_metrics = cycle_return_metrics(solution)
    return_classifications = classify_return_sequences(
        return_metrics,
        tolerance=return_tolerance,
    )

    if termination_boundaries:
        kinds = ", ".join(boundary["kind"] for boundary in termination_boundaries)
        return _assessment(
            outcome=OutcomeClass.PHYSICAL_DOMAIN_TERMINATION,
            reason=(
                f"Integration reached declared domain boundary set [{kinds}] at "
                f"t={float(solution.termination_time):.6g}. This is an analysis boundary, "
                "not a spacetime singularity claim."
            ),
            numerically_valid=False,
            event_sequence=event_sequence,
            max_constraint=max_constraint,
            return_classifications=return_classifications,
        )

    bounce_count = event_sequence.count("bounce")
    turnaround_count = event_sequence.count("turnaround")
    hubble = arrays[2]

    if not event_sequence:
        if np.all(hubble > hubble_zero_tolerance):
            outcome = OutcomeClass.MONOTONIC_EXPANSION
            reason = "The Hubble parameter remains strictly positive on the integrated interval."
        elif np.all(hubble < -hubble_zero_tolerance):
            outcome = OutcomeClass.MONOTONIC_CONTRACTION
            reason = "The Hubble parameter remains strictly negative on the integrated interval."
        else:
            outcome = OutcomeClass.QUASI_STATIC_OR_AMBIGUOUS
            reason = "No turning event is recorded, but H is not uniformly separated from zero."
    elif bounce_count == 0 and turnaround_count == 1:
        outcome = OutcomeClass.RECOLLAPSE_WITHOUT_BOUNCE
        reason = "One turnaround is recorded and no subsequent bounce occurs in the interval."
    elif bounce_count == 1 and turnaround_count == 0:
        outcome = OutcomeClass.ONE_OFF_BOUNCE
        reason = "One bounce is recorded and no turnaround occurs in the interval."
    elif bounce_count == 1 and turnaround_count == 1 and len(event_sequence) == 2:
        outcome = OutcomeClass.SINGLE_BOUNCE_TURNAROUND_PAIR
        reason = "Exactly one bounce and one turnaround are recorded; this is not repeated recurrence."
    else:
        outcome = OutcomeClass.REPEATED_TURNING_POINTS
        reason = (
            "More than two nondegenerate turning events, or repeated events of one kind, "
            "are recorded."
        )

    return _assessment(
        outcome=outcome,
        reason=reason,
        numerically_valid=True,
        event_sequence=event_sequence,
        max_constraint=max_constraint,
        return_classifications=return_classifications,
    )
