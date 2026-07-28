"""Conservative outcome classification for homogeneous background trajectories.

The classifier separates numerical invalidity, unresolved event detection, and
physical trajectory morphology. It does not infer a spacetime singularity from
a solver exception and does not label repeated turning points as a cyclic
cosmology.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from scpc.models.phase import SCPCSolution
from scpc.numerics.cycles import classify_return_sequences, cycle_return_metrics
from scpc.scans.errors import ResultIntegrityError


class OutcomeClass(StrEnum):
    """Mutually exclusive high-level classes for completed integrations."""

    NONFINITE_STATE = "nonfinite_state"
    NONPOSITIVE_SCALE_FACTOR = "nonpositive_scale_factor"
    CONSTRAINT_VIOLATION = "constraint_violation"
    DEGENERATE_TURNING_EVENT = "degenerate_turning_event"
    UNRESOLVED_EVENT_DETECTION = "unresolved_event_detection"
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


@dataclass(frozen=True)
class OutcomeAssessment:
    """Machine-readable summary of one successfully returned solution."""

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
    """Classify one completed integration without making a cyclicity claim."""

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
