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


def _has_unresolved_hubble_crossing(hubble: np.ndarray, tolerance: float) -> bool:
    positive = hubble > tolerance
    negative = hubble < -tolerance
    return bool(np.any(positive[:-1] & negative[1:]) or np.any(negative[:-1] & positive[1:]))


def assess_solution(
    solution: SCPCSolution,
    *,
    constraint_threshold: float = 1.0e-7,
    hubble_zero_tolerance: float = 1.0e-10,
    return_tolerance: float = 1.0e-3,
) -> OutcomeAssessment:
    """Classify one completed integration without making a cyclicity claim."""

    if constraint_threshold <= 0.0:
        raise ValueError("constraint_threshold must be positive")
    if hubble_zero_tolerance <= 0.0:
        raise ValueError("hubble_zero_tolerance must be positive")
    if return_tolerance <= 0.0:
        raise ValueError("return_tolerance must be positive")

    arrays = _state_arrays(solution)
    lengths = {array.size for array in arrays}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("All stored trajectory arrays must be nonempty and have equal lengths")

    event_sequence = tuple(solution.turning_kinds)
    bounce_count = event_sequence.count("bounce")
    turnaround_count = event_sequence.count("turnaround")
    degenerate_count = event_sequence.count("degenerate")
    unknown_kinds = set(event_sequence) - {"bounce", "turnaround", "degenerate"}
    if unknown_kinds:
        raise ValueError(f"Unknown turning-event kinds: {sorted(unknown_kinds)}")

    return_metrics = cycle_return_metrics(solution)
    return_classifications = classify_return_sequences(
        return_metrics,
        tolerance=return_tolerance,
    )
    constraint = np.asarray(solution.constraint_residual, dtype=float)
    finite_constraint = constraint[np.isfinite(constraint)]
    max_constraint = (
        float(np.max(np.abs(finite_constraint))) if finite_constraint.size else float("inf")
    )

    if any(np.any(~np.isfinite(array)) for array in arrays):
        return OutcomeAssessment(
            outcome=OutcomeClass.NONFINITE_STATE,
            reason="At least one stored state, density, time, or constraint value is nonfinite.",
            numerically_valid=False,
            bounce_count=bounce_count,
            turnaround_count=turnaround_count,
            degenerate_count=degenerate_count,
            event_sequence=event_sequence,
            max_abs_constraint_residual=max_constraint,
            return_sequence_classifications=return_classifications,
        )

    if np.any(np.asarray(solution.a, dtype=float) <= 0.0):
        return OutcomeAssessment(
            outcome=OutcomeClass.NONPOSITIVE_SCALE_FACTOR,
            reason="The stored trajectory contains a nonpositive scale factor.",
            numerically_valid=False,
            bounce_count=bounce_count,
            turnaround_count=turnaround_count,
            degenerate_count=degenerate_count,
            event_sequence=event_sequence,
            max_abs_constraint_residual=max_constraint,
            return_sequence_classifications=return_classifications,
        )

    if max_constraint > constraint_threshold:
        return OutcomeAssessment(
            outcome=OutcomeClass.CONSTRAINT_VIOLATION,
            reason=(
                f"Maximum absolute Friedmann residual {max_constraint:.6g} exceeds "
                f"the declared threshold {constraint_threshold:.6g}."
            ),
            numerically_valid=False,
            bounce_count=bounce_count,
            turnaround_count=turnaround_count,
            degenerate_count=degenerate_count,
            event_sequence=event_sequence,
            max_abs_constraint_residual=max_constraint,
            return_sequence_classifications=return_classifications,
        )

    if degenerate_count:
        return OutcomeAssessment(
            outcome=OutcomeClass.DEGENERATE_TURNING_EVENT,
            reason="At least one H=0 event has unresolved sign in dH/dt.",
            numerically_valid=False,
            bounce_count=bounce_count,
            turnaround_count=turnaround_count,
            degenerate_count=degenerate_count,
            event_sequence=event_sequence,
            max_abs_constraint_residual=max_constraint,
            return_sequence_classifications=return_classifications,
        )

    hubble = np.asarray(solution.H, dtype=float)
    if not event_sequence:
        if _has_unresolved_hubble_crossing(hubble, hubble_zero_tolerance):
            outcome = OutcomeClass.UNRESOLVED_EVENT_DETECTION
            reason = "The sampled Hubble parameter changes sign but no root-localized event was recorded."
            valid = False
        elif np.all(hubble > hubble_zero_tolerance):
            outcome = OutcomeClass.MONOTONIC_EXPANSION
            reason = "The Hubble parameter remains strictly positive on the integrated interval."
            valid = True
        elif np.all(hubble < -hubble_zero_tolerance):
            outcome = OutcomeClass.MONOTONIC_CONTRACTION
            reason = "The Hubble parameter remains strictly negative on the integrated interval."
            valid = True
        else:
            outcome = OutcomeClass.QUASI_STATIC_OR_AMBIGUOUS
            reason = "No turning event is recorded, but H is not uniformly separated from zero."
            valid = True
    elif bounce_count == 0 and turnaround_count == 1:
        outcome = OutcomeClass.RECOLLAPSE_WITHOUT_BOUNCE
        reason = "One turnaround is recorded and no subsequent bounce occurs in the interval."
        valid = True
    elif bounce_count == 1 and turnaround_count == 0:
        outcome = OutcomeClass.ONE_OFF_BOUNCE
        reason = "One bounce is recorded and no turnaround occurs in the interval."
        valid = True
    elif bounce_count == 1 and turnaround_count == 1 and len(event_sequence) == 2:
        outcome = OutcomeClass.SINGLE_BOUNCE_TURNAROUND_PAIR
        reason = "Exactly one bounce and one turnaround are recorded; this is not repeated recurrence."
        valid = True
    else:
        outcome = OutcomeClass.REPEATED_TURNING_POINTS
        reason = "More than two nondegenerate turning events, or repeated events of one kind, are recorded."
        valid = True

    return OutcomeAssessment(
        outcome=outcome,
        reason=reason,
        numerically_valid=valid,
        bounce_count=bounce_count,
        turnaround_count=turnaround_count,
        degenerate_count=degenerate_count,
        event_sequence=event_sequence,
        max_abs_constraint_residual=max_constraint,
        return_sequence_classifications=return_classifications,
    )
