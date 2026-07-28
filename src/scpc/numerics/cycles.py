"""Diagnostics for repeated turning-point returns.

The routines in this module do not infer a cosmological cycle from a visually
oscillatory curve. They compare states at repeated turning points and report
explicit return errors. Even repeated close returns in one integration are not
proof of recurrence, perturbative stability, or an attractor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scpc.models.phase import SCPCSolution


@dataclass(frozen=True)
class TurningState:
    """Homogeneous state at a classified H=0 event."""

    time: float
    kind: str
    scale_factor: float
    field: float
    field_velocity: float


@dataclass(frozen=True)
class CycleReturnMetric:
    """Dimensionless mismatch between two same-kind turning states."""

    kind: str
    start_time: float
    end_time: float
    period: float
    target_space: str
    relative_scale_factor_error: float
    field_error: float
    field_winding: int
    relative_field_velocity_error: float
    maximum_error: float


def wrapped_phase_difference(phi_a: float, phi_b: float, circumference: float) -> float:
    """Return the signed shortest separation on an explicitly circular target."""

    if not np.isfinite(circumference) or circumference <= 0.0:
        raise ValueError("circumference must be finite and positive")
    return float((phi_b - phi_a + 0.5 * circumference) % circumference - 0.5 * circumference)


def turning_states(solution: SCPCSolution) -> tuple[TurningState, ...]:
    """Return exact solver event states, with interpolation as legacy fallback."""

    event_count = len(solution.turning_times)
    if event_count != len(solution.turning_kinds):
        raise ValueError("turning_times and turning_kinds must have equal length")

    exact_states: np.ndarray | None = None
    if solution.turning_state_vectors is not None:
        exact_states = np.asarray(solution.turning_state_vectors, dtype=float)
        if exact_states.shape != (event_count, 4):
            raise ValueError("turning_state_vectors must have shape (events, 4)")

    states: list[TurningState] = []
    for index, (event_time, kind) in enumerate(
        zip(solution.turning_times, solution.turning_kinds, strict=True)
    ):
        time = float(event_time)
        if time < solution.t[0] or time > solution.t[-1]:
            raise ValueError(f"Turning event at t={time} lies outside the stored trajectory")
        if exact_states is None:
            scale_factor = float(np.interp(time, solution.t, solution.a))
            field = float(np.interp(time, solution.t, solution.phi))
            field_velocity = float(np.interp(time, solution.t, solution.phi_dot))
        else:
            scale_factor = float(exact_states[index, 0])
            field = float(exact_states[index, 2])
            field_velocity = float(exact_states[index, 3])
        states.append(
            TurningState(
                time=time,
                kind=kind,
                scale_factor=scale_factor,
                field=field,
                field_velocity=field_velocity,
            )
        )
    return tuple(states)


def _field_return_error(
    previous: float,
    current: float,
    solution: SCPCSolution,
) -> tuple[float, int]:
    potential = solution.parameters.potential
    displacement = current - previous
    if potential.target_space == "real":
        return abs(displacement) / potential.potential_period, 0

    circumference = potential.target_circumference
    if circumference is None:  # pragma: no cover - protected by target-space validation
        raise RuntimeError("Circular target space has no circumference")
    wrapped = wrapped_phase_difference(previous, current, circumference)
    winding = int(np.rint((displacement - wrapped) / circumference))
    return abs(wrapped) / potential.potential_period, winding


def cycle_return_metrics(
    solution: SCPCSolution,
    *,
    kind: str | None = None,
    relative_floor: float = 1.0e-12,
) -> tuple[CycleReturnMetric, ...]:
    """Compare consecutive turning states of the same kind.

    Real scalar fields retain their full displacement even when the potential
    is periodic. A circular target space must be declared explicitly; only a
    full target circumference is then identified, and nonzero winding is
    recorded rather than hidden.
    """

    if relative_floor <= 0.0:
        raise ValueError("relative_floor must be positive")

    selected = [state for state in turning_states(solution) if kind is None or state.kind == kind]
    target_space = solution.parameters.potential.target_space

    metrics: list[CycleReturnMetric] = []
    previous_by_kind: dict[str, TurningState] = {}
    for current in selected:
        previous = previous_by_kind.get(current.kind)
        previous_by_kind[current.kind] = current
        if previous is None:
            continue

        period = current.time - previous.time
        if period <= 0.0:
            raise ValueError("Turning events must be strictly ordered in time")

        scale_error = abs(current.scale_factor - previous.scale_factor) / max(
            abs(current.scale_factor), abs(previous.scale_factor), relative_floor
        )
        field_error, field_winding = _field_return_error(previous.field, current.field, solution)
        velocity_error = abs(current.field_velocity - previous.field_velocity) / max(
            abs(current.field_velocity), abs(previous.field_velocity), relative_floor
        )
        maximum_error = max(scale_error, field_error, velocity_error)
        metrics.append(
            CycleReturnMetric(
                kind=current.kind,
                start_time=previous.time,
                end_time=current.time,
                period=period,
                target_space=target_space,
                relative_scale_factor_error=float(scale_error),
                field_error=float(field_error),
                field_winding=field_winding,
                relative_field_velocity_error=float(velocity_error),
                maximum_error=float(maximum_error),
            )
        )
    return tuple(metrics)


def classify_recurrence(
    metrics: tuple[CycleReturnMetric, ...],
    *,
    tolerance: float = 1.0e-3,
) -> str:
    """Summarize one run's return sequence without claiming recurrence.

    At least two return metrics, requiring at least three turning points of the
    same kind, are required for a positive close-return summary. A recurrence
    candidate additionally requires reproduction across solver tolerances and
    independent methods, which is intentionally outside this function.
    """

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if not metrics:
        return "insufficient_turning_points"
    if len(metrics) < 2:
        return "insufficient_repeated_returns"

    errors = np.asarray([metric.maximum_error for metric in metrics], dtype=float)
    if np.any(~np.isfinite(errors)):
        return "invalid_return_metric"
    has_winding = any(metric.field_winding != 0 for metric in metrics)
    if np.all(errors <= tolerance):
        return "repeated_close_winding_returns" if has_winding else "repeated_close_returns"
    if np.all(np.diff(errors) < 0.0):
        return "return_errors_decreasing_but_unresolved"
    return "nonclosing_or_drifting_returns"
