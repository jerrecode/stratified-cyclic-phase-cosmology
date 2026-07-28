"""Diagnostics for recurrent background trajectories.

The routines in this module do not infer a cosmological cycle from a visually
oscillatory curve. They compare states at repeated turning points and report
explicit return errors. A low return error is only a candidate recurrence; it
is not a perturbative-stability or attractor proof.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scpc.models.phase import SCPCSolution


@dataclass(frozen=True)
class TurningState:
    """Interpolated homogeneous state at a classified H=0 event."""

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
    relative_scale_factor_error: float
    wrapped_phase_error: float
    relative_field_velocity_error: float
    maximum_error: float


def wrapped_phase_difference(phi_a: float, phi_b: float, period: float) -> float:
    """Return the signed shortest separation on a periodic field domain."""

    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("period must be finite and positive")
    return float((phi_b - phi_a + 0.5 * period) % period - 0.5 * period)


def turning_states(solution: SCPCSolution) -> tuple[TurningState, ...]:
    """Interpolate the serialized trajectory at all detected turning events."""

    if len(solution.turning_times) != len(solution.turning_kinds):
        raise ValueError("turning_times and turning_kinds must have equal length")

    states: list[TurningState] = []
    for event_time, kind in zip(solution.turning_times, solution.turning_kinds, strict=True):
        time = float(event_time)
        if time < solution.t[0] or time > solution.t[-1]:
            raise ValueError(f"Turning event at t={time} lies outside the stored trajectory")
        states.append(
            TurningState(
                time=time,
                kind=kind,
                scale_factor=float(np.interp(time, solution.t, solution.a)),
                field=float(np.interp(time, solution.t, solution.phi)),
                field_velocity=float(np.interp(time, solution.t, solution.phi_dot)),
            )
        )
    return tuple(states)


def cycle_return_metrics(
    solution: SCPCSolution,
    *,
    kind: str | None = None,
    relative_floor: float = 1.0e-12,
) -> tuple[CycleReturnMetric, ...]:
    """Compare consecutive turning states of the same kind.

    The field mismatch is wrapped by the period of the stratification
    potential, so two states separated by an integer number of strata periods
    are treated as topologically equivalent in the compact phase coordinate.
    """

    if relative_floor <= 0.0:
        raise ValueError("relative_floor must be positive")

    selected = [state for state in turning_states(solution) if kind is None or state.kind == kind]
    field_period = (
        2.0
        * np.pi
        * solution.parameters.potential.field_scale
        / solution.parameters.potential.strata_count
    )

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
        phase_error = abs(
            wrapped_phase_difference(previous.field, current.field, field_period)
        ) / field_period
        velocity_error = abs(current.field_velocity - previous.field_velocity) / max(
            abs(current.field_velocity), abs(previous.field_velocity), relative_floor
        )
        maximum_error = max(scale_error, phase_error, velocity_error)
        metrics.append(
            CycleReturnMetric(
                kind=current.kind,
                start_time=previous.time,
                end_time=current.time,
                period=period,
                relative_scale_factor_error=float(scale_error),
                wrapped_phase_error=float(phase_error),
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
    """Classify background recurrence without claiming nonlinear stability."""

    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if not metrics:
        return "insufficient_turning_points"

    errors = np.asarray([metric.maximum_error for metric in metrics], dtype=float)
    if np.any(~np.isfinite(errors)):
        return "invalid_return_metric"
    if np.all(errors <= tolerance):
        return "recurrent_candidate"
    if errors.size >= 2 and errors[-1] < errors[0] and errors[-1] <= 10.0 * tolerance:
        return "converging_recurrence_candidate"
    return "nonrecurrent_or_drifting"
