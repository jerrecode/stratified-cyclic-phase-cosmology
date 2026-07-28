"""Numerical-convergence diagnostics for homogeneous SCPC backgrounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scpc.models.phase import SCPCParameters, SCPCSolution, integrate_scpc


@dataclass(frozen=True)
class SolutionDifference:
    """Global normalized L-infinity differences on a common time grid."""

    scale_factor: float
    hubble: float
    wrapped_phase: float
    field_velocity: float
    maximum: float


@dataclass(frozen=True)
class ConvergenceResult:
    """One integration result relative to a declared reference run."""

    label: str
    method: str
    rtol: float
    atol: float
    max_abs_constraint_residual: float
    difference_to_reference: SolutionDifference | None


def _relative_linf(reference: np.ndarray, candidate: np.ndarray, floor: float) -> float:
    numerator = float(np.max(np.abs(candidate - reference)))
    denominator = max(float(np.max(np.abs(reference))), floor)
    return numerator / denominator


def compare_solutions(
    reference: SCPCSolution,
    candidate: SCPCSolution,
    *,
    relative_floor: float = 1.0e-12,
) -> SolutionDifference:
    """Compare two solutions sampled on the same time grid.

    The scalar field is compared modulo the period of the stratification
    potential. This avoids treating equivalent compact-phase representatives
    as a large numerical disagreement.
    """

    if relative_floor <= 0.0:
        raise ValueError("relative_floor must be positive")
    if reference.t.shape != candidate.t.shape or not np.allclose(
        reference.t,
        candidate.t,
        rtol=0.0,
        atol=10.0 * np.finfo(float).eps,
    ):
        raise ValueError("Solutions must use an identical stored time grid")

    reference_period = (
        2.0
        * np.pi
        * reference.parameters.potential.field_scale
        / reference.parameters.potential.strata_count
    )
    candidate_period = (
        2.0
        * np.pi
        * candidate.parameters.potential.field_scale
        / candidate.parameters.potential.strata_count
    )
    if not np.isclose(reference_period, candidate_period, rtol=1.0e-14, atol=0.0):
        raise ValueError("Solutions use incompatible stratification-field periods")

    phase_delta = (
        candidate.phi - reference.phi + 0.5 * reference_period
    ) % reference_period - 0.5 * reference_period
    scale_error = _relative_linf(reference.a, candidate.a, relative_floor)
    hubble_error = _relative_linf(reference.H, candidate.H, relative_floor)
    phase_error = float(np.max(np.abs(phase_delta))) / reference_period
    velocity_error = _relative_linf(reference.phi_dot, candidate.phi_dot, relative_floor)
    maximum = max(scale_error, hubble_error, phase_error, velocity_error)
    return SolutionDifference(
        scale_factor=scale_error,
        hubble=hubble_error,
        wrapped_phase=phase_error,
        field_velocity=velocity_error,
        maximum=maximum,
    )


def run_tolerance_ladder(
    parameters: SCPCParameters,
    *,
    integration_options: dict[str, Any],
    tolerances: tuple[tuple[str, float, float], ...],
    method: str = "DOP853",
) -> tuple[ConvergenceResult, ...]:
    """Run a declared tolerance ladder and compare against its tightest run."""

    if len(tolerances) < 2:
        raise ValueError("At least two tolerance levels are required")
    labels = [label for label, _, _ in tolerances]
    if len(labels) != len(set(labels)):
        raise ValueError("Tolerance labels must be unique")
    for _, rtol, atol in tolerances:
        if rtol <= 0.0 or atol <= 0.0:
            raise ValueError("All tolerances must be positive")

    base_options = dict(integration_options)
    for reserved in ("method", "rtol", "atol"):
        base_options.pop(reserved, None)

    solutions: dict[str, SCPCSolution] = {}
    for label, rtol, atol in tolerances:
        solutions[label] = integrate_scpc(
            parameters,
            method=method,
            rtol=rtol,
            atol=atol,
            **base_options,
        )

    reference_label, _, _ = min(tolerances, key=lambda item: (item[1], item[2]))
    reference = solutions[reference_label]
    results: list[ConvergenceResult] = []
    for label, rtol, atol in tolerances:
        solution = solutions[label]
        difference = None if label == reference_label else compare_solutions(reference, solution)
        results.append(
            ConvergenceResult(
                label=label,
                method=method,
                rtol=rtol,
                atol=atol,
                max_abs_constraint_residual=float(
                    np.max(np.abs(solution.constraint_residual))
                ),
                difference_to_reference=difference,
            )
        )
    return tuple(results)


def run_cross_solver_comparison(
    parameters: SCPCParameters,
    *,
    integration_options: dict[str, Any],
    methods: tuple[str, ...],
    rtol: float,
    atol: float,
    reference_method: str = "DOP853",
) -> tuple[ConvergenceResult, ...]:
    """Compare independent SciPy integrators at fixed tolerances."""

    if len(methods) < 2:
        raise ValueError("At least two solver methods are required")
    if reference_method not in methods:
        raise ValueError("reference_method must be present in methods")
    if len(methods) != len(set(methods)):
        raise ValueError("Solver methods must be unique")

    base_options = dict(integration_options)
    for reserved in ("method", "rtol", "atol"):
        base_options.pop(reserved, None)
    solutions = {
        method: integrate_scpc(
            parameters,
            method=method,
            rtol=rtol,
            atol=atol,
            **base_options,
        )
        for method in methods
    }
    reference = solutions[reference_method]
    return tuple(
        ConvergenceResult(
            label=method,
            method=method,
            rtol=rtol,
            atol=atol,
            max_abs_constraint_residual=float(
                np.max(np.abs(solutions[method].constraint_residual))
            ),
            difference_to_reference=(
                None
                if method == reference_method
                else compare_solutions(reference, solutions[method])
            ),
        )
        for method in methods
    )
