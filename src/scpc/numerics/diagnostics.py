"""Numerical credibility diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpc.theory.background import BackgroundParameters, BackgroundState, relative_friedmann_residual
from scpc.theory.potentials import ScalarPotential


@dataclass(frozen=True)
class ConstraintSummary:
    maximum_absolute_relative_residual: float
    rms_relative_residual: float
    final_relative_residual: float


def summarize_friedmann_constraint(
    trajectory: NDArray[np.float64],
    parameters: BackgroundParameters,
    potential: ScalarPotential,
) -> ConstraintSummary:
    residuals = np.asarray(
        [
            relative_friedmann_residual(
                BackgroundState.from_array(trajectory[:, index]), parameters, potential
            )
            for index in range(trajectory.shape[1])
        ]
    )
    return ConstraintSummary(
        maximum_absolute_relative_residual=float(np.max(np.abs(residuals))),
        rms_relative_residual=float(np.sqrt(np.mean(residuals**2))),
        final_relative_residual=float(residuals[-1]),
    )
