from __future__ import annotations

import numpy as np

from scpc.theory.potentials import PeriodicStratifiedPotential


def test_periodic_potential_gradient_matches_finite_difference() -> None:
    potential = PeriodicStratifiedPotential(
        amplitude=0.7, offset=0.2, periodicity=3, field_scale=1.4, tilt=0.05
    )
    phi = 0.37
    step = 1.0e-6
    numerical = (potential.value(phi + step) - potential.value(phi - step)) / (2.0 * step)
    assert np.isclose(potential.gradient(phi), numerical, rtol=1.0e-8, atol=1.0e-9)


def test_periodic_potential_rejects_invalid_scale() -> None:
    try:
        PeriodicStratifiedPotential(amplitude=1.0, field_scale=0.0)
    except ValueError:
        return
    raise AssertionError("Expected invalid field scale to raise ValueError")
