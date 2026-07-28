"""Small effective-loop-quantum-cosmology comparator utilities.

This module is a benchmark equation, not a complete LQC implementation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def effective_hubble_squared(
    density: ArrayLike,
    critical_density: float,
    reduced_planck_mass: float = 1.0,
) -> NDArray[np.float64]:
    """Return H^2 = rho/(3 M_pl^2) * (1 - rho/rho_c)."""
    rho = np.asarray(density, dtype=float)
    if critical_density <= 0.0 or reduced_planck_mass <= 0.0:
        raise ValueError("critical_density and reduced_planck_mass must be positive")
    return rho / (3.0 * reduced_planck_mass**2) * (1.0 - rho / critical_density)
