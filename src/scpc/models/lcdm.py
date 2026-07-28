"""Lambda-CDM background comparator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scpc.models.base import validate_scale_factor


@dataclass(frozen=True)
class LambdaCDM:
    """Homogeneous expansion history including radiation and curvature."""

    omega_matter: float
    omega_radiation: float
    omega_curvature: float = 0.0
    omega_lambda: float | None = None
    hubble_0_km_s_mpc: float = 67.36
    name: str = "Lambda-CDM"

    def __post_init__(self) -> None:
        values = (self.omega_matter, self.omega_radiation)
        if any(value < 0.0 for value in values):
            raise ValueError("Matter and radiation density fractions must be non-negative")
        if self.hubble_0_km_s_mpc <= 0.0:
            raise ValueError("H0 must be positive")

    @property
    def resolved_omega_lambda(self) -> float:
        if self.omega_lambda is not None:
            return self.omega_lambda
        return 1.0 - self.omega_matter - self.omega_radiation - self.omega_curvature

    def dimensionless_hubble_squared(self, scale_factor: ArrayLike) -> NDArray[np.float64]:
        a = validate_scale_factor(scale_factor)
        return (
            self.omega_radiation * a**-4
            + self.omega_matter * a**-3
            + self.omega_curvature * a**-2
            + self.resolved_omega_lambda
        )

    def dimensionless_hubble(self, scale_factor: ArrayLike) -> NDArray[np.float64]:
        e2 = self.dimensionless_hubble_squared(scale_factor)
        if np.any(e2 < 0.0):
            raise ValueError("Expansion history contains H^2 < 0")
        return np.sqrt(e2)
