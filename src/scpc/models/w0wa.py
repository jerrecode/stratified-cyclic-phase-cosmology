"""CPL w0-wa dark-energy background comparator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scpc.models.base import validate_scale_factor


@dataclass(frozen=True)
class W0WaCDM:
    omega_matter: float
    omega_radiation: float
    w0: float
    wa: float
    omega_curvature: float = 0.0
    omega_dark_energy: float | None = None
    hubble_0_km_s_mpc: float = 67.36
    name: str = "w0wa-CDM"

    @property
    def resolved_omega_dark_energy(self) -> float:
        if self.omega_dark_energy is not None:
            return self.omega_dark_energy
        return 1.0 - self.omega_matter - self.omega_radiation - self.omega_curvature

    def dark_energy_scaling(self, scale_factor: ArrayLike) -> NDArray[np.float64]:
        a = validate_scale_factor(scale_factor)
        power = -3.0 * (1.0 + self.w0 + self.wa)
        return a**power * np.exp(-3.0 * self.wa * (1.0 - a))

    def dimensionless_hubble(self, scale_factor: ArrayLike) -> NDArray[np.float64]:
        a = validate_scale_factor(scale_factor)
        e2 = (
            self.omega_radiation * a**-4
            + self.omega_matter * a**-3
            + self.omega_curvature * a**-2
            + self.resolved_omega_dark_energy * self.dark_energy_scaling(a)
        )
        if np.any(e2 < 0.0):
            raise ValueError("Expansion history contains H^2 < 0")
        return np.sqrt(e2)
