"""Homogeneous FLRW comparison models and distance observables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

from scpc.constants import C_KM_S


@dataclass(frozen=True)
class ExpansionParameters:
    """Late-time FLRW parameters using the CPL dark-energy parameterization.

    Densities are defined at a=1. Omega_de is inferred from closure when omitted.
    Omega_k follows the standard convention Omega_k = -k c^2/(a0 H0)^2.
    """

    H0: float = 67.4
    omega_m: float = 0.315
    omega_r: float = 9.0e-5
    omega_k: float = 0.0
    omega_de: float | None = None
    w0: float = -1.0
    wa: float = 0.0
    name: str = "flat_lcdm"

    def __post_init__(self) -> None:
        if self.H0 <= 0:
            raise ValueError("H0 must be positive")
        if self.omega_m < 0 or self.omega_r < 0:
            raise ValueError("Matter and radiation densities must be non-negative")
        if self.resolved_omega_de < 0:
            raise ValueError("Resolved dark-energy density must be non-negative")

    @property
    def resolved_omega_de(self) -> float:
        if self.omega_de is not None:
            return float(self.omega_de)
        return 1.0 - self.omega_m - self.omega_r - self.omega_k


class FLRWExpansion:
    """Background expansion and geometrical distances for CPL cosmologies."""

    def __init__(self, parameters: ExpansionParameters):
        self.p = parameters

    def dark_energy_scaling(self, a: np.ndarray | float) -> np.ndarray:
        a_arr = np.asarray(a, dtype=float)
        if np.any(a_arr <= 0):
            raise ValueError("Scale factor must be positive")
        return a_arr ** (-3.0 * (1.0 + self.p.w0 + self.p.wa)) * np.exp(
            -3.0 * self.p.wa * (1.0 - a_arr)
        )

    def e2_of_a(self, a: np.ndarray | float) -> np.ndarray:
        a_arr = np.asarray(a, dtype=float)
        e2 = (
            self.p.omega_r * a_arr**-4
            + self.p.omega_m * a_arr**-3
            + self.p.omega_k * a_arr**-2
            + self.p.resolved_omega_de * self.dark_energy_scaling(a_arr)
        )
        if np.any(e2 <= 0):
            raise ValueError("E(a)^2 became non-positive in the requested domain")
        return e2

    def e_of_z(self, z: np.ndarray | float) -> np.ndarray:
        z_arr = np.asarray(z, dtype=float)
        if np.any(z_arr < 0):
            raise ValueError("This background helper expects z >= 0")
        return np.sqrt(self.e2_of_a(1.0 / (1.0 + z_arr)))

    def hubble(self, z: np.ndarray | float) -> np.ndarray:
        return self.p.H0 * self.e_of_z(z)

    def distance_table(self, z: np.ndarray) -> dict[str, np.ndarray]:
        z_arr = np.asarray(z, dtype=float)
        if z_arr.ndim != 1 or z_arr.size < 2:
            raise ValueError("z must be a one-dimensional grid with at least two points")
        if np.any(np.diff(z_arr) <= 0) or z_arr[0] < 0:
            raise ValueError("z must be strictly increasing and non-negative")

        e = self.e_of_z(z_arr)
        chi = cumulative_trapezoid(1.0 / e, z_arr, initial=0.0)
        d_h = C_KM_S / self.p.H0
        ok = self.p.omega_k
        if np.isclose(ok, 0.0):
            d_m = d_h * chi
        elif ok > 0.0:
            root = np.sqrt(ok)
            d_m = d_h * np.sinh(root * chi) / root
        else:
            root = np.sqrt(-ok)
            d_m = d_h * np.sin(root * chi) / root

        d_l = (1.0 + z_arr) * d_m
        mu = np.full_like(d_l, np.nan)
        positive = d_l > 0
        mu[positive] = 5.0 * np.log10(d_l[positive]) + 25.0
        return {
            "redshift": z_arr,
            "E": e,
            "H_km_s_Mpc": self.p.H0 * e,
            "chi_dimensionless": chi,
            "D_M_Mpc": d_m,
            "D_L_Mpc": d_l,
            "distance_modulus_mag": mu,
        }
