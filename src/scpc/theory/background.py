"""Closed-FLRW homogeneous equations for the canonical SCPC baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpc.theory.potentials import ScalarPotential


@dataclass(frozen=True)
class BackgroundParameters:
    """Physical parameters in reduced-Planck natural units."""

    reduced_planck_mass: float = 1.0
    curvature_k: int = 1

    def __post_init__(self) -> None:
        if self.reduced_planck_mass <= 0.0:
            raise ValueError("reduced_planck_mass must be positive")
        if self.curvature_k not in {-1, 0, 1}:
            raise ValueError("curvature_k must be -1, 0, or +1")


@dataclass(frozen=True)
class BackgroundState:
    scale_factor: float
    hubble: float
    phi: float
    phi_dot: float
    rho_matter: float
    rho_radiation: float

    def as_array(self) -> NDArray[np.float64]:
        return np.asarray(
            [
                self.scale_factor,
                self.hubble,
                self.phi,
                self.phi_dot,
                self.rho_matter,
                self.rho_radiation,
            ],
            dtype=float,
        )

    @classmethod
    def from_array(cls, values: NDArray[np.float64]) -> "BackgroundState":
        if values.shape != (6,):
            raise ValueError(f"Expected six background variables, received {values.shape}")
        return cls(*(float(value) for value in values))


def scalar_density_pressure(
    phi: float, phi_dot: float, potential: ScalarPotential
) -> tuple[float, float]:
    kinetic = 0.5 * phi_dot**2
    value = float(potential.value(phi))
    return kinetic + value, kinetic - value


def total_density_pressure(
    state: BackgroundState, potential: ScalarPotential
) -> tuple[float, float]:
    rho_phi, p_phi = scalar_density_pressure(state.phi, state.phi_dot, potential)
    rho = state.rho_matter + state.rho_radiation + rho_phi
    pressure = state.rho_radiation / 3.0 + p_phi
    return rho, pressure


def friedmann_constraint(
    state: BackgroundState,
    parameters: BackgroundParameters,
    potential: ScalarPotential,
) -> float:
    """Return 3 M_pl^2 (H^2 + k/a^2) - rho_total."""
    if state.scale_factor <= 0.0:
        return float("nan")
    rho, _ = total_density_pressure(state, potential)
    geometric = 3.0 * parameters.reduced_planck_mass**2 * (
        state.hubble**2 + parameters.curvature_k / state.scale_factor**2
    )
    return geometric - rho


def relative_friedmann_residual(
    state: BackgroundState,
    parameters: BackgroundParameters,
    potential: ScalarPotential,
    floor: float = 1.0e-14,
) -> float:
    rho, _ = total_density_pressure(state, potential)
    return friedmann_constraint(state, parameters, potential) / max(abs(rho), floor)


def consistent_hubble(
    state_without_hubble: BackgroundState,
    parameters: BackgroundParameters,
    potential: ScalarPotential,
    branch: str,
) -> float:
    """Solve the Friedmann constraint for the initial Hubble value."""
    rho, _ = total_density_pressure(state_without_hubble, potential)
    h2 = rho / (3.0 * parameters.reduced_planck_mass**2)
    h2 -= parameters.curvature_k / state_without_hubble.scale_factor**2
    if h2 < -1.0e-13:
        raise ValueError(
            "Initial state does not admit a real Hubble parameter: " f"H^2={h2:.6e}"
        )
    magnitude = float(np.sqrt(max(h2, 0.0)))
    if branch == "expanding":
        return magnitude
    if branch == "contracting":
        return -magnitude
    if branch == "static":
        if magnitude > 1.0e-10:
            raise ValueError("Static branch requested but the constraint requires nonzero H")
        return 0.0
    raise ValueError("branch must be expanding, contracting, or static")


def background_rhs(
    _time: float,
    values: NDArray[np.float64],
    parameters: BackgroundParameters,
    potential: ScalarPotential,
) -> NDArray[np.float64]:
    state = BackgroundState.from_array(values)
    if state.scale_factor <= 0.0:
        raise FloatingPointError("Scale factor became non-positive")

    rho, pressure = total_density_pressure(state, potential)
    m2 = parameters.reduced_planck_mass**2

    scale_factor_dot = state.scale_factor * state.hubble
    hubble_dot = -0.5 * (rho + pressure) / m2
    hubble_dot += parameters.curvature_k / state.scale_factor**2
    phi_dot = state.phi_dot
    phi_ddot = -3.0 * state.hubble * state.phi_dot - float(potential.gradient(state.phi))
    matter_dot = -3.0 * state.hubble * state.rho_matter
    radiation_dot = -4.0 * state.hubble * state.rho_radiation

    return np.asarray(
        [scale_factor_dot, hubble_dot, phi_dot, phi_ddot, matter_dot, radiation_dot],
        dtype=float,
    )
