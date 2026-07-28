"""Canonical covariant stratification-field baseline in FLRW spacetime.

Natural units c = hbar = M_pl = 1 are used internally. This module is a
background-theory baseline, not a claim that stable cyclic solutions exist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import xarray as xr
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class PeriodicPotential:
    offset: float = 3.0
    amplitude: float = 0.1
    strata_count: int = 4
    field_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.amplitude < 0 or self.field_scale <= 0 or self.strata_count < 1:
            raise ValueError("Potential parameters are outside their allowed domain")

    def value(self, phi: np.ndarray | float) -> np.ndarray:
        x = self.strata_count * np.asarray(phi, dtype=float) / self.field_scale
        return self.offset + self.amplitude * (1.0 - np.cos(x))

    def derivative(self, phi: np.ndarray | float) -> np.ndarray:
        x = self.strata_count * np.asarray(phi, dtype=float) / self.field_scale
        return self.amplitude * self.strata_count * np.sin(x) / self.field_scale


@dataclass(frozen=True)
class SCPCParameters:
    spatial_curvature_k: int = 1
    rho_m_ref: float = 0.03
    rho_r_ref: float = 0.0
    a_ref: float = 1.0
    potential: PeriodicPotential = PeriodicPotential()

    def __post_init__(self) -> None:
        if self.spatial_curvature_k not in (-1, 0, 1):
            raise ValueError("spatial_curvature_k must be -1, 0, or 1")
        if self.rho_m_ref < 0 or self.rho_r_ref < 0 or self.a_ref <= 0:
            raise ValueError("Reference densities must be non-negative and a_ref positive")

    def matter_density(self, a: np.ndarray | float) -> np.ndarray:
        return self.rho_m_ref * (self.a_ref / np.asarray(a, dtype=float)) ** 3

    def radiation_density(self, a: np.ndarray | float) -> np.ndarray:
        return self.rho_r_ref * (self.a_ref / np.asarray(a, dtype=float)) ** 4


@dataclass
class SCPCSolution:
    t: np.ndarray
    a: np.ndarray
    H: np.ndarray
    phi: np.ndarray
    phi_dot: np.ndarray
    rho_m: np.ndarray
    rho_r: np.ndarray
    rho_phi: np.ndarray
    p_phi: np.ndarray
    constraint_residual: np.ndarray
    turning_times: np.ndarray
    turning_kinds: tuple[str, ...]
    parameters: SCPCParameters
    solver_metadata: dict[str, Any]

    def to_xarray(self) -> xr.Dataset:
        return xr.Dataset(
            data_vars={
                "scale_factor": ("time", self.a, {"units": "1"}),
                "hubble": ("time", self.H, {"units": "M_pl"}),
                "stratification_field": ("time", self.phi, {"units": "M_pl"}),
                "field_velocity": ("time", self.phi_dot, {"units": "M_pl^2"}),
                "matter_density": ("time", self.rho_m, {"units": "M_pl^4"}),
                "radiation_density": ("time", self.rho_r, {"units": "M_pl^4"}),
                "field_density": ("time", self.rho_phi, {"units": "M_pl^4"}),
                "field_pressure": ("time", self.p_phi, {"units": "M_pl^4"}),
                "friedmann_constraint_residual": ("time", self.constraint_residual, {"units": "1"}),
            },
            coords={"time": ("time", self.t, {"units": "M_pl^-1"})},
            attrs={
                "model": "canonical_scpc_phase_baseline",
                "unit_system": "natural units c=hbar=M_pl=1",
                "max_abs_constraint_residual": float(np.max(np.abs(self.constraint_residual))),
                "parameters": str(asdict(self.parameters)),
                **self.solver_metadata,
            },
        )


def _components(a: np.ndarray, phi: np.ndarray, v: np.ndarray, p: SCPCParameters):
    rho_m = p.matter_density(a)
    rho_r = p.radiation_density(a)
    potential = p.potential.value(phi)
    rho_phi = 0.5 * v**2 + potential
    p_phi = 0.5 * v**2 - potential
    return rho_m, rho_r, rho_phi, p_phi


def initial_hubble(
    a0: float,
    phi0: float,
    phi_dot0: float,
    parameters: SCPCParameters,
    branch: int = 1,
) -> float:
    if a0 <= 0 or branch not in (-1, 1):
        raise ValueError("a0 must be positive and branch must be +1 or -1")
    rho_m, rho_r, rho_phi, _ = _components(
        np.asarray(a0), np.asarray(phi0), np.asarray(phi_dot0), parameters
    )
    h2 = (rho_m + rho_r + rho_phi) / 3.0 - parameters.spatial_curvature_k / a0**2
    if h2 < 0:
        raise ValueError(f"Initial state violates the Friedmann constraint: H^2={float(h2)}")
    return float(branch * np.sqrt(h2))


def _rhs(_t: float, y: np.ndarray, p: SCPCParameters) -> np.ndarray:
    a, H, phi, v = y
    if a <= 0:
        raise FloatingPointError("Scale factor reached a non-positive value")
    rho_m = float(p.matter_density(a))
    rho_r = float(p.radiation_density(a))
    return np.asarray(
        [
            a * H,
            p.spatial_curvature_k / a**2 - 0.5 * (rho_m + 4.0 * rho_r / 3.0 + v**2),
            v,
            -3.0 * H * v - float(p.potential.derivative(phi)),
        ]
    )


def integrate_scpc(
    parameters: SCPCParameters,
    *,
    t_span: tuple[float, float],
    a0: float,
    phi0: float,
    phi_dot0: float,
    branch: int = 1,
    samples: int = 2001,
    rtol: float = 1e-9,
    atol: float = 1e-11,
    method: str = "DOP853",
) -> SCPCSolution:
    """Integrate the homogeneous canonical SCPC baseline and monitor H=0 events."""

    H0 = initial_hubble(a0, phi0, phi_dot0, parameters, branch)
    t_eval = np.linspace(t_span[0], t_span[1], samples)

    def turning_event(t: float, y: np.ndarray) -> float:
        del t
        return float(y[1])

    turning_event.terminal = False  # type: ignore[attr-defined]
    turning_event.direction = 0  # type: ignore[attr-defined]

    sol = solve_ivp(
        lambda t, y: _rhs(t, y, parameters),
        t_span,
        np.asarray([a0, H0, phi0, phi_dot0], dtype=float),
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        events=turning_event,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(f"Background integration failed: {sol.message}")

    a, H, phi, v = sol.y
    rho_m, rho_r, rho_phi, p_phi = _components(a, phi, v, parameters)
    lhs = H**2 + parameters.spatial_curvature_k / a**2
    rhs = (rho_m + rho_r + rho_phi) / 3.0
    scale = np.maximum(np.abs(rhs), 1e-15)
    residual = (lhs - rhs) / scale

    event_times = sol.t_events[0] if sol.t_events else np.asarray([])
    kinds: list[str] = []
    if sol.sol is not None:
        for event_time in event_times:
            state = sol.sol(float(event_time))
            hdot = _rhs(float(event_time), state, parameters)[1]
            kinds.append("bounce" if hdot > 0 else "turnaround" if hdot < 0 else "degenerate")

    return SCPCSolution(
        t=sol.t,
        a=a,
        H=H,
        phi=phi,
        phi_dot=v,
        rho_m=rho_m,
        rho_r=rho_r,
        rho_phi=rho_phi,
        p_phi=p_phi,
        constraint_residual=residual,
        turning_times=np.asarray(event_times),
        turning_kinds=tuple(kinds),
        parameters=parameters,
        solver_metadata={
            "solver_method": method,
            "solver_rtol": rtol,
            "solver_atol": atol,
            "solver_nfev": int(sol.nfev),
            "solver_status": int(sol.status),
        },
    )
