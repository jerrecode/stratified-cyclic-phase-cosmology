"""Symmetry-reduced scalar test field on a fixed R x S^4 spacetime.

This module is deliberately not a five-dimensional gravity theory.  It implements the
maximum higher-dimensional calculation justified by the declared fixed geometry: an
S^3-invariant scalar field with a finite-volume radial operator whose discrete energy
uses the same face-gradient form as the Laplacian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class S4RadialGrid:
    cells: int = 96
    radius: float = 1.0

    def __post_init__(self) -> None:
        if self.cells < 8 or self.radius <= 0:
            raise ValueError("S4RadialGrid requires cells >= 8 and positive radius")

    @property
    def faces(self) -> np.ndarray:
        return np.linspace(0.0, np.pi, self.cells + 1)

    @property
    def centers(self) -> np.ndarray:
        f = self.faces
        return 0.5 * (f[:-1] + f[1:])

    @property
    def dchi(self) -> float:
        return np.pi / self.cells

    @property
    def cell_weights(self) -> np.ndarray:
        """Exact integral of sin^3(chi) over every finite-volume cell."""
        faces = self.faces

        def primitive(chi: np.ndarray) -> np.ndarray:
            return -np.cos(chi) + np.cos(chi) ** 3 / 3.0

        return np.diff(primitive(faces))

    @property
    def face_measure(self) -> np.ndarray:
        return np.sin(self.faces) ** 3

    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """Radial Laplace-Beltrami operator with regular zero flux at both poles."""
        phi = np.asarray(field, dtype=float)
        if phi.shape != (self.cells,):
            raise ValueError(f"field must have shape {(self.cells,)}, got {phi.shape}")
        flux = np.zeros(self.cells + 1, dtype=float)
        flux[1:-1] = self.face_measure[1:-1] * np.diff(phi) / self.dchi
        return np.diff(flux) / (self.radius**2 * self.cell_weights)

    def gradient_quadratic(self, field: np.ndarray) -> float:
        """Integral of sin^3(chi) (d_phi/d_chi)^2 using the operator's face form."""
        phi = np.asarray(field, dtype=float)
        differences = np.diff(phi)
        return float(np.sum(self.face_measure[1:-1] * differences**2 / self.dchi))


@dataclass(frozen=True)
class LayeredScalarParameters:
    mass: float = 0.35
    coupling: float = 0.05
    damping: float = 0.0

    def __post_init__(self) -> None:
        if self.mass < 0 or self.coupling < 0 or self.damping < 0:
            raise ValueError("mass, coupling, and damping must be non-negative")


def potential(field: np.ndarray, parameters: LayeredScalarParameters) -> np.ndarray:
    phi = np.asarray(field, dtype=float)
    return 0.5 * parameters.mass**2 * phi**2 + 0.25 * parameters.coupling * phi**4


def layered_energy(
    grid: S4RadialGrid,
    field: np.ndarray,
    velocity: np.ndarray,
    parameters: LayeredScalarParameters,
) -> float:
    """Continuum-normalized semi-discrete energy associated with the FV operator."""
    phi = np.asarray(field, dtype=float)
    vel = np.asarray(velocity, dtype=float)
    if phi.shape != (grid.cells,) or vel.shape != (grid.cells,):
        raise ValueError("field and velocity must match the grid")
    cell_term = np.sum(grid.cell_weights * (0.5 * vel**2 + potential(phi, parameters)))
    gradient_term = 0.5 / grid.radius**2 * grid.gradient_quadratic(phi)
    return float(2.0 * np.pi**2 * grid.radius**4 * (cell_term + gradient_term))


def integrate_layered_pulse(
    *,
    grid: S4RadialGrid | None = None,
    parameters: LayeredScalarParameters | None = None,
    t_end: float = 6.0,
    samples: int = 321,
    amplitude: float = 0.2,
    center: float = 0.42,
    width: float = 0.13,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
) -> dict[str, np.ndarray | float | int]:
    """Evolve a smooth radial pulse and return arrays plus energy diagnostics."""
    grid = grid or S4RadialGrid()
    parameters = parameters or LayeredScalarParameters()
    if t_end <= 0 or samples < 3 or width <= 0:
        raise ValueError("invalid integration controls")
    chi = grid.centers
    phi0 = amplitude * np.exp(-0.5 * ((chi - center) / width) ** 2)
    velocity0 = np.zeros_like(phi0)
    y0 = np.concatenate([phi0, velocity0])

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        field = state[: grid.cells]
        velocity = state[grid.cells :]
        acceleration = (
            grid.laplacian(field)
            - parameters.damping * velocity
            - parameters.mass**2 * field
            - parameters.coupling * field**3
        )
        return np.concatenate([velocity, acceleration])

    times = np.linspace(0.0, t_end, samples)
    result = solve_ivp(
        rhs,
        (0.0, t_end),
        y0,
        t_eval=times,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not result.success:
        raise RuntimeError(f"Layered scalar integration failed: {result.message}")
    field = result.y[: grid.cells].T
    velocity = result.y[grid.cells :].T
    energies = np.asarray(
        [layered_energy(grid, field[i], velocity[i], parameters) for i in range(times.size)],
        dtype=float,
    )
    reference = max(abs(float(energies[0])), np.finfo(float).tiny)
    relative_energy_error = energies / energies[0] - 1.0
    return {
        "time": times,
        "chi": chi,
        "field": field,
        "velocity": velocity,
        "energy": energies,
        "relative_energy_error": relative_energy_error,
        "max_abs_relative_energy_drift": float(np.max(np.abs(energies - energies[0])) / reference),
        "nfev": int(result.nfev),
        "cells": int(grid.cells),
        "radius": float(grid.radius),
    }
