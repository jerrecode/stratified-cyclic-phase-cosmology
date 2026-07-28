"""Scalar-field potentials used to represent phase strata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

ScalarOrArray = float | NDArray[np.float64]


class ScalarPotential(Protocol):
    """Interface required by homogeneous background equations."""

    def value(self, phi: ArrayLike) -> ScalarOrArray: ...

    def gradient(self, phi: ArrayLike) -> ScalarOrArray: ...

    def hessian(self, phi: ArrayLike) -> ScalarOrArray: ...


@dataclass(frozen=True)
class PeriodicStratifiedPotential:
    r"""Periodic phase-stratum potential.

    V(phi) = V0 + A[1-cos(N phi/f)] + tilt*phi.

    All values are expressed in the model's declared natural-unit system.
    """

    amplitude: float
    offset: float = 0.0
    periodicity: int = 1
    field_scale: float = 1.0
    tilt: float = 0.0

    def __post_init__(self) -> None:
        if self.periodicity < 1:
            raise ValueError("periodicity must be a positive integer")
        if self.field_scale <= 0.0:
            raise ValueError("field_scale must be positive")
        if self.amplitude < 0.0:
            raise ValueError("amplitude must be non-negative")

    def _argument(self, phi: ArrayLike) -> NDArray[np.float64]:
        return self.periodicity * np.asarray(phi, dtype=float) / self.field_scale

    def value(self, phi: ArrayLike) -> ScalarOrArray:
        array = np.asarray(phi, dtype=float)
        result = self.offset + self.amplitude * (1.0 - np.cos(self._argument(array)))
        result = result + self.tilt * array
        return float(result) if result.ndim == 0 else result

    def gradient(self, phi: ArrayLike) -> ScalarOrArray:
        array = np.asarray(phi, dtype=float)
        prefactor = self.amplitude * self.periodicity / self.field_scale
        result = prefactor * np.sin(self._argument(array)) + self.tilt
        return float(result) if result.ndim == 0 else result

    def hessian(self, phi: ArrayLike) -> ScalarOrArray:
        array = np.asarray(phi, dtype=float)
        prefactor = self.amplitude * (self.periodicity / self.field_scale) ** 2
        result = prefactor * np.cos(self._argument(array))
        return float(result) if result.ndim == 0 else result


@dataclass(frozen=True)
class QuadraticPotential:
    """Quadratic verification potential V = offset + mass_squared * phi^2 / 2."""

    mass_squared: float
    offset: float = 0.0

    def value(self, phi: ArrayLike) -> ScalarOrArray:
        array = np.asarray(phi, dtype=float)
        result = self.offset + 0.5 * self.mass_squared * array**2
        return float(result) if result.ndim == 0 else result

    def gradient(self, phi: ArrayLike) -> ScalarOrArray:
        array = np.asarray(phi, dtype=float)
        result = self.mass_squared * array
        return float(result) if result.ndim == 0 else result

    def hessian(self, phi: ArrayLike) -> ScalarOrArray:
        array = np.asarray(phi, dtype=float)
        result = np.full_like(array, self.mass_squared, dtype=float)
        return float(result) if result.ndim == 0 else result
