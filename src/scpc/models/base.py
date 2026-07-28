"""Interfaces shared by observational background comparators."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ExpansionHistory(Protocol):
    name: str

    def dimensionless_hubble(self, scale_factor: ArrayLike) -> NDArray[np.float64]: ...


def validate_scale_factor(scale_factor: ArrayLike) -> NDArray[np.float64]:
    values = np.asarray(scale_factor, dtype=float)
    if np.any(values <= 0.0):
        raise ValueError("Scale factor values must be positive")
    return values
