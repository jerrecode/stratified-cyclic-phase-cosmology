"""Event functions for cosmological turning points and invalid states."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

Event = Callable[[float, NDArray[np.float64]], float]


def bounce_event(_time: float, values: NDArray[np.float64]) -> float:
    return float(values[1])


bounce_event.direction = 1.0  # type: ignore[attr-defined]
bounce_event.terminal = False  # type: ignore[attr-defined]


def turnaround_event(_time: float, values: NDArray[np.float64]) -> float:
    return float(values[1])


turnaround_event.direction = -1.0  # type: ignore[attr-defined]
turnaround_event.terminal = False  # type: ignore[attr-defined]


def invalid_scale_factor_event(_time: float, values: NDArray[np.float64]) -> float:
    return float(values[0])


invalid_scale_factor_event.direction = -1.0  # type: ignore[attr-defined]
invalid_scale_factor_event.terminal = True  # type: ignore[attr-defined]
