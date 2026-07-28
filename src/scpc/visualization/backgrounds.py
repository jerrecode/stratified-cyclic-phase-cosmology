"""Background-evolution and model-comparison figures."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from scpc.models.base import ExpansionHistory
from scpc.numerics.integrate import BackgroundSolution


def plot_background_solution(
    solution: BackgroundSolution, destination: str | Path
) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    time = solution.time
    scale_factor, hubble, phi, phi_dot, rho_m, rho_r = solution.values

    figure, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    series = (
        (scale_factor, r"$a$"),
        (hubble, r"$H$"),
        (phi, r"$\phi$"),
        (phi_dot, r"$\dot\phi$"),
        (rho_m, r"$\rho_m$"),
        (rho_r, r"$\rho_r$"),
    )
    for axis, (values, label) in zip(axes.flat, series, strict=True):
        axis.plot(time, values)
        axis.set_xlabel("Natural time")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    figure.suptitle("Canonical closed-FLRW SCPC baseline")
    figure.savefig(destination)
    plt.close(figure)


def plot_expansion_comparison(
    scale_factor: np.ndarray,
    models: Mapping[str, ExpansionHistory],
    destination: str | Path,
) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for label, model in models.items():
        axis.plot(scale_factor, model.dimensionless_hubble(scale_factor), label=label)
    axis.set_xlabel("Scale factor $a$")
    axis.set_ylabel(r"$E(a)=H(a)/H_0$")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.savefig(destination)
    plt.close(figure)
