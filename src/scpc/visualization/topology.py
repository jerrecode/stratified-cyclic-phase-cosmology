"""Topology and fixed-background layered-demonstrator figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_s4_foliation(output: str | Path) -> None:
    """Plot an exact meridional S4 foliation schematic without identifying chi with time."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    ax = axes[0]
    theta = np.linspace(0.0, 2.0 * np.pi, 600)
    ax.plot(np.cos(theta), np.sin(theta), linewidth=1.2)
    for chi in np.linspace(0.15 * np.pi, 0.85 * np.pi, 6):
        y = np.cos(chi)
        radius = np.sin(chi)
        ax.plot([-radius, radius], [y, y], linewidth=0.9, alpha=0.65)
    ax.set_aspect("equal")
    ax.set_xlabel("meridional embedding coordinate")
    ax.set_ylabel("polar embedding coordinate")
    ax.set_title(r"Meridional schematic of $S^4$ foliated by $S^3$ leaves")
    ax.grid(alpha=0.2)

    chi = np.linspace(0.0, np.pi, 500)
    axes[1].plot(chi / np.pi, np.sin(chi))
    axes[1].set_xlabel(r"Spatial polar coordinate $\chi/\pi$")
    axes[1].set_ylabel(r"$A(\chi)/R=\sin\chi$")
    axes[1].set_title(r"Exact $S^3$ areal radius")
    axes[1].grid(alpha=0.2)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_layered_propagation(result: dict[str, object], output: str | Path) -> None:
    """Plot the causal test-field propagation and its discrete-energy error."""
    time = np.asarray(result["time"], dtype=float)
    chi = np.asarray(result["chi"], dtype=float)
    field = np.asarray(result["field"], dtype=float)
    relative_energy_error = np.asarray(result["relative_energy_error"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), constrained_layout=True)
    image = axes[0, 0].imshow(
        field.T,
        origin="lower",
        aspect="auto",
        extent=[time[0], time[-1], chi[0], chi[-1]],
    )
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel(r"$\chi$")
    axes[0, 0].set_title("S$^3$-invariant scalar propagation on fixed $R\times S^4$")
    fig.colorbar(image, ax=axes[0, 0], label=r"$\varphi$")

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        index = min(int(round(fraction * (time.size - 1))), time.size - 1)
        axes[0, 1].plot(chi, field[index], label=rf"$t={time[index]:.2f}$")
    axes[0, 1].set_xlabel(r"$\chi$")
    axes[0, 1].set_ylabel(r"$\varphi$")
    axes[0, 1].set_title("Selected radial profiles")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(alpha=0.2)

    axes[1, 0].plot(time, np.max(np.abs(field), axis=1))
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel(r"$\max_\chi|\varphi|$")
    axes[1, 0].set_title("Pulse amplitude")
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].plot(time, np.abs(relative_energy_error))
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel(r"$|E/E_0-1|$")
    axes[1, 1].set_title("Finite-volume energy diagnostic")
    axes[1, 1].grid(alpha=0.2)

    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
