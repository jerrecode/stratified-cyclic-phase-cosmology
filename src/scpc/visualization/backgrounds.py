"""Publication-oriented background plots from serialized scientific arrays."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scpc.models.phase import SCPCSolution


def plot_expansion_comparison(tables: dict[str, dict[str, np.ndarray]], output: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for name, table in tables.items():
        ax.plot(table["redshift"], table["H_km_s_Mpc"], label=name)
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_scpc_background(solution: SCPCSolution, output: str | Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), sharex=True)
    axes[0, 0].plot(solution.t, solution.a)
    axes[0, 0].set_ylabel("$a$")
    axes[0, 1].plot(solution.t, solution.H)
    axes[0, 1].axhline(0.0, linewidth=0.8)
    axes[0, 1].set_ylabel("$H$ [$M_{\\rm Pl}$]")
    axes[1, 0].plot(solution.t, solution.phi)
    axes[1, 0].set_ylabel(r"$\phi$ [$M_{\rm Pl}$]")
    axes[1, 1].plot(solution.t, np.abs(solution.constraint_residual))
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("Friedmann residual")
    for ax in axes[1, :]:
        ax.set_xlabel(r"$t$ [$M_{\rm Pl}^{-1}$]")
    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
