"""Publication-oriented background plots from serialized scientific arrays."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scpc.models.phase import SCPCSolution


def plot_expansion_comparison(tables: dict[str, dict[str, np.ndarray]], output: str | Path) -> None:
    """Keep the standard-model comparison as a pipeline diagnostic, not a principal result."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    reference = next(iter(tables.values()))["H_km_s_Mpc"]
    for name, table in tables.items():
        relative = table["H_km_s_Mpc"] / reference - 1.0
        ax.plot(table["redshift"], 100.0 * relative, label=name)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel(r"$100[H/H_{\rm first}-1]$ [\%]")
    ax.set_title("Standard-comparator expansion residuals")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_scpc_background(solution: SCPCSolution, output: str | Path) -> None:
    """Plot the canonical trajectory with event and constraint information made explicit."""
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.9), sharex=True)
    axes[0, 0].plot(solution.t, np.log(solution.a))
    axes[0, 0].set_ylabel(r"$N=\ln a$")
    axes[0, 0].set_title("Expansion history")

    axes[0, 1].plot(solution.t, solution.H)
    axes[0, 1].axhline(0.0, linewidth=0.9)
    axes[0, 1].set_ylabel("$H$ [$M_{\\rm Pl}$]")
    axes[0, 1].set_title("Turning-event diagnostic")
    if solution.turning_times.size:
        for time, kind in zip(solution.turning_times, solution.turning_kinds, strict=True):
            axes[0, 1].axvline(time, linestyle="--", linewidth=0.8, alpha=0.65)
            axes[0, 1].text(time, 0.02, str(kind), rotation=90, fontsize=7, va="bottom")
    else:
        axes[0, 1].text(0.04, 0.08, "No root-localized $H=0$ event", transform=axes[0, 1].transAxes, fontsize=8)

    axes[1, 0].plot(solution.t, solution.phi)
    axes[1, 0].set_ylabel(r"$\phi$ [$M_{\rm Pl}$]")
    axes[1, 0].set_title("Scalar trajectory")

    plot_floor = 1.0e-16
    residual = np.maximum(np.abs(solution.constraint_residual), plot_floor)
    axes[1, 1].plot(solution.t, residual)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylim(plot_floor, max(1.0e-9, 10.0 * float(np.max(residual))))
    axes[1, 1].set_ylabel(r"$|\epsilon_F|$")
    axes[1, 1].set_title("Friedmann-constraint residual")
    axes[1, 1].text(
        0.04,
        0.08,
        r"Display floor $10^{-16}$; reported maximum uses unfloored residuals",
        transform=axes[1, 1].transAxes,
        fontsize=7,
    )

    for ax in axes[1, :]:
        ax.set_xlabel(r"$t$ [$M_{\rm Pl}^{-1}$]")
    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Canonical SCPC baseline: numerical consistency without recurrence", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
