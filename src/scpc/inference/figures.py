"""Publication-oriented DESI DR2 + BBN posterior and background figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from scpc.inference.camb_background import CambLCDMParameters, background_history, build_camb_background
from scpc.inference.desi_chains import ChainSet, weighted_correlation, weighted_mean, weighted_quantile

DESI_EFFECTIVE_REDSHIFTS = np.asarray([0.295, 0.510, 0.706, 0.934, 1.321, 1.484, 2.330])


@dataclass(frozen=True)
class PosteriorSummary:
    mean: float
    q16: float
    median: float
    q84: float


def summarize(values: np.ndarray, weights: np.ndarray) -> PosteriorSummary:
    q16, median, q84 = weighted_quantile(values, weights, (0.16, 0.5, 0.84))
    return PosteriorSummary(weighted_mean(values, weights), float(q16), float(median), float(q84))


def _density_grid(x: np.ndarray, y: np.ndarray, weights: np.ndarray, n: int = 180):
    """Weighted posterior-density grid from the complete chain."""
    x_lo, x_hi = weighted_quantile(x, weights, (0.001, 0.999))
    y_lo, y_hi = weighted_quantile(y, weights, (0.001, 0.999))
    pad_x = 0.08 * (x_hi - x_lo)
    pad_y = 0.08 * (y_hi - y_lo)
    x_edges = np.linspace(x_lo - pad_x, x_hi + pad_x, n + 1)
    y_edges = np.linspace(y_lo - pad_y, y_hi + pad_y, n + 1)
    histogram, _, _ = np.histogram2d(x, y, bins=(x_edges, y_edges), weights=weights)
    density = gaussian_filter(histogram.T, sigma=1.25, mode="nearest")
    gx = 0.5 * (x_edges[:-1] + x_edges[1:])
    gy = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(gx, gy)
    flat = density.ravel()
    order = np.argsort(flat)[::-1]
    cumulative = np.cumsum(flat[order])
    cumulative /= cumulative[-1]
    thresholds = {}
    for probability in (0.68, 0.95):
        thresholds[probability] = float(flat[order[np.searchsorted(cumulative, probability)]])
    return xx, yy, density, thresholds


def _draw_contours(ax, x: np.ndarray, y: np.ndarray, weights: np.ndarray, *, xlabel: str, ylabel: str) -> None:
    xx, yy, density, thresholds = _density_grid(x, y, weights)
    levels = [thresholds[0.95], thresholds[0.68], float(np.max(density))]
    ax.contourf(xx, yy, density, levels=levels, alpha=0.35)
    ax.contour(xx, yy, density, levels=levels[:-1], linewidths=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)


def _resample_indices(weights: np.ndarray, count: int, seed: int = 23117) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probability = np.asarray(weights, dtype=float)
    probability /= probability.sum()
    return rng.choice(probability.size, size=count, replace=True, p=probability)


def _low_z_hubble(samples: ChainSet, z: np.ndarray, *, count: int = 6000) -> np.ndarray:
    """Posterior draws of H(z) over the DESI BAO range.

    At z<=2.33 the radiation contribution is negligible and the 0.06 eV neutrino is
    non-relativistic. The high-redshift panel is generated separately from CAMB and
    therefore does not use this low-z closure approximation.
    """
    indices = _resample_indices(samples.weights, min(count, max(1000, samples.weights.size)))
    H0 = samples.h0()[indices, None]
    omega_m = samples.omega_m()[indices, None]
    zp1 = 1.0 + np.asarray(z, dtype=float)[None, :]
    return H0 * np.sqrt(omega_m * zp1**3 + (1.0 - omega_m))


def make_main_figure(
    bao_chain: ChainSet,
    bbn_chain: ChainSet,
    output: Path,
    *,
    map_result: dict[str, object] | None = None,
) -> dict[str, object]:
    """Create the four-panel publication figure recommended by the scientific audit."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), constrained_layout=True)
    ax_bao, ax_bbn, ax_h, ax_early = axes.ravel()

    omega_m_bao = bao_chain.omega_m()
    hrd_bao = bao_chain.h_r_drag_mpc()
    _draw_contours(
        ax_bao,
        omega_m_bao,
        hrd_bao,
        bao_chain.weights,
        xlabel=r"$\Omega_m$",
        ylabel=r"$h\,r_d\;[\mathrm{Mpc}]$",
    )
    corr_bao = weighted_correlation(omega_m_bao, hrd_bao, bao_chain.weights)
    ax_bao.set_title("DESI DR2 BAO-only posterior")
    ax_bao.text(0.03, 0.03, rf"weighted $\rho={corr_bao:.2f}$", transform=ax_bao.transAxes, fontsize=9)

    omega_m_bbn = bbn_chain.omega_m()
    H0_bbn = bbn_chain.h0()
    _draw_contours(
        ax_bbn,
        omega_m_bbn,
        H0_bbn,
        bbn_chain.weights,
        xlabel=r"$\Omega_m$",
        ylabel=r"$H_0\;[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$",
    )
    ax_bbn.set_title("DESI DR2 BAO + BBN posterior")

    z = np.linspace(0.0, 2.55, 220)
    h_draws = _low_z_hubble(bbn_chain, z)
    median_curve = np.median(h_draws, axis=0)
    residual = h_draws / median_curve[None, :] - 1.0
    q025, q16, q84, q975 = np.percentile(residual, [2.5, 16.0, 84.0, 97.5], axis=0)
    ax_h.fill_between(z, 100.0 * q025, 100.0 * q975, alpha=0.16, label="95% posterior predictive")
    ax_h.fill_between(z, 100.0 * q16, 100.0 * q84, alpha=0.30, label="68% posterior predictive")
    ax_h.axhline(0.0, linewidth=1.0)
    ax_h.axvspan(DESI_EFFECTIVE_REDSHIFTS.min(), DESI_EFFECTIVE_REDSHIFTS.max(), alpha=0.07)
    for zeff in DESI_EFFECTIVE_REDSHIFTS:
        ax_h.axvline(zeff, linewidth=0.6, alpha=0.35)
    ax_h.set_xlabel("Redshift $z$")
    ax_h.set_ylabel(r"$100\,[H(z)/H_{\rm med}(z)-1]$ [%]")
    ax_h.set_title("Model-implied expansion over the DESI BAO lever arm")
    ax_h.legend(fontsize=8, loc="upper right")
    ax_h.grid(alpha=0.2)

    H0_summary = summarize(H0_bbn, bbn_chain.weights)
    om_summary = summarize(omega_m_bbn, bbn_chain.weights)
    ob_summary = summarize(bbn_chain.omega_b_h2(), bbn_chain.weights)
    central = CambLCDMParameters(
        H0=H0_summary.median,
        omega_m=om_summary.median,
        omega_b_h2=ob_summary.median,
    )
    z_early = np.geomspace(1.0e-4, 1.0e5, 600) - 1.0e-4
    history = background_history(central, z_early)
    x = 1.0 + z_early
    ax_early.semilogx(x, history["omega_cb"], label=r"$\Omega_{cb}$")
    ax_early.semilogx(x, history["omega_gamma"], label=r"$\Omega_\gamma$")
    ax_early.semilogx(x, history["omega_nu"], label=r"$\Omega_\nu$")
    ax_early.semilogx(x, history["omega_lambda"], label=r"$\Omega_\Lambda$")
    marker_specs = (
        ("z_acc", r"$z_{\rm acc}$", 0.98),
        ("z_drag", r"$z_d$", 0.86),
        ("z_star", r"$z_*$", 0.74),
        ("z_eq", r"$z_{\rm eq}$", 0.62),
    )
    for key, label, text_y in marker_specs:
        value = float(history[key])
        if np.isfinite(value) and value >= 0:
            ax_early.axvline(1.0 + value, linewidth=0.8, linestyle="--", alpha=0.65)
            ax_early.text(1.0 + value, text_y, label, rotation=90, va="top", ha="right", fontsize=8)
    ax_early.set_ylim(-0.02, 1.03)
    ax_early.set_xlabel(r"$1+z$")
    ax_early.set_ylabel("Fraction of critical density")
    ax_early.set_title("CAMB background with massive-neutrino transition")
    ax_early.legend(fontsize=8, ncol=2)
    ax_early.grid(alpha=0.2)

    fig.suptitle(r"DESI DR2 (2025 BAO likelihood) + BBN constraints in flat $\Lambda$CDM", fontsize=15)
    note = (
        "Top: marginalized official DESI MCMC posteriors. Bottom left: posterior-predictive low-z model band; "
        "vertical lines mark DESI effective redshifts. Bottom right: CAMB-derived species fractions; the 0.06 eV "
        "massive-neutrino density is not forced to scale as a^-3 at early times."
    )
    if map_result is not None:
        note += (
            f" Independent pinned-likelihood MAP check: chi2_BAO={float(map_result['chi_square_bao']):.2f}, "
            f"nu={int(map_result['bao_degrees_of_freedom'])}, "
            f"p={float(map_result['bao_goodness_of_fit_p_value']):.3f}."
        )
    fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=8, wrap=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "bao_only": {
            "omega_m": summarize(omega_m_bao, bao_chain.weights).__dict__,
            "h_r_drag_Mpc": summarize(hrd_bao, bao_chain.weights).__dict__,
            "correlation_omega_m_h_r_drag": corr_bao,
        },
        "bao_bbn": {
            "omega_m": om_summary.__dict__,
            "H0": H0_summary.__dict__,
            "omega_b_h2": ob_summary.__dict__,
        },
        "camb_central": {
            "H0": central.H0,
            "omega_m": central.omega_m,
            "omega_b_h2": central.omega_b_h2,
            "r_drag_Mpc": float(history["r_drag_Mpc"]),
            "z_acc": float(history["z_acc"]),
            "z_drag": float(history["z_drag"]),
            "z_star": float(history["z_star"]),
            "z_eq": float(history["z_eq"]),
        },
    }


def make_kinematic_figure(bbn_chain: ChainSet, output: Path) -> dict[str, float]:
    """Move the redundant q/w_tot diagnostic to a clearly labeled supplementary figure."""
    output.parent.mkdir(parents=True, exist_ok=True)
    central = CambLCDMParameters(
        H0=float(weighted_quantile(bbn_chain.h0(), bbn_chain.weights, (0.5,))[0]),
        omega_m=float(weighted_quantile(bbn_chain.omega_m(), bbn_chain.weights, (0.5,))[0]),
        omega_b_h2=float(weighted_quantile(bbn_chain.omega_b_h2(), bbn_chain.weights, (0.5,))[0]),
    )
    _, results = build_camb_background(central)
    z = np.linspace(0.0, 3.0, 500)
    H = np.asarray(results.hubble_parameter(z), dtype=float)
    dH_dz = np.gradient(H, z, edge_order=2)
    q = (1.0 + z) * dH_dz / H - 1.0
    w_tot = (2.0 * q - 1.0) / 3.0
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    ax.plot(z, q, label=r"$q(z)$")
    ax.plot(z, w_tot, label=r"$w_{\rm tot}(z)=p_{\rm tot}/\rho_{\rm tot}$")
    ax.axhline(0.0, linewidth=0.9, label=r"$q=0$")
    ax.axhline(-1.0 / 3.0, linewidth=0.9, linestyle="--", label=r"$w_{\rm tot}=-1/3$")
    ax.set_xlabel("Redshift $z$")
    ax.set_ylabel("Kinematic value")
    ax.set_title(r"Supplementary flat-$\Lambda$CDM kinematics")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    transition = np.nan
    crossings = np.where(np.diff(np.signbit(q)))[0]
    if crossings.size:
        i = int(crossings[0])
        transition = float(z[i] - q[i] * (z[i + 1] - z[i]) / (q[i + 1] - q[i]))
    return {"q0": float(q[0]), "w_tot0": float(w_tot[0]), "z_acc_numeric": transition}


def aubourg_r_drag_mpc(
    H0: np.ndarray,
    omega_m: np.ndarray,
    omega_b_h2: np.ndarray,
    sum_mnu_eV: float = 0.06,
) -> np.ndarray:
    """Aubourg et al. (2015) Eq. 16; validation approximation only, never the fit backend."""
    h = np.asarray(H0, dtype=float) / 100.0
    omega_nu = sum_mnu_eV / 93.14
    omega_cb = np.asarray(omega_m, dtype=float) * h**2 - omega_nu
    if np.any(omega_cb <= 0):
        raise ValueError("Aubourg approximation received non-positive omega_cb")
    return 55.154 * np.exp(-72.3 * (omega_nu + 0.0006) ** 2) / (
        omega_cb**0.25351 * np.asarray(omega_b_h2, dtype=float) ** 0.12807
    )


def make_aubourg_validation_figure(bbn_chain: ChainSet, output: Path, *, samples: int = 48) -> dict[str, float]:
    """Compare the legacy Aubourg approximation against CAMB at posterior draws."""
    output.parent.mkdir(parents=True, exist_ok=True)
    indices = _resample_indices(bbn_chain.weights, max(8, int(samples)), seed=31891)
    H0 = bbn_chain.h0()[indices]
    omega_m = bbn_chain.omega_m()[indices]
    omega_b = bbn_chain.omega_b_h2()[indices]
    approx = aubourg_r_drag_mpc(H0, omega_m, omega_b)
    exact = np.empty_like(approx)
    for i in range(exact.size):
        _, results = build_camb_background(
            CambLCDMParameters(H0=float(H0[i]), omega_m=float(omega_m[i]), omega_b_h2=float(omega_b[i]))
        )
        derived = results.get_derived_params()
        exact[i] = float(derived["rdrag"])
    fractional = approx / exact - 1.0
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    ax.scatter(omega_m, 100.0 * fractional, s=14, alpha=0.65)
    ax.axhline(0.0, linewidth=0.9)
    ax.set_xlabel(r"$\Omega_m$ posterior draw")
    ax.set_ylabel(r"$100\,(r_d^{\rm Aubourg}/r_d^{\rm CAMB}-1)$ [%]")
    ax.set_title("Supplementary sound-horizon backend validation")
    ax.grid(alpha=0.2)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return {
        "samples": int(exact.size),
        "mean_fractional_difference": float(np.mean(fractional)),
        "rms_fractional_difference": float(np.sqrt(np.mean(fractional**2))),
        "max_abs_fractional_difference": float(np.max(np.abs(fractional))),
    }
