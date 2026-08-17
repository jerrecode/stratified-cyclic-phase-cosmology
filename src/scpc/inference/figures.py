"""Publication-oriented DESI DR2 + BBN posterior and validation figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scpc.inference.camb_background import CambLCDMParameters, background_history, build_camb_background
from scpc.inference.desi_chains import ChainSet, weighted_correlation, weighted_mean, weighted_quantile
from scpc.inference.desi_likelihood import DESIDR2BAOLikelihood


@dataclass(frozen=True)
class PosteriorSummary:
    mean: float
    q16: float
    median: float
    q84: float


def summarize(values: np.ndarray, weights: np.ndarray) -> PosteriorSummary:
    q16, median, q84 = weighted_quantile(values, weights, (0.16, 0.5, 0.84))
    return PosteriorSummary(weighted_mean(values, weights), float(q16), float(median), float(q84))


def _canonical(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def getdist_parameter_name(samples, *aliases: str) -> str:
    names = [item.name for item in samples.getParamNames().names]
    lookup = {_canonical(name): name for name in names}
    for alias in aliases:
        key = _canonical(alias)
        if key in lookup:
            return lookup[key]
    raise KeyError(f"None of {aliases!r} found in GetDist parameters {names!r}")


def add_getdist_derived(samples, values: np.ndarray, name: str, label: str) -> str:
    """Add one derived vector to a GetDist sample object unless already present."""
    try:
        return getdist_parameter_name(samples, name)
    except KeyError:
        pass
    vector = np.asarray(values, dtype=float)
    if vector.shape != (samples.numrows,):
        raise ValueError(f"Derived parameter {name} has shape {vector.shape}; expected {(samples.numrows,)}")
    samples.addDerived(vector, name=name, label=label)
    return name


def _draw_getdist_contours(ax, samples, x_name: str, y_name: str, *, xlabel: str, ylabel: str) -> None:
    """Draw GetDist's autocorrelation-aware KDE and 68/95 percent credible regions."""
    density = samples.get2DDensityGridData(x_name, y_name, num_plot_contours=2)
    if density is None or density.contours is None or len(density.contours) < 2:
        raise RuntimeError(f"GetDist could not construct two contours for {x_name}, {y_name}")
    xx, yy = np.meshgrid(density.x, density.y)
    contour_levels = sorted(float(value) for value in density.contours[:2])
    fill_levels = [contour_levels[0], contour_levels[1], float(np.nanmax(density.P))]
    ax.contourf(xx, yy, density.P, levels=fill_levels, alpha=0.35)
    ax.contour(xx, yy, density.P, levels=contour_levels, linewidths=1.2)
    ax.text(0.97, 0.96, "68%, 95% marginalized", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)


def _posterior_subset(chain: ChainSet, maximum: int = 8000) -> tuple[np.ndarray, np.ndarray]:
    count = chain.weights.size
    if count <= maximum:
        indices = np.arange(count)
    else:
        indices = np.unique(np.linspace(0, count - 1, maximum, dtype=int))
    return indices, chain.weights[indices]


def _flat_lcdm_bao_predictions(
    likelihood: DESIDR2BAOLikelihood,
    H0: np.ndarray,
    omega_m: np.ndarray,
    r_drag: np.ndarray,
) -> np.ndarray:
    """Vectorized low-z flat-LCDM BAO prediction used only for posterior predictive checks.

    Radiation is negligible over the DESI BAO redshift range.  The independent CAMB MAP
    calculation remains the regression backend for the exact published likelihood check.
    """
    H0 = np.asarray(H0, dtype=float)
    omega_m = np.asarray(omega_m, dtype=float)
    r_drag = np.asarray(r_drag, dtype=float)
    z_unique = likelihood.unique_redshifts
    nodes, weights = np.polynomial.legendre.leggauss(48)
    c_km_s = 299792.458
    dm = np.empty((H0.size, z_unique.size), dtype=float)
    dh = np.empty_like(dm)
    dv = np.empty_like(dm)
    for j, z in enumerate(z_unique):
        integration_z = 0.5 * z * (nodes + 1.0)
        e_grid = np.sqrt(omega_m[:, None] * (1.0 + integration_z[None, :]) ** 3 + (1.0 - omega_m[:, None]))
        integral = 0.5 * z * np.sum(weights[None, :] / e_grid, axis=1)
        dm[:, j] = c_km_s / H0 * integral
        e_here = np.sqrt(omega_m * (1.0 + z) ** 3 + (1.0 - omega_m))
        dh[:, j] = c_km_s / (H0 * e_here)
        dv[:, j] = np.cbrt(z * dh[:, j] * dm[:, j] ** 2)
    index = {float(z): j for j, z in enumerate(z_unique)}
    prediction = np.empty((H0.size, len(likelihood.rows)), dtype=float)
    for j, row in enumerate(likelihood.rows):
        k = index[row.redshift]
        if row.quantity == "DM_over_rs":
            prediction[:, j] = dm[:, k] / r_drag
        elif row.quantity == "DH_over_rs":
            prediction[:, j] = dh[:, k] / r_drag
        elif row.quantity == "DV_over_rs":
            prediction[:, j] = dv[:, k] / r_drag
        else:  # pragma: no cover - likelihood loader validates supported quantities
            raise ValueError(f"Unsupported BAO observable {row.quantity}")
    return prediction


def _whitened_posterior_predictive(
    chain: ChainSet,
    likelihood: DESIDR2BAOLikelihood,
    *,
    maximum: int = 8000,
) -> tuple[np.ndarray, np.ndarray]:
    indices, weights = _posterior_subset(chain, maximum=maximum)
    prediction = _flat_lcdm_bao_predictions(
        likelihood,
        chain.h0()[indices],
        chain.omega_m()[indices],
        chain.r_drag_mpc()[indices],
    )
    residual = prediction - likelihood.data[None, :]
    whitened = np.linalg.solve(likelihood.cholesky, residual.T).T
    return whitened, weights


def _column_quantiles(matrix: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, ...]:
    results = [
        weighted_quantile(matrix[:, j], weights, (0.025, 0.16, 0.5, 0.84, 0.975))
        for j in range(matrix.shape[1])
    ]
    return tuple(np.asarray(results).T)


def make_main_figure(
    bao_chain: ChainSet,
    bbn_chain: ChainSet,
    bao_getdist,
    bbn_getdist,
    likelihood: DESIDR2BAOLikelihood,
    output: Path,
    *,
    map_result: dict[str, object] | None = None,
) -> dict[str, object]:
    """Create the principal reproduction/validation figure."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.6), constrained_layout=True)
    ax_bao, ax_bbn, ax_resid, ax_early = axes.ravel()

    omega_m_bao = bao_chain.omega_m()
    hrd_bao = bao_chain.h_r_drag_mpc()
    omega_m_bao_name = getdist_parameter_name(bao_getdist, "omegam", "omega_m", "Omega_m")
    hrd_name = add_getdist_derived(bao_getdist, hrd_bao, "h_r_drag", r"h r_d")
    _draw_getdist_contours(
        ax_bao,
        bao_getdist,
        omega_m_bao_name,
        hrd_name,
        xlabel=r"$\Omega_m$",
        ylabel=r"$h\,r_d\;[\mathrm{Mpc}]$",
    )
    corr_bao = weighted_correlation(omega_m_bao, hrd_bao, bao_chain.weights)
    ax_bao.set_title("DESI DR2 BAO-only posterior")
    ax_bao.text(0.03, 0.03, rf"weighted $\rho={corr_bao:.2f}$", transform=ax_bao.transAxes, fontsize=9)

    omega_m_bbn = bbn_chain.omega_m()
    H0_bbn = bbn_chain.h0()
    omega_m_bbn_name = getdist_parameter_name(bbn_getdist, "omegam", "omega_m", "Omega_m")
    H0_name = getdist_parameter_name(bbn_getdist, "H0", "hubble")
    _draw_getdist_contours(
        ax_bbn,
        bbn_getdist,
        omega_m_bbn_name,
        H0_name,
        xlabel=r"$\Omega_m$",
        ylabel=r"$H_0\;[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$",
    )
    ax_bbn.set_title("DESI DR2 BAO + BBN posterior")

    whitened, pp_weights = _whitened_posterior_predictive(bbn_chain, likelihood)
    q025, q16, median, q84, q975 = _column_quantiles(whitened, pp_weights)
    x = np.arange(1, whitened.shape[1] + 1)
    ax_resid.fill_between(x, q025, q975, alpha=0.16, label="95% posterior interval")
    ax_resid.fill_between(x, q16, q84, alpha=0.30, label="68% posterior interval")
    ax_resid.plot(x, median, marker="o", linewidth=1.0, markersize=3.5, label="posterior median")
    ax_resid.axhline(0.0, linewidth=1.0)
    labels = [f"{row.quantity.split('_')[0]}\n{row.redshift:.3g}" for row in likelihood.rows]
    ax_resid.set_xticks(x, labels, fontsize=7)
    ax_resid.set_ylabel(r"Whitened residual $[L^{-1}(m-d)]_i$")
    ax_resid.set_xlabel("DESI observable and effective redshift")
    ax_resid.set_title("Full-covariance posterior predictive check")
    ax_resid.legend(fontsize=8, loc="upper right")
    ax_resid.grid(alpha=0.2)
    if map_result is not None:
        ax_resid.text(
            0.02,
            0.04,
            rf"Independent CAMB MAP: $\chi^2_{{\rm BAO}}/\nu="
            rf"{float(map_result['chi_square_bao']):.2f}/{int(map_result['bao_degrees_of_freedom'])}$, "
            rf"$p={float(map_result['bao_goodness_of_fit_p_value']):.3f}$",
            transform=ax_resid.transAxes,
            fontsize=8,
        )

    H0_summary = summarize(H0_bbn, bbn_chain.weights)
    om_summary = summarize(omega_m_bbn, bbn_chain.weights)
    ob_summary = summarize(bbn_chain.omega_b_h2(), bbn_chain.weights)
    central = CambLCDMParameters(
        H0=H0_summary.median,
        omega_m=om_summary.median,
        omega_b_h2=ob_summary.median,
    )
    z_early = np.geomspace(1.0e-4, 1.0e5, 700) - 1.0e-4
    history = background_history(central, z_early)
    zp1 = 1.0 + z_early
    ax_early.semilogx(zp1, history["omega_cb"], label=r"$\Omega_{cb}$")
    ax_early.semilogx(zp1, history["omega_gamma"], label=r"$\Omega_\gamma$")
    ax_early.semilogx(zp1, history["omega_nu_massless"], label=r"$\Omega_{\nu,\rm massless}$")
    ax_early.semilogx(zp1, history["omega_nu_massive"], label=r"$\Omega_{\nu,\rm massive}$")
    ax_early.semilogx(zp1, history["omega_lambda"], label=r"$\Omega_\Lambda$")
    marker_specs = (
        ("z_acc", r"$z_{\rm acc}$", 0.98, "right"),
        ("z_nu_thermal", r"$z_{\nu,\rm th}$", 0.88, "right"),
        ("z_drag", r"$z_d$", 0.77, "right"),
        ("z_star", r"$z_*$", 0.66, "left"),
        ("z_eq", r"$z_{\rm eq}$", 0.55, "right"),
    )
    for key, label, text_y, horizontal in marker_specs:
        value = float(history[key])
        if np.isfinite(value) and value >= 0:
            ax_early.axvline(1.0 + value, linewidth=0.8, linestyle="--", alpha=0.55)
            ax_early.text(
                1.0 + value,
                text_y,
                label,
                rotation=90,
                va="top",
                ha=horizontal,
                fontsize=7.5,
            )
    ax_early.set_ylim(-0.02, 1.03)
    ax_early.set_xlabel(r"$1+z$")
    ax_early.set_ylabel("Fraction of critical density")
    ax_early.set_title("CAMB species fractions and characteristic epochs")
    ax_early.legend(fontsize=7.5, ncol=2)
    ax_early.grid(alpha=0.2)

    fig.suptitle(
        r"Reproduction of DESI DR2 BAO + BBN flat-$\Lambda$CDM constraints (2025 Results I/II likelihood)",
        fontsize=14,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    median_whitened_chi2 = float(np.sum(median**2))
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
        "posterior_predictive": {
            "whitened_residual_median": median.tolist(),
            "median_vector_squared_norm": median_whitened_chi2,
            "draws_used": int(whitened.shape[0]),
        },
        "camb_central": {
            "H0": central.H0,
            "omega_m": central.omega_m,
            "omega_b_h2": central.omega_b_h2,
            "r_drag_Mpc": float(history["r_drag_Mpc"]),
            "z_acc": float(history["z_acc"]),
            "z_nu_thermal": float(history["z_nu_thermal"]),
            "z_drag": float(history["z_drag"]),
            "z_star": float(history["z_star"]),
            "z_eq": float(history["z_eq"]),
        },
    }


def posterior_kinematics(bbn_chain: ChainSet) -> dict[str, object]:
    """Propagate the released flat-LCDM posterior into low-z kinematic summaries."""
    om = bbn_chain.omega_m()
    weights = bbn_chain.weights
    q0 = 1.5 * om - 1.0
    w_tot0 = om - 1.0
    z_acc = np.cbrt(2.0 * (1.0 - om) / om) - 1.0
    return {
        "q0": summarize(q0, weights).__dict__,
        "w_tot0": summarize(w_tot0, weights).__dict__,
        "z_acc": summarize(z_acc, weights).__dict__,
        "relation": "late-time flat-LambdaCDM approximation; radiation is negligible at z~0",
    }


def make_kinematic_figure(bbn_chain: ChainSet, output: Path) -> dict[str, object]:
    """Generate a supplementary posterior distribution for the acceleration transition."""
    output.parent.mkdir(parents=True, exist_ok=True)
    om = bbn_chain.omega_m()
    z_acc = np.cbrt(2.0 * (1.0 - om) / om) - 1.0
    summary = posterior_kinematics(bbn_chain)
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    bins = np.linspace(*weighted_quantile(z_acc, bbn_chain.weights, (0.002, 0.998)), 70)
    ax.hist(z_acc, bins=bins, weights=bbn_chain.weights, density=True, histtype="step", linewidth=1.4)
    z_summary = summary["z_acc"]
    ax.axvline(float(z_summary["median"]), linewidth=1.0, label="posterior median")
    ax.axvspan(float(z_summary["q16"]), float(z_summary["q84"]), alpha=0.2, label="68% interval")
    ax.set_xlabel(r"Acceleration-transition redshift $z_{\rm acc}$")
    ax.set_ylabel("Posterior density")
    ax.set_title(r"Supplementary DESI DR2 + BBN flat-$\Lambda$CDM kinematics")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return summary


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


def make_aubourg_validation_figure(bbn_chain: ChainSet, output: Path, *, samples: int = 128) -> dict[str, float]:
    """Compare the legacy Aubourg approximation against CAMB on posterior-spanning draws."""
    output.parent.mkdir(parents=True, exist_ok=True)
    count = max(16, int(samples))
    indices = np.unique(np.linspace(0, bbn_chain.weights.size - 1, count, dtype=int))
    H0 = bbn_chain.h0()[indices]
    omega_m = bbn_chain.omega_m()[indices]
    omega_b = bbn_chain.omega_b_h2()[indices]
    approx = aubourg_r_drag_mpc(H0, omega_m, omega_b)
    exact = np.empty_like(approx)
    for i in range(exact.size):
        _, results = build_camb_background(
            CambLCDMParameters(H0=float(H0[i]), omega_m=float(omega_m[i]), omega_b_h2=float(omega_b[i]))
        )
        exact[i] = float(results.get_derived_params()["rdrag"])
    fractional = approx / exact - 1.0
    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    ax.scatter(exact, 100.0 * fractional, s=12, alpha=0.60)
    ax.axhline(0.0, linewidth=0.9)
    ax.set_xlabel(r"CAMB $r_d$ [Mpc]")
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
        "r_drag_camb_min_Mpc": float(np.min(exact)),
        "r_drag_camb_max_Mpc": float(np.max(exact)),
    }
