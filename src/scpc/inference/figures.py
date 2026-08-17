"""Publication-oriented DESI DR2 + BBN posterior and validation figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2 as chi2_distribution

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


def systematic_posterior_indices(weights: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic posterior-representative rows and their multiplicities.

    The positions are equally spaced in the normalized cumulative weight rather than in
    MCMC row number. This avoids treating compressed Cobaya rows as equally probable.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0 or np.any(~np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be a finite one-dimensional non-negative vector with positive sum")
    n = max(1, int(count))
    targets = (np.arange(n, dtype=float) + 0.5) * (w.sum() / n)
    selected = np.searchsorted(np.cumsum(w), targets, side="left")
    selected = np.minimum(selected, w.size - 1)
    indices, multiplicities = np.unique(selected, return_counts=True)
    return indices.astype(int), multiplicities.astype(float)


def _camb_posterior_residuals(
    chain: ChainSet,
    likelihood: DESIDR2BAOLikelihood,
    *,
    samples: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate exact CAMB BAO residuals on deterministic posterior-representative samples."""
    indices, multiplicities = systematic_posterior_indices(chain.weights, samples)
    predictions = np.empty((indices.size, len(likelihood.rows)), dtype=float)
    H0 = chain.h0()
    omega_m = chain.omega_m()
    omega_b = chain.omega_b_h2()
    for out_index, chain_index in enumerate(indices):
        parameters = CambLCDMParameters(
            H0=float(H0[chain_index]),
            omega_m=float(omega_m[chain_index]),
            omega_b_h2=float(omega_b[chain_index]),
        )
        predictions[out_index] = likelihood.prediction(parameters)
    residual = predictions - likelihood.data[None, :]
    whitened = np.linalg.solve(likelihood.cholesky, residual.T).T
    return whitened, multiplicities


def _column_quantiles(matrix: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, ...]:
    results = [
        weighted_quantile(matrix[:, j], weights, (0.025, 0.16, 0.5, 0.84, 0.975))
        for j in range(matrix.shape[1])
    ]
    return tuple(np.asarray(results).T)


def _background_posterior(
    chain: ChainSet,
    z: np.ndarray,
    *,
    samples: int = 48,
) -> dict[str, object]:
    """Propagate released posterior uncertainty through exact CAMB background histories."""
    indices, multiplicities = systematic_posterior_indices(chain.weights, samples)
    species_keys = (
        "omega_cb",
        "omega_gamma",
        "omega_nu_massless",
        "omega_nu_massive",
        "omega_lambda",
    )
    epoch_keys = ("z_acc", "z_nu_thermal", "z_drag", "z_star", "z_eq", "r_drag_Mpc")
    histories = {key: np.empty((indices.size, z.size), dtype=float) for key in species_keys}
    epochs = {key: np.empty(indices.size, dtype=float) for key in epoch_keys}
    H0 = chain.h0()
    omega_m = chain.omega_m()
    omega_b = chain.omega_b_h2()
    for out_index, chain_index in enumerate(indices):
        parameters = CambLCDMParameters(
            H0=float(H0[chain_index]),
            omega_m=float(omega_m[chain_index]),
            omega_b_h2=float(omega_b[chain_index]),
        )
        history = background_history(parameters, z)
        for key in species_keys:
            histories[key][out_index] = np.asarray(history[key], dtype=float)
        for key in epoch_keys:
            epochs[key][out_index] = float(history[key])

    species_summary: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, matrix in histories.items():
        q16 = np.empty(z.size, dtype=float)
        median = np.empty(z.size, dtype=float)
        q84 = np.empty(z.size, dtype=float)
        for column in range(z.size):
            q16[column], median[column], q84[column] = weighted_quantile(
                matrix[:, column], multiplicities, (0.16, 0.5, 0.84)
            )
        species_summary[key] = (q16, median, q84)

    epoch_summary = {key: summarize(values, multiplicities).__dict__ for key, values in epochs.items()}
    return {
        "indices": indices,
        "weights": multiplicities,
        "species": species_summary,
        "epochs": epoch_summary,
        "requested_systematic_positions": int(max(1, samples)),
        "unique_camb_evaluations": int(indices.size),
    }


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

    whitened, residual_weights = _camb_posterior_residuals(bbn_chain, likelihood)
    q025, q16, median, q84, q975 = _column_quantiles(whitened, residual_weights)
    x = np.arange(1, whitened.shape[1] + 1)
    ax_resid.axhspan(-1.96, 1.96, alpha=0.045, label=r"$N(0,1)$ 95% reference")
    ax_resid.fill_between(x, q025, q975, alpha=0.14, label="95% posterior model interval")
    ax_resid.fill_between(x, q16, q84, alpha=0.28, label="68% posterior model interval")
    ax_resid.plot(x, median, marker="o", linewidth=1.0, markersize=3.5, label="posterior median")
    ax_resid.axhline(0.0, linewidth=1.0)
    labels = [f"{i}\n{row.quantity.split('_')[0]} {row.redshift:.3g}" for i, row in enumerate(likelihood.rows, 1)]
    ax_resid.set_xticks(x, labels, fontsize=6.7)
    ax_resid.set_ylabel(r"Cholesky-whitened residual $[L^{-1}(m-d)]_i$")
    ax_resid.set_xlabel("Whitened mode i; second line identifies row i in the pinned ordering")
    ax_resid.set_title("Posterior-propagated full-covariance residual diagnostic")
    ax_resid.legend(fontsize=7.5, loc="upper right")
    ax_resid.grid(alpha=0.2)
    chi2_values = np.sum(whitened**2, axis=1)
    chi2_summary = summarize(chi2_values, residual_weights)
    posterior_predictive_tail = weighted_mean(
        chi2_distribution.sf(chi2_values, df=len(likelihood.rows)), residual_weights
    )
    annotation = (
        rf"CAMB posterior: $\chi^2_{{\rm BAO}}={chi2_summary.median:.2f}$ "
        rf"$[{chi2_summary.q16:.2f},{chi2_summary.q84:.2f}]$; "
        rf"$p_{{\rm PPC}}={posterior_predictive_tail:.3f}$"
    )
    if map_result is not None:
        annotation += (
            "\n"
            rf"Independent MAP: $\chi^2/\nu={float(map_result['chi_square_bao']):.2f}/"
            rf"{int(map_result['bao_degrees_of_freedom'])}$, "
            rf"$p_{{\rm gof}}={float(map_result['bao_goodness_of_fit_p_value']):.3f}$"
        )
    ax_resid.text(0.02, 0.035, annotation, transform=ax_resid.transAxes, fontsize=7.4, va="bottom")
    ax_resid.text(
        0.02,
        0.965,
        r"Mode $i$ mixes covariance rows $1\ldots i$; labels are ordering aids, not one-to-one observables.",
        transform=ax_resid.transAxes,
        fontsize=6.8,
        va="top",
    )

    H0_summary = summarize(H0_bbn, bbn_chain.weights)
    om_summary = summarize(omega_m_bbn, bbn_chain.weights)
    ob_summary = summarize(bbn_chain.omega_b_h2(), bbn_chain.weights)
    z_early = np.geomspace(1.0e-4, 1.0e5, 700) - 1.0e-4
    bg = _background_posterior(bbn_chain, z_early, samples=48)
    zp1 = 1.0 + z_early
    species_specs = (
        ("omega_cb", r"$\Omega_{cb}$"),
        ("omega_gamma", r"$\Omega_\gamma$"),
        ("omega_nu_massless", r"$\Omega_{\nu,\rm massless}$"),
        ("omega_nu_massive", r"$\Omega_{\nu,\rm massive}$"),
        ("omega_lambda", r"$\Omega_\Lambda$"),
    )
    for key, label in species_specs:
        low, center, high = bg["species"][key]
        (line,) = ax_early.semilogx(zp1, center, label=label)
        ax_early.fill_between(zp1, low, high, color=line.get_color(), alpha=0.08, linewidth=0)

    marker_specs = (
        ("z_acc", r"$z_{\rm acc}$", 0.98, "right"),
        ("z_nu_thermal", r"$z_{\nu,\rm th}$", 0.88, "right"),
        ("z_drag", r"$z_d$", 0.77, "right"),
        ("z_star", r"$z_*$", 0.66, "left"),
        ("z_eq", r"$z_{\rm eq}$", 0.55, "right"),
    )
    for key, label, text_y, horizontal in marker_specs:
        summary = bg["epochs"][key]
        value = float(summary["median"])
        if np.isfinite(value) and value >= 0:
            low = float(summary["q16"])
            high = float(summary["q84"])
            if np.isfinite(low) and np.isfinite(high) and high > low:
                ax_early.axvspan(1.0 + low, 1.0 + high, alpha=0.055, linewidth=0)
            ax_early.axvline(1.0 + value, linewidth=0.8, linestyle="--", alpha=0.60)
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
    ax_early.set_title("CAMB background extrapolation with 68% posterior bands")
    ax_early.legend(fontsize=7.5, ncol=2)
    ax_early.grid(alpha=0.2)

    fig.suptitle(
        r"Reproduction of DESI DR2 BAO + BBN flat-$\Lambda$CDM constraints (2025 Results I/II likelihood)",
        fontsize=14,
    )
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
        "posterior_residual_diagnostic": {
            "method": "exact CAMB predictions on deterministic systematic posterior-weight positions; Cholesky whitening",
            "requested_systematic_positions": 512,
            "unique_camb_evaluations": int(whitened.shape[0]),
            "whitened_residual_median": median.tolist(),
            "chi_square": chi2_summary.__dict__,
            "bayesian_posterior_predictive_tail_probability": float(posterior_predictive_tail),
            "note": "The component labels identify the pinned row ordering; Cholesky mode i mixes rows 1..i.",
        },
        "camb_background_posterior": {
            "method": "exact CAMB histories on deterministic systematic posterior-weight positions",
            "requested_systematic_positions": int(bg["requested_systematic_positions"]),
            "unique_camb_evaluations": int(bg["unique_camb_evaluations"]),
            "characteristic_epochs": bg["epochs"],
            "epistemic_status": "model-derived standard-LambdaCDM extrapolation, not direct high-redshift DESI measurement",
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
    """Compare the legacy Aubourg approximation against CAMB on weighted posterior representatives."""
    output.parent.mkdir(parents=True, exist_ok=True)
    count = max(16, int(samples))
    indices, multiplicities = systematic_posterior_indices(bbn_chain.weights, count)
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
    mean_fractional = weighted_mean(fractional, multiplicities)
    rms_fractional = float(np.sqrt(np.average(fractional**2, weights=multiplicities)))
    return {
        "requested_systematic_positions": int(count),
        "samples": int(exact.size),
        "mean_fractional_difference": float(mean_fractional),
        "rms_fractional_difference": rms_fractional,
        "max_abs_fractional_difference": float(np.max(np.abs(fractional))),
        "r_drag_camb_min_Mpc": float(np.min(exact)),
        "r_drag_camb_max_Mpc": float(np.max(exact)),
    }
