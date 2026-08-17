"""Pinned DESI DR2 Gaussian BAO likelihood and CAMB-calibrated flat-LCDM MAP check."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from scpc.inference.camb_background import CambLCDMParameters, camb_bao_observables


@dataclass(frozen=True)
class BAORow:
    redshift: float
    value: float
    quantity: str


def load_mean(path: Path) -> tuple[BAORow, ...]:
    rows: list[BAORow] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            z, value, quantity = stripped.split()
            rows.append(BAORow(float(z), float(value), quantity))
    if not rows:
        raise ValueError("DESI BAO mean file contains no rows")
    return tuple(rows)


def load_covariance(path: Path, n: int) -> np.ndarray:
    covariance = np.loadtxt(path, dtype=float)
    if covariance.shape != (n, n):
        raise ValueError(f"Expected {n}x{n} covariance, got {covariance.shape}")
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14):
        raise ValueError("DESI covariance is not symmetric")
    np.linalg.cholesky(covariance)
    return covariance


class DESIDR2BAOLikelihood:
    def __init__(self, mean_path: Path, covariance_path: Path):
        self.rows = load_mean(mean_path)
        self.data = np.asarray([row.value for row in self.rows], dtype=float)
        self.covariance = load_covariance(covariance_path, len(self.rows))
        self.cholesky = np.linalg.cholesky(self.covariance)
        self.unique_redshifts = np.unique([row.redshift for row in self.rows])

    def prediction(self, parameters: CambLCDMParameters) -> np.ndarray:
        observables = camb_bao_observables(parameters, self.unique_redshifts)
        index = {float(z): i for i, z in enumerate(self.unique_redshifts)}
        lookup = {
            "DM_over_rs": np.asarray(observables["D_M_over_rd"]),
            "DH_over_rs": np.asarray(observables["D_H_over_rd"]),
            "DV_over_rs": np.asarray(observables["D_V_over_rd"]),
        }
        result = []
        for row in self.rows:
            if row.quantity not in lookup:
                raise ValueError(f"Unsupported BAO observable {row.quantity!r}")
            result.append(float(lookup[row.quantity][index[row.redshift]]))
        return np.asarray(result)

    def whitened_residual(self, parameters: CambLCDMParameters) -> np.ndarray:
        residual = self.prediction(parameters) - self.data
        return np.linalg.solve(self.cholesky, residual)

    def chi_square(self, parameters: CambLCDMParameters) -> float:
        residual = self.whitened_residual(parameters)
        return float(residual @ residual)


def fit_flat_lcdm_bbn_map(
    likelihood: DESIDR2BAOLikelihood,
    *,
    initial: tuple[float, float, float] = (68.5, 0.298, 0.02218),
    bounds: tuple[tuple[float, float], ...] = ((45.0, 90.0), (0.1, 0.6), (0.015, 0.03)),
    bbn_mean: float = 0.02218,
    bbn_sigma: float = 0.00055,
    sum_mnu_eV: float = 0.06,
    N_eff: float = 3.044,
    T_cmb_K: float = 2.7255,
) -> dict[str, object]:
    """Reproduce the deterministic MAP check; posterior inference uses released DESI chains."""

    def objective(theta: np.ndarray) -> float:
        H0, omega_m, omega_b_h2 = map(float, theta)
        inside_bounds = (
            bounds[0][0] <= H0 <= bounds[0][1]
            and bounds[1][0] <= omega_m <= bounds[1][1]
            and bounds[2][0] <= omega_b_h2 <= bounds[2][1]
        )
        if not inside_bounds:
            return 1.0e30
        params = CambLCDMParameters(
            H0=H0,
            omega_m=omega_m,
            omega_b_h2=omega_b_h2,
            sum_mnu_eV=sum_mnu_eV,
            N_eff=N_eff,
            T_cmb_K=T_cmb_K,
        )
        try:
            chi_bao = likelihood.chi_square(params)
        except (ValueError, RuntimeError):
            return 1.0e30
        chi_prior = ((omega_b_h2 - bbn_mean) / bbn_sigma) ** 2
        return chi_bao + chi_prior

    result = minimize(
        objective,
        np.asarray(initial, dtype=float),
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": 1000, "xatol": 1.0e-7, "fatol": 1.0e-7},
    )
    if not result.success:
        result = minimize(objective, np.asarray(initial, dtype=float), method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise RuntimeError(f"DESI+BBN MAP optimization failed: {result.message}")
    H0, omega_m, omega_b_h2 = map(float, result.x)
    params = CambLCDMParameters(
        H0=H0,
        omega_m=omega_m,
        omega_b_h2=omega_b_h2,
        sum_mnu_eV=sum_mnu_eV,
        N_eff=N_eff,
        T_cmb_K=T_cmb_K,
    )
    chi_bao = likelihood.chi_square(params)
    chi_prior = ((omega_b_h2 - bbn_mean) / bbn_sigma) ** 2
    obs = camb_bao_observables(params, np.asarray([0.295]))
    r_drag = float(obs["r_drag_Mpc"])
    dof_bao = len(likelihood.rows) - 2
    return {
        "success": True,
        "parameters": {"H0": H0, "omega_m": omega_m, "omega_b_h2": omega_b_h2},
        "derived": {"r_drag_Mpc": r_drag, "h_r_drag_Mpc": H0 / 100.0 * r_drag},
        "chi_square_bao": chi_bao,
        "chi_square_bbn_prior": chi_prior,
        "chi_square_total": chi_bao + chi_prior,
        "bao_degrees_of_freedom": dof_bao,
        "bao_goodness_of_fit_p_value": float(chi2.sf(chi_bao, dof_bao)),
        "optimizer_message": str(result.message),
    }
