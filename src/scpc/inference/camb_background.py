"""CAMB-backed background quantities for publication-level standard-cosmology checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class CambLCDMParameters:
    H0: float
    omega_m: float
    omega_b_h2: float
    sum_mnu_eV: float = 0.06
    N_eff: float = 3.044
    T_cmb_K: float = 2.7255

    def __post_init__(self) -> None:
        if self.H0 <= 0 or not 0 < self.omega_m < 1 or self.omega_b_h2 <= 0:
            raise ValueError("Invalid flat-LCDM background parameters")
        if self.sum_mnu_eV < 0 or self.N_eff <= 0 or self.T_cmb_K <= 0:
            raise ValueError("Invalid early-Universe parameters")


def _import_camb():
    try:
        import camb  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("CAMB is required for this calculation; install the 'inference' extra") from exc
    return camb


def build_camb_background(parameters: CambLCDMParameters):
    """Return CAMB results while matching the requested present-day Omega_m.

    CAMB computes the massive-neutrino density from the mass hierarchy and temperature.
    We therefore iterate once on omch2 rather than permanently assigning a 0.06 eV
    neutrino to an a^-3 matter term at all redshifts.
    """
    camb = _import_camb()
    h = parameters.H0 / 100.0
    omega_nu_seed = parameters.sum_mnu_eV / 93.14
    omch2 = parameters.omega_m * h**2 - parameters.omega_b_h2 - omega_nu_seed
    if omch2 <= 0:
        raise ValueError("Requested Omega_m leaves no positive CDM density")

    def run(omega_c_h2: float):
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=parameters.H0,
            ombh2=parameters.omega_b_h2,
            omch2=omega_c_h2,
            mnu=parameters.sum_mnu_eV,
            nnu=parameters.N_eff,
            TCMB=parameters.T_cmb_K,
            tau=0.054,
        )
        pars.set_dark_energy(w=-1.0, wa=0.0, dark_energy_model="fluid")
        return pars, camb.get_background(pars)

    pars, results = run(omch2)
    omega_nu_exact_h2 = float(results.get_Omega("nu", z=0.0)) * h**2
    corrected_omch2 = parameters.omega_m * h**2 - parameters.omega_b_h2 - omega_nu_exact_h2
    if corrected_omch2 <= 0:
        raise ValueError("CAMB massive-neutrino density leaves no positive CDM density")
    pars, results = run(corrected_omch2)
    return pars, results


def _derived_value(derived: dict[str, float], *aliases: str) -> float:
    canonical = {
        "".join(ch for ch in key.lower() if ch.isalnum()): value
        for key, value in derived.items()
    }
    for alias in aliases:
        key = "".join(ch for ch in alias.lower() if ch.isalnum())
        if key in canonical:
            return float(canonical[key])
    raise KeyError(f"None of {aliases!r} found in CAMB derived parameters {tuple(derived)}")


def background_history(parameters: CambLCDMParameters, z: np.ndarray) -> dict[str, np.ndarray | float]:
    z_arr = np.asarray(z, dtype=float)
    if z_arr.ndim != 1 or np.any(z_arr < 0):
        raise ValueError("z must be a one-dimensional non-negative array")
    _, results = build_camb_background(parameters)
    H = np.asarray(results.hubble_parameter(z_arr), dtype=float)
    omega_b = np.asarray(results.get_Omega("baryon", z_arr), dtype=float)
    omega_c = np.asarray(results.get_Omega("cdm", z_arr), dtype=float)
    omega_gamma = np.asarray(results.get_Omega("photon", z_arr), dtype=float)
    omega_massless_nu = np.asarray(results.get_Omega("neutrino", z_arr), dtype=float)
    omega_massive_nu = np.asarray(results.get_Omega("nu", z_arr), dtype=float)
    omega_lambda = np.asarray(results.get_Omega("de", z_arr), dtype=float)
    derived = results.get_derived_params()

    def q_of_z(redshift: float) -> float:
        dz = max(1.0e-5, 2.0e-4 * (1.0 + redshift))
        lo = max(0.0, redshift - dz)
        hi = redshift + dz
        h_lo = float(results.hubble_parameter(lo))
        h_hi = float(results.hubble_parameter(hi))
        derivative = (h_hi - h_lo) / (hi - lo)
        h_here = float(results.hubble_parameter(redshift))
        return (1.0 + redshift) * derivative / h_here - 1.0

    try:
        z_acc = float(brentq(q_of_z, 0.0, 3.0))
    except ValueError:
        z_acc = float("nan")

    return {
        "redshift": z_arr,
        "E": H / parameters.H0,
        "H_km_s_Mpc": H,
        "omega_cb": omega_b + omega_c,
        "omega_gamma": omega_gamma,
        "omega_nu": omega_massless_nu + omega_massive_nu,
        "omega_lambda": omega_lambda,
        "z_acc": z_acc,
        "z_drag": _derived_value(derived, "zdrag"),
        "z_star": _derived_value(derived, "zstar"),
        "z_eq": _derived_value(derived, "zeq"),
        "r_drag_Mpc": _derived_value(derived, "rdrag"),
    }


def camb_bao_observables(parameters: CambLCDMParameters, redshift: np.ndarray) -> dict[str, np.ndarray | float]:
    """Return D_M/r_d, D_H/r_d and D_V/r_d from the same CAMB background."""
    z = np.asarray(redshift, dtype=float)
    _, results = build_camb_background(parameters)
    derived = results.get_derived_params()
    r_drag = _derived_value(derived, "rdrag")
    d_m = np.asarray(results.comoving_radial_distance(z), dtype=float)
    H = np.asarray(results.hubble_parameter(z), dtype=float)
    c_km_s = 299792.458
    d_h = c_km_s / H
    d_v = np.cbrt(z * d_h * d_m**2)
    return {
        "D_M_over_rd": d_m / r_drag,
        "D_H_over_rd": d_h / r_drag,
        "D_V_over_rd": d_v / r_drag,
        "r_drag_Mpc": r_drag,
    }
