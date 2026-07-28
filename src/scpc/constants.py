"""Physical constants and explicit unit conversions used by SCPC."""

from __future__ import annotations

MPC_IN_METERS = 3.0856775814913673e22
KM_IN_METERS = 1_000.0
SECOND_IN_GYR = 3.15576e16


def hubble_km_s_mpc_to_si(value: float) -> float:
    """Convert km s^-1 Mpc^-1 to s^-1."""
    return value * KM_IN_METERS / MPC_IN_METERS


def hubble_si_to_km_s_mpc(value: float) -> float:
    """Convert s^-1 to km s^-1 Mpc^-1."""
    return value * MPC_IN_METERS / KM_IN_METERS
