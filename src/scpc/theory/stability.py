"""Stability result containers and baseline canonical-field checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StabilityReport:
    ghost_free_scalar: bool
    gradient_stable_scalar: bool
    ghost_free_tensor: bool
    gradient_stable_tensor: bool
    notes: tuple[str, ...]


def canonical_gr_scalar_report() -> StabilityReport:
    """Return formal local kinetic-sign checks for GR plus a canonical scalar.

    This does not replace a perturbation evolution calculation through a bounce.
    """
    return StabilityReport(
        ghost_free_scalar=True,
        gradient_stable_scalar=True,
        ghost_free_tensor=True,
        gradient_stable_tensor=True,
        notes=(
            "Canonical kinetic terms have positive local kinetic coefficients.",
            "Global perturbative regularity and strong coupling remain to be tested.",
        ),
    )
