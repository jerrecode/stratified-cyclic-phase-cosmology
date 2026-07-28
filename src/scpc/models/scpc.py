"""Construction of the canonical closed-FLRW SCPC baseline from configuration."""

from __future__ import annotations

from typing import Any

from scpc.theory.background import BackgroundParameters, BackgroundState, consistent_hubble
from scpc.theory.potentials import PeriodicStratifiedPotential


def build_scpc_background(
    config: dict[str, Any],
) -> tuple[BackgroundParameters, PeriodicStratifiedPotential, BackgroundState]:
    model = config["model"]
    potential_config = model["potential"]
    initial = config["initial_state"]

    parameters = BackgroundParameters(
        reduced_planck_mass=float(model.get("reduced_planck_mass", 1.0)),
        curvature_k=int(model.get("curvature_k", 1)),
    )
    potential = PeriodicStratifiedPotential(
        amplitude=float(potential_config["amplitude"]),
        offset=float(potential_config.get("offset", 0.0)),
        periodicity=int(potential_config.get("periodicity", 1)),
        field_scale=float(potential_config.get("field_scale", 1.0)),
        tilt=float(potential_config.get("tilt", 0.0)),
    )

    provisional = BackgroundState(
        scale_factor=float(initial["scale_factor"]),
        hubble=0.0,
        phi=float(initial["phi"]),
        phi_dot=float(initial["phi_dot"]),
        rho_matter=float(initial.get("rho_matter", 0.0)),
        rho_radiation=float(initial.get("rho_radiation", 0.0)),
    )
    if "hubble" in initial:
        hubble = float(initial["hubble"])
    else:
        hubble = consistent_hubble(
            provisional,
            parameters,
            potential,
            branch=str(initial.get("hubble_branch", "expanding")),
        )
    state = BackgroundState(
        scale_factor=provisional.scale_factor,
        hubble=hubble,
        phi=provisional.phi,
        phi_dot=provisional.phi_dot,
        rho_matter=provisional.rho_matter,
        rho_radiation=provisional.rho_radiation,
    )
    return parameters, potential, state
