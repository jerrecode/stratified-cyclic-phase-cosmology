"""Analytic necessary conditions for homogeneous Hubble turning points.

These certificates are conservative preflight results derived directly from the
Friedmann constraint.  They can rule out a future turnaround on a declared
branch without integrating farther in time, but they never prove recurrence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from scpc.models.phase import SCPCParameters, initial_hubble


@dataclass(frozen=True)
class TurningFeasibilityCertificate:
    """Necessary-condition audit for regular homogeneous turning points."""

    spatial_curvature_k: int
    potential_minimum: float
    nonnegative_total_density_guaranteed: bool
    positive_density_turning_requires_closed_geometry: bool
    turning_scale_factor_upper_bound: float | None
    initial_scale_factor: float
    initial_hubble: float
    initial_branch: int
    future_turnaround_excluded: bool
    exclusion_reason: str | None
    scientific_scope: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def turning_feasibility_certificate(
    parameters: SCPCParameters,
    *,
    a0: float,
    phi0: float,
    phi_dot0: float,
    branch: int,
) -> TurningFeasibilityCertificate:
    """Derive conservative turning-point bounds from the declared model.

    For the implemented potential with nonnegative amplitude,
    ``V(phi) >= offset = V_min``.  With nonnegative dust/radiation and canonical
    kinetic energy, ``rho_total >= V_min``.  At a regular H=0 event in reduced
    Planck units,

        rho_total = 3 k / a^2.

    Hence positive-density turning requires ``k=+1``.  If additionally
    ``V_min>0``, every turning point satisfies ``a <= sqrt(3/V_min)``.  An
    expanding branch that already starts at or above this bound cannot encounter
    a future turnaround because ``a`` increases monotonically until such an
    event would occur.
    """

    potential_minimum = float(parameters.potential.offset)
    nonnegative_total = bool(
        potential_minimum >= 0.0
        and parameters.rho_m_ref >= 0.0
        and parameters.rho_r_ref >= 0.0
    )
    closed_required = bool(nonnegative_total)
    upper_bound: float | None = None
    if parameters.spatial_curvature_k == 1 and potential_minimum > 0.0:
        upper_bound = float(np.sqrt(3.0 / potential_minimum))

    hubble0 = initial_hubble(a0, phi0, phi_dot0, parameters, branch)
    excluded = False
    reason: str | None = None
    if hubble0 > 0.0 and nonnegative_total and parameters.spatial_curvature_k <= 0:
        excluded = True
        reason = "positive-density regular H=0 turning is impossible for k<=0"
    elif hubble0 > 0.0 and upper_bound is not None and float(a0) >= upper_bound:
        excluded = True
        reason = (
            "expanding branch starts at or above the Friedmann upper bound "
            "a_turn<=sqrt(3/V_min), so no future turnaround can occur"
        )

    return TurningFeasibilityCertificate(
        spatial_curvature_k=int(parameters.spatial_curvature_k),
        potential_minimum=potential_minimum,
        nonnegative_total_density_guaranteed=nonnegative_total,
        positive_density_turning_requires_closed_geometry=closed_required,
        turning_scale_factor_upper_bound=upper_bound,
        initial_scale_factor=float(a0),
        initial_hubble=float(hubble0),
        initial_branch=int(branch),
        future_turnaround_excluded=excluded,
        exclusion_reason=reason,
        scientific_scope=(
            "necessary-condition certificate from the homogeneous Friedmann constraint; "
            "not a recurrence or stability proof"
        ),
    )
