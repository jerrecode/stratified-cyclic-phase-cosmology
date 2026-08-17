from __future__ import annotations

import math

from scpc.models.phase import PeriodicPotential, SCPCParameters
from scpc.numerics.turning_feasibility import turning_feasibility_certificate


def test_canonical_baseline_is_analytically_above_any_turning_scale() -> None:
    parameters = SCPCParameters(
        spatial_curvature_k=1,
        rho_m_ref=0.03,
        rho_r_ref=0.0001,
        a_ref=1.0,
        potential=PeriodicPotential(offset=3.5, amplitude=0.08, strata_count=4),
    )
    certificate = turning_feasibility_certificate(
        parameters,
        a0=1.0,
        phi0=0.15,
        phi_dot0=0.0,
        branch=1,
    )
    assert math.isclose(certificate.turning_scale_factor_upper_bound or 0.0, math.sqrt(3.0 / 3.5))
    assert certificate.initial_scale_factor > (certificate.turning_scale_factor_upper_bound or 0.0)
    assert certificate.future_turnaround_excluded
    assert certificate.exclusion_reason is not None


def test_open_geometry_with_nonnegative_energy_excludes_positive_density_turning() -> None:
    parameters = SCPCParameters(
        spatial_curvature_k=0,
        rho_m_ref=0.0,
        rho_r_ref=0.0,
        potential=PeriodicPotential(offset=3.0, amplitude=0.0),
    )
    certificate = turning_feasibility_certificate(
        parameters,
        a0=1.0,
        phi0=0.0,
        phi_dot0=0.0,
        branch=1,
    )
    assert certificate.future_turnaround_excluded
    assert certificate.turning_scale_factor_upper_bound is None


def test_negative_potential_floor_does_not_issue_positive_energy_certificate() -> None:
    parameters = SCPCParameters(
        spatial_curvature_k=1,
        rho_m_ref=0.0,
        rho_r_ref=0.0,
        potential=PeriodicPotential(offset=-0.2, amplitude=0.4),
    )
    certificate = turning_feasibility_certificate(
        parameters,
        a0=1.0,
        phi0=0.0,
        phi_dot0=2.0,
        branch=1,
    )
    assert not certificate.nonnegative_total_density_guaranteed
    assert certificate.turning_scale_factor_upper_bound is None
    assert not certificate.future_turnaround_excluded
