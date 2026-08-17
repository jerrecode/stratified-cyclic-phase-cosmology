from __future__ import annotations

import math

from scpc.models.phase import PeriodicPotential, SCPCParameters
from scpc.numerics.turning_feasibility import turning_feasibility_certificate


def test_canonical_baseline_has_strict_future_hubble_gap() -> None:
    parameters = SCPCParameters(
        spatial_curvature_k=1,
        rho_m_ref=0.03,
        rho_r_ref=0.0001,
        potential=PeriodicPotential(offset=3.5, amplitude=0.08, strata_count=4),
    )
    certificate = turning_feasibility_certificate(
        parameters,
        a0=1.0,
        phi0=0.15,
        phi_dot0=0.0,
        branch=1,
    )
    assert certificate.future_turnaround_excluded
    assert math.isclose(certificate.future_hubble_squared_lower_bound or 0.0, 1.0 / 6.0)
    assert math.isclose(
        certificate.future_abs_hubble_lower_bound or 0.0,
        math.sqrt(1.0 / 6.0),
    )
