from __future__ import annotations

import numpy as np

from scpc.models.phase import PeriodicPotential, SCPCParameters, SCPCSolution
from scpc.numerics.turning_audit import audit_turning_points


def _solution(*, kind: str, potential_offset: float, velocity: float) -> SCPCSolution:
    parameters = SCPCParameters(
        spatial_curvature_k=1,
        rho_m_ref=0.0,
        rho_r_ref=0.0,
        potential=PeriodicPotential(offset=potential_offset, amplitude=0.0),
    )
    time = np.asarray([0.0, 0.5, 1.0])
    return SCPCSolution(
        t=time,
        a=np.ones_like(time),
        H=np.asarray([0.1, 0.0, -0.1]) if kind == "turnaround" else np.asarray([-0.1, 0.0, 0.1]),
        phi=np.zeros_like(time),
        phi_dot=np.full_like(time, velocity),
        rho_m=np.zeros_like(time),
        rho_r=np.zeros_like(time),
        rho_phi=np.full_like(time, 0.5 * velocity**2 + potential_offset),
        p_phi=np.full_like(time, 0.5 * velocity**2 - potential_offset),
        constraint_residual=np.zeros_like(time),
        turning_times=np.asarray([0.5]),
        turning_kinds=(kind,),
        parameters=parameters,
        solver_metadata={},
        turning_state_vectors=np.asarray([[1.0, 0.0, 0.0, velocity]]),
        requested_end_time=1.0,
    )


def test_bounce_identity_is_recomputed_from_exact_state() -> None:
    audit = audit_turning_points(_solution(kind="bounce", potential_offset=3.0, velocity=0.0))[0]
    assert audit.theorem_kind == "bounce"
    assert audit.kind_consistent
    assert audit.friedmann_surface_residual == 0.0
    assert audit.hdot_raychaudhuri == 1.0
    assert audit.hdot_turning_identity == 1.0
    assert audit.bounce_margin == 3.0
    assert audit.nec_density == 0.0
    assert audit.sec_density == -6.0


def test_turnaround_identity_is_recomputed_from_exact_state() -> None:
    audit = audit_turning_points(_solution(kind="turnaround", potential_offset=1.0, velocity=2.0))[0]
    assert audit.theorem_kind == "turnaround"
    assert audit.kind_consistent
    assert audit.friedmann_surface_residual == 0.0
    assert audit.hdot_raychaudhuri == -1.0
    assert audit.hdot_turning_identity == -1.0
    assert audit.bounce_margin == -3.0
    assert audit.nec_density == 4.0
    assert audit.sec_density == 6.0


def test_wrong_recorded_kind_is_exposed_not_silently_relabelled() -> None:
    solution = _solution(kind="bounce", potential_offset=1.0, velocity=2.0)
    audit = audit_turning_points(solution)[0]
    assert audit.recorded_kind == "bounce"
    assert audit.theorem_kind == "turnaround"
    assert not audit.kind_consistent
