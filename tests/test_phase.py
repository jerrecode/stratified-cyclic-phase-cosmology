import numpy as np

from scpc.models.phase import PeriodicPotential, SCPCParameters, integrate_scpc


def test_periodic_potential_derivative() -> None:
    potential = PeriodicPotential(offset=1.0, amplitude=0.2, strata_count=3, field_scale=1.7)
    phi = 0.31
    step = 1e-6
    numerical = (potential.value(phi + step) - potential.value(phi - step)) / (2 * step)
    assert np.isclose(potential.derivative(phi), numerical, rtol=1e-6)


def test_exact_flat_de_sitter_limit() -> None:
    parameters = SCPCParameters(
        spatial_curvature_k=0,
        rho_m_ref=0.0,
        rho_r_ref=0.0,
        potential=PeriodicPotential(offset=3.0, amplitude=0.0, strata_count=1),
    )
    solution = integrate_scpc(
        parameters,
        t_span=(0.0, 1.0),
        samples=101,
        a0=1.0,
        phi0=0.0,
        phi_dot0=0.0,
        rtol=1e-11,
        atol=1e-13,
    )
    assert np.allclose(solution.H, 1.0, rtol=1e-9, atol=1e-10)
    assert np.allclose(solution.a, np.exp(solution.t), rtol=2e-9, atol=1e-10)
    assert np.max(np.abs(solution.constraint_residual)) < 1e-9
