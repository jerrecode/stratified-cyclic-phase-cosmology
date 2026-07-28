import numpy as np

from scpc.models.phase import PeriodicPotential, SCPCParameters, SCPCSolution
from scpc.numerics.convergence import (
    compare_solutions,
    run_cross_solver_comparison,
    run_tolerance_ladder,
)


def _de_sitter_parameters() -> SCPCParameters:
    return SCPCParameters(
        spatial_curvature_k=0,
        rho_m_ref=0.0,
        rho_r_ref=0.0,
        potential=PeriodicPotential(offset=3.0, amplitude=0.0, strata_count=1),
    )


def _integration_options() -> dict[str, object]:
    return {
        "t_span": (0.0, 1.0),
        "samples": 101,
        "a0": 1.0,
        "phi0": 0.0,
        "phi_dot0": 0.0,
        "branch": 1,
    }


def _constant_solution(*, field_offset: float = 0.0) -> SCPCSolution:
    potential = PeriodicPotential(
        offset=3.0,
        amplitude=0.0,
        strata_count=4,
        field_scale=1.0,
        target_space="circle",
    )
    parameters = SCPCParameters(
        spatial_curvature_k=0,
        rho_m_ref=0.0,
        rho_r_ref=0.0,
        potential=potential,
    )
    time = np.linspace(0.0, 1.0, 11)
    return SCPCSolution(
        t=time,
        a=np.exp(time),
        H=np.ones_like(time),
        phi=np.full_like(time, field_offset),
        phi_dot=np.zeros_like(time),
        rho_m=np.zeros_like(time),
        rho_r=np.zeros_like(time),
        rho_phi=np.full_like(time, 3.0),
        p_phi=np.full_like(time, -3.0),
        constraint_residual=np.zeros_like(time),
        turning_times=np.asarray([]),
        turning_kinds=(),
        parameters=parameters,
        solver_metadata={},
    )


def test_tolerance_ladder_uses_tightest_run_as_reference() -> None:
    results = run_tolerance_ladder(
        _de_sitter_parameters(),
        integration_options=_integration_options(),
        tolerances=(
            ("coarse", 1.0e-6, 1.0e-8),
            ("fine", 1.0e-10, 1.0e-12),
        ),
    )
    assert results[0].difference_to_reference is not None
    assert results[0].difference_to_reference.maximum < 1.0e-5
    assert results[1].difference_to_reference is None
    assert max(result.max_abs_constraint_residual for result in results) < 1.0e-9


def test_independent_solvers_agree_in_exact_de_sitter_limit() -> None:
    results = run_cross_solver_comparison(
        _de_sitter_parameters(),
        integration_options=_integration_options(),
        methods=("DOP853", "RK45"),
        reference_method="DOP853",
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    dop853, rk45 = results
    assert dop853.difference_to_reference is None
    assert rk45.difference_to_reference is not None
    assert rk45.difference_to_reference.maximum < 1.0e-8
    assert np.isfinite(rk45.max_abs_constraint_residual)


def test_solver_comparison_preserves_unwrapped_winding_history() -> None:
    reference = _constant_solution()
    circumference = reference.parameters.potential.target_circumference
    assert circumference is not None
    shifted = _constant_solution(field_offset=circumference)
    difference = compare_solutions(reference, shifted)
    assert np.isclose(difference.field, 4.0)
    assert difference.maximum >= difference.field
