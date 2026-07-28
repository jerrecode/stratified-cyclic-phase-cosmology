import numpy as np

from scpc.models.phase import PeriodicPotential, SCPCParameters
from scpc.numerics.convergence import (
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
