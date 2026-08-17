from __future__ import annotations

import math

import pytest

from scpc.models.phase import PeriodicPotential, SCPCParameters, initial_hubble, integrate_scpc


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("offset", math.nan),
        ("amplitude", math.inf),
        ("field_scale", math.nan),
    ],
)
def test_periodic_potential_rejects_nonfinite_parameters(keyword: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        PeriodicPotential(**{keyword: value})


def test_scpc_parameters_reject_nonfinite_reference_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        SCPCParameters(rho_m_ref=math.nan)
    with pytest.raises(ValueError, match="integer"):
        SCPCParameters(spatial_curvature_k=1.0)  # type: ignore[arg-type]


def test_initial_hubble_rejects_nonfinite_state_and_noninteger_branch() -> None:
    parameters = SCPCParameters()
    with pytest.raises(ValueError, match="finite"):
        initial_hubble(1.0, math.nan, 0.0, parameters)
    with pytest.raises(ValueError, match="integer"):
        initial_hubble(1.0, 0.0, 0.0, parameters, branch=1.0)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"t_span": (0.0, math.inf)},
        {"t_span": (1.0, 0.0)},
        {"samples": 10.5},
        {"samples": True},
        {"domain_check_substeps": 4.5},
        {"rtol": math.nan},
        {"atol": 0.0},
    ],
)
def test_integrator_rejects_invalid_controls_before_solver(kwargs: dict[str, object]) -> None:
    parameters = SCPCParameters()
    call = {
        "parameters": parameters,
        "t_span": (0.0, 0.1),
        "a0": 1.0,
        "phi0": 0.0,
        "phi_dot0": 0.0,
        "samples": 11,
    }
    call.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        integrate_scpc(**call)  # type: ignore[arg-type]
