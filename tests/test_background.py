from __future__ import annotations

from scpc.config import load_yaml
from scpc.models.scpc import build_scpc_background
from scpc.numerics.integrate import IntegrationSettings, integrate_background
from scpc.theory.background import relative_friedmann_residual


def test_initial_state_satisfies_friedmann_constraint() -> None:
    config = load_yaml("configs/baseline/scpc_closed.yaml")
    parameters, potential, state = build_scpc_background(config)
    assert abs(relative_friedmann_residual(state, parameters, potential)) < 1.0e-12


def test_short_integration_preserves_constraint() -> None:
    config = load_yaml("configs/baseline/scpc_closed.yaml")
    parameters, potential, state = build_scpc_background(config)
    settings = IntegrationSettings(
        time_start=0.0,
        time_end=0.5,
        samples=100,
        relative_tolerance=1.0e-11,
        absolute_tolerance=1.0e-13,
        max_step=0.01,
    )
    solution = integrate_background(state, parameters, potential, settings)
    assert solution.constraint.maximum_absolute_relative_residual < 1.0e-8
