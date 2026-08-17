import numpy as np

from scpc.models.layered import LayeredScalarParameters, S4RadialGrid, integrate_layered_pulse


def test_s4_radial_laplacian_annihilates_constant_field() -> None:
    grid = S4RadialGrid(cells=32)
    result = grid.laplacian(np.ones(grid.cells))
    assert np.allclose(result, 0.0, atol=1.0e-14)
    assert np.isclose(np.sum(grid.cell_weights), 4.0 / 3.0)


def test_undamped_layered_pulse_has_small_energy_drift() -> None:
    result = integrate_layered_pulse(
        grid=S4RadialGrid(cells=32),
        parameters=LayeredScalarParameters(mass=0.2, coupling=0.02, damping=0.0),
        t_end=1.0,
        samples=41,
        rtol=1.0e-9,
        atol=1.0e-11,
    )
    assert float(result["max_abs_relative_energy_drift"]) < 2.0e-6
    assert int(result["nfev"]) > 0
