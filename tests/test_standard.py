import numpy as np

from scpc.models.standard import ExpansionParameters, FLRWExpansion


def test_flat_model_is_normalized_today() -> None:
    model = FLRWExpansion(ExpansionParameters())
    assert np.isclose(model.e_of_z(0.0), 1.0)


def test_einstein_de_sitter_limit() -> None:
    p = ExpansionParameters(H0=70.0, omega_m=1.0, omega_r=0.0, omega_k=0.0, omega_de=0.0)
    model = FLRWExpansion(p)
    z = np.asarray([0.0, 0.5, 2.0])
    assert np.allclose(model.e_of_z(z), (1.0 + z) ** 1.5)


def test_luminosity_distance_is_monotonic() -> None:
    z = np.linspace(0.0, 3.0, 301)
    table = FLRWExpansion(ExpansionParameters()).distance_table(z)
    assert np.all(np.diff(table["D_L_Mpc"]) > 0)
