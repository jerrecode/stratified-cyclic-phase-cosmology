import numpy as np

from scpc.constants import C_KM_S
from scpc.models.standard import ExpansionParameters, FLRWExpansion


def test_sparse_positive_redshift_grid_is_anchored_at_zero() -> None:
    model = FLRWExpansion(
        ExpansionParameters(H0=70.0, omega_m=1.0, omega_r=0.0, omega_k=0.0, omega_de=0.0)
    )
    z = np.asarray([0.295, 0.706, 2.33])
    table = model.distance_table(z)
    exact = 2.0 * C_KM_S / 70.0 * (1.0 - 1.0 / np.sqrt(1.0 + z))
    assert np.allclose(table["D_M_Mpc"], exact, rtol=2e-10, atol=1e-9)
    assert table["D_M_Mpc"][0] > 0.0
