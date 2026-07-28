from __future__ import annotations

import numpy as np

from scpc.models.lcdm import LambdaCDM
from scpc.models.w0wa import W0WaCDM


def test_lcdm_is_normalized_at_present() -> None:
    model = LambdaCDM(omega_matter=0.315, omega_radiation=9.2e-5)
    assert np.isclose(model.dimensionless_hubble(np.asarray([1.0]))[0], 1.0)


def test_w0wa_reduces_to_lcdm_for_minus_one_zero() -> None:
    scale_factor = np.geomspace(0.05, 1.5, 100)
    lcdm = LambdaCDM(omega_matter=0.3, omega_radiation=1.0e-4)
    cpl = W0WaCDM(
        omega_matter=0.3,
        omega_radiation=1.0e-4,
        w0=-1.0,
        wa=0.0,
    )
    assert np.allclose(lcdm.dimensionless_hubble(scale_factor), cpl.dimensionless_hubble(scale_factor))
