from pathlib import Path

import numpy as np

from scpc.inference.desi_chains import load_cobaya_chains, weighted_correlation, weighted_quantile
from scpc.inference.desi_likelihood import load_covariance, load_mean
from scpc.inference.figures import aubourg_r_drag_mpc


def test_cobaya_chain_reader_and_derived_h_rd(tmp_path: Path) -> None:
    content = "# weight minuslogpost H0 omegam ombh2 rdrag\n1 2 68 0.30 0.022 147\n2 3 70 0.31 0.023 148\n"
    first = tmp_path / "chain.1.txt"
    second = tmp_path / "chain.2.txt"
    first.write_text(content, encoding="utf-8")
    second.write_text(content, encoding="utf-8")
    chain = load_cobaya_chains([first, second])
    assert chain.values.shape == (4, 6)
    assert np.allclose(chain.h_r_drag_mpc()[:2], [99.96, 103.6])
    assert np.isclose(weighted_quantile(chain.h0(), chain.weights, (0.5,))[0], 70.0)


def test_weighted_correlation_has_expected_sign() -> None:
    x = np.asarray([0.0, 1.0, 2.0])
    y = np.asarray([2.0, 1.0, 0.0])
    w = np.ones(3)
    assert np.isclose(weighted_correlation(x, y, w), -1.0)


def test_pinned_desi_files_have_expected_dimensions() -> None:
    rows = load_mean(Path("data/analysis_ready/desi_dr2_bao/mean.txt"))
    covariance = load_covariance(Path("data/analysis_ready/desi_dr2_bao/covariance.txt"), len(rows))
    assert len(rows) == 13
    assert covariance.shape == (13, 13)
    assert rows[0].redshift == 0.295
    assert rows[-1].quantity == "DM_over_rs"


def test_aubourg_crosscheck_returns_physical_drag_scale() -> None:
    rd = aubourg_r_drag_mpc(np.asarray([68.5]), np.asarray([0.297]), np.asarray([0.02218]))
    assert 140.0 < rd[0] < 155.0
