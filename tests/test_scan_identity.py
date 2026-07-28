import json

import numpy as np
import pytest

from scpc.scans.identity import canonical_run_identity, normalize_background_specification


def test_mapping_order_does_not_change_identity() -> None:
    first = canonical_run_identity(
        {
            "model": {"offset": 3.0, "amplitude": 0.1},
            "initial": {"a": 1.0, "phi": 0.2},
        }
    )
    second = canonical_run_identity(
        {
            "initial": {"phi": 0.2, "a": 1.0},
            "model": {"amplitude": 0.1, "offset": 3.0},
        }
    )
    assert first == second
    assert first.run_id.startswith("scpc-")
    assert len(first.sha256) == 64


def test_numerically_distinct_floats_do_not_collapse() -> None:
    first = canonical_run_identity({"phi": 0.1})
    second = canonical_run_identity({"phi": np.nextafter(0.1, 1.0)})
    assert first.sha256 != second.sha256


def test_execution_equivalent_integer_representations_share_identity() -> None:
    integer = canonical_run_identity({"run": {"samples": 101}})
    floating = canonical_run_identity({"run": {"samples": 101.0}})
    assert integer == floating


def test_nonintegral_integer_execution_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite integer"):
        normalize_background_specification({"run": {"samples": 101.5}})


def test_solver_settings_are_part_of_experiment_identity() -> None:
    dop853 = canonical_run_identity({"run": {"method": "DOP853", "rtol": 1.0e-9}})
    rk45 = canonical_run_identity({"run": {"method": "RK45", "rtol": 1.0e-9}})
    assert dop853.run_id != rk45.run_id


def test_canonical_json_is_machine_readable_and_namespaced() -> None:
    identity = canonical_run_identity({"samples": np.int64(101), "span": (0.0, 1.0)})
    payload = json.loads(identity.canonical_json)
    assert payload["namespace"] == "scpc-background-v1"
    assert payload["specification"]["samples"] == 101
    assert payload["specification"]["span"][0] == {"__float_hex__": "0x0.0p+0"}


def test_nonfinite_floats_are_rejected() -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        canonical_run_identity({"value": float("nan")})


def test_short_or_overlong_prefix_is_rejected() -> None:
    with pytest.raises(ValueError, match="prefix_length"):
        canonical_run_identity({}, prefix_length=8)
    with pytest.raises(ValueError, match="prefix_length"):
        canonical_run_identity({}, prefix_length=65)


def test_non_string_mapping_keys_are_rejected() -> None:
    with pytest.raises(TypeError, match="string keys"):
        canonical_run_identity({1: "invalid"})  # type: ignore[dict-item]
