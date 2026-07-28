import pytest

from scpc.scans.grid import expand_parameter_grid


def _base() -> dict[str, object]:
    return {
        "model": {
            "potential": {"amplitude": 0.1, "offset": 3.0},
            "background": {"spatial_curvature_k": 1},
        },
        "initial_conditions": {"phi": 0.0, "a": 1.0},
        "run": {"method": "DOP853", "samples": 101},
    }


def test_grid_order_is_lexical_and_reproducible() -> None:
    axes = {
        "initial_conditions.phi": [0.0, 0.2],
        "model.potential.amplitude": [0.05, 0.1],
    }
    first = expand_parameter_grid(_base(), axes)
    second = expand_parameter_grid(_base(), dict(reversed(list(axes.items()))))

    assert [point.identity.sha256 for point in first] == [point.identity.sha256 for point in second]
    assert [point.coordinates for point in first] == [
        {
            "initial_conditions.phi": 0.0,
            "model.potential.amplitude": 0.05,
        },
        {
            "initial_conditions.phi": 0.0,
            "model.potential.amplitude": 0.1,
        },
        {
            "initial_conditions.phi": 0.2,
            "model.potential.amplitude": 0.05,
        },
        {
            "initial_conditions.phi": 0.2,
            "model.potential.amplitude": 0.1,
        },
    ]


def test_base_configuration_is_not_mutated() -> None:
    base = _base()
    points = expand_parameter_grid(base, {"initial_conditions.phi": [0.2]})
    assert base["initial_conditions"] == {"phi": 0.0, "a": 1.0}
    assert points[0].specification["initial_conditions"]["phi"] == 0.2  # type: ignore[index]


def test_empty_axes_yield_one_identity_for_base_specification() -> None:
    points = expand_parameter_grid(_base(), {})
    assert len(points) == 1
    assert points[0].coordinates == {}


def test_missing_or_nonmapping_path_is_rejected() -> None:
    with pytest.raises(KeyError, match="does not exist"):
        expand_parameter_grid(_base(), {"model.potential.missing": [1.0]})
    with pytest.raises(KeyError, match="does not resolve"):
        expand_parameter_grid(_base(), {"run.method.name": ["invalid"]})


def test_empty_axis_and_invalid_dotted_path_are_rejected() -> None:
    with pytest.raises(ValueError, match="nonempty list"):
        expand_parameter_grid(_base(), {"initial_conditions.phi": []})
    with pytest.raises(ValueError, match="Invalid dotted"):
        expand_parameter_grid(_base(), {"initial_conditions..phi": [0.1]})


def test_duplicate_complete_specifications_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate complete specifications"):
        expand_parameter_grid(_base(), {"initial_conditions.phi": [0.1, 0.1]})


def test_scan_size_limit_is_checked_before_execution() -> None:
    with pytest.raises(ValueError, match="exceeding max_runs"):
        expand_parameter_grid(
            _base(),
            {
                "initial_conditions.phi": [0.0, 0.1, 0.2],
                "model.potential.amplitude": [0.01, 0.02],
            },
            max_runs=5,
        )
