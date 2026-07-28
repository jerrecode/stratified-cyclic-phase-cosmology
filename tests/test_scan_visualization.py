import csv
import json

import pytest

from scpc.visualization.scans import plot_scan_outcome_map


FIELDNAMES = ["coordinates", "status", "outcome", "failure_class"]


def _write_index(path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "coordinates": json.dumps(row["coordinates"], sort_keys=True),
                    "status": row["status"],
                    "outcome": row.get("outcome", ""),
                    "failure_class": row.get("failure_class", ""),
                }
            )


def _complete_rows():
    return [
        {
            "coordinates": {"offset": 0.5, "amplitude": 0.0},
            "status": "failed",
            "failure_class": "invalid_initial_constraint",
        },
        {
            "coordinates": {"offset": 3.5, "amplitude": 0.0},
            "status": "completed",
            "outcome": "monotonic_expansion",
        },
        {
            "coordinates": {"offset": 0.5, "amplitude": 0.08},
            "status": "failed",
            "failure_class": "invalid_initial_constraint",
        },
        {
            "coordinates": {"offset": 3.5, "amplitude": 0.08},
            "status": "completed",
            "outcome": "monotonic_expansion",
        },
    ]


def test_complete_cartesian_index_generates_nonempty_figure(tmp_path) -> None:
    index = tmp_path / "index.csv"
    destination = tmp_path / "outcome-map.png"
    _write_index(index, _complete_rows())

    result = plot_scan_outcome_map(
        index,
        destination,
        x_axis="offset",
        y_axis="amplitude",
    )

    assert result == destination
    assert destination.stat().st_size > 0


def test_missing_cartesian_cell_is_rejected(tmp_path) -> None:
    index = tmp_path / "index.csv"
    _write_index(index, _complete_rows()[:-1])
    with pytest.raises(ValueError, match="complete Cartesian"):
        plot_scan_outcome_map(index, tmp_path / "map.png", x_axis="offset", y_axis="amplitude")


def test_duplicate_coordinate_is_rejected(tmp_path) -> None:
    index = tmp_path / "index.csv"
    rows = _complete_rows()
    rows[-1] = {
        "coordinates": {"offset": 0.5, "amplitude": 0.0},
        "status": "completed",
        "outcome": "monotonic_expansion",
    }
    _write_index(index, rows)
    with pytest.raises(ValueError, match="Duplicate scan coordinate"):
        plot_scan_outcome_map(index, tmp_path / "map.png", x_axis="offset", y_axis="amplitude")


def test_nonnumeric_axis_is_rejected(tmp_path) -> None:
    index = tmp_path / "index.csv"
    rows = _complete_rows()
    rows[0]["coordinates"]["offset"] = "low"
    _write_index(index, rows)
    with pytest.raises(TypeError, match="must be numeric"):
        plot_scan_outcome_map(index, tmp_path / "map.png", x_axis="offset", y_axis="amplitude")
