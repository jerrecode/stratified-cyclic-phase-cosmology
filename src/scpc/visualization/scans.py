"""Visualization of parameter-scan outcome classifications."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def _load_rows(index_path: str | Path) -> list[dict[str, Any]]:
    with Path(index_path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Scan index is empty")
    for row in rows:
        coordinates = json.loads(row["coordinates"])
        if not isinstance(coordinates, dict):
            raise TypeError("Scan coordinates must decode to a mapping")
        row["decoded_coordinates"] = coordinates
    return rows


def _numeric_coordinate(row: dict[str, Any], axis: str) -> float:
    coordinates = row["decoded_coordinates"]
    if axis not in coordinates:
        raise KeyError(f"Scan index row is missing configured axis {axis!r}")
    value = coordinates[axis]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Outcome-map axis {axis!r} must be numeric, received {value!r}")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Outcome-map axis {axis!r} contains a nonfinite value")
    return number


def _classification_label(row: dict[str, Any]) -> str:
    status = row["status"]
    if status == "completed":
        return row["outcome"] or "completed:unclassified"
    if status == "rejected":
        return f"rejected:{row['outcome'] or 'unclassified'}"
    if status == "failed":
        return f"failed:{row['failure_class'] or 'unclassified'}"
    return f"unknown-status:{status}"


def _cell_edges(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Axis values must be a nonempty one-dimensional array")
    if values.size == 1:
        width = max(abs(float(values[0])) * 0.1, 0.5)
        return np.asarray([values[0] - width, values[0] + width])
    differences = np.diff(values)
    if np.any(differences <= 0.0):
        raise ValueError("Axis values must be strictly increasing after deduplication")
    midpoints = values[:-1] + 0.5 * differences
    return np.concatenate(
        (
            [values[0] - 0.5 * differences[0]],
            midpoints,
            [values[-1] + 0.5 * differences[-1]],
        )
    )


def plot_scan_outcome_map(
    index_path: str | Path,
    output_path: str | Path,
    *,
    x_axis: str,
    y_axis: str,
    annotate: bool = True,
) -> Path:
    """Plot a complete two-dimensional Cartesian outcome classification map."""

    if x_axis == y_axis:
        raise ValueError("x_axis and y_axis must be different")
    rows = _load_rows(index_path)
    x_values = np.asarray(sorted({_numeric_coordinate(row, x_axis) for row in rows}))
    y_values = np.asarray(sorted({_numeric_coordinate(row, y_axis) for row in rows}))
    expected_cells = x_values.size * y_values.size
    if len(rows) != expected_cells:
        raise ValueError(
            f"Outcome map requires a complete Cartesian index: expected {expected_cells} rows, "
            f"found {len(rows)}"
        )

    labels = sorted({_classification_label(row) for row in rows})
    label_codes = {label: index for index, label in enumerate(labels)}
    grid = np.full((y_values.size, x_values.size), -1, dtype=int)
    seen: set[tuple[float, float]] = set()
    for row in rows:
        x_value = _numeric_coordinate(row, x_axis)
        y_value = _numeric_coordinate(row, y_axis)
        coordinate = (x_value, y_value)
        if coordinate in seen:
            raise ValueError(f"Duplicate scan coordinate in outcome map: {coordinate}")
        seen.add(coordinate)
        x_index = int(np.searchsorted(x_values, x_value))
        y_index = int(np.searchsorted(y_values, y_value))
        grid[y_index, x_index] = label_codes[_classification_label(row)]
    if np.any(grid < 0):
        raise ValueError("Outcome map contains unfilled Cartesian cells")

    base_cmap = plt.get_cmap("tab20", max(len(labels), 1))
    cmap = ListedColormap([base_cmap(index) for index in range(len(labels))])
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.pcolormesh(
        _cell_edges(x_values),
        _cell_edges(y_values),
        grid,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(labels) - 0.5,
        shading="flat",
    )
    axis.set_xlabel(x_axis)
    axis.set_ylabel(y_axis)
    axis.set_title("SCPC background outcome classification")

    if annotate and expected_cells <= 100:
        for y_index, y_value in enumerate(y_values):
            for x_index, x_value in enumerate(x_values):
                label = labels[grid[y_index, x_index]]
                axis.text(
                    x_value,
                    y_value,
                    label.replace(":", ":\n"),
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    handles = [Patch(facecolor=cmap(index), label=label) for label, index in label_codes.items()]
    axis.legend(handles=handles, title="Classification", loc="best", fontsize=8)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination
