"""Command-line interface for simulations, comparisons, and data releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scpc import __version__
from scpc.config import load_yaml
from scpc.data.fetch import AccessInstructionRequired, fetch_product
from scpc.data.manifest import (
    DEFAULT_MANIFEST,
    find_product,
    find_release,
    iter_releases,
    validate_manifest,
)
from scpc.models.lcdm import LambdaCDM
from scpc.models.scpc import build_scpc_background
from scpc.models.w0wa import W0WaCDM
from scpc.numerics.integrate import IntegrationSettings, integrate_background, write_solution
from scpc.visualization.backgrounds import plot_background_solution, plot_expansion_comparison


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _simulate(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    config = load_yaml(config_path)
    parameters, potential, initial_state = build_scpc_background(config)
    integration = config["integration"]
    settings = IntegrationSettings(
        time_start=float(integration["time_start"]),
        time_end=float(integration["time_end"]),
        samples=int(integration.get("samples", 2000)),
        method=str(integration.get("method", "DOP853")),
        relative_tolerance=float(integration.get("relative_tolerance", 1.0e-10)),
        absolute_tolerance=float(integration.get("absolute_tolerance", 1.0e-12)),
        max_step=float(integration.get("max_step", np.inf)),
    )
    solution = integrate_background(initial_state, parameters, potential, settings)
    provenance: dict[str, Any] = {
        "scpc_version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "configuration_path": str(config_path),
        "configuration_sha256": _sha256(config_path),
        "configuration": config,
    }
    output = Path(args.output)
    write_solution(solution, output, provenance)
    plot_background_solution(solution, output / "background_evolution.pdf")
    print(json.dumps({"output": str(output), "constraint": solution.constraint.__dict__}, indent=2))
    return 0


def _build_comparison_model(specification: dict[str, Any]) -> LambdaCDM | W0WaCDM:
    kind = specification["kind"]
    parameters = specification["parameters"]
    common = {
        "omega_matter": float(parameters["omega_matter"]),
        "omega_radiation": float(parameters.get("omega_radiation", 0.0)),
        "omega_curvature": float(parameters.get("omega_curvature", 0.0)),
        "hubble_0_km_s_mpc": float(parameters.get("hubble_0_km_s_mpc", 67.36)),
        "name": str(specification.get("label", kind)),
    }
    if kind == "lcdm":
        return LambdaCDM(omega_lambda=parameters.get("omega_lambda"), **common)
    if kind == "w0wa_cdm":
        return W0WaCDM(
            w0=float(parameters["w0"]),
            wa=float(parameters["wa"]),
            omega_dark_energy=parameters.get("omega_dark_energy"),
            **common,
        )
    raise ValueError(f"Unsupported comparison model kind: {kind}")


def _compare(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)
    grid = config["grid"]
    scale_factor = np.geomspace(
        float(grid["scale_factor_min"]),
        float(grid["scale_factor_max"]),
        int(grid["samples"]),
    )
    models = {
        str(item["label"]): _build_comparison_model(item) for item in config["models"]
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    table = np.column_stack(
        [scale_factor] + [model.dimensionless_hubble(scale_factor) for model in models.values()]
    )
    header = "scale_factor," + ",".join(models.keys())
    np.savetxt(output / "expansion_histories.csv", table, delimiter=",", header=header, comments="")
    plot_expansion_comparison(scale_factor, models, output / "expansion_histories.pdf")
    return 0


def _data_validate(args: argparse.Namespace) -> int:
    manifest = validate_manifest(args.manifest)
    print(f"Validated {len(list(iter_releases(manifest)))} releases: {args.manifest}")
    return 0


def _data_list(args: argparse.Namespace) -> int:
    manifest = validate_manifest(args.manifest)
    for release in iter_releases(manifest):
        roles = ",".join(release["scientific_roles"])
        print(f"{release['id']:<28} {release['priority']:<14} {roles}")
    return 0


def _data_show(args: argparse.Namespace) -> int:
    manifest = validate_manifest(args.manifest)
    release = find_release(args.release_id, manifest)
    print(yaml.safe_dump(release, sort_keys=False, allow_unicode=True))
    return 0


def _data_fetch(args: argparse.Namespace) -> int:
    manifest = validate_manifest(args.manifest)
    release = find_release(args.release_id, manifest)
    product = find_product(release, args.product)
    try:
        path = fetch_product(product, args.destination)
    except AccessInstructionRequired as error:
        print(str(error), file=sys.stderr)
        return 2
    print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scpc")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="Integrate an SCPC background")
    simulate.add_argument("config")
    simulate.add_argument("--output", required=True)
    simulate.set_defaults(handler=_simulate)

    compare = subparsers.add_parser("compare", help="Generate standardized comparator curves")
    compare.add_argument("config")
    compare.add_argument("--output", required=True)
    compare.set_defaults(handler=_compare)

    data = subparsers.add_parser("data", help="Inspect or acquire registered releases")
    data_subparsers = data.add_subparsers(dest="data_command", required=True)
    for name in ("validate", "list"):
        command = data_subparsers.add_parser(name)
        command.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
        command.set_defaults(handler=_data_validate if name == "validate" else _data_list)
    show = data_subparsers.add_parser("show")
    show.add_argument("release_id")
    show.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    show.set_defaults(handler=_data_show)
    fetch = data_subparsers.add_parser("fetch")
    fetch.add_argument("release_id")
    fetch.add_argument("--product")
    fetch.add_argument("--destination", required=True)
    fetch.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    fetch.set_defaults(handler=_data_fetch)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
