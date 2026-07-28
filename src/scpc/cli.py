"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scpc.data.manifest import select_products, validate_manifest
from scpc.scans.runner import run_background_scan
from scpc.workflows import compare_models, run_scpc_background, verify_scpc_background


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scpc")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-manifest", help="Validate data/releases.yaml")
    validate.add_argument("--manifest", default="data/releases.yaml")
    validate.add_argument("--schema", default="data/manifest.schema.json")

    list_data = sub.add_parser("list-data", help="List manifest products")
    list_data.add_argument("--manifest", default="data/releases.yaml")
    list_data.add_argument("--role", action="append")
    list_data.add_argument("--domain", action="append")
    list_data.add_argument("--required-only", action="store_true")

    compare = sub.add_parser("compare-models", help="Generate standard-model comparison grid")
    compare.add_argument("--config", default="configs/model_comparisons.yaml")
    compare.add_argument("--output", default="results/model_comparison")

    background = sub.add_parser("run-background", help="Integrate the canonical SCPC background")
    background.add_argument("--config", default="configs/scpc_baseline.yaml")
    background.add_argument("--output", default="results/scpc_baseline")

    verify = sub.add_parser(
        "verify-background",
        help="Run tolerance and cross-solver verification for the SCPC background",
    )
    verify.add_argument("--config", default="configs/scpc_verification.yaml")
    verify.add_argument("--output", default="results/scpc_verification")

    scan = sub.add_parser(
        "scan-background",
        help="Run or resume a deterministic failure-preserving background parameter scan",
    )
    scan.add_argument("--config", default="configs/scans/stage1_smoke.yaml")
    scan.add_argument("--output", default="results/stage1_smoke_scan")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-manifest":
        manifest = validate_manifest(args.manifest, args.schema)
        print(f"valid: {len(manifest['releases'])} releases")
    elif args.command == "list-data":
        manifest = validate_manifest(args.manifest)
        for product in select_products(
            manifest,
            roles=args.role,
            domains=args.domain,
            required_only=args.required_only,
        ):
            print(f"{product['release']}::{product['id']} — {product['purpose']}")
    elif args.command == "compare-models":
        print(compare_models(args.config, args.output))
    elif args.command == "run-background":
        print(run_scpc_background(args.config, args.output))
    elif args.command == "verify-background":
        report_path = verify_scpc_background(args.config, args.output)
        print(report_path)
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        return 0 if report["passed"] else 2
    elif args.command == "scan-background":
        print(run_background_scan(args.config, args.output))
    else:  # pragma: no cover
        raise RuntimeError(f"Unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
