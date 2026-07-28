"""Command-line entry point."""

from __future__ import annotations

import argparse

from scpc.data.manifest import select_products, validate_manifest
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
        print(verify_scpc_background(args.config, args.output))
    else:  # pragma: no cover
        raise RuntimeError(f"Unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
