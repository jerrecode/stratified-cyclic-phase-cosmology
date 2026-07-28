"""Load, validate, and query the scientific data-release manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import jsonschema
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "data" / "manifest" / "releases.yaml"
DEFAULT_SCHEMA = REPOSITORY_ROOT / "data" / "manifest" / "releases.schema.json"


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Manifest root must be a mapping")
    return payload


def validate_manifest(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    schema_path: str | Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(instance=manifest, schema=schema)
    return manifest


def iter_releases(manifest: dict[str, Any]) -> Iterable[dict[str, Any]]:
    releases = manifest.get("releases", [])
    if not isinstance(releases, list):
        raise ValueError("Manifest releases must be a list")
    for release in releases:
        if not isinstance(release, dict):
            raise ValueError("Each release must be a mapping")
        yield release


def find_release(release_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    for release in iter_releases(manifest):
        if release.get("id") == release_id:
            return release
    raise KeyError(f"Unknown release id: {release_id}")


def find_product(
    release: dict[str, Any], product_id: str | None
) -> dict[str, Any]:
    products = release.get("products", [])
    if not products:
        raise KeyError(f"Release {release.get('id')} has no products")
    if product_id is None:
        if len(products) != 1:
            available = ", ".join(str(product.get("id")) for product in products)
            raise KeyError(f"Select a product explicitly; available: {available}")
        return products[0]
    for product in products:
        if product.get("id") == product_id:
            return product
    raise KeyError(f"Unknown product {product_id!r} for release {release.get('id')}")
