"""Validation and querying of the public cosmology data-release manifest."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import jsonschema
import yaml


def load_manifest(path: str | Path = "data/releases.yaml") -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError("Manifest root must be a mapping")
    if "release_files" in data:
        releases = []
        for relative in data["release_files"]:
            release_path = manifest_path.parent / relative
            with release_path.open("r", encoding="utf-8") as handle:
                release = yaml.safe_load(handle)
            if not isinstance(release, dict):
                raise TypeError(f"Release file {release_path} must contain a mapping")
            releases.append(release)
        data = {**data, "releases": releases}
    return data


def validate_manifest(
    manifest_path: str | Path = "data/releases.yaml",
    schema_path: str | Path = "data/manifest.schema.json",
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        index = yaml.safe_load(handle)
    with Path(schema_path).open("r", encoding="utf-8") as handle:
        schema = yaml.safe_load(handle)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(index, schema)
    manifest = load_manifest(manifest_path)
    release_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": schema["$defs"],
        **schema["$defs"]["release"],
    }
    for release in manifest["releases"]:
        jsonschema.validate(release, release_schema)
    return manifest


def select_products(
    manifest: dict[str, Any],
    *,
    roles: Iterable[str] | None = None,
    domains: Iterable[str] | None = None,
    required_only: bool = False,
) -> list[dict[str, Any]]:
    role_set = set(roles or [])
    domain_set = set(domains or [])
    selected: list[dict[str, Any]] = []
    for release in manifest.get("releases", []):
        if domain_set and not domain_set.intersection(release.get("scientific_domains", [])):
            continue
        if role_set and not role_set.intersection(release.get("roles", [])):
            continue
        for product in release.get("products", []):
            if required_only and not product.get("required", False):
                continue
            selected.append({"release": release["id"], **product})
    return selected
