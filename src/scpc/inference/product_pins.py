"""Cryptographic pin validation for externally hosted publication inputs."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProductPin:
    dataset: str
    filename: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ProductPinManifest:
    path: Path
    base_url: str
    pins: Mapping[tuple[str, str], ProductPin]


def load_product_pin_manifest(path: Path) -> ProductPinManifest:
    """Load and strictly validate the machine-readable external-product manifest."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported product-pin schema_version")
    if payload.get("hash_algorithm") != "sha256":
        raise ValueError("Product-pin manifest must use SHA-256")
    base_url = payload.get("base_url")
    if not isinstance(base_url, str) or not base_url.startswith("https://data.desi.lbl.gov/"):
        raise ValueError("Product-pin manifest has an unexpected DESI base URL")
    products = payload.get("products")
    if not isinstance(products, list) or not products:
        raise ValueError("Product-pin manifest contains no products")

    pins: dict[tuple[str, str], ProductPin] = {}
    for item in products:
        if not isinstance(item, dict):
            raise ValueError("Every product pin must be a mapping")
        dataset = item.get("dataset")
        filename = item.get("filename")
        size = item.get("bytes")
        digest = item.get("sha256")
        if not isinstance(dataset, str) or not dataset or "/" in dataset or ".." in dataset:
            raise ValueError(f"Invalid pinned dataset name {dataset!r}")
        if not isinstance(filename, str) or not filename or "/" in filename or ".." in filename:
            raise ValueError(f"Invalid pinned filename {filename!r}")
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"Invalid pinned byte length for {dataset}/{filename}")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Invalid pinned SHA-256 for {dataset}/{filename}")
        key = (dataset, filename)
        if key in pins:
            raise ValueError(f"Duplicate product pin for {dataset}/{filename}")
        pins[key] = ProductPin(dataset=dataset, filename=filename, bytes=size, sha256=digest)
    return ProductPinManifest(path=source, base_url=base_url.rstrip("/"), pins=pins)


def verify_dataset_provenance(
    manifest: ProductPinManifest,
    dataset: str,
    provenance: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Require provenance for a dataset to match every and only its pinned products."""
    expected = {filename: pin for (pin_dataset, filename), pin in manifest.pins.items() if pin_dataset == dataset}
    if not expected:
        raise ValueError(f"No product pins declared for dataset {dataset!r}")

    actual: dict[str, Mapping[str, object]] = {}
    for item in provenance:
        filename = Path(str(item.get("path", ""))).name
        if not filename:
            raise ValueError(f"Provenance entry has no usable path: {item!r}")
        if filename in actual:
            raise ValueError(f"Duplicate provenance entry for {dataset}/{filename}")
        actual[filename] = item

    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise RuntimeError(
            f"Pinned product set mismatch for {dataset}: missing={missing}, unexpected={unexpected}"
        )

    verified: list[dict[str, object]] = []
    for filename in sorted(expected):
        pin = expected[filename]
        item = actual[filename]
        try:
            observed_bytes = int(item["bytes"])
            observed_sha = str(item["sha256"])
            observed_url = str(item["url"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed provenance for {dataset}/{filename}: {item!r}") from exc
        expected_url = f"{manifest.base_url}/{dataset}/{filename}"
        if observed_bytes != pin.bytes:
            raise RuntimeError(
                f"Byte-length mismatch for {dataset}/{filename}: observed {observed_bytes}, pinned {pin.bytes}"
            )
        if observed_sha != pin.sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {dataset}/{filename}: observed {observed_sha}, pinned {pin.sha256}"
            )
        if observed_url != expected_url:
            raise RuntimeError(
                f"Source URL mismatch for {dataset}/{filename}: observed {observed_url!r}, expected {expected_url!r}"
            )
        verified.append(
            {
                "dataset": dataset,
                "filename": filename,
                "bytes": observed_bytes,
                "sha256": observed_sha,
                "url": observed_url,
                "verified": True,
            }
        )
    return verified
