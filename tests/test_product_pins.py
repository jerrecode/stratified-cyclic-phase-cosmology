import json
from pathlib import Path

import pytest

from scpc.inference.product_pins import load_product_pin_manifest, verify_dataset_provenance


def _manifest(tmp_path: Path) -> Path:
    path = tmp_path / "pins.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hash_algorithm": "sha256",
                "base_url": "https://data.desi.lbl.gov/public/example",
                "products": [
                    {
                        "dataset": "sample",
                        "filename": "chain.1.txt",
                        "bytes": 3,
                        "sha256": "a" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_product_pin_verification_accepts_exact_provenance(tmp_path: Path) -> None:
    manifest = load_product_pin_manifest(_manifest(tmp_path))
    verified = verify_dataset_provenance(
        manifest,
        "sample",
        [
            {
                "path": "/cache/sample/chain.1.txt",
                "url": "https://data.desi.lbl.gov/public/example/sample/chain.1.txt",
                "bytes": 3,
                "sha256": "a" * 64,
            }
        ],
    )
    assert verified == [
        {
            "dataset": "sample",
            "filename": "chain.1.txt",
            "bytes": 3,
            "sha256": "a" * 64,
            "url": "https://data.desi.lbl.gov/public/example/sample/chain.1.txt",
            "verified": True,
        }
    ]


def test_product_pin_verification_rejects_digest_mismatch(tmp_path: Path) -> None:
    manifest = load_product_pin_manifest(_manifest(tmp_path))
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_dataset_provenance(
            manifest,
            "sample",
            [
                {
                    "path": "/cache/sample/chain.1.txt",
                    "url": "https://data.desi.lbl.gov/public/example/sample/chain.1.txt",
                    "bytes": 3,
                    "sha256": "b" * 64,
                }
            ],
        )


def test_product_pin_verification_rejects_unexpected_product(tmp_path: Path) -> None:
    manifest = load_product_pin_manifest(_manifest(tmp_path))
    with pytest.raises(RuntimeError, match="product set mismatch"):
        verify_dataset_provenance(
            manifest,
            "sample",
            [
                {
                    "path": "/cache/sample/chain.1.txt",
                    "url": "https://data.desi.lbl.gov/public/example/sample/chain.1.txt",
                    "bytes": 3,
                    "sha256": "a" * 64,
                },
                {
                    "path": "/cache/sample/unpinned.txt",
                    "url": "https://data.desi.lbl.gov/public/example/sample/unpinned.txt",
                    "bytes": 1,
                    "sha256": "c" * 64,
                },
            ],
        )
