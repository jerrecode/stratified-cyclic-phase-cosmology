from __future__ import annotations

from scpc.data.manifest import iter_releases, validate_manifest


def test_release_manifest_validates_and_ids_are_unique() -> None:
    manifest = validate_manifest()
    ids = [release["id"] for release in iter_releases(manifest)]
    assert len(ids) >= 10
    assert len(ids) == len(set(ids))


def test_manifest_contains_core_probe_families() -> None:
    manifest = validate_manifest()
    ids = {release["id"] for release in iter_releases(manifest)}
    required = {
        "planck-2018-pr3",
        "act-dr6.02",
        "desi-dr2-cosmology",
        "pantheon-plus-shoes",
        "des-y6",
        "bicep-keck-bk18",
    }
    assert required <= ids
