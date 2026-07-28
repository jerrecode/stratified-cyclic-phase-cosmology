from scpc.data.manifest import select_products, validate_manifest


def test_manifest_validates() -> None:
    manifest = validate_manifest()
    assert manifest["manifest_version"] == "0.1.0"
    assert len(manifest["releases"]) >= 10


def test_required_fit_products_are_queryable() -> None:
    manifest = validate_manifest()
    products = select_products(manifest, roles=["parameter_fit"], required_only=True)
    ids = {(item["release"], item["id"]) for item in products}
    assert ("pantheon_plus", "distances_and_covariance") in ids
    assert ("desi_dr1_bao_cosmology", "bao_measurements_and_covariance") in ids
