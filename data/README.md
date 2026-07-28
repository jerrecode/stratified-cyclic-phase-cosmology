# External scientific data

- `releases.yaml`: curated release and product manifest.
- `manifest.schema.json`: machine validation contract.
- `raw/`: immutable downloaded source products; ignored by Git.
- `processed/`: deterministic transformations with provenance; ignored by Git.

Use `scpc validate-manifest` before any retrieval. Use `scpc list-data` to inspect roles and access methods. The downloader defaults to a dry run because many cosmological archives are very large or require query-specific selection.
