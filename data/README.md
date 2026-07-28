# Data policy

`data/manifest/releases.yaml` is the authoritative registry of external releases. `data/external/` is a local cache and is ignored by Git.

Never commit multi-gigabyte public products. Every analysis run must record:

- manifest release and product IDs;
- immutable version, DOI, archive path, or Git commit;
- file sizes and cryptographic checksums;
- access date and query/selection parameters;
- transformations applied;
- citations and acknowledgements.

Compact paper-specific derived tables may be copied into `paper/data/` only when their derivation and provenance are documented.
