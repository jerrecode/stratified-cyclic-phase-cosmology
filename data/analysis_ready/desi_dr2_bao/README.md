# Pinned DESI DR2 BAO Gaussian likelihood

These two small likelihood-critical files are pinned from `CobayaSampler/bao_data` commit
`bb0c1c9009dc76d1391300e169e8df38fd1096db`, directory `desi_bao_dr2/`:

- `desi_gaussian_bao_ALL_GCcomb_mean.txt` -> `mean.txt`
- `desi_gaussian_bao_ALL_GCcomb_cov.txt` -> `covariance.txt`

SHA-256 checksums are recorded in `configs/inference/desi_dr2_bbn.yaml`. The row order in
`mean.txt` is authoritative for the covariance. DESI DR1 and DR2 overlap and must not be
multiplied as independent likelihoods. The 2026 DESI Lyman-alpha AP+BAO update is tracked
separately and is not silently inserted into this pinned 2025 covariance.

The much larger official DESI posterior chains are deliberately not vendored. The reproduction
script downloads them from the DESI public data server into `data/external/` and records hashes
in the generated inference report.
