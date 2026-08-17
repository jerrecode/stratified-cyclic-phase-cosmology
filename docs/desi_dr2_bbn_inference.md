# DESI DR2 BAO + BBN inference contract

This analysis separates three epistemically different objects that were previously easy to
conflate: the released DESI posterior, the deterministic repository likelihood cross-check, and
model-implied expansion histories.

The **main posterior figure** uses the official DESI DR2 Cobaya chains released for Results II.
The BAO-only panel shows the marginalized `(Omega_m, h r_d)` posterior. The BBN-calibrated panel
shows `(Omega_m, H0)`. BBN is represented through the Schoneberg 2024 prior on `omega_b_h2`; it
is never drawn as if it directly measured `r_d`.

The **MAP regression check** uses the pinned 13-element DESI Gaussian vector and full covariance,
with CAMB providing `r_d`, distances, and the early-Universe background. It exists to catch code
regressions; local optimizer curvature is not used as a substitute for marginalized posterior
coverage.

The **background panels** are explicitly model-implied. The low-redshift panel shows posterior
predictive `H(z)` variation across the DESI effective-redshift range. The high-redshift panel is
computed from CAMB and plots `Omega_cb`, `Omega_gamma`, total `Omega_nu`, and `Omega_Lambda`.
This avoids assigning the 0.06 eV massive neutrino to an `a^-3` component while it is still
relativistic.

`q(z)` and `w_tot(z)=p_tot/rho_tot` are moved to a supplementary diagnostic because they are
algebraically redundant in a flat GR background. The Aubourg sound-horizon approximation is
also supplementary and is compared against CAMB on posterior draws; it is never the inference
backend.

## Reproduction

```bash
python -m pip install -e '.[inference]'
python scripts/reproduce_desi_dr2_bbn.py --download-official-chains
```

This creates `results/desi_dr2_bbn/desi_dr2_bao_bbn_posteriors.png`, supplementary figures,
`inference_summary.json`, and exact source hashes for each downloaded chain file. The figures are
also copied into `paper/generated/` for optional manuscript inclusion.

The analysis intentionally uses the pinned 2025 Results I/II BAO likelihood. The later 2026
DESI Lyman-alpha AP+BAO result is not multiplied into that covariance unless a release-native
joint product supports doing so.
