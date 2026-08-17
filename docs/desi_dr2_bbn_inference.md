# DESI DR2 BAO + BBN inference contract

This analysis separates four epistemically different objects that must not be conflated:

1. the released DESI posterior;
2. the deterministic repository likelihood/MAP cross-check;
3. posterior-propagated data-model residual diagnostics;
4. model-implied standard-cosmology background histories.

None of these is an SCPC observational fit. They are a calibration and reproducibility benchmark for the observational pipeline.

## Released posterior

The main posterior figure uses the official DESI DR2 Cobaya chains released for Results II. The BAO-only panel shows the marginalized `(Omega_m, h r_d)` posterior. The BBN-calibrated panel shows `(Omega_m, H0)`. GetDist constructs the marginalized 68% and 95% credible regions.

BAO alone identifies the ruler combination `eta = H0 r_d / c` together with late-time shape information; it does not separately measure `H0` and `r_d`. BBN enters through the Schöneberg 2024 PRyMordial prior on `omega_b_h2`. It is never represented as if BBN directly measured `r_d`; the drag scale follows only after the declared standard early-Universe assumptions and CAMB calculation are supplied.

A fixed burn fraction is not assumed. The pipeline tests a predeclared sequence of candidate cuts and retains the earliest cut satisfying multivariate GetDist `R-1 <= 0.01`. An explicit burn override is accepted only if it passes the same threshold.

## External-product integrity

The exact sample and release-side metadata products used for publication are declared in:

`configs/inference/desi_dr2_chain_pins.json`

For every selected file the manifest pins the expected byte length and SHA-256 digest. `scripts/prefetch_desi_dr2_chains.sh` performs resumable, retrying retrieval and validates both quantities before promoting a file into the cache. The Python reproduction independently verifies that every and only the declared products are present and that byte length, SHA-256, and official source URL all match the manifest. A mismatch is a hard failure.

The generated `inference_summary.json` records the pin-manifest checksum, observed product checksums, source URLs, chain convergence diagnostics, runtime versions, configuration checksum, and numerical outputs.

## MAP regression check

The independent MAP regression check uses the pinned 13-element DESI Gaussian BAO vector and full covariance, with CAMB providing `r_d` and the BAO distance predictions. It exists to catch likelihood/data/backend regressions. Local optimizer curvature is not used as a substitute for marginalized posterior coverage.

The frequentist BAO goodness-of-fit probability from this predeclared MAP check is reported separately from the Bayesian posterior-predictive diagnostic described below.

## Posterior residual diagnostic

For `C = L L^T`, the residual vector is

`r_w(theta) = L^-1 [m(theta) - d]`.

Exact CAMB predictions are evaluated at deterministic systematic positions in cumulative posterior weight. This is important because compressed Cobaya rows carry nonuniform weights; equally spaced row numbers are not equiprobable posterior draws.

The main figure displays posterior intervals of the Cholesky-whitened residual components. Component `i` mixes covariance rows `1..i`, so the short observable/redshift text below a component is only an ordering aid and must not be interpreted as an independent one-observable residual.

The pipeline also reports a Bayesian posterior-predictive tail probability based on the quadratic discrepancy `T = r_w^T r_w`. For fixed `theta`, a Gaussian replicated BAO vector has `T_rep ~ chi2_n`. Averaging the corresponding upper-tail probability over the posterior gives the reported posterior-predictive diagnostic. Because the observed data enter both the posterior and the discrepancy, this quantity is generally conservative and is not interchangeable with the frequentist MAP `p_gof`.

## CAMB background extrapolation

The high-redshift panel is explicitly model-implied. CAMB provides the momentum-integrated background densities and the panel separates:

- `Omega_cb`;
- `Omega_gamma`;
- massless-neutrino density;
- massive-neutrino density;
- `Omega_Lambda`.

Posterior uncertainty is propagated through these histories and through characteristic epochs such as acceleration onset, drag, photon decoupling, and matter-radiation equality. This avoids assigning the fixed 0.06 eV massive-neutrino species to a pure `a^-3` component while it is relativistic. The resulting high-redshift histories are standard-flat-LambdaCDM extrapolations conditioned on the released DESI+BBN posterior, not direct DESI measurements.

`q0`, `w_tot,0`, and the acceleration-transition posterior are supplementary because they are algebraically related in the late-time flat-GR background. The Aubourg sound-horizon approximation is also supplementary and is compared against CAMB on posterior-weight representative points; it is never the calibrated inference backend.

## Reproduction

```bash
python -m pip install -e '.[inference]'
python scripts/reproduce.py
bash scripts/prefetch_desi_dr2_chains.sh
python scripts/reproduce_desi_dr2_bbn.py --download-official-chains
```

This creates `results/desi_dr2_bbn/desi_dr2_bao_bbn_posteriors.png`, supplementary diagnostics, generated LaTeX tables, and `inference_summary.json`. Paper-local copies are written to `paper/generated/`.

The analysis intentionally uses the pinned 2025 Results I/II BAO likelihood. The later 2026 DESI Lyman-alpha AP+BAO result remains a separate release product and is not multiplied into that covariance unless a release-native joint covariance or likelihood supports doing so.
