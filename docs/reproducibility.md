# Reproducibility standard

Every scientific run is configuration-driven and must record, where applicable:

- source Git commit and whether the checkout represents a branch head or integration/merge ref;
- configuration file and SHA-256 digest;
- Python, package, and operating-system versions;
- solver, tolerances, evaluation grid, and event settings;
- random seeds where stochastic algorithms are used;
- external release identifiers, exact source URLs, byte lengths, and local SHA-256 checksums;
- convergence diagnostics and declared acceptance thresholds for posterior products;
- wall-clock/runtime context and output checksums for load-bearing products.

## Numerical acceptance

A numerical result intended for the paper must have:

1. an analytical-limit or regression test;
2. a tolerance or resolution-convergence result;
3. a constraint-residual diagnostic;
4. an independent implementation, solver, or backend cross-check for load-bearing claims;
5. a serialized data product from which the figure is regenerated;
6. explicit failure semantics for invalid states, domain exits, and output-integrity failures.

A declared numerical domain boundary is an analysis limit. Reaching it must not be silently interpreted as a physical singularity, instability, or proof of geodesic incompleteness.

## Posterior-inference acceptance

A posterior result intended for the manuscript must additionally record:

1. the exact released likelihood/chain product and release version;
2. prior assumptions and which parameters are fixed, sampled, or derived;
3. the convergence diagnostic and predeclared pass threshold;
4. the treatment of sample weights and burn-in;
5. whether quoted intervals are marginalized posterior intervals, local Gaussian approximations, profile intervals, or another construction;
6. the full covariance treatment for correlated data;
7. the distinction between frequentist goodness-of-fit quantities and Bayesian posterior/model-check quantities;
8. the epistemic status of plotted curves: directly data-constrained, posterior-derived, model-extrapolated, or numerical-validation-only.

Finite expensive-backend evaluations used to approximate a weighted posterior must respect posterior weights. For compressed Cobaya chains in the DESI reproduction, deterministic systematic positions in cumulative posterior weight are used instead of equal raw-row spacing.

## Data integrity

Downloaded products are treated as immutable publication inputs. If an upstream collaboration publishes checksums, record and verify them as upstream checksums. If it does not, compute a local SHA-256 digest, label it as a locally recorded pin, and keep the exact source URL and byte length alongside it.

For the DESI DR2 publication reproduction, `configs/inference/desi_dr2_chain_pins.json` is the machine-readable pin manifest. The prefetch workflow requires an exact byte-length and SHA-256 match before a file enters the usable cache, and the Python inference layer independently verifies the complete expected product set, source URLs, byte lengths, and SHA-256 digests. A truncated, corrupted, unexpected, missing, or byte-mutated product is a hard failure.

A cache is a performance optimization, not a source of truth. Cached external products must be revalidated against the publication pins before use.

## Paper reproduction

The complete publication path is:

```bash
python scripts/reproduce.py
bash scripts/prefetch_desi_dr2_chains.sh
python scripts/reproduce_desi_dr2_bbn.py --download-official-chains
latexmk -pdf -cd paper/main.tex
```

The GitHub Actions CI, DESI-posterior, and Paper workflows independently exercise the relevant parts of this path. A manuscript revision should not be described as publication-reproducible until the complete inference workflow and complete paper build both succeed on the same source head.
