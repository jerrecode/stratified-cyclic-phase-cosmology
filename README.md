# Stratified Cyclic Phase Cosmology

**Stratified Cyclic Phase Cosmology (SCPC)** is a research-software project for constructing, simulating, testing, and falsifying covariant cosmological models whose dynamical state space contains distinguishable phase strata and whose solutions may admit nonsingular cyclic evolution.

The repository deliberately separates four structures that must not be conflated:

1. Lorentzian spacetime geometry;
2. phase- or field-space stratification;
3. thermodynamic state-space geometry;
4. numerical discretization.

The initial implementation is a scientifically conservative baseline. It contains standard FLRW comparison models, a canonical scalar stratification field in FLRW spacetime, constraint diagnostics, turning-point return diagnostics, numerical-convergence checks, reproducible model-comparison grids, a machine-validated public-data manifest, and a modular LaTeX paper. It does **not** claim that a recurrent or stable cycle, a fundamental discrete time, or a physical vortex has already been demonstrated.

## Research questions

The codebase is organized to answer, in order:

- Does a declared action produce a mathematically consistent background cosmology?
- Are bounce and turnaround solutions nonsingular and dynamically stable?
- Are stress-energy conservation and entropy-production conditions satisfied?
- Do perturbations avoid ghosts, gradient instabilities, and strong coupling?
- Does the model recover GR and standard cosmology in a controlled limit?
- Which observables distinguish SCPC from established models?
- Which public data releases constrain or disprove the viable parameter domain?

The dependency-ordered implementation and evidence gates are defined in [`docs/research_roadmap.md`](docs/research_roadmap.md).

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

scpc validate-manifest
scpc compare-models --config configs/model_comparisons.yaml --output results/model_comparison
scpc run-background --config configs/scpc_baseline.yaml --output results/scpc_baseline
scpc verify-background --config configs/scpc_verification.yaml --output results/scpc_verification
pytest
```

Generate the reproducible paper inputs:

```bash
python scripts/reproduce.py
scpc verify-background
latexmk -pdf -cd paper/main.tex
```

## Core outputs

- Standard background quantities on common redshift grids: `H(z)`, `E(z)`, comoving distance, luminosity distance, and distance modulus.
- SCPC phase-field trajectories: `a(t)`, `H(t)`, `phi(t)`, `dphi/dt`, energy components, pressure, equation of state, and Friedmann-constraint residual.
- Root-localized turning-point states classified as candidate bounces or turnarounds.
- Same-kind return metrics that preserve real-field displacement and report circular winding explicitly.
- Return-sequence summaries that require repeated close returns but do not claim recurrence from one run.
- Tolerance-ladder and cross-solver verification reports with explicit acceptance checks and unwrapped-field comparison.
- NetCDF and CSV products with explicit units and coordinates.
- Publication-ready figures generated from the same arrays used in numerical analysis.
- A versioned manifest describing public observational releases, products, values, units, dimensions, access protocols, and reproducible retrieval instructions.

## Repository map

```text
src/scpc/                  importable scientific package
configs/                   versioned model, run, and verification configurations
data/                      release manifest, schema, checksums, local-data policy
scripts/                   reproducible workflow entry points
tests/                     analytical, convergence, unit, regression, and manifest tests
paper/                     modular LaTeX manuscript and paper-local products
docs/                      theory, architecture, conventions, governance, and research gates
results/                   generated outputs; large results are not committed
.github/workflows/         continuous integration and paper-build workflows
```

## Scientific nonclaims

The analytic cyclic reference curve supplied for plotting and pipeline testing is not an observationally fitted cosmological theory. Likewise, the first canonical SCPC action is a baseline whose failure is scientifically informative. New terms may be added only with an explicit action, dimensions, conservation law, stability conditions, and a stated standard-model limit.

A close return in one integration is not a recurrence result. Candidate recurrence requires repeated close returns reproduced under tighter tolerances and independent solvers; a stable limit-cycle claim additionally requires a Poincare map, variational equations, converged Floquet multipliers, and perturbative-stability checks.

A periodic scalar potential does not make the field compact. The baseline declares a real target space. Circular topology must be selected explicitly, adjacent potential minima remain distinct strata, and nonzero winding is retained as a separate diagnostic.

## Reproducibility

Every run should be driven by a committed YAML configuration and should emit a provenance record containing the Git commit, dependency versions, configuration hash, solver settings, platform information, and output checksums. See `docs/reproducibility.md` and `workflows/reproduce.yaml`.

## License and citation

Code is released under the BSD 3-Clause License. Dataset licenses and acknowledgements remain those of the originating collaborations and are recorded per release in `data/releases.yaml`. Cite the software using `CITATION.cff` and cite every observational release used in an analysis.
