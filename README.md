# Stratified Cyclic Phase Cosmology

**Stratified Cyclic Phase Cosmology (SCPC)** is a research-software and manuscript project for constructing, verifying, simulating, testing, and falsifying covariant cosmological models whose dynamical state space contains distinguishable phase strata and whose solutions may admit nonsingular cyclic evolution.

The repository deliberately separates four structures that must not be conflated:

1. Lorentzian spacetime geometry;
2. phase- or field-space stratification;
3. thermodynamic state-space geometry;
4. numerical discretization.

The initial implementation is a scientifically conservative baseline. It provides a closed-FLRW background with a canonical stratification field, explicit stress-energy conservation, Friedmann-constraint diagnostics, event-based bounce and turnaround detection, standardized comparison histories, a machine-validated public-data manifest, and a modular LaTeX paper. It does **not** claim that a stable cycle, fundamental discrete time, physical vortex, or observational preference has already been demonstrated.

## Research questions

The codebase is organized to answer, in order:

- Does a declared action produce a mathematically consistent background cosmology?
- Does it admit nonsingular bounces and turnarounds in a declared parameter domain?
- Are recurrent trajectories stable under homogeneous and inhomogeneous perturbations?
- Are stress-energy conservation and any entropy-production conditions satisfied?
- Does the effective theory avoid ghosts, gradient instabilities, and strong coupling?
- Does it recover general relativity and standard cosmology in a controlled limit?
- Which observables distinguish SCPC from established non-cyclic and cyclic models?
- Which public data releases constrain or disprove the viable parameter domain?
- Are apparent spectral or topological features invariant under solver, resolution, and sampling changes?

## Repository map

```text
src/scpc/                 Installable scientific Python package
configs/                  Versioned model, solver, and comparison configurations
data/manifest/            Scientific release manifest and JSON Schema
data/external/            Ignored local cache for external products
scripts/                  Reproduction entry points
results/                  Generated run products; not committed by default
tests/                    Analytical, conservation, schema, and regression tests
paper/                    Modular LaTeX manuscript and paper-local products
workflows/                Reproduction and release workflow descriptions
docs/                     Theory, architecture, conventions, and governance
.github/workflows/         Continuous integration and paper builds
```

## Installation

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

With `uv`:

```bash
uv sync --all-extras
```

## Baseline simulation

```bash
scpc simulate configs/baseline/scpc_closed.yaml --output results/baseline
```

The run writes:

- `trajectory.nc`: labeled scientific array data with units and metadata;
- `trajectory.csv`: compact interoperability export;
- `diagnostics.json`: constraint residuals and event summary;
- `run.json`: configuration and software provenance;
- `background_evolution.pdf`: publication-oriented diagnostic figure.

The core trajectory contains `a(t)`, `H(t)`, `phi(t)`, `dphi/dt`, matter and radiation densities, turning-point events, and the Friedmann-constraint residual. A candidate bounce or turnaround is an event classification, not by itself evidence of a stable cosmological cycle.

## Data manifest

Validate and inspect the external data-release registry:

```bash
scpc data validate
scpc data list
scpc data show desi-dr2-cosmology
```

Fetch products with direct or Git access:

```bash
scpc data fetch pantheon-plus-shoes --product repository --destination data/external
```

Archive-query, TAP, Globus, DataFind, Zenodo, and asynchronous products intentionally produce explicit retrieval instructions rather than pretending to be static files. Dataset licenses and acknowledgements remain those of the originating collaborations and are recorded per release in `data/manifest/releases.yaml`.

## Model comparison

```bash
scpc compare configs/comparison/background_models.yaml \
  --output results/model_comparison
```

All curves are evaluated on the same configured scale-factor grid. Published benchmark parameters and illustrative, non-fitted parameters are labeled separately. Dimensional and dimensionless quantities are never mixed silently.

## Verification

```bash
pytest
ruff check .
mypy src
scpc data validate
```

Every result-worthy run should be driven by a committed YAML configuration and emit provenance containing the Git commit, dependency versions, configuration hash, solver settings, platform information, external product identifiers, and output checksums.

## Paper

```bash
make paper
```

All paper figures and numerical tables should be generated from immutable run configurations. Exploratory notebooks are not authoritative computational sources.

## Scientific nonclaims

The canonical scalar closed-FLRW implementation is a reference baseline, not yet the final SCPC theory. Its failure would be scientifically informative. New physical terms may be introduced only through a covariant action or explicitly declared effective equations, with dimensions, conservation identities, stability conditions, limiting cases, numerical verification, and predeclared rejection criteria.

## License and citation

Source code is distributed under the BSD 3-Clause License. Manuscript text and original project documentation are intended for release under CC BY 4.0 unless a file states otherwise. Cite the software using `CITATION.cff` and cite every observational release used in an analysis.
