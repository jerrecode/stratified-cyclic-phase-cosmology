# Stratified Cyclic Phase Cosmology

**SCPC** is a research-software and manuscript repository for constructing, verifying, simulating, and empirically testing covariant cosmological models whose state space contains distinguishable phase strata and whose background solutions may exhibit nonsingular cyclic evolution.

The repository deliberately separates four structures that are often conflated in speculative cosmology:

1. Lorentzian spacetime geometry;
2. phase or field-space stratification;
3. thermodynamic state-space geometry;
4. numerical discretization.

The initial implementation is a scientifically conservative baseline. It provides a closed-FLRW background with a canonical stratification field, explicit stress-energy conservation, Friedmann-constraint diagnostics, event-based bounce and turnaround detection, comparator expansion histories, a machine-readable scientific-data release manifest, and a modular LaTeX paper. It does **not** claim that a viable cyclic solution or an observational detection has already been established.

## Scientific questions

The project is organized around falsifiable questions:

- Does a declared SCPC action admit nonsingular bounces and turnarounds?
- Are recurrent trajectories stable under homogeneous and inhomogeneous perturbations?
- Is the effective theory free of ghost, gradient, and strong-coupling pathologies?
- Does it recover general relativity and standard cosmology in a controlled limit?
- Does it predict observables distinguishable from widely used non-cyclic and cyclic models?
- Are apparent spectral or topological features invariant under solver, resolution, and sampling changes?

## Repository map

```text
src/scpc/                 Installable Python package
configs/                  Versioned model, solver, and comparison configurations
data/manifest/            Scientific release manifest and JSON Schema
scripts/                  Reproduction entry points
results/                  Generated run products; not committed by default
tests/                    Analytical, conservation, schema, and regression tests
paper/                    Modular LaTeX manuscript, figures, tables, and paper-local data
workflows/                Reproduction and release workflow descriptions
docs/                     Architecture and scientific-scope documentation
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

Archive-query, TAP, Globus, and asynchronous products intentionally produce explicit instructions rather than pretending to be static files.

## Model comparison

```bash
scpc compare configs/comparison/background_models.yaml \
  --output results/model_comparison
```

All curves are evaluated on the same configured scale-factor grid. Dimensional and dimensionless quantities are never mixed silently.

## Verification

```bash
pytest
ruff check .
mypy src
```

## Paper

```bash
make paper
```

All paper figures and tables should be generated from immutable run configurations. Exploratory notebooks are not authoritative computational sources.

## Scientific status

The repository is at **framework/bootstrap** status. The canonical scalar closed-FLRW implementation is a reference baseline, not yet the final SCPC theory. New physical terms must be introduced through a covariant action or an explicitly declared effective equation set, accompanied by dimensional analysis, conservation checks, stability conditions, limiting cases, and numerical verification.

## Licensing

Source code is distributed under the BSD 3-Clause License. Manuscript text and original project documentation are intended for release under CC BY 4.0 unless a file states otherwise. External datasets retain their original licenses and citation requirements.
