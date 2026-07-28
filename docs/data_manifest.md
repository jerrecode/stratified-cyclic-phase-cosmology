# Scientific data manifest

The manifest is a release-level registry, not a substitute for each collaboration's documentation. It records:

- release identity and scientific role;
- authoritative landing and documentation URLs;
- access method and request instructions;
- product formats and approximate scale;
- key variables, units, dimensions, coordinates, and relationships;
- whether the variable inventory is complete;
- license, citation, acknowledgement, and verification date;
- whether a product is suitable for fitting, validation, visualization, or historical comparison.

## Priority tiers

- `core`: compact likelihoods, covariances, chains, and summary measurements needed for parameter inference.
- `extended`: catalogs or maps useful for independent reanalysis and validation.
- `optional_large`: high-volume products not needed for the baseline pipeline.
- `context_only`: useful documentation or non-cosmology releases that must not be treated as fitting data.

## Dimensional vocabulary

Dimension exponents use SI base dimensions:

```text
M: mass
L: length
T: time
Theta: thermodynamic temperature
A: angle (treated as a declared semantic dimension)
1: dimensionless
```

For example, the Hubble parameter has `T^-1`; a power spectrum P(k) commonly has `L^3`; angular spectra C_ell may be dimensionless or carry temperature-squared units depending on convention.
