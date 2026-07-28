# Scientific data manifest

`data/releases.yaml` is a curated, machine-readable registry of public cosmological releases relevant to parameter fitting, likelihood evaluation, validation, transfer-function construction, and comparative visualization.

The manifest records:

- provider, release identifier, date, license, citation, and documentation;
- access protocol and whether authentication or an interactive archive request is required;
- reproducible command templates where a stable direct endpoint exists;
- product selection and expected data scale;
- contained quantities, symbols, units, physical dimensions, shapes, and defining relations;
- intended use in SCPC and whether a product is core or optional.

It intentionally does not fabricate direct URLs for archives that require basket, TAP, Globus, or query-based selection. Such products are marked with the appropriate request method.

The initial registry is curated rather than exhaustive. New releases must pass schema validation and link review before use in a published run.
