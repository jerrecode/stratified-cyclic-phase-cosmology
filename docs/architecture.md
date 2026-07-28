# Architecture

The package follows a theory-to-observable dependency direction:

```text
theory -> model -> numerics -> diagnostics -> observables -> inference -> paper assets
```

`theory/` contains equations and physical definitions. `models/` binds those definitions to parameterized model families. `numerics/` integrates equations without changing their physical meaning. `diagnostics/` checks constraints, conservation, events, and numerical validity. `visualization/` consumes stored result datasets and never becomes the sole source of a numerical result.

External data are registered in `data/manifest/releases.yaml`. The manifest records whether a product is a direct file, Git repository, asynchronous archive request, TAP query, Globus collection, or manual/interactive archive selection. A product is not marked automatically downloadable unless the access method is machine-resolvable.

Paper figures are generated into `paper/generated_figures/`; they are not manually edited copies of exploratory plots. Paper-local tables and small derived datasets are stored separately from the external archive cache.
