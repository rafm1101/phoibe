# 0008 Manifest, contract and versioning

Status: Accepted

## Decision

Each artifact carries one manifest. It starts with the `contract`, generated from the specification and identical for all runs of a version, followed by `meta`, `parameters`, `spatial_context`, `run` and `artifacts`. The keys are described in the `manifest` entry of the product definition.
TODO: drift test against the real manifest.

The product version is `version` in the definition. Breaking changes of the contract (removing or renaming a column, changing file or layer names, changing what a number means) raise the major version, new optional outputs the minor version, wording the patch version. The baseline is 1.1. The first release with the changes in the changelog is 2.0.0. The contract of each release is stored as a snapshot and compared (D-A3).

## Consequences

- Changing any output means: definition, manifest keys, changelog, version.
