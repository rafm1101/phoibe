# 0003 The product specification is the single source

---
type: ADR
id: "0003"
title: "The product specification is the single source"
status: active
date: 2026-10-23
superseded_by: null
---

## Context

Names of columns, files and layers, severities and fixed parameters were spread over several modules and a `Keys` class.

## Decision

The product definition is a nested dict, parsed into frozen types (`ProductSpecification`). Unknown keys are errors. Table and column names, filenames, layers, severities, check thresholds and fixed parameters are looked up there and nowhere else. The contract is generated from it. `pipelines/requirements.py` verifies when the analyzer is created that
the specification contains everything the code uses, and reports all gaps at once.

## Consequences

- Removing a column is an edit of the definition. Adding one needs a builder that provides the quantity.
- Drift tests are planned: manifest keys (D-A1), written files (D-A2), contract snapshots per release (D-A3).
- TODO: move the validation of cross references from `from_dict` into `ProductSpecification.__post_init__`, so it holds however a specification is built.
