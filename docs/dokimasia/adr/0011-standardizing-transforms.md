---
type: ADR
id: "0011"
title: "Standardizing transforms between layers"
status: implemented
date: 2026-11-22
superseded_by: null
---

# ADR-0011: Standardizing transforms between layers

**Depends on:** [ADR-0001](0001-domain-model.md), [ADR-0006](0006-attestation-and-lineage.md).

## Context

The demonstration notebook moves from raw to bronze with a transform. It renames detected columns to catalog names (in the run shown, only `power` becomes `power_kw`), and its text names deduplication and time zones, the latter skipped. Moving to the next layer is described as transformation next to profiling and validation. The notebook also builds a validation context by hand only to read the detected mapping. Cleaning and imputation stay with the human for now.

## Decision

- A **transform** is a deterministic, parameterized function from one dataset version to the next. Its description (name, parameters, software version) and the output fingerprint go into the provenance record ([ADR-0006](0006-attestation-and-lineage.md)).
- Two classes are distinguished.
  - **Standardizing** transforms are in scope, because the next layer's profile assumes standardized signals. Renaming to catalog names comes first. Localization of time to UTC follows, with deduplication only as far as localization needs it, which means resolving the repeated hour. Other duplicate rows are a finding, not a transform. Sorting and unit conversion are not planned for now.
  - **Cleaning and imputation** are out of scope; a human does them in the notebook, and the changed fingerprint is the only trace.
- A transform is not part of the gate. The gate judges the data after it. Transforms live in the `standardize` package, which imports `core` only ([ADR-0007](0007-package-structure-and-dependency-direction.md)). The signal mapping that detection produces is data in `core`, so renaming needs nothing from `rules`. A domain pack may add transforms of its own, and `advance` calls them.
- Signal detection runs once, at raw ([ADR-0001](0001-domain-model.md)). After the standardizing transform, later layers address signals by catalog name, with no detection and no ambiguity.
- A public function returns the detected mapping, so a notebook needs no hand-built context.
- **The repeated hour.** The transform converts to UTC and keeps the site time zone as a subject attribute. Daylight-saving duplicates disappear once the time axis is in UTC. If the raw timestamps are naive local time, the repeated hour must be disambiguated before the conversion, because pandas refuses to localize ambiguous times by default. It is resolved by order (`ambiguous="infer"`), which holds when the data is sorted and complete around the change. The rule applied is recorded in the provenance. When it cannot be applied, the transform stops with a finding and the outcome is not evaluable. It never guesses.

## Consequences

- Raw profiles run on unstandardized data and later ones on standardized data.
- Localizing time needs the site time zone from the subject specification.
- Real data may behave differently from the generator; this is for the reality check.

## Implementation

`phoibe.dokimasia.standardize` (`rename`, `time`, `chain`) and `phoibe.dokimasia.core.advance_dataset`. The scope (standardizing transforms in, cleaning out) is confirmed by the project owner.
