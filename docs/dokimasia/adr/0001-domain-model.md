---
type: ADR
id: "0001"
title: "Domain model"
status: implemented
date: 2026-11-22
superseded_by: null
---

# ADR-0001: Domain model

**Depends on:** none.

## Context

Rules, contract and exporters of the legacy package (`phoibe.layered`) refer to "variables" by name (`power_kw`, `wind_speed`). The synthetic-data generator emits `_min` and `_max` columns for the same quantity and puts `datetime` in the index. The legacy code uses `turbine_id` throughout, and its detector silently takes the first matching column. In the demonstration notebook the patterns for date and time both match the datetime column, so "15/15 variables detected" counts one column three times (legacy defects 11 and 13, see [legacy review](../legacy-review.md)).

## Decision

- A **subject** (id, device type and a specification such as rated power, rotor diameter and site time zone) replaces `turbine_id`. One run covers one subject and one dataset.
- A **dataset** is an immutable time series plus a simple fingerprint. The time axis is a property of the dataset, whether it was an index or a column in the source. The loader normalizes it; no rule searches for it.
- A **signal** is a quantity and a statistic (mean, min, max, std) with a unit and, when sources are combined, a measurement point such as source or height. Rules ask for signals, never for column names.
- The **catalog** of signals, units and detection patterns is shared by all layers of a contract and is joint when data sources are combined. It belongs to a domain pack; wind is the first. Detection reports ambiguity when several columns match.
- Detection runs once, at the raw layer. A standardizing transform ([ADR-0011](0011-standardizing-transforms.md)) then renames columns to catalog names, so later layers address signals directly.
- Layer and aspect are identifiers in the model; their content is defined by the contract ([ADR-0004](0004-contract.md)).

## Consequences

- Wind specifics (power, wind speed, rotor speed, pitch angle, curtailment) live in the wind pack. A solar or other time-series domain needs a new pack, not a core change.
- Legacy defect 11 (time axis in the index is not found) disappears.
- Rule authors learn one more concept.

## Open points

- `Subject.specification` must stay a plain, generic mapping. If wind terms leak into the core model, the domain switch is no longer cheap.
- With a joint catalog, a dataset may carry columns from several sources, and the fingerprint then has to say which. Not yet exercised.

## Implementation

`phoibe.dokimasia.core` (`Subject`, `Dataset`, `Signal`), `phoibe.dokimasia.wind.create_catalog`, `phoibe.dokimasia.rules` (detection). Detection reports ambiguity and never picks a column. No known gaps.
