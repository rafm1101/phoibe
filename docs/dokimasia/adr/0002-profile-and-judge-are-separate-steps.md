---
type: ADR
id: "0002"
title: "Profile and judge are separate steps"
status: implemented
date: 2026-11-22
superseded_by: null
---

# ADR-0002: Profile and judge are separate steps

**Depends on:** [ADR-0001](0001-domain-model.md).

## Context

Each legacy rule measures, applies thresholds and awards points in one call. The validation mode only toggles the gate; the rules never see it. As a result `data_gaps` ignores its thresholds and `ranges` never judges (legacy defects 2 and 8). The intended loop is: measure, then judge, and after a fix measure and judge again.

## Decision

- `profile` runs the layer's measurements with the contract's measurement parameters and returns a **Profile**: summaries and findings per aspect. It returns no verdicts.
- `judge` evaluates the layer's criteria on a profile and returns **Verdicts**. It never reads the dataset.
- `validate` is `judge` after `profile`. There is no validation mode.
- A profile without a contract is allowed, with default parameters. That is the exploratory case.
- After every human fix the same two steps run again on the new dataset version.
- A measurement outputs whatever its criteria need; criteria that need row-level detail (for example "no frozen run longer than n steps") are expressed on findings. `judge` never goes back to the data.

## Consequences

- A measurement yields typed results, not statuses. "Not checked" moves to the judgment and means "not evaluable, because the signal is missing"; it can no longer pass silently ([ADR-0005](0005-gate-is-a-policy-over-verdicts.md)).
- A person can change a threshold and re-judge an existing profile in a notebook without re-measuring.
- Measurement kernels are pure functions on arrays, so a notebook experiment becomes a rule without a rewrite.
- Measurements are tested against the generator, criteria against hand-made profiles ([ADR-0009](0009-testing-strategy-generator-as-oracle.md)).

## Implementation

`phoibe.dokimasia.core` (profile, measurement), `phoibe.dokimasia.contract.judge_layer`, `phoibe.dokimasia.compose` (`profile_layer`, `validate_layer`). Judge never reads data. No known gaps.
