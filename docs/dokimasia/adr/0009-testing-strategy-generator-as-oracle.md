---
type: ADR
id: "0009"
title: "Testing strategy, the generator as oracle"
status: partly-implemented
date: 2026-11-22
superseded_by: null
---

# ADR-0009: Testing strategy, the generator as oracle

**Depends on:** [ADR-0003](0003-findings-are-located-score-is-informational.md).

## Context

The synthetic-data generator builds a seeded SCADA series and applies messup steps (`Freeze`, `Spike`, `RowDrop` and others). Each step knows which segments it corrupts and then discards that knowledge. The `incidence` and `level` arguments scale how many faults occur and how long they last. Before this work, no test used the generator and the declared pytest markers were unused.

## Decision

- Every messup step records **ground-truth labels**: fault kind, columns, start and end timestamps. The pipeline returns them next to the data (`MessUpPipeline.apply_labelled`; `apply` is unchanged).
- **Detection tests** compare findings with labels at interval level and report recall and precision per rule. Seeds are fixed, and `incidence` and `level` vary to show how detection degrades.
- **Criteria tests** run on hand-made profiles, with no data needed.
- **Contract tests** check that the contract, the registered rules and the schemas agree, and that every serialized artifact round-trips without loss.
- **Golden profiles** for one fixed seed catch unintended changes.
- The generator is tested as an asset. It fixes `CurtailmentNight` (hard-coded hours) and its copied docstring, documents whether the time axis is index or column, and marks unused steps (`ZeroOut`, `BernoulliStartSegments`) for use or removal.
- Defect tests for the legacy package live in `tests/characterization/` as `xfail(strict=True)` cases and shrink as defects are fixed.
- Real SCADA data is not used as a fixture for now.

## Consequences

- The fault coverage table is executable (`tests/unit_tests/dokimasia/rules/test_coverage.py`), and a new rule is accepted when it detects its injected fault with the agreed recall and precision. See [architecture](../architecture.md#fault-coverage).
- The `level` knob gives a principled way to calibrate thresholds, but only on generator data. Real-data floors need the reality check ([open questions](../open-questions.md)).
- The notebook run with incidence 7 and level 5, which the legacy bronze gate passed with 30 of 30 points, fails the draft bronze contract's gate on every seed, by measurement.

## Open points

Detection targets are decided per rule when the rule is written.

## Implementation

Labels, the executable coverage table, and criteria tests exist.

Gaps:
- Golden profiles for a fixed seed do not exist.
- The generator is not yet cleaned up: the hard-coded night hours of `CurtailmentNight` and the unused steps remain. The cleanup is a separate package of work.
