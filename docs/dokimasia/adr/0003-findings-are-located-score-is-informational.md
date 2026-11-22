---
type: ADR
id: "0003"
title: "Findings are located; the score is informational"
status: implemented
date: 2026-11-22
superseded_by: null
---

# ADR-0003: Findings are located; the score is informational

**Depends on:** [ADR-0001](0001-domain-model.md), [ADR-0002](0002-profile-and-judge-are-separate-steps.md).

## Context

Aggregates were a first step so that a human could act; too much information would prevent action, but more detail is wanted. A legacy measurement returns counts and a free-form `details` dict, and the exporters read it by rule name and key (legacy defect 7). Summary statistics are to be kept, and scoring is informal.

## Decision

- A measurement returns **summaries** and **findings**: kind, signals, start, end, number of steps, magnitude (with unit).
- Findings are run-length intervals, never one per row. "Gap 10:20 to 11:10, 6 steps" is one finding.
- The profile shows summaries by default. Findings are one call away as a table sorted by size, plus a lazily built row mask per finding kind for plotting and fixing.
- The **score** is an informal summary derived from aspect verdicts (the share of criteria met; acceptable counts as met). It is displayed in the profile and never used by the gate.
- Points per rule, including ad hoc deductions such as "minus 2 for no timezone", are removed.

## Consequences

- Exporters read typed summaries by metric key, not by rule name and `details` key.
- Findings are also the interface through which the generator's ground truth is compared with detection ([ADR-0009](0009-testing-strategy-generator-as-oracle.md)): `Score(n_truth, n_flagged, n_hit)` gives recall and precision per rule.

## Open points

The order in which findings are listed (length, magnitude or both) and how many are shown by default.

## Implementation

`phoibe.dokimasia.core` (`Finding`, `build_finding_mask`, `scoring`). No known gaps.
