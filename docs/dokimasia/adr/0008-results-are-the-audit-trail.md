---
type: ADR
id: "0008"
title: "Results are the audit trail; notebook-first API"
status: implemented
date: 2026-11-22
superseded_by: null
---

# ADR-0008: Results are the audit trail; notebook-first API

**Depends on:** [ADR-0002](0002-profile-and-judge-are-separate-steps.md), [ADR-0003](0003-findings-are-located-score-is-informational.md).

## Context

The interface is the notebook and there is to be no black box. The legacy logging package (about 350 lines) writes console, rotating-file and JSONL audit logs. Importing it creates `logs/` in the working directory. Every `get_logger` call rebuilds handlers, and `propagate=False` hides the messages from the host (legacy defect 12). In the demonstration notebook each run prints about ten INFO lines that the report already holds.

## Decision

- **Profile, verdicts, gate result and attestation are the audit trail.** They are inspectable objects with a notebook-friendly `repr` and a `to_frame()` for summaries and findings. Persistence goes through [ADR-0010](0010-serialization-and-odcs-alignment.md).
- The package uses the standard `logging` module the way a library should: module loggers, a `NullHandler`, no files, no directories, no `propagate` changes. Messages are for diagnostics only. The notebook configures logging if it wants output.
- The duration of each measurement is recorded in the profile, not only in log lines.
- The legacy `JSONFormatter`, `ContextualLogger`, `RuleExecutionTracker` and handler factories are removed.
- The public API stays small: load a contract, profile, judge, validate, attest, advance.

## Consequences

- The audit trail exists only for what the user saves. That is explicit: profiles and attestations are written deliberately, not as a side effect.
- About 350 lines of code disappear.

## Open points

Whether plotting helpers for findings belong in the package as an optional extra or stay in notebooks. Deferred.

## Implementation

`repr` and `to_frame` on the core objects, durations in the profile, `NullHandler` on the package logger. No known gaps.
