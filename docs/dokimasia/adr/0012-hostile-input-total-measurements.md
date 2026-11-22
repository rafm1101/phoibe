---
type: ADR
id: "0012"
title: "Hostile input, every measurement is total"
status: partly-implemented
date: 2026-11-22
superseded_by: null
---

# ADR-0012: Hostile input, every measurement is total

**Depends on:** [ADR-0002](0002-profile-and-judge-are-separate-steps.md), [ADR-0005](0005-gate-is-a-policy-over-verdicts.md).

## Context

The package is to expect everything of data. Some of this existed in the legacy package: the validator turns any crash into an error result, and several rules cope with empty frames, NaN and single rows. The rest was scattered. Timestamps are parsed with `errors="coerce"`, so unparseable values silently become missing, and format guessing falls back to parsing element by element. Detection reads columns only, and values are assumed numeric. A crashed rule is recorded as critical whatever its configuration (legacy defect 10). The demonstration notebook adds a case: the time axis has no time zone while the site is in Europe/Berlin, so the form of the time axis is something to measure, not to assume.

## Decision

- **A measurement is total.** For any input it returns a result or a typed outcome, `not evaluable` with a reason. It never raises to the caller and never coerces silently.
- **Coercion is itself a measurement.** When text is read as numbers or timestamps, the count and the location of values that could not be read are findings. They are not dropped rows.
- **Structure comes first.** The first measurement of the raw layer describes what arrived: which signals were found, by column or by index; the type of each; the form of the time axis (index or column, naive or aware, share unparseable); duplicate column names; empty and constant columns. Every later measurement declares the structure it needs, and without it the outcome is `not evaluable`, which the gate treats like a failure for a critical check ([ADR-0005](0005-gate-is-a-policy-over-verdicts.md)).
- **Failures are contained.** One failing measurement does not stop the others. An unexpected exception becomes an `error` outcome carrying the measurement id and the exception type, and the severity comes from the check, not from a default. An expected problem such as a missing column gives `not evaluable`; only an unexpected exception gives `error`.
- **Limits are optional.** A contract may declare row count and memory limits, and there is no default. They are not expected to be reached. Beyond a declared limit the outcome is `not evaluable` instead of a hang.
- **Detecting is not repairing.** Sentinel values such as -9999, comma decimals and mixed units are findings in the raw layer. Only the deterministic transforms of [ADR-0011](0011-standardizing-transforms.md) change data.

## Consequences

Totality is tested with a hostile corpus: an empty frame, one row, all missing, text in numeric columns including comma decimals, infinities, mixed time zones, unsorted and duplicate timestamps, epoch numbers, duplicate column names and a large frame. The corpus runs through every registered measurement and asserts three things: no exception, a valid result, and never a pass when nothing could be evaluated. Property-based testing with Hypothesis would find more, but it is a new dependency.

## Implementation

`phoibe.dokimasia.core.run_measure` is the single `except Exception` in the package.

Gaps:
- Declared row and memory limits are not implemented; `limit_exceeded` exists only as a reason.
- Only `ranges` counts the values it could not read (`n_unreadable`). The other measures that read numbers treat unreadable text as missing without counting it, which goes against "never coerces silently".
- Measuring the structure first is partial.
