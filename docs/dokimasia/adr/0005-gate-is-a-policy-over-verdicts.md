---
type: ADR
id: "0005"
title: "Gate is a policy over verdicts"
status: implemented
date: 2026-11-22
superseded_by: null
---

# ADR-0005: Gate is a policy over verdicts

**Depends on:** [ADR-0002](0002-profile-and-judge-are-separate-steps.md), [ADR-0004](0004-contract.md).

## Context

The legacy gate means "no CRITICAL rule failed or errored". It fails open when a CRITICAL rule is `NOT_CHECKED` or when no rules are configured (legacy defect 4), and a crashed rule is always recorded as CRITICAL (defect 10). The gate is meant to be made up of the aspects profiled, and a violated rule may have exceptions through other measurements or even other time series. There is no complete overview of those exceptions yet.

## Decision

- A **gate** is a policy object. It takes the verdicts of a layer's checks and returns an outcome (pass, conditional or fail) with its reasons.
- A check has a verdict (good, acceptable, bad, not evaluable or error) and a **severity** that the contract states for it. There is no default severity; the contract must say.
- The first policy maps verdict and severity to an effect and takes the worst effect over all checks (table below). The acceptable band of a two-band check is the conditional pass. Conditions are recorded in the attestation.
- "Not evaluable" and "error" never pass silently. A missing signal or a crashed measurement follows the check's severity, not a fixed CRITICAL.
- A layer without checks is a contract error, not a pass.
- The policy is the extension point. A later compensation policy can read other checks' verdicts, other measurements or other datasets, because profiles and verdicts are plain, addressable objects.

Effect of a check on the gate, in the first policy:

| Verdict | Critical | Warning | Info |
|---|---|---|---|
| Good | pass | pass | pass |
| Acceptable | conditional | pass, noted | pass |
| Bad | fail | conditional | pass, noted |
| Not evaluable or error | fail | conditional | pass, noted |

## Consequences

- `LayerGateKeeper` and `LayerGateFailureError` of the legacy package merge into one gate result with an explicit `raise_if_failed()`. The decision is returned, not hidden behind a flag.
- The vocabulary no longer uses "warning" for both a status and a severity: verdicts say acceptable, severities say warning.

## Open points

What a compensation looks like (other measurements, other time series, a recorded human waiver) stays deferred; it is a [non-goal](../goals.md) for now.

## Implementation

`phoibe.dokimasia.core.gate` (`GatePolicy`, `WorstEffectPolicy`, `GateResult`, `get_effect`). Compensation is deferred.
