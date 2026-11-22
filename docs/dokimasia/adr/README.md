# Architecture decision records

One file per decision, named `NNNN-short-title.md`. Read them in order: each depends only on those above it.

| ID | Title | Status |
|---|---|---|
| [0001](0001-domain-model.md) | Domain model | implemented |
| [0002](0002-profile-and-judge-are-separate-steps.md) | Profile and judge are separate steps | implemented |
| [0003](0003-findings-are-located-score-is-informational.md) | Findings are located; the score is informational | implemented |
| [0004](0004-contract.md) | Contract | implemented |
| [0005](0005-gate-is-a-policy-over-verdicts.md) | Gate is a policy over verdicts | implemented |
| [0006](0006-attestation-and-lineage.md) | Attestation and lineage | implemented |
| [0007](0007-package-structure-and-dependency-direction.md) | Package structure and enforced dependency direction | implemented |
| [0008](0008-results-are-the-audit-trail.md) | Results are the audit trail; notebook-first API | implemented |
| [0009](0009-testing-strategy-generator-as-oracle.md) | Testing strategy, the generator as oracle | partly-implemented |
| [0010](0010-serialization-and-odcs-alignment.md) | Serialization and ODCS/ODPS alignment | implemented |
| [0011](0011-standardizing-transforms.md) | Standardizing transforms between layers | implemented |
| [0012](0012-hostile-input-total-measurements.md) | Hostile input, every measurement is total | partly-implemented |

## Header

Every ADR starts with this front matter:

```yaml
---
type: ADR
id: "NNNN"
title: "Short title"
status: proposed
date: 2026-01-31
superseded_by: null
---
```

`date` is the date of the last status change. `superseded_by` holds the id of the replacing ADR, or `null`.

## Lifecycle

`proposed` → `active` → `partly-implemented` → `implemented`, and from any state after `proposed` to `superseded` or `retired`.

| Status | Meaning |
|---|---|
| `proposed` | Written, not yet accepted. |
| `active` | Accepted and binding; nothing or little is built yet. |
| `partly-implemented` | Accepted and binding; the code realizes it in part. The gaps are listed under "Implementation" in the ADR and in the [open questions](../open-questions.md). |
| `implemented` | Accepted and fully realized in the code. Known deviations, if any, are listed under "Implementation". |
| `superseded` | Replaced by another ADR; `superseded_by` names it. |
| `retired` | No longer applies, with no replacement. |

## Body

Each ADR has the sections **Context**, **Decision**, **Consequences**, optionally **Open points**, and **Implementation** (where the code lives and which gaps remain).
