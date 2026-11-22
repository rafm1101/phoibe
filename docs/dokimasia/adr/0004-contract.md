---
type: ADR
id: "0004"
title: "Contract"
status: implemented
date: 2026-11-22
superseded_by: null
---

# ADR-0004: Contract

**Depends on:** [ADR-0001](0001-domain-model.md), [ADR-0002](0002-profile-and-judge-are-separate-steps.md), [ADR-0003](0003-findings-are-located-score-is-informational.md).

## Context

A legacy layer is a list of `{name, params}` rules. Judgment parameters are mixed into measurement parameters, there is no aspect level, and the only version is the config's `version`. Tuples such as `(0, 60)` do not survive YAML or JSON (legacy defects 6, 8, 9). The real contracts are authored by the user after the technical requirements TR6 or TR10, from fragments for single layers.

## Decision

- The contract is a serializable (YAML or JSON), versioned document authored by the user. A tool may later derive a skeleton from a profile, but the contract is never generated implicitly.
- Structure: contract (id, version, domain) → layers. Each layer has two separate sections.
  - **Measures** have an id, an optional signal, a rule and its parameters, and no thresholds.
  - **Checks** have an id, the one measure they read, which value of its result they read, one threshold band or two (good, acceptable), a **severity** (critical, warning or info) and a reference to the source clause.
  - Results of two measures are never merged, even when both describe the same signal ([ADR-0010](0010-serialization-and-odcs-alignment.md)).
- The clause reference is free text, for example the document and section of TR6 or TR10. The cited documents are fixed, but their internal structure differs, so no structure is imposed.
- Layer names and count come from the contract. The medallion names (raw, bronze, silver, gold) are a convention. The gold contract defines readiness for further processing; that processing itself is outside this package.
- The signal catalog is referenced by the contract, shared by all layers and joint across combined sources ([ADR-0001](0001-domain-model.md)).
- A parameter may refer to a subject attribute instead of a fixed number, so one contract serves turbines of different types. The form is fixed: `{subject: <attribute>, factor: <number>, offset: <number>}`, read as a dataclass `SubjectRef(attribute, factor=1.0, offset=0.0)`. It needs no parser, serializes as plain data, and can be checked at load: the attribute must exist in the subject specification. An expression syntax is only worth adding when a real contract needs two attributes in one limit.
- Rule parameters are validated against a typed parameter model per rule when the contract is loaded, so an error appears at load time, not mid-run.
- Presentation options such as `locale` are not contract content.
- The registry maps one rule name to its class and its parameter model. There is a single source for names.
- The schema is plain dataclasses with explicit validation at load. There is no new dependency.

### Example: a limit that depends on the turbine

A bronze power limit is a share of the rated power. With rated power 5,600 kW, the following resolves to 5,880 kW. The profile records the resolved value and where it came from ("max 5,880 kW = 1.05 × rated_power_kw 5,600 kW"), so the judgment still never reads the dataset or the subject.

```yaml
measures:
  power_limits:
    rule: ranges
    signal: power
    params:
      min: 0
      max: {subject: rated_power_kw, factor: 1.05}
checks:
  power_within_range:
    measure: power_limits
    value: share_out_of_range
    severity: critical
    clause: "free text, e.g. the TR section"
    good: {max: 0.001}
    acceptable: {max: 0.01}
```

## Consequences

- `required_variable` becomes a check over a list of signals.
- The contract id and version appear in every verdict and attestation ([ADR-0006](0006-attestation-and-lineage.md)).
- Severity is kept, but it moves from the rule's constructor to the check, where the contract states what a violation does to the gate ([ADR-0005](0005-gate-is-a-policy-over-verdicts.md)). One measurement can then serve a critical check in one contract and an informational one in another.

## Implementation

`phoibe.dokimasia.contract` (schema, build, judge, limits), `phoibe.dokimasia.adapters` (YAML and JSON reading), `phoibe.dokimasia.compose` (parameter validation).

Gap: the bronze contract `contracts/wind-scada-10min.yaml` (version 0.1.0) is a draft. Its severities and bands are proposals measured on generator data; they wait for the technical requirements (see [open questions](../open-questions.md)). Clause references stay free text.
