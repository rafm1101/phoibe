# Goals, non-goals, quality attributes and working model

`phoibe.dokimasia` assists data preparation. It tells a human what is wrong with a wind-turbine time series, and it decides whether the data may pass into the next quality layer. Terms are defined in the [glossary](glossary.md); the decisions behind this page are the [ADRs](adr/).

## Goals

| ID | Goal | Meaning |
|---|---|---|
| G1 | **Profile** | Describe a dataset per quality aspect so that a human knows what to fix. Measurement only, no verdict. |
| G2 | **Validate** | Judge a profile against a contract. Each layer ends in a gate whose criteria are made of the aspects profiled in that layer. |
| G3 | **Medallion** | Layers (raw, bronze, silver, gold) are quality stages whose gates address different aspects. The profiler of layer N+1 waits behind the gate of layer N. Gold is the state in which data is ready for further processing; that processing (long-term correlation, turbine assessment) is beyond this project. |
| G4 | **Human in the loop** | A human acts on the profile. After the fix, the same layer is measured and judged again. |
| G5 | **Actionable and transparent** | Findings say where; summaries say how much. Every verdict traces to a measurement, a parameter and a contract clause. Used in notebooks, no black box. |
| G6 | **Contract-driven** | The contract is user-authored (for example after TR6 or TR10) and versioned. It defines rules and parameters per layer. |
| G7 | **Extensible** | New rules and aspects are added without touching the core. |
| G8 | **Domain switch tractable** | Wind now; solar or other time series later without redesign, if that is cheap. The signal catalog is shared by all layers and is joint when data sources are combined. |

## Non-goals

- Automated cleaning or imputation. It is further away; the design must not block it. The standardizing transforms of [ADR-0011](adr/0011-standardizing-transforms.md) are not cleaning.
- Analytics on gold data, such as long-term correlation or turbine assessment.
- A pipeline orchestrator or scheduler. Layers are linked by gates and attestations, not run by this package.
- Fleet-level checks in the first version. They come as a layer after the single-turbine layers.
- A command-line interface. The interface is the notebook.
- Compensation rules, meaning exceptions to a violated criterion justified by other measurements or other time series. Not designed now; the gate design leaves room for them ([ADR-0005](adr/0005-gate-is-a-policy-over-verdicts.md)).
- Backward compatibility with the legacy report formats, and compliance with ODCS or ODPS as such. Both are design inputs, not constraints ([ADR-0010](adr/0010-serialization-and-odcs-alignment.md)).
- Real SCADA data as test fixtures, for now.

## Quality attributes, in priority order

1. **Traceability.** A verdict leads back to its measurement, its parameters and its contract clause.
2. **Reproducibility.** The same data and contract version give the same profile and verdict.
3. **Notebook usability.** Results are objects a person can inspect, filter and plot.
4. **Testability.** Detection is tested against known, injected faults ([ADR-0009](adr/0009-testing-strategy-generator-as-oracle.md)).
5. **Extensibility.** See G7.

Performance is not a driver. In the demonstration notebook, one year of ten-minute data (51,122 rows) takes at most 89 ms per rule (legacy measurement).

## Boundaries

One run covers one turbine and one dataset. The domain is wind. Python is 3.13 only (`requires-python`). The code lives as a subpackage of `phoibe`, a library that also carries geospatial dependencies. The legacy package `phoibe.layered` stays untouched.

## Working model

One layer is a loop on one dataset version; layers are chained only by attestations.

```
Dataset ──profile──▶ Profile ──judge──▶ Verdicts ──gate──▶ pass | conditional | fail
   ▲                                                              │
   └───────────── human fix (new dataset version) ◀───────────────┘
                                                                  │ pass or conditional
                                                    attestation ──▼── advance (+ transform) ──▶ next layer
```

- Profile and judge are separate steps. Judge can run again on an existing profile under a changed threshold or contract version. Every fix re-enters at the dataset and produces a new version ([ADR-0002](adr/0002-profile-and-judge-are-separate-steps.md)).
- The same layer measures and judges again after each human interaction; earlier layers are not re-run ([ADR-0006](adr/0006-attestation-and-lineage.md)).
- The contract feeds both steps: measurement parameters to profile, checks with their severity to judge ([ADR-0004](adr/0004-contract.md)).
- The gate is a policy over verdicts that returns pass, conditional or fail. It is the extension point for compensation ([ADR-0005](adr/0005-gate-is-a-policy-over-verdicts.md)).
- Between layers a standardizing transform may run; it is recorded as provenance with the attestation ([ADR-0011](adr/0011-standardizing-transforms.md)).
- A later fleet layer takes several single-turbine attestations as input.
