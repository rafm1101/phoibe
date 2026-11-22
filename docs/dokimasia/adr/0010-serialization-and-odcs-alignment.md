---
type: ADR
id: "0010"
title: "Serialization and ODCS/ODPS alignment"
status: implemented
date: 2026-11-22
superseded_by: null
---

# ADR-0010: Serialization and ODCS/ODPS alignment

**Depends on:** [ADR-0003](0003-findings-are-located-score-is-informational.md), [ADR-0004](0004-contract.md), [ADR-0006](0006-attestation-and-lineage.md).

## Context

People are the target readers, nothing depends on the legacy formats, and the Open Data Contract Standard (ODCS) and the Open Data Product Specification (ODPS, the Bitol standard tied to ODCS) are to serve as the framework. The legacy `ProfileExporter` writes an "ODCS-style" fragment and `QualityAttestationWriter` an "ODPS-style" attestation, both with custom keys. The same report is serialized three times with different key styles, the round trip is lossy (legacy defect 5), and nothing validates the output against a schema.

## Decision

- One canonical, versioned, lossless schema per artifact: contract, profile (measurements), verdicts (checks and gate result), attestation. Every file carries its schema version; one `schema_version` covers all artifact schemas and a file of another version is refused.
- One mapper per artifact in `adapters`. Exporters become mappers.
- Reports are keyed by subject, layer and dataset fingerprint, not by turbine id.
- A **run index** lists many runs (subject, layer, fingerprint, outcome). It indexes stored artifacts and does not copy them.
- Timestamps with a time zone are stored in UTC with offset. `NaN` is handled explicitly, because JSON cannot hold it. Lists in subjects and transform parameters read back as tuples.
- YAML is the default, for readability. JSON is available.
- Layer names use letters, digits, `_` and `.`, because `-` separates the parts of a file name.
- Alignment follows the specification spike below: the contract has its own dataclass schema with an ODCS-shaped envelope and a one-way export to ODCS. Profile, verdicts and attestation have their own schemas, because neither standard defines results. Contracts are referenced by id and version, as ODPS ports do.

### Layout of the three documents

Names are keys. The id is the key, and `rule` defaults to the key, so the common case reads like the original fragments. A second use of the same rule gets its own key and an explicit `rule:`. A mapping cannot hold a key twice, so the uniqueness of ids comes for free. Thresholds below are illustrative.

```yaml
# contract: what to measure, and, separately, what to check
layers:
  bronze:
    measures:
      wind_speed_limits:
        rule: ranges              # the rule defaults to the key, so a single use can omit it
        signal: wind_speed
        params: {min: 0, max: 60}
      essential_range:
        signal: wind_speed
      curtailment_levels:
        signal: power_kw
    checks:
      no_wind_speed_outliers:
        measure: wind_speed_limits
        value: n_out_of_range
        severity: critical
        good: {max: 0}
        acceptable: {max: 10}
        clause: "free text"
      full_load_matches_rating:
        measure: curtailment_levels
        value: highest
        severity: warning
        good: {min: {subject: rated_power, factor: 0.99}, max: {subject: rated_power, factor: 1.01}}

# profile: measured, no judgment
layer: bronze
summary:                           # one headline per measurement
  wind_speed_limits: {n_out_of_range: 3}
  essential_range: {low: 0.367, high: 20.085}
  curtailment_levels: {highest: 5600.0}
measurements:
  wind_speed_limits:
    signal: wind_speed
    result: {n_out_of_range: 3}
  essential_range:
    signal: wind_speed
    result: {low: 0.367, high: 20.085}
  curtailment_levels:
    signal: power_kw
    result: {levels: [0.0, 3200.0, 5600.0], highest: 5600.0}

# verdicts: judged from the profile alone
summary:
  by_verdict: {good: 1, acceptable: 1}
  by_severity: {critical: 1, warning: 1}
  gate: {outcome: conditional, conditions: [no_wind_speed_outliers]}
checks:
  no_wind_speed_outliers: {measurement: wind_speed_limits, value: 3, verdict: acceptable, effect: conditional}
  full_load_matches_rating: {measurement: curtailment_levels, value: 5600.0, verdict: good, effect: pass}
```

Three rules keep the sections apart:

- **A measurement is one entry**, keyed by an id that is unique in its layer. Its result is never merged with another measurement's. Two measures on one signal are two entries.
- **A check reads exactly one value of exactly one measurement.** Two checks may read the same measurement. A check that reads several measurements is compensation, which is deferred ([ADR-0005](0005-gate-is-a-policy-over-verdicts.md)); the separate section is where it would be added.
- **The profile holds no verdicts.** The verdicts file holds only the value a check read and its outcome, and refers to the measurement by id.

The contract is one file with measures and checks as two sections, because they are authored together and a check refers to a measure by key. The profile, the verdicts and the attestation are one file each. Keeping profile and verdicts apart lets the same profile be judged again against a new contract version without measuring again. Merging them later is a mechanical change.

Both files open with a summary: the profile with one headline value per measurement, the verdicts with counts by verdict and by severity, the gate outcome with its conditions, and the informational score.

### Findings of the specification spike

The spike read the ODCS data quality page (ODCS v3.2.0 at the time) and the ODPS product pages (v1.1.0). It did not check the JSON Schema. Two standards share the acronym ODPS; the Bitol one is meant here.

| Dokimasia element | In ODCS or ODPS | Fit |
|---|---|---|
| Contract id, name, version, status | ODCS `id`, `name`, `version`, `status`, `apiVersion`, `kind` | Direct |
| Signal in the catalog | ODCS schema property with `logicalType` and `required` | Direct for presence; quantity, statistic and unit go in `customProperties` |
| Check with a threshold and severity | ODCS quality rule with `dimension`, `severity`, an operator such as `mustBeGreaterOrEqualTo`, and `unit` | Partial: one pass condition per rule |
| Good and acceptable bands, conditional outcome | Nothing | Missing; kept in `customProperties` |
| Gap, availability and curtailment measures | Library metrics cover only nulls, missing and invalid values, duplicates and row count | Through `type: custom` with `engine` and `implementation` |
| Clause reference (TR6, TR10) | `authoritativeDefinitions` takes a link | Free text fits `customProperties` |
| Layer | ODPS output port with `contractId` and `version` | Plausible; not needed now |
| Profile, verdicts, gate result, attestation | ODCS defines no result field; ODPS ports carry no quality or run fields | Missing; own schema |
| Dataset fingerprint and provenance | Nothing | Missing; own schema |

What follows:

- **Contract.** The own dataclass schema is the source, with an ODCS-shaped envelope and a one-way export. The export is lossy by design: the good and acceptable bands and the clause text travel as `dokimasia.*` custom properties, so an ODCS tool can read the file but cannot run the time-series rules. The export puts no numeric operator on a rule, because ODCS refuses operators on custom rules. Each check becomes one quality rule that names its measure by id in `implementation`; the measures travel as a `dokimasia.measures` custom property.
- **Results** reference a contract by id and version, as ODPS ports do.
- **ODPS** has no object for an attestation. Treat it as a later option for describing layers as ports, not as a target.
- **Dependency.** The reference ODCS model is pydantic-based. The export writes plain dictionaries without it, and validating against it is an optional test.

## Consequences

- A round-trip test is a contract test ([ADR-0009](0009-testing-strategy-generator-as-oracle.md)).
- The attestation format of [ADR-0006](0006-attestation-and-lineage.md) is settled here.

## Implementation

`phoibe.dokimasia.adapters` (`artifacts`, `odcs`, `store`). The ODCS export is one-way; there is no import.
