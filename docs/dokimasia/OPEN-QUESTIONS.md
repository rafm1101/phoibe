# Open questions

The project closes before the reality check. The reality check against real turbine data (the TR check) is a new project that builds on `phoibe.dokimasia`. This page lists what is still open, who can close it, what it blocks, and what the code does until then. Everything else is decided; see [Closed](#closed).

How to read it: **Needs** says what closes the question (`TR` the technical requirements, `experts` domain experts, `users`, `data` real turbine data, `owner` a decision by the project owner). **Default today** is the behavior in the code and in the draft contract `contracts/wind-scada-10min.yaml`.

## Open: needs a person or real data

| ID | Question | Needs | Blocks | Default today |
|---|---|---|---|---|
| OQ-01 | **Frozen values.** Which signals take part in a held state, and for how many steps does it count? | TR | The bands of the `no_frozen_state` check | `frozen_runs` has no default for `signals` or `min_steps`; every contract must name both. The draft uses power, wind speed, rotor speed and pitch angle with `min_steps: 2`, a placeholder. A frozen state must be multi-signal (the same state of all named detectors); a single signal held is not flagged because a stopped turbine does that too. |
| OQ-02 | **Spikes.** What exactly is a spike? | experts | A spike rule | Nothing is built. The generator's `Spike` step is uncovered in the coverage table. |
| OQ-03 | **Missing turbine numbers.** Do rated power, rated wind speed, cut-in and cut-out speed ever lack in practice, and how do users deal with it? | experts, users | Subject-reference checks on turbines with incomplete specifications | A missing subject attribute gives an `error` verdict, so the gate never passes silently. The attributes are `rated_power_kw`, `rated_wind_speed_mps`, `cut_in_wind_speed_mps`, `cut_out_wind_speed_mps`. |
| OQ-04 | **Zeroing of power.** How do cut-in and cut-out speed enter the rule that finds a turbine zeroed while it should run? | users | A zeroing rule and its window | Nothing is built. The generator's `CurtailmentToZero` step is uncovered. |
| OQ-05 | **Conditional availability.** What is the context and the measurement of conditional availabilities? | users | A rule for it | Parked. Nothing is built. |
| OQ-06 | **Power and wind-speed limits.** Which limits does the TR set for power and for wind speed? | TR | The `ranges` checks of the bronze contract | Power −5 % to +105 % of rated power (the −5 % is an assumption) and wind speed 0 to 60 m/s, both marked "to be verified with the TR". Rotor speed and pitch angle have no range conditions (decided). `no_wind_speed_outliers` reads `share_out_of_range` with an acceptable band of 0.001, a placeholder. |
| OQ-07 | **Rounding tolerance for statistic order.** Does real data need a tolerance for "min ≤ mean ≤ max"? | data | Critical order checks on real data | `statistic_order` has `tolerance=0.0`, enough for generator data. The bronze checks are critical and tolerance-free, so rounding in real data would fail the gate. |
| OQ-08 | **Level method for power levels.** Which method finds curtailed power levels on real data, `count` or `kde`? | data | `full_load_reaches_rated_power` | `count` is the default. On generator data `count` misses short plateaus (recall 0.93); `kde`, the legacy method, flags one stopped-turbine row (precision 0.98, one false alarm on clean data). With `count`, real power that scatters by tens of kW may share no value among five rows, so the check would be bad on healthy data. |
| OQ-09 | **High-wind threshold and level defaults.** Does `min_wind_speed` equal the rated wind speed, and do the rounding to 10 kW and a minimum of 5 rows per level hold on real data? | data | `power_levels`, `night_limit` | `min_wind_speed` is `{subject: rated_wind_speed_mps}` in the draft; the legacy threshold of 14 m/s is not carried over. |
| OQ-10 | **Night hours for other sites, and where the time zone lives.** Which hours define the night, and which subject attribute carries the zone? | owner | Using `night_limit` for sites outside Germany | `night_limit` takes `time_zone` and local hours. The draft sets 22 to 6 in `Europe/Berlin`, kept at the same local hours across daylight saving time. The subject attribute `site_timezone` is not read yet, because the registry has no text-valued subject references. |
| OQ-11 | **Start, end and time zone of the time axis.** What start, end and zone should each layer have? | TR, users | A `time_span` check | `time_span` reports start and end as UTC epoch seconds and reads a naive axis as UTC. The draft checks only the span. |
| OQ-12 | **Recall and precision floors.** Which floors should the coverage table assert? | data | A strict coverage table | The table asserts only coarse floors from generator data (count: recall ≥ 0.9; kde: precision ≥ 0.95, at most one false alarm on clean data). Real floors come from the first measured values on real data. |
| OQ-13 | **Bronze contract bands.** Which severities and bands does the TR set? | TR | The agreement itself, with clause references | The contract is version 0.1.0, a draft. All bands are placeholders measured on generator data. |

## Open: design points that do not block anything

| ID | Question | Where | Default today |
|---|---|---|---|
| DP-01 | In which order, and how many, are findings listed by default? | [ADR-0003](adr/0003-findings-are-located-score-is-informational.md) | Sorted by size; no cap documented. |
| DP-02 | Should an opt-in re-check of lower gates exist? | [ADR-0006](adr/0006-attestation-and-lineage.md) | No. Gates are not cumulative. |
| DP-03 | Does a recorded human waiver belong in the attestation, and what does a compensation look like? | [ADR-0005](adr/0005-gate-is-a-policy-over-verdicts.md), [ADR-0006](adr/0006-attestation-and-lineage.md) | Deferred; the gate policy is the extension point. |
| DP-04 | Do packs become separate distributions? | [ADR-0007](adr/0007-package-structure-and-dependency-direction.md) | Subpackages of `phoibe`. |
| DP-05 | Do plotting helpers for findings and masks belong in the package? | [ADR-0008](adr/0008-results-are-the-audit-trail.md) | No; they need their own design step. |
| DP-06 | Should expected problems give `not evaluable` and only unexpected exceptions give `error`? Was "expect everything of data" meant as input handling? | [ADR-0012](adr/0012-hostile-input-total-measurements.md) | Implemented as proposed (input handling; `not evaluable` for expected problems). Confirm or change. |
| DP-07 | Does a fleet layer need `Subject` to allow a group, and must a joint catalog's fingerprint say which source a column came from? | [Architecture](architecture.md#where-the-package-grows) | Not exercised. |

## Known gaps (decided, not yet built)

| Gap | Where |
|---|---|
| Golden profiles for a fixed seed | [ADR-0009](adr/0009-testing-strategy-generator-as-oracle.md) |
| Generator cleanup: `CurtailmentNight` ignores its hour arguments, `ZeroOut` and `BernoulliStartSegments` are unused, and the extended pipeline draws separate segments for a signal and its minimum and maximum | [ADR-0009](adr/0009-testing-strategy-generator-as-oracle.md) |
| Declared row and memory limits (`limit_exceeded` exists only as a reason) | [ADR-0012](adr/0012-hostile-input-total-measurements.md) |
| Counting unreadable values in every measure that reads numbers (only `ranges` does) | [ADR-0012](adr/0012-hostile-input-total-measurements.md) |
| Measuring the structure first, fully | [ADR-0012](adr/0012-hostile-input-total-measurements.md) |
| ODCS import (the export is one-way by design) | [ADR-0010](adr/0010-serialization-and-odcs-alignment.md) |
| CI optional test dependencies: `nbformat`, `nbclient`, `ipykernel`, `jsonschema`, `open-data-contract-standard`; the notebook and ODCS tests skip without them | project setup |

## Upcoming initiatives

1. **Reality check on real data** (new project): measure the covered rules on real turbines, close OQ-01 to OQ-13, set the floors.
2. **Remaining rules:** spikes (OQ-02) and the zeroing window (OQ-04), then conditional availability (OQ-05).
3. **Bronze contract agreement** with clause references (OQ-13).
4. **Generator cleanup** to the coding conventions, which also shrinks the legacy ignore list.
5. **Plotting helpers** for findings and masks (DP-05).
6. **CI dependencies** for the optional tests.
7. **Documentation:** a user guide for the notebook workflow, a rule reference with the metrics each rule reports, and a guide to writing a contract.

## Closed

| Question | Answer | ADR |
|---|---|---|
| Is the acceptable band the conditional pass? | Yes. | [0005](adr/0005-gate-is-a-policy-over-verdicts.md) |
| How are clause references written? | Free text. | [0004](adr/0004-contract.md) |
| Is the signal catalog shared or per layer? | Shared by all layers, joint when sources are combined. | [0001](adr/0001-domain-model.md) |
| Where are attestations stored, and how is data hashed? | Storage is the user's; hashing is kept light. | [0006](adr/0006-attestation-and-lineage.md) |
| How is the gold contract authored? | Gold means ready for further processing; that processing is out of scope. | [0004](adr/0004-contract.md) |
| Which detection targets, and real SCADA data? | Decided per rule; no real SCADA data now. | [0009](adr/0009-testing-strategy-generator-as-oracle.md) |
| Is a spike on ODCS and ODPS acceptable, and which ODPS? | Yes; the Bitol standard. Own schemas, ODCS export. | [0010](adr/0010-serialization-and-odcs-alignment.md) |
| What is the package name? | `phoibe.dokimasia`. | [0007](adr/0007-package-structure-and-dependency-direction.md) |
| Contract schema technology? | Dataclasses; no pydantic. | [0004](adr/0004-contract.md) |
| Is the warning column of the effect table right? | Yes. | [0005](adr/0005-gate-is-a-policy-over-verdicts.md) |
| Are standardizing transforms in scope? | Yes. Renaming first, then localization; deduplication only as far as localization needs it. | [0011](adr/0011-standardizing-transforms.md) |
| Is the default store layout acceptable? | Reduced to a flat folder. | [0006](adr/0006-attestation-and-lineage.md) |
| Are import-linter contracts wanted? | No; an `ast` test enforces the direction. | [0007](adr/0007-package-structure-and-dependency-direction.md) |
| Subject-attribute references: fixed forms or expressions? | Fixed form `{subject, factor, offset}`. | [0004](adr/0004-contract.md) |
| How are daylight-saving duplicates treated? | Conversion to UTC; naive local time is disambiguated by order and the rule is recorded. | [0011](adr/0011-standardizing-transforms.md) |
| Are defect tests strict expected failures in `tests/characterization`? | Yes, for the moment. | [0009](adr/0009-testing-strategy-generator-as-oracle.md) |
| Are profile and verdicts separate files, with one contract file of two sections? | Yes. | [0010](adr/0010-serialization-and-odcs-alignment.md) |
| Which limits for large inputs? | None by default. | [0012](adr/0012-hostile-input-total-measurements.md) |
| Frozen values must be single or multi-signal? | Multi-signal. | [OQ-01](#open-needs-a-person-or-real-data) |
| Is the min ≤ mean ≤ max side effect of the generator's extended pipeline acceptable? | Yes, desired. | [0009](adr/0009-testing-strategy-generator-as-oracle.md) |
| Which night hours? | 22 to 6 local time, as for a site in Germany, kept across daylight saving time. | [OQ-10](#open-needs-a-person-or-real-data) |
| Rotor speed and pitch angle range conditions? | None. | [OQ-06](#open-needs-a-person-or-real-data) |
| Recall and precision floors first or measurement first? | Measure first; set floors later from measured values. | [OQ-12](#open-needs-a-person-or-real-data) |
