# Architecture

`phoibe.dokimasia` has one small core that knows nothing about files, YAML or wind, and replaceable parts around it. Wind is one of those parts, not the shape of the package. This page puts [ADR-0001](adr/0001-domain-model.md) to [ADR-0012](adr/0012-hostile-input-total-measurements.md) into one picture. Vocabulary is in the [glossary](glossary.md), the aims in [goals](goals.md).

## Accountabilities

| Package | Accountable for | May import |
|---|---|---|
| `core` | The model: subject, dataset with fingerprint and provenance, signal, measurement outcome, finding, profile, verdict, gate and its policy, attestation, scoring, the transform protocol, and `run_measure`, the one place that contains failures. No file I/O, no YAML. | nothing of ours |
| `contract` | The contract schema, its structural validation, and `judge_layer`, which turns a profile and a layer into verdicts. Never reads data. | `core` |
| `standardize` | Recorded dataset-to-dataset transforms: rename to catalog names, localize time to UTC, chain. | `core` |
| `rules` | Generic measures and signal detection: temporal, completeness, validity, consistency, distribution, structure. | `core` |
| `wind` | The domain pack: signal catalog and patterns, curtailment rules (`power_levels`, `night_limit`). | `core`, `rules` |
| `adapters` | Everything that touches the outside: loading contract files (YAML, JSON), artifact mappers, `Store` and `FlatStore`, the one-way ODCS export. | `core`, `contract` |
| `compose` | The only place that knows all packages: builds the rule registry, validates rule parameters against the contract, and offers `profile_layer`, `validate_layer`, `attest_validation` and `build_run`. | all of the above |

An `ast`-based test enforces the import direction ([ADR-0007](adr/0007-package-structure-and-dependency-direction.md)). The architecture is ports and adapters (protocols in `core`, implementations in `adapters`), with a composition root (`compose`) and plug-in domain packs (`wind`). The core and the generic `rules` are the kernel that packs extend.

```
                         compose
        ┌───────┬──────────┼───────────┬──────────┐
     adapters   wind     rules    standardize   contract
        │  └──────┴────────┤           │           │
        └──────────────────┴───────────┴───────────┘
                          core
```

## Major objects

| Object | Role | Made by |
|---|---|---|
| `Subject` | The asset and its specification (rated power, time zone, and so on) | the user |
| `Dataset` | Immutable time series, fingerprint, optional provenance | an adapter, a transform |
| Contract | Layers with measures and checks | the user; `adapters.load_contract` |
| `Profile` | Outcomes of the layer's measures: metrics and findings, durations, no verdicts | `profile_layer` |
| Verdicts and `GateResult` | One verdict per check; gate outcome with reasons | `judge_layer` and the gate policy |
| `Attestation` | Passed gate bound to a fingerprint, contract version and parent | `attest_validation` |
| `Run` and run index | Profile, verdicts and attestation as stored files | `build_run`, `FlatStore` |

Measurements return a typed outcome: measured (metrics and findings) or not evaluable with a reason. Checks read exactly one metric of exactly one measure.

## One run, end to end

1. **Load.** An adapter turns a file or a frame into an immutable `Dataset` with a fingerprint. The user supplies the `Subject` ([ADR-0001](adr/0001-domain-model.md), [ADR-0006](adr/0006-attestation-and-lineage.md)).
2. **Standardize (optional).** Recorded transforms map the dataset onto catalog names, time axis and units ([ADR-0011](adr/0011-standardizing-transforms.md)).
3. **Profile.** The measures named by the contract's layer run on the dataset and return metrics and located findings. No judgment ([ADR-0002](adr/0002-profile-and-judge-are-separate-steps.md), [ADR-0003](adr/0003-findings-are-located-score-is-informational.md)).
4. **Judge.** The checks of the layer are applied to the profile alone, which gives verdicts and a gate result by the effect table ([ADR-0005](adr/0005-gate-is-a-policy-over-verdicts.md)). Judging never reads the dataset, so it can be repeated at no cost.
5. **Act.** In the notebook the user reads the profile, fixes the data and returns to step 1 or 3. The same layer is measured and judged again.
6. **Attest and advance.** A passed or conditional gate yields an attestation. One function hands the dataset and the attestation to the next layer, with an optional transform and a provenance record.

## The public surface

The notebook calls the steps one by one, so there is no black box. The package configures no logging and writes no file by itself.

```python
contract = dokimasia.load_contract(path)
detection = dokimasia.wind.detect_wind_signals(dataset.signal_names)
validation = dokimasia.validate_layer(dataset, subject, contract, "raw")  # profile_layer, then judge
attestation = dokimasia.attest_validation(validation, created_at=now, software_version=version)
store.save_run(dokimasia.build_run(validation, attestation), created_at=now)
bronze = dokimasia.advance_dataset(dataset, attestation, transform, software_version=version)
```

`profile_layer` runs the measures alone. `validate_layer` is profile followed by judge.

## Rules in the default registry

`essential_range`, `frozen_runs`, `missing_runs`, `night_limit`, `power_levels`, `ranges`, `required_signals`, `statistic_order`, `time_gaps`, `time_span`. Their metrics and parameters are reported by the registry; a rule reference document is not yet written.

## Where the package grows

| Change | Where it goes | Core touched? |
|---|---|---|
| A new rule | `rules` if generic, a domain pack if domain-specific; registered through `compose` | No |
| Solar or another time series | A new pack with its catalog and rules; generic `rules` must not assume wind names | No, if `Subject.specification` stays a plain mapping |
| A fleet layer | A layer in the contract that consumes the attestations of several turbines | Probably: `Subject` must allow a group |
| A database or object store | Another `Store` in `adapters` | No |
| ODCS or ODPS output | An exporter in `adapters` ([ADR-0010](adr/0010-serialization-and-odcs-alignment.md)) | No |
| Compensation of a violated check | A richer gate policy ([ADR-0005](adr/0005-gate-is-a-policy-over-verdicts.md)) | No |
| Automated cleaning | A new function kind next to transforms; out of scope now | Probably |

Two places can still break this picture and should be tested early. `Subject.specification` must stay generic. And a joint catalog across combined sources means a dataset may carry columns from several sources, so the fingerprint then has to say which.

## Fault coverage

The coverage table is executable (`tests/unit_tests/dokimasia/rules/test_coverage.py`). Values are pooled over four seeds and a grid of incidence {1, 3} and level {1, 2}, scored by recall and precision on rows. Generator data only.

| Fault class | Rule | Recall | Precision | False alarms on clean data |
|---|---|---|---|---|
| Dropped rows | `time_gaps` | 1.00 | 1.00 | 0 |
| Deleted entries | `missing_runs` | 1.00 | 1.00 | 0 |
| State held by all signals | `frozen_runs` | 1.00 | 1.00 | 0 |
| Mean outside minimum and maximum | `statistic_order` | 1.00 | 1.00 | 0 |
| Night power limit | `night_limit` | 1.00 | 1.00 | 0 |
| Held power level (method `count`) | `power_levels` | 0.93 | 1.00 | 0 |
| Held power level (method `kde`) | `power_levels` | 1.00 | 0.98 | 1 row |
| Single-signal freeze | none; a stopped turbine does that too | | | |
| Spikes | none; waits for the domain experts | | | |
| Whole-turbine zeroing | none; waits for the users | | | |

The generator's notebook run with incidence 7 and level 5 fails the draft bronze contract's gate on every seed.

## Legacy structure and what changed

The legacy README described `core` ← `application` ← `infrastructure` and `rules`: the dependency rule of clean, onion and hexagonal architecture. The new design keeps that rule and differs in three places.

| Legacy | Now | Why |
|---|---|---|
| `core`: entities, interfaces, metadata | `core` | Same idea. Site and device metadata become the subject's specification; report and gate entities become profile, verdict and gate. |
| `application`: config, context, factory, registry, validator | `compose` and `contract` | The config becomes the user-authored contract. The context becomes dataset, subject and signal mapping in `core`. Factory and registry become the composition module. The validator splits into profile and judge. |
| `rules` | `rules` and `wind` | Generic measures stay; wind-specific ones, such as curtailment, move to the pack. |
| `infrastructure`: detector, io, export | `adapters`, `rules`, `standardize` | I/O and exporters go to `adapters`. Detection becomes a structure measurement in `rules`, with patterns from the pack. Renaming goes to `standardize`. |
| `logging` | no package | Modules use the standard library logger ([ADR-0008](adr/0008-results-are-the-audit-trail.md)). |
| nothing | `contract`, `standardize`, `wind` | New concepts: the versioned contract, transforms and the domain switch. |
