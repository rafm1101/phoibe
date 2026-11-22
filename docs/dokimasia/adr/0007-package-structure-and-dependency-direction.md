---
type: ADR
id: "0007"
title: "Package structure and enforced dependency direction"
status: implemented
date: 2026-11-22
superseded_by: null
---

# ADR-0007: Package structure and enforced dependency direction

**Depends on:** [ADR-0001](0001-domain-model.md) to [ADR-0006](0006-attestation-and-lineage.md).

## Context

The legacy README claims `core` ← `application` ← `infrastructure` and `rules`. The code differs: rules import `application` (context, registry), the registry imports `rules`, the factory imports `infrastructure`, and logging is imported everywhere. Nothing enforces the direction. The legacy rule registry is class-level state and is empty on a fresh import (legacy defect 1).

## Decision

Six packages and a composition module under `phoibe.dokimasia`, with one direction of dependency.

| Package | Holds | May import |
|---|---|---|
| `core` | Model, profile, finding, verdict, gate, attestation, protocols. No file I/O, no YAML; pandas for the dataset. | nothing of ours |
| `contract` | Contract schema, structural validation, the judge. | `core` |
| `standardize` | Recorded dataset-to-dataset transforms: rename to catalog names, localize time to UTC ([ADR-0011](0011-standardizing-transforms.md)). | `core` |
| `rules` | Generic measures and signal detection: temporal, completeness, ranges. | `core` |
| `wind` | Domain pack: signal catalog and patterns, curtailment and power rules. | `core`, `rules` |
| `adapters` | Loaders, `Store`, YAML and JSON mappers, including contract loading. | `core`, `contract` |
| `compose` | The only place that knows all packages: builds the registry, validates rule parameters, offers profile, judge, attest and advance. | all of the above |

- The composition module builds the registry from an explicit list of packs. The registry is an instance, not class-level state, so a `clear()` for tests is not needed and defect 1 cannot occur.
- Rule parameters are validated in `compose`, because only this module knows both the contract and the registry.
- The package name is `phoibe.dokimasia`. The legacy name `layered` described the architecture, not the function. Considered and not taken: `phoibe.quality`, `phoibe.pylai`, `phoibe.medallion`.
- No import-linter contracts: they are not related to this project and were dropped.

**Enforcement.** A pytest test parses the imports of each package with the standard library `ast` module and checks the table above. It needs no extra tool.

## Consequences

- A new domain is a new pack.
- Naming and directory layout are secondary and can follow.
- Six packages is more structure than the code strictly needs. If that feels heavy, they can be modules of one package; the import direction is what matters, and it survives a later split.

## Open points

Whether packs stay subpackages of `phoibe` or become separate distributions. Subpackages suffice for now.

## Implementation

Six packages and `compose` exist in `phoibe.dokimasia`; the import-direction test passes. No known gaps.
