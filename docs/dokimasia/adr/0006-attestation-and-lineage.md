---
type: ADR
id: "0006"
title: "Attestation and lineage"
status: implemented
date: 2026-11-22
superseded_by: null
---

# ADR-0006: Attestation and lineage

**Depends on:** [ADR-0001](0001-domain-model.md), [ADR-0005](0005-gate-is-a-policy-over-verdicts.md).

## Context

The profiler of layer N+1 waits behind the gate of layer N, and the fixes between them happen outside this package. The same layer is re-run after a human interaction, and gates do not accumulate. The legacy attestation writer only formats a report and nothing checks it.

## Decision

- A passed gate, including a conditional pass, yields an **attestation**: subject, layer, dataset fingerprint, contract id and version, gate outcome with conditions, summary of verdicts, UTC timestamp, software version and the parent attestation.
- The **fingerprint** is simple and documented. It is a consistency check, not tamper protection: a hash of the file bytes for file sources, and for in-memory frames a pandas row hash over index and values plus the column names. No canonical serialization is built.
- Moving to the next layer goes through one function that takes the dataset and its attestation, applies an optional transform ([ADR-0011](0011-standardizing-transforms.md)) and returns the next dataset with a **provenance** record: parent attestation, transform description and output fingerprint.
- The profile of layer N+1 reports whether its input carries provenance and whether the fingerprint matches. By default this is information; a strict mode refuses a mismatch.
- Edits made while working on layer N+1 produce a new dataset version. The attestation of layer N+1 binds to its final fingerprint and records its parent.
- Gates are **not cumulative**. Passing the gate of layer N+1 does not show that the gate of layer N still holds.
- **Storage is the user's.** A small `Store` protocol writes and reads profiles, verdicts and attestations. The default is deliberately plain: one folder the user names, holding `<layer>-<UTC timestamp>-profile.yaml`, `-verdicts.yaml` and `-attestation.yaml` per run. The subject is a field inside the files, not a folder level. A richer layout can be added as another `Store` without changing the core.
- A fleet layer later consumes the attestations of several single-turbine datasets.

## Consequences

- The non-cumulative rule is an accepted risk: a fix at bronze, such as setting outliers to NaN, can break a raw guarantee unnoticed.
- Because fingerprint checks are informational by default, a lineage can be broken by hand; the profile then says so instead of stopping the work.
- Nothing is persisted unless the user passes a store.

## Open points

- Whether an opt-in re-check of lower gates is worth adding later.
- Whether a recorded human waiver belongs in the attestation (see [ADR-0005](0005-gate-is-a-policy-over-verdicts.md)).

## Implementation

`phoibe.dokimasia.core` (`attestation`, `dataset`, `advance_dataset`), `phoibe.dokimasia.adapters` (`Store`, `FlatStore`). No known gaps.
