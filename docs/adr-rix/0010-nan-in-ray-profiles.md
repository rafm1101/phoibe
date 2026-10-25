# 0010 NaN in ray profiles

---
type: ADR
id: "0010"
title: "NaN in ray profiles"
status: draft
date: 2026-10-24
superseded_by: null
---

## Context

Rays leave the field or meet data gaps, so sampled profiles contain NaN. The product fixes `nan_policy` to `mask` (ADR 0004): NaN stays in the profile. This ADR records what follows from that for the numbers, so that nobody has to reconstruct it from the code. It describes the current behaviour. Nothing here is tested on purpose (see Consequences).

## Decision

With `mask`, the ruggedness of a ray is computed as follows:

- A slope with a NaN end point is NaN. A NaN slope is not steep, because the comparison with the critical slope is false.
- The length of a segment with a NaN slope stays in the total length of the ray. The ruggedness is the steep length divided by the full length, never by the valid length only. It is therefore a lower bound of the ruggedness the ray would have with complete data.
- A ray without any segment (a profile of fewer than two points) has ruggedness NaN. The RIX of a location is the mean over its rays, so one such ray makes it NaN (check `rix_nan`, ADR 0005).
- How much of a ray is affected is reported by the check `nan_count`. Its threshold decides whether the run warns or aborts. The computation itself does not look at it.

The policies `truncate` and `error` exist in the domain but are not settable in the product.

## Consequences

- A ray with a few NaN values contributes a slightly low ruggedness and a finding, not an error. A location with many NaN values is flagged by `nan_count` and needs a look at the DEM coverage.
- No unit test pins the ruggedness values of profiles with interior NaN. A change of this behaviour is a decision of the product, to be made here and in the changelog, not a failing test.
- Level crossings (`create_levelcrossing`, not used by the product) handle NaN differently: an interior NaN silently drops the crossings of its segments, a NaN at the first point is copied into the result. Undecided. The counts in `SamplingInfo` of a level-crossing profile are those of the discarded helper points.

## Open

- Whether a ray with NaN values should be excluded from the mean over rays instead of being counted low.