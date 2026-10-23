# 0007 CRS and reprojection

---
type: ADR
id: "0007"
title: "CRS and reprojection"
status: active
date: 2026-10-23
superseded_by: null
---

## Decision

Locations are expected in a projected metric CRS, supplied by the caller. The field may carry a different CRS: rays are reprojected for sampling. Sites and reference locations must share one CRS (check `crs_site_vs_reference`). A site CRS that differs from the configured `crs` is reported (`crs_vs_config`). Both may indicate undesired behaviour. No adapter reprojects inputs. Distances between sites and references are Euclidean in the x/y plane of that CRS.

## Consequences

- A non-metric CRS gives meaningless distances. This is stated in the notes of the inputs in the definition.
- Reprojecting inputs could become an adapter later. Revisit when partitioned DEM data of varying CRS arrives.
