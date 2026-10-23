# 0005 Findings and severities

Status: Accepted

## Context

Data anomalies (NaN along rays, deviating centre elevation, a changed locked parameter) should not stop a run by default, but must be visible. Whether something is acceptable is a decision of the product, not of the domain.

## Decision

Domain functions return `Finding`s, which are facts. The definition assigns each check a severity ("abort" or "warn") and thresholds. The use case collects located findings (collection, location, ray direction), stops before the expensive part if an input or parameter finding is "abort", stops again after computing if any finding is "abort", and otherwise completes. Findings are logged once per location and check and recorded in the manifest: counts always, the list capped at 100 per check.

Checks:

- `crs_site_vs_reference`
- `crs_vs_config`
- `unique_ids`
- `n_angle`
- `nan_count` (threshold `max_fraction`)
- `rix_nan`
- `centre_elevation`
- `locked_parameter_changed`

Severities as defined in the product definition.

`rix_nan` reports a location whose RIX is `NaN` (a ray without a segment), once per location. `centre_elevation` reports rays whose origin lies at a different elevation, which indicates an error, and a missing elevation at the origin. Both make TRIX `NaN`; neither is necessarily covered by `nan_count`, whose threshold tolerates single NaN value.

NaN handling: `nan_policy` is fixed to `mask`. A transfer that cannot be established, because the distance or a limit is NaN, is classified 0 (no transfer), never "unconditional". The NaN stays visible in the columns `trix`, `A`, `B`.

## Consequences

- Errors are reserved for violated input requirements, technical failures and contract violations.
- A new check needs a definition entry, a domain function and an entry in `requirements.py`.
