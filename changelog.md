# Changelog

## T-RIX assessment

All notable changes to the T-RIX assessment product are documented here.
A version bump is required in each of these cases:

1. Any artifact schema changes (columns, layers, filenames).
2. The parameter interface changes (keys, units, locked status).

Internal refactoring does not require a bump.

## [Unreleased/0.5]

### Added

- `PRODUCT_DEFINITION` in `schema.py` as single source of truth for version, parameters, artifact schema, and field definitions.
- `ANALYZER_DEFAULTS` derived from `PRODUCT_DEFINITION["parameters"]` via `_get_parameter()` instead of hardcoded values.
- `version` and `definitions` block in `summary.yaml`, sourced from `PRODUCT_DEFINITION`.
<!--
- `ArtifactKeys` and `InternalKeys` split from `ColumnKeys` to make the boundary between published and internal fields explicit.
- `RunContext` with `DEMContext` and `AlignmentContext` for structured collection of spatial metadata during a run.
-->

### Changed

- `summary.yaml` restructured: nested `meta`, `parameters`, `spatial_context`, and `artifacts` blocks replace flat layout.
- Writer derives all column definitions and filenames from `PRODUCT_DEFINITION`.

---

## [1.0/0.6]: Unreleased baseline

Initial product definition establishing:

- RIX per site: mean, std, min, max across ray directions.
- T-RIX pairwise table: transferability, distance, A, B thresholds.
- GeoPackage with layers: `locations_site`, `locations_reference`, `ruggedness`, `trix`.
- Writer profiles: `SUMMARY` (csv + manifest) and `FULL` (+ geopackage).
- Parameters per TR6: `n_angles = 72`, `R_km = 3.5`, `slope_critical = 0.033`.

---

## [1.1/0.7]: Reviewed baseline

Reviewed and revised product:

- Artifact schema and parameter schema unified.
- Refactor of public Python API.
- Fixes.

### Changed

- Filename: `rix_summary.csv` -> `rix_table.csv`.
- Schema key: `rix_summary` -> `rix_table`.
- Schema of the RIX table `location_id` now is `site_id`.
- `summary.yaml` structure: `project_name` is top level. Added block `run`. for additional diagnostics.

### Fixes

- Removed `NameErrors` in single collection path.
- `RayRuggedness.meta`/`RadialRuggedness.meta` unrobust w/o `rioxarray`.
- Circular imports `results`/`evaluate` removed.
