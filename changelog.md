# Changelog

## T-RIX assessment

All notable changes to the T-RIX assessment product are documented here.
A version bump is required in each of these cases:

1. Any artifact schema changes (columns, layers, filenames).
2. The parameter interface changes (keys, units, locked status).

Internal refactoring does not require a bump.

## [2.0.0/0.8] unreleased

### Changed (breaking)

- Manifest restructured. New first section `contract`. `project_name` moved to `run.project_name`.
  `run.diagnostics` has `nan_count` and `n_locations_with_nans` per collection of locations (was `n_sites_with_nans`, sites only), `finding_counts` and `findings_omitted`. New list `run.findings`. `spatial_context.source_dem.extent` and `.resolution` are mappings (`west/south/east/north`, `dx/dy`), single-element lists became plain values, `source_ray.nan_count` moved to the diagnostics, `created_at` is ISO 8601 in UTC, `artifacts` has `table_records` and  `layer_records` (was `geopackage_records_per_layer`). A missing alignment message is an empty list (was the text "None"). `n_oob` is called `nan_count`.
- GeoPackage layer `trix` is called `trix_table` and is a table without geometry (was a layer with empty geometries).
- Layers `locations_site` and `locations_reference` carry the identifier and the geometry only. Other columns of the input are not written.
- Transferability is 0 if the distance or a limit is `NaN` (was 2).
- Distances between sites and references are computed in the x/y plane (were 3D for points with a z coordinate).
- Which artifacts and layers depend on reference locations is declared in the product definition (to confirm for the final definition).
- A CRS mismatch between sites and references and duplicate location identifiers raise `CheckAborted` (a `ValueError`), were a plain `ValueError`.

### Added

- Command line (`run`, `contract`), configuration file, DEM and locations readers.
- Contract generated from the product definition and embedded in the manifest.
- Findings with severities from the definition: `nan_count`, `rix_nan`, `centre_elevation`, `n_angle`, `crs_site_vs_reference`, `crs_vs_config`, `unique_ids`, `locked_parameter_changed`.
- Check of the written artifacts against the contract, based on reading them back.
- Fixed parameters `trix.decimals` and `sampler.nan_policy` in the definition (values unchanged: 1 and mask).

### Removed

- Column `elevation_std` of `rix_table`. The origins of all rays of a location must lie at the same elevation; a deviation indicates an error and is reported by the check `centre_elevation`.

### Fixed

- `plot_polar` failed when an axis was passed.


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

---

## [1.0/0.6]: Unreleased baseline

Initial product definition establishing:

- RIX per site: mean, std, min, max across ray directions.
- T-RIX pairwise table: transferability, distance, A, B thresholds.
- GeoPackage with layers: `locations_site`, `locations_reference`, `ruggedness`, `trix`.
- Writer profiles: `SUMMARY` (csv + manifest) and `FULL` (+ geopackage).
- Parameters per TR6: `n_angles = 72`, `R_km = 3.5`, `slope_critical = 0.033`.

---

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
