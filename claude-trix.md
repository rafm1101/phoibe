# T-RIX Assessment — Session Summary

## Rules

- Interview until 95% context is captured. Do not guess.
- If less than 99% certain, correct the statement rather than giving a wrong answer.
- Do not say what the user wants to hear. Say what a smart skeptic would say.
- Weigh every statement against an alternative and choose the stronger one.
- No abstraction without justification: an abstraction is justified only if it carries state or behaviour reused in more than one place.
- No future-proofing without a concrete trigger: build for today's use case, not a hypothetical one.

---

## Context

**Product:** T-RIX assessment — computes RIX per site and pairwise TRIX between two collections of locations over a DEM, following TR6 Rev. 12.

**Users:** Internal data scientists / researchers (1–2 people).

**Scale:** 5–20 sites (collection A) vs. 5–10 wind reference sites (collection B). DEM < 1 GB. Runs once per study or project.

**Existing code:** A class-based structure for single-location RIX computation already exists, producing `RadialRixResult` with summary statistics and a detail `GeoDataFrame` of steep segments as `LineString` geometries.

**RIX definition:** Terrain Ruggedness Index, Riley et al. 1999.

**TRIX definition:** Pairwise comparison of RIX values between collection A and B. Produces transferability classification `{0, 1, 2}` based on TRIX, distance, and thresholds A and B derived from TRIX.

**TR6:** Technische Richtlinie 6. Locks `n_angles = 72`, `R_km = 3.5`, `slope_critical = 0.033`. These parameters must not be overridden silently.

---

## Architecture decisions

**What was rejected and why:**

- Factory + Builder pattern: unjustified for one flow, one input type, internal users.
- Provenance hash + deterministic serialization: unclear benefit, high implementation risk, no peer-review requirement today.
- Parquet for TRIX: overkill for ≤ 200 pairs.
- Writer as a collection of per-artifact methods: replaced by profile-based single `write()` call.
- `RIXRunner` as class name: replaced by `RIXAnalyzer` — more semantically precise for an output that includes geometry, not just a scalar.

**What was kept and why:**

- Separation of summary vs. detail: summary is a compact scalar metric; detail is geometries for inspection.
- Compute pure / in-memory, IO separate: `RIXAnalyzer` returns `ResultSummary`; `RIXWriter` serializes.
- `ResultSummary` as `frozen=True` dataclass: prevents mutation after compute.
- CRS validation: silent CRS errors are the most common failure mode in geo pipelines.
- `config.yaml` as single source of parameter input for the user.

---

## Module structure

```
schema.py     — PRODUCT_DEFINITION (no dependencies within package)
interface.py  — ArtifactKeys, InternalKeys (depends on schema.py) CHANGED: Keys only.
config.py     — ANALYZER_DEFAULTS derived from PRODUCT_DEFINITION (depends on schema.py)
geometry.py   — RayGeometry (no dependencies within package)
fieldsampler.py — FieldSampler, RegularGridXYSampler (depends on interface.py and config.py)
profiles.py   — RayProfile (depends on fieldsampler.py, geometry.py and interface.py)
results.py    — RayProfileMeta, RayRuggedness, RadialRuggedness (depends on profiles.py and interface.py)
evaluate.py   — contains basic computations only (depends on geometry.py, interface.py, profiles.py)
trix.py       — compute_trix, compute_trix_limit_distances, evaluate_transferability_limits (no dependencies within package)
analyzer.py   — TRIXAnalyzer, ResultSummary, RunContext, compute_regular_rix (orchestration geometry, sampling, ruggedness as this is an analyzer task) (depends on almost entire subpackage)
writer.py     — RIXWriter, WriterProfile (depends on rix_analyzer.py, config.py, interface.py)
```

No circular dependencies. `schema.py` is the base of the dependency hierarchy.

---

## Public API

```python
# Compute
analyzer = TRIXAnalyzer.from_config("config.yaml")
result = analyzer.run(dem, locations_a, locations_b=None)

result.summary      # DataFrame: location_id, rix, rix_std, rix_min, rix_max,
                    #            n_rays, slope_critical
result.detail       # GeoDataFrame: location_id, theta, segment_id, geometry
result.trix         # DataFrame: location_id, reference_id, trix, distance,
                    #            A, B, transferability — or None
result.context      # RunContext: DEMContext + AlignmentContext
result.config       # dict — parameters used in this run
result.timestamp    # UTC datetime

# Export
writer = RIXWriter(result, profile=WriterProfile.SUMMARY)
writer.write(directory="output/")
```

---

## PRODUCT_DEFINITION

Central structure in `schema.py`. Single source of truth for:

- `version` — string `"1.0"`, bumped on any schema or parameter interface change.
- `references` — TR6, Riley 1999.
- `parameters` — grouped as `ray`, `slope`, `sampler`. Each parameter carries `value`, `unit`, `description`, `dtype`, `range`. TR6 parameters additionally carry `source: "TR6"` and `locked: True`.
- `inputs` — `dem`, `locations_site`, `locations_reference` (optional). Each carries `required`, `conditions`, `notes`.
- `artifacts` — `profiles` (SUMMARY / FULL), `filenames`, `layers`.
- `schema` — `manifest`, `rix_table`, `trix`, `geopackage`. Each column carries `description`, `unit`, `dtype` (where non-obvious), `range` or `values` (mutually exclusive), never both.

**Key invariants:**
- `locked: True` and `range: [x, x]` are redundant — use `locked` alone.
- `range` and `values` are mutually exclusive.
- `unit: None` for dimensionless quantities and identifiers. Never `""`.
- `geometry_type: None` (Python None), never the string `"None"`.
- `site_id: str` identifies in single-role tables each row, `site_id` and `reference_id` in pairwise tables to allow joins across files.

**Dependency rule:** `ArtifactKeys` is the source of column name strings. `PRODUCT_DEFINITION` is built using `ArtifactKeys` values as keys — not hardcoded strings. `ANALYZER_DEFAULTS` is derived from `PRODUCT_DEFINITION["parameters"]` via `_get_parameter()`.

---

## Key helper: `_get_parameter`

```python
def _get_parameter(definition: dict, *path: str, strict: bool = True) -> Any:
    """Navigate nested dict. strict=False returns None if the last key is absent."""
```

`strict=False` only suppresses `KeyError` on the final key. Intermediate missing keys always raise — they indicate a structural defect in `PRODUCT_DEFINITION`.

---

## ColumnKeys split

`ColumnKeys` is split into:

- `ArtifactKeys` — keys appearing in published artifacts. Changes require a version bump.
- `InternalKeys` — keys used internally (e.g. `x`, `y`, `nan_count`, `theta`).

A unit test monitors `ArtifactKeys` against a known hash, failing visibly when external keys change without a version bump.

---

## WriterProfile

```python
class WriterProfile(enum.StrEnum):
    SUMMARY = "summary"   # manifest + rix_table.csv + trix.csv
    FULL    = "full"      # SUMMARY + rix_results.gpkg
```

Profile membership defined in `PRODUCT_DEFINITION["artifacts"]["profiles"]`. Writer reads from there — no hardcoded artifact lists in the writer.

**GeoPackage layers (FULL only):**
- `locations_site` — Point
- `locations_reference` — Point (omitted if `locations_b` is None)
- `ruggedness` — LineString (steep segments)
- `trix` — attribute table, no geometry

---

## summary.yaml structure

```yaml
project_name: ~
meta:
  name: T-RIX assessment
  version: "1.0"
  description: "TRIX, TR6 Rev.12"
  timestamp: "2026-07-03T17:18:04Z"

parameters:
  ray:
    n_angles: 72
    R_km: 3.5
    dr_km: 0.05
    crs: "EPSG:32633"
  slope:
    slope_critical: 0.033
  sampler:
    method: linear

spatial_context:
  dem:
    source: ~
    description: ~
    crs: "EPSG:4326"
    extent:
      west: 10.996
      south: 51.002
      east: 13.996
      north: 53.002
    resolution:
      dx: 0.00746
      dy: -0.00498
    nan_count: 0
  rays:
    crs: "EPSG:32633"
  alignment:
    messages:
      - "Transformed ray CRS EPSG:32633 to DEM CRS EPSG:4326 for sampling."

run:
  n_sites: 7
  n_references: 3
  computed: ["rix_site", "rix_reference", "trix"]
  diagnostics:
    n_sites_with_nans: 
    transferability_counts: {0: 5, 1: 5, 2: 11}

artifacts:
  profile: full
  files:
    manifest: summary.yaml
    rix_table: rix_table.csv
    trix: trix.csv
    geopackage: rix_details.gpkg
```

---

## RunContext

```python
@dataclasses.dataclass
class DEMContext:
    source: Optional[str]
    description: Optional[str]
    crs: Optional[str]
    extent: Optional[dict]    # {west, south, east, north}
    resolution: Optional[dict] # {dx, dy}
    nan_count: int

@dataclasses.dataclass
class AlignmentContext:
    messages: list[str]
    crs_dem: Optional[str]
    crs_ray: Optional[str]

@dataclasses.dataclass
class RunContext:
    dem: DEMContext
    alignment: AlignmentContext
```

DEM source and description are supplied by the user via optional `dem_metadata` input to `RIXAnalyzer.run()` or `config.yaml`. Extent, resolution, CRS, and NaN count are derived from the DataArray by the sampler.

---

## Open ticket: PRODUCT_DEFINITION integration

**Story:** The product has no central definition of itself. Parameter defaults, artifact schema, and output field semantics are scattered. A schema change does not trigger a visible version bump.

**Acceptance criteria:**
- [ ] `PRODUCT_DEFINITION` exists in `schema.py` with version, parameters, artifact schema, and field definitions.
- [ ] Unit test monitors `Keys` hash; fails visibly on unbumped changes.
- [ ] `summary.yaml` contains `schema_version` and `definitions` block from `PRODUCT_DEFINITION`.
- [ ] Writer derives all field definitions and filenames from `PRODUCT_DEFINITION`; no hardcoded strings.

**Scope (future):** Compatibility check when loading a `summary.yaml` from a previous version; separate changelog convention for schema vs. parameter bumps.

---

## Demonstration notebook outline

```markdown
# T-RIX Assessment — Demonstration

This notebook demonstrates the T-RIX assessment workflow for a set of sites
and wind data base references using a synthetic DEM with analytically known
RIX values.

**Key results:** _To be filled after running the notebook._

**Contents:**
1. Preparations — imports, configuration, synthetic DEM.
2. Examples:
   - Standard case: DEM and sites in UTM.
   - Reprojection: DEM in GCS, sites in UTM.
   - Edge cases: flat plane (RIX = 0), steep plane (RIX = 1),
     site at DEM boundary, identical collections (TRIX = 0).
   - Non-examples: sites in GCS, missing CRS.
3. Key results — verifiable RIX and TRIX outcomes across all examples.
```

**Analytical RIX for a planar DEM:**

```python
def expected_rix(slope_x, slope_y, slope_critical, n_angles):
    angles = np.linspace(0, 360, n_angles, endpoint=False)
    slopes = slope_x * np.sin(np.deg2rad(angles)) \
           + slope_y * np.cos(np.deg2rad(angles))
    return float(np.mean(np.abs(slopes) > slope_critical))
```

Note: expected RIX must be computed in the CRS in which rays are evaluated (UTM), not in GCS. After reprojection, `slope_x` and `slope_y` rotate slightly with latitude.

---

## Known issues in existing code

- `RadialRixResult.__post_init__` sets `_ray_by_angle` via `object.__setattr__` as a workaround for `frozen=True`. Consider converting `_ray_by_angle` to a plain `@property`.

---

## Test organizsation

- Tests stay organized by test type on top level (tests/unit/, tests/contract/, tests/integration/), not according to module (refactor not suitable atm, decision boundary yet to be clarified).
- Added Pytest marker rio for tests that rely on  `rioxarray`'s `.rio` accessor (registered via `pyproject.toml/pytest.ini/conftest.py`'s `pytest_configure`). Required as `rioxarray` registers the accessor globally for the entire process as soon as a random test imports it (`pytest-xdists`, `--dist=load`, may mix non-rio and rio tests and allow contamination; `--cov-append` added to join coverage reports).

---

## Backlog

- NaN-Handling for `nan_policy="mask"` in `compute_regular_rix`: Sampling may treat point close to the DEM boundary silently; clarify.
- `dummy_location` fixture (block-1 `conftest.py`) not touched, delete at the end if not necessary.
- `Keys.ruggedness`/`Keys.ruggedness_layer` got same value `ruggedness`.
- `Keys.alignment` has value `internal_alignment` that differs from key as the only occurrence.
- `make_levelcrossing_profile` fixture `test_rayprofiles_contract.py` unused, wrong parameter name (`ray_geometry` instead of `ray`).
- `_get_steep_segments`/`_epsg_int` (`analyzer.py`) seem to be unused. Clarify.
- `RayProfileMeta.n_oob`s type corrected (`int | None`), name mixes NaN counter and „out of bounds". Clarify.
- Three unreachable checks in `evaluate.py` (`slopes`' `dr>0`, `ruggedness`' `length_m<=0`, `steep_ray_segments`' `len(xs)>=2`). Clarify.

---

## Open ticket: Scientific validation

**Story:** The main focus up to now lied on the implementation itself: robustness, architecture, testing. A comparison against a reference remains waiting (already used together with CopDEM).

**Acceptance criteria:**
- [ ] Compare against a reference implementation for a real site.
- [ ] Sensitivity analysed on CopDEM dataset: Stability of RIX and T-RIX against `n_angles`, resolution and interpolation method.
- [ ] Document limits of the method, e.g. DEM resolution.
- [ ] Uncertainty considerations of transferability thresholds against measurement and model uncertainty in distance and T-RIX near A/B cut-offs.