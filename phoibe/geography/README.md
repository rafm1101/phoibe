# Geography

## Overview

**Quantify terrain complexity at any location and wind transferability.** This toolkit computes the Ruggedness Index (RIX) and wind representability (TRIX) -- metrics that measure how steep and broken terrain is based on slope analysis along radial rays and how applicable a transfer of wind measurements between sites is. Use it for **site suitability assessment for wind parks**.

Key insights:

1. **Single complexity score**.
1. **Composed complexity score and representativity tables**.
1. **Detailed ray-by-ray breakdowns** to understand exactly where terrain becomes problematic.

## What You Can Do

- **Assess terrain complexity** with RIX: a single 0-1 score indicating proportion of steep slopes.
- **Assess wind data base representativity** with T-RIX: a single score translating directly into distance thresholds.
- **Drill into results** with per-ray statistics, elevation profiles, and steep-segment geometry.
- **Work across coordinate systems**: your site and elevation model can use different CRS; the toolkit handles conversion.
- **Visualize findings**: Plot polar diagrams, overlay steep segments on maps, export results as GeoDataFrames.
- **Customize analysis**: Adjust ray count, search radius, grid spacing, and slope thresholds to your use case.

## Quick Start

1. **Quick access:**

Analyzer:

```python
import geopandas as gpd
import shapely
from phoibe.geography.complexity.rix import analyzer
from phoibe.geography.complexity.rix.config import ANALYZER_DEFAULTS

sites = gpd.GeoDataFrame(data={"location_id":["WTG1"], "geometry":[shapely.Point(0,0)]}, geometry="geometry")
references = gpd.GeoDataFrame(data={"location_id":["WDB"], "geometry":[shapely.Point(5,3)]}, geometry="geometry")

rix_analyzer = analyzer.TRIXAnalyzer(config=ANALYZER_DEFAULTS)
results = rix_analyzer.run(dem=elevation_map, locations_site=sites, locations_reference=references)
```

Functional:

```python
from phoibe.geography.complexity.rix import RegularGridXYSampler, compute_regular_rix
import shapely

location = shapely.Point(0, 0)
sampler = RegularGridXYSampler(da=elevation_map, method="linear")

result = compute_regular_rix(
    location, slope_critical=0.033, crs=None, sampler=sampler, n_angles=72, R_km=3.5, dr_km=0.01
)

print(result.rix)
```

2. **Explore your results:**

Analyzer:

```python
results.summary
results.trix_table
```

Functional:

```python
result.describe()

result.to_dataframe()

result.steep_segments_geodataframe()
```

3. **Plots:**
_Note: `geopandas` and `matplotlib` are not default dependencies in this package._

```python
from phoibe.geography.plot import plot_raster
result.plot_polar()

_, ax = plot_raster(da=elevation_map)
result.steep_segments_geodataframe().plot(ax=ax, color="r", linewidth=1, label="ray's steep parts")
```

4. **Store your results:**

```python
from phoibe.geography.complexity.rix import writer

writer.RIXWriter(result=results, profile=writer.WriterProfile.FULL).write(
   directory="trix_assessment", project_name="WP Planar I"
)
```

## Architecture

| Subpackage | Purpose |
| ------------| ---------|
| **complexity.rix** | Core RIX computations and result objects |
| **crs** | Coordinate system reprojections |
| **plot** | Raster visualization with CRS awareness |

Within `rix` hexagonal (see ADR 0001), dependencies point inwards:

| Layer | Responsibility | Allowed third-party | Forbidden |
|---|---|---|---|
| `computations` | Scientific rules and computations on in-memory values | numpy, scipy, shapely, pyproj, xarray (loaded arrays) | pandas, geopandas, yaml, matplotlib, rioxarray, pyogrio, file or network I/O |
| `pipeline` | Use case: order of steps, severities of checks, tables, run record, manifest, contract. Defines specification, ports and errors | pandas, geopandas (as data structures) | I/O libraries, adapters |
| `adapters` | Implementations for the outside world: reading rasters and locations, writing files | rioxarray, pyogrio, geopandas I/O, yaml | importing each other or `entrypoints` |
| `entrypoints` | Command line, configuration file, notebook interface | argparse, yaml, matplotlib | importing adapters |
| `main.py` | Composition root: wires adapters into the use case, hosts `main()` | all | business logic |

```
/
├── computations/
│   ├── geometry.py  fieldsampler.py  profiles.py     rays, field, sampled profiles
│   ├── evaluate.py  results.py  rix.py  trix.py      metrics, results, RIX, T-RIX
│   └── checks.py                                     rules that report findings
├── pipeline/
│   ├── analyzer.py          the use case, knows all others
│   ├── requirements.py      checks that the specification has what the code uses
│   ├── ports.py  errors.py
│   ├── product/             WHAT the product is: specification.py, definition.py
│   ├── checks/              findings.py, inputs.py
│   ├── results/             tables.py, record.py, result_summary.py
│   └── publishing/          manifest.py, contract.py, contract_check.py
├── adapters/                dem_reader.py, locations_reader.py, writer_files.py
├── entrypoints/             cli.py, config_file.py, notebook.py
└── main.py
```

Inside `pipeline`, dependencies run downwards only: `analyzer` → `publishing` → `ports` → `results` → `checks` →
`product`/`errors`.

Flow of a run: the entrypoint reads parameters and files through adapters → the use case checks inputs and parameters → domain functions compute RIX per location → findings are collected → tables, TRIX and the run record are built → `publish`: the writer writes the data files and reads them back → the contract check compares them with the specification → the manifest is written last.

Within `crs`:

- `reproject`: Reproject raster data.

Within `plot`:

- `raster`: Plot raster data bringing their own CRS or not.
- `landmarks`: Plot landmarks (on some raster plot).

### Best practices and considerations

1. Central anchor for configuration, version, parameters, artifact schema, and field definitions and descriptions are contained in `schema.py`.
   - A rule or computation on values goes into `computations`, as a function if it needs nothing else.
   - Something that needs a resource (file, clock) is a service: the contract goes into `pipeline`, the implementation into `adapters`.
   - A name, filename, severity, threshold or fixed parameter goes into the product definition, never into code.
   - A view for people (DataFrame, plot, manifest) goes into `pipeline` or `entrypoints`, never onto a domain value.
   - An anomaly that should be visible but not stop the run is a `Finding` from the domain, with its severity in the
   definition.
1. DEM metadata remains to be refined.
   - Includes: Source and metadata.
1. Multi-CRS supported. Provide assessed sites in a projected CRS measuring in meters. Maps in a geographic CRS are handled accordingly by transforming the sites' coordinates and sampling these.
1. Interpolation and grid spacing matter: Along the individual rays, resampling the elevations at its internal grid points, may have some hidden sensitivities. The following combinations should be used carefully:
   - Interpolation method `nearest` w/ a small grid spacing (leads to jumps on short segments).
   - Level crossing profiles w/ a coarse grid of levels (loss of information).
   - Interpolation method `nearest` for level crossing profiles (loads of jumps).
1. Along the individual rays, two basic philosophies lead to different kinds of grids:
   - Regular: Equidistant discretisation along the ray.
   - Level crossing: A vertical discretisation pinning available elevations to a given discrete set of values. This seems to be more prone to loosing details.
1. Pitfalls:
   1. The assessment permits computing RIX in GCS coordinates. However, expect serious distortions in the computations. _Avoid unless you have a specific reason._
   1. Coarse level-crossing grids + nearest-neighbor: Extreme loss of slope detail. Use linear interpolation and finer grids instead.
   1. Mismatched ray resolution: Very short `dr_km` with coarse elevation data means you're interpolating between widely-spaced source points. Choose consistent scales.
1. Objects separate concerns:
   - `RayGeometry` represents a single ray only. It may change its representation to another CRS.
   - `RayProfile` represents the elevation profile along its ray. For its instantiation, a `FieldSampler` is required.
   - `FieldSampler` samples coordinates from a 2D field. In case of a CRS mismatch, it provides a matching representation of the coordinates of the ray in the field's CRS.
   - `RayRuggedness` evaluates a single `RayProfile` instance.
   - `RadialRuggedness` collects multiples `RayRuggedness`es.
   - `TRIXAnalyzer` manages the RIX assessment, and if also reference locations are provided, also the full TRIX assessment of representativity.
   - `RIXWriter` serializes the results.
