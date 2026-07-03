# Geography

## Overview

**Quantify terrain complexity at any location and wind transferability.** This toolkit computes the Ruggedness Index (RIX) and wind representability (TRIX) -- metrics that measure how steep and broken terrain is based on slope analysis along radial rays and how applicable a transfer of wind measurements between sites is. Use it for **site suitability assessment for wind parks**.

Key insights:

1. **Single complexity score**.
1. **Composed complexity score and representativity tables**.
1. **Detailed ray-by-ray breakdowns** to understand exactly where terrain becomes problematic.

## What You Can Do

- **Assess terrain complexity** with RIX: a single 0-1 score indicating proportion of steep slopes.
- **Drill into results** with per-ray statistics, elevation profiles, and steep-segment geometry.
- **Work across coordinate systems**: your site and elevation model can use different CRS; the toolkit handles conversion.
- **Visualize findings**: Plot polar diagrams, overlay steep segments on maps, export results as GeoDataFrames
- **Customize analysis**: Adjust ray count, search radius, grid spacing, and slope thresholds to your use case

## Quick Start

1. **Quick access:**

Analyzer:

```python
import geopandas as gpd
import shapely
from complexity.rix import analyzer

sites = gpd.GeoDataFrame(data={"location_id":["WTG1"], "geometry":[shapely.Point(0,0)]}, geometry="geometry)
references = gpd.GeoDataFrame(data={"location_id":["WDB"], "geometry":[shapely.Point(5,3)]}, geometry="geometry)

rix_analyzer = analyzer.TRIXAnalyzer()
results = rix_analyzer.run(dem=elevation_map, locations_site=sites, locations_reference=references)
```

Functional:

```python
from complexity.rix import RegularGridXYSampler, compute_regular_rix
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
result.plot_polar()

_, ax = plot_geodata(da=elevation_map)
result_regular.steep_segments_geodataframe().plot(ax=ax, color="r", linewidth=1, label="ray's steep parts")
```

## Architecture

| Subpackage | Purpose |
| ------------| ---------|
| **complexity.rix** | Core RIX computations and result objects |
| **crs** | Coordinate system reprojections |
| **plot** | Raster visualization with CRS awareness |

Within `rix`:

- `config`: Gathered configurations used throughout the subpackage.
- `geometry`: Definition of rays as representatives in 2D world.
- `profiles`: Generate 1D profiles along rays.
- `fieldsampler`: Sample from 2D fields.
- `results`: Gather and provide results for single locations.
- `evaluate`: Computations, functional interface for single locations.
- `trix`: T-RIX computations and evaluations.
- `analyzer`: Interface for full T-RIX assessment.
- `writer`: Serialise the assessment results.

Within `crs`:

- `reproject`: Reproject raster data.

Within `plot`:

- `raster`: Plot raster data bringing their own CRS or not.
- `landmarks`: Plot landmarks (on some raster plot).

### Best practices and considerations

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
   - `RayGeometry` represents a ray only. It may change its representation to another CRS.
   - `RayProfile` represents the elevation profile along its ray. For its instantiation, a `FieldSampler` is required.
   - `FieldSampler` samples coordinates from a 2D field. In case of a CRS mismatch, it requests a matching representation of the coordinates in its own CRS.
   - `RayResult` evaluates a single `RayProfile` instance.
   - `RadialRixResult` collects multiples `RayResult`s.
   - `RIXAnalyzer` manages the RIX assessment, and if also reference locations are provided, also the full TRIX assessment of representativity.
   - `RIXWriter` serializes the results.
