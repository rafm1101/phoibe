# Geography


## Structure

### Actual

- `complexity`: Assessment of terrain complexity.
  - `rix`: Ruggedness index computation.
    - `geometry`: Definition of rays as representatives in 2D world.
    - `fieldsampler`: Sample from 2D fields.
    - `profiles`: Generate profiles along rays.
    - `analyse`: Computations.
    - `results`: Gather and provide results.


## Topics

### RIX computation

The RIX assesses terrain complexity via evaluation slopes along rays originating from some source location. Roughly speaking, it is the proportion of steep segments given the total length of all rays.

This implementation provides some tools to assess more detailed results than just the single number between zero and one.

Dependencies that do not provide core functionalities are loaded lazily (`geopandas` and `matplotlib`). Install them in your working environment in case you want to use them.

1. **Quick access:** Ensure your elevation map is given in Cartesian coordinates.

```python
LOCATION = ergaleiothiki.perdix.LocationCCS(easting=0, northing=0, zone=32)
elevation_sampler = RegularGridXYSampler(da=elevation_map, method="linear")
kwargs_rix = {"sampler": elevation_sampler, "n_angles": 72, "R_km": 3.5, "dr_km": 0.01}

result = compute_regular_rix(LOCATION, slope_critical=0.033, **kwargs_rix)
result.rix
```
2. **Retrieve detailed results:** `.describe()` shows some summary statistics, while `.to_dataframe()` provides results for each ray. For even more details, `.steep_segments_geodataframe()` provides a Geodataframe holding all steep segments as geometric objects (`shapely.geometry.LineString`) ready to be plotted on maps. Pass the CRS to align coordinates w/ w map.

```python
result.describe()

result.to_dataframe()

result.steep_segments_geodataframe(crs=None)
```

3. **Plots:** The individual ray's ruggednesses can be plotted via `.plot_polar()`. Assuming `plot_geodata` plots some elevation map given as some `xarray.DataArray` w/ CRS information, steep parts may be plotted on top of the map. _Note: `geopandas` and `matplotlib` are not default dependencies in this package._

```python
result.plot_polar()

_, ax = plot_geodata(da=elevation_map)
CRS = elevations.rio.crs
result_regular.steep_segments_geodataframe(crs=CRS).plot(ax=ax, color="r", linewidth=1, label="ray's steep parts")
```
### RIX discussions

1. The assessment requires several approximations to be aware of:
   1. If the elevation map is given in a GCS coordinate system, a conversion to a Cartesian CRS is necessary beforehand. Typically, there is a natural choice given by the context.
      - The reprojection is an upcoming subject.
      - The current choice is `scipy.interpolate.RegularGridInterpolator` w/ a linear interpolation.
   1. Along the individual rays, two basic philosophies lead to different kinds of grids:
      - Regular: Equidistant discretisation along the ray.
      - Level crossing: A vertical discretisation pinning available elevations to a given discrete set of values. This seems to be more prone to loosing details.
   1. Along the individual rays, resampling the elevations at its internal grid points, may have some hidden sensitivities. The following combinations should be used carefully:
      - Interpolation method `nearest` w/ a small grid spacing (leads to jumps on short segments).
      - Level crossing profiles w/ a coarse grid of levels (loss of information).
      - Interpolation method `nearest` for level crossing profiles (loads of jumps).
