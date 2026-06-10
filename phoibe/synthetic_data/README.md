# Synthetic data

## Summary

Generate synthetic data for experiments and demonstrations.

## Structure

- `fields`: Generate 2D fields. W/ or w/o coordinate reference system information.
- `sites`: Generate sites on these fields.
- `turbine`: Generate wtg scada data and blurr them.

## Topics

### Fields

Generate 2D fields to simulate elevation models. Fields are generated as data arrays. Any such data array can be equipped w/ crs information for any downstream reprojection.

Available fields are:

- *Planar field* sloping in any desired direction.
- *Eggbox field* oscillating at any frequency in the major directions.
- *Radial wave field* oscillating circularly.

These fields are returned as plain `xarray.DataArray`. As a postprocessing, they can be mapped to some box in a given CRS.

```python
# Planar field sloping in x direction.
NX, NY = 201, 201
DX, DY = 10, 10
SLOPE_X, SLOPE_Y = 0.5, 0
plane = phoibe.synthetic_data.fields.make_planar_field(nx=NX, ny=NY, dx=DX, dy=DY, slope_x=SLOPE_X, slope_y=SLOPE_Y)

# Eggbox field.
NX, NY = 30, 20
eggbox = phoibe.synthetic_data.fields.make_eggbox_field(nx=NX, ny=NY, dx=0.1, dy=0.1, freq_x=3, freq_y=3)

# Map entire field to the box in geographic coordinates.
bounds = 8, 48, 9, 49
eggbox = phoibe.synthetic_data.fields.make_field_rio(da=eggbox, bounds=bounds, crs=4326, dtype="float", nodata=np.nan)

# Wave map.
NX, NY = 201, 201
DX, DY = 10, 10
FREQ = 4
radial_wave = phoibe.synthetic_data.fields.make_radial_wave_field(nx=NX, ny=NY, dx=DX, dy=DY, freq=FREQ)
```

### Sites

Sample sites in a given bounding box at random.

```python
# Generate 7 sites within on the given map w/ 0.2 degrees empty space at the boundary.
sites = phoibe.synthetic_data.sites.make_sites(sites=7, bounds=eggbox.rio.bounds(), buffer=0.2, seed=23, crs=eggbox.rio.crs)
```

### Wind turbine generator data

Generate timeseries of wind turbine generator data:

- Wind speeds are given as a stochastic process following Weibull statistics.
- Power is computed from a synthetic power curve.
- Optionally, rotor speed can be added.
- A sampling from a higher frequency w/ added min and max for each variable is optional.
- Various kinds of noise can be applied to the timeseries at a desired strength. Noise pipelines are provided.
