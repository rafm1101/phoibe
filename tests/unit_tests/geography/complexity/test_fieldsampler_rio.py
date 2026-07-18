"""Tests for RegularGridXYSampler's crs/meta properties when a real rioxarray CRS is set.

Kept in its own file, separate from test_fieldsampler.py: rioxarray registers its
`.rio` accessor globally for the whole test process once imported anywhere, which
would silently invalidate the "no accessor" tests in test_fieldsampler.py if the
two were mixed and executed in the wrong order.

NOT VERIFIED BY EXECUTION -- rioxarray is unavailable in the environment these
tests were written in. In particular, double-check locally:
- The exact sign/values returned by `.rio.bounds()` and `.rio.resolution()` for
  this coordinate layout (ascending x/y), which depend on rioxarray's affine
  transform conventions.
- That `.rio.write_crs(...)` is the correct call for your rioxarray version.
"""

import numpy as np
import pytest
import xarray

from phoibe.geography.complexity.rix.fieldsampler import RegularGridXYSampler


@pytest.fixture
def planar_field_with_crs():
    import rioxarray  # noqa: F401  (registers the .rio accessor)

    x = np.arange(-100, 100)
    y = np.arange(-100, 100)
    xx, yy = np.meshgrid(x, y)
    z = 0.5 * xx
    da = xarray.DataArray(data=z, coords={"x": x, "y": y}, dims=("y", "x"), name="field")
    return da.rio.write_crs("EPSG:32633")


def test_fieldsampler_crs_returns_the_written_crs_string(planar_field_with_crs):
    sampler = RegularGridXYSampler(da=planar_field_with_crs, method="linear")
    assert sampler.crs is not None
    assert "32633" in sampler.crs.to_string()


def test_meta_populates_dem_record_with_crs_extent_and_resolution(planar_field_with_crs):
    sampler = RegularGridXYSampler(da=planar_field_with_crs, method="linear")
    meta = sampler.meta

    dem_record = meta[sampler.keys.dem]
    assert "32633" in dem_record[sampler.keys.crs_dem]

    extent = dem_record[sampler.keys.extent_dem]
    assert len(extent) == 4

    resolution = dem_record[sampler.keys.resolution_dem]
    assert len(resolution) == 2
    assert abs(resolution[0]) == pytest.approx(1.0)
    assert abs(resolution[1]) == pytest.approx(1.0)
