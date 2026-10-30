import sys

import numpy as np
import pytest

from phoibe.geography.complexity.rix.adapters.dem_reader import read_field
from phoibe.geography.complexity.rix.computations import RegularGridXYSampler
from phoibe.geography.complexity.rix.pipelines import errors

rio = pytest.mark.rio
X = 500_000 + 15 + 30 * np.arange(4)  # cell centres of four 30 m cells
Y = 5_000_000 - 15 - 30 * np.arange(6)  # descending, as a GeoTIFF is read
VALUES = np.arange(24, dtype="float32").reshape(6, 4)


@pytest.fixture
def make_geotiff(tmp_path):
    import rioxarray  # noqa: F401  # registers the .rio accessor for the whole process
    import xarray

    def make(data=VALUES, crs="EPSG:32633", nodata=None, name="dem.tif"):
        data = np.asarray(data)
        dims, coords = ("y", "x"), {"y": Y, "x": X}
        if data.ndim == 3:
            dims, coords = ("band", "y", "x"), {"band": np.arange(1, len(data) + 1), **coords}
        da = xarray.DataArray(data, coords=coords, dims=dims)
        if crs is not None:
            da = da.rio.write_crs(crs)
        if nodata is not None:
            da = da.rio.write_nodata(nodata)
        path = tmp_path / name
        da.rio.to_raster(path)
        return path

    return make


@rio
def test_read_field_returns_the_values_crs_extent_and_resolution_of_the_file(make_geotiff):
    field = read_field(make_geotiff())
    np.testing.assert_allclose(field.da.values, VALUES)
    assert field.info.crs == "EPSG:32633"
    assert field.info.resolution == pytest.approx((30.0, -30.0))  # y decreases with the row
    assert field.info.extent == pytest.approx((500_000, 4_999_820, 500_120, 5_000_000))  # left, bottom, right, top


@rio
def test_read_field_returns_a_field_that_can_be_sampled_given_y_decreasing_with_the_row(make_geotiff):
    sampler = RegularGridXYSampler(read_field(make_geotiff()), "linear")
    z, nan_count = sampler.sample(xs=[X[0], X[3]], ys=[Y[0], Y[5]])
    assert (z.tolist(), nan_count) == ([VALUES[0, 0], VALUES[5, 3]], 0)


@rio
@pytest.mark.parametrize("dtype, nodata", [("float32", -9999.0), ("int16", -1)])
def test_read_field_returns_nan_for_nodata(make_geotiff, dtype, nodata):
    data = VALUES.astype(dtype)
    data[2, 1] = nodata
    field = read_field(make_geotiff(data, nodata=nodata))
    assert np.isnan(field.da.values[2, 1]) and int(np.isnan(field.da.values).sum()) == 1


@rio
def test_read_field_returns_the_band_that_was_asked_for(make_geotiff):
    path = make_geotiff(np.stack([VALUES, VALUES + 100]))
    assert read_field(path).da.values[0, 0] == 0
    assert read_field(path, band=2).da.values[0, 0] == 100


@rio
def test_read_field_returns_no_crs_given_a_file_without_one(make_geotiff):
    assert read_field(make_geotiff(crs=None)).info.crs is None


@rio
def test_read_field_returns_the_data_loaded_and_not_lazy(make_geotiff):
    field = read_field(make_geotiff())
    assert field.da.chunks is None  # no dask
    assert isinstance(
        field.da.variable._data, np.ndarray
    )  # no public API tells "loaded": lazy data is a cached wrapper


@rio
def test_read_field_raises_a_port_error_given_a_band_that_does_not_exist(make_geotiff):
    with pytest.raises(errors.PortError, match="Could not read a DEM"):
        read_field(make_geotiff(), band=3)


@rio
def test_read_field_raises_a_port_error_given_a_missing_file(tmp_path):
    with pytest.raises(errors.PortError, match="Could not read a DEM"):
        read_field(tmp_path / "missing.tif")


@rio
def test_read_field_raises_a_port_error_given_a_file_that_is_no_raster(tmp_path):
    path = tmp_path / "text.tif"
    path.write_text("this is not a GeoTIFF")
    with pytest.raises(errors.PortError, match="Could not read a DEM"):
        read_field(path)


def test_read_field_raises_a_port_error_naming_the_optional_dependency_given_no_rioxarray(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "rioxarray", None)  # an import then raises ImportError
    with pytest.raises(errors.PortError, match="optional dependency 'rioxarray'"):
        read_field(tmp_path / "dem.tif")
