import numpy as np
import pyproj
import pytest
import xarray

from phoibe.geography.complexity.rix.computations.fieldsampler import Field, RegularGridXYSampler

pytestmark = pytest.mark.rio


@pytest.fixture
def make_rio_da():
    import rioxarray  # noqa: F401  # registers the .rio accessor for the whole process

    default_y = np.arange(-100, 100)

    def make(y=default_y, crs="EPSG:32633"):
        x = np.arange(-100, 100)
        da = xarray.DataArray(0.5 * x[None, :] + 0 * y[:, None], coords={"x": x, "y": y}, dims=("y", "x"))
        return da if crs is None else da.rio.write_crs(crs)

    return make


@pytest.mark.parametrize(
    "y, expected_resolution",
    [(np.arange(-100, 100), (1.0, 1.0)), (np.arange(99, -101, -1), (1.0, -1.0))],
    ids=["ascending y", "descending y"],  # a GeoTIFF is read with descending y: dy is negative
)
def test_from_array_returns_crs_extent_and_resolution_given_rio_accessor(make_rio_da, y, expected_resolution):
    info = Field.from_array(make_rio_da(y)).info
    assert info.crs == "EPSG:32633"
    assert info.extent == pytest.approx((-100.5, -100.5, 99.5, 99.5))  # left, bottom, right, top
    assert info.resolution == pytest.approx(expected_resolution)


def test_from_array_returns_no_crs_given_no_crs_written(make_rio_da):
    info = Field.from_array(make_rio_da(crs=None)).info
    assert info.crs is None
    assert info.resolution == pytest.approx((1.0, 1.0))


def test_crs_returns_the_written_crs(make_rio_da):
    sampler = RegularGridXYSampler(Field.from_array(make_rio_da()), "linear")
    assert sampler.crs.to_string() == "EPSG:32633"


def test_crs_returns_a_pyproj_crs(make_rio_da):
    assert isinstance(RegularGridXYSampler(Field.from_array(make_rio_da()), "linear").crs, pyproj.CRS)


def test_crs_returns_none_given_no_crs_written(make_rio_da):
    assert RegularGridXYSampler(Field.from_array(make_rio_da(crs=None)), "linear").crs is None
