import numpy as np
import pytest
import xarray

from phoibe.geography.complexity.rix.computations.fieldsampler import Field, FieldInfo, RegularGridXYSampler

NAN = np.nan


def make_da(y, x_name="x", y_name="y"):
    """Planar field z = 0.5 x + 2 y on x = -100...99 and the given y."""
    x = np.arange(-100, 100)
    y = np.asarray(y)
    z = 0.5 * x[None, :] + 2.0 * y[:, None]
    return xarray.DataArray(data=z, coords={x_name: x, y_name: y}, dims=(y_name, x_name), name="field")


@pytest.fixture
def no_rio_da():
    da = make_da(np.arange(-100, 100))
    assert not hasattr(da, "rio"), "rioxarray was imported in this process: run with -m 'not rio'"
    return da


@pytest.mark.parametrize(
    "method, xs, ys, expected_z, expected_nan_count",
    [
        ("linear", [0.0, 10.0, 20.0], [0.0, 0.0, 0.0], [0.0, 5.0, 10.0], 0),
        ("linear", [0.5, 0.5, 31.5], [0.0, 0.5, 0.5], [0.25, 1.25, 16.75], 0),
        ("linear", [105.0, -101.0], [0.0, 3.0], [NAN, NAN], 2),
        ("linear", [0.0, 105.0, 10.0], [0.0, 0.0, 0.0], [0.0, NAN, 5.0], 1),  # NaN stays in place, nothing is cut
        ("linear", [], [], [], 0),
        ("nearest", [0.0, 10.0, 20.0], [0.0, 0.0, 0.0], [0.0, 5.0, 10.0], 0),
        ("nearest", [0.4, 0.4, 0.6, 31.4], [0.0, 0.4, 0.0, 0.4], [0.0, 0.0, 0.5, 15.5], 0),
        ("nearest", [105.0, -101.0], [0.0, 3.0], [NAN, NAN], 2),
    ],
)
def test_sample_returns_the_values_and_the_nan_count(no_rio_da, method, xs, ys, expected_z, expected_nan_count):
    z, nan_count = RegularGridXYSampler(Field.from_array(no_rio_da), method).sample(xs=xs, ys=ys)
    np.testing.assert_allclose(z, expected_z)
    assert nan_count == expected_nan_count


@pytest.mark.parametrize("y", [np.arange(-100, 100), np.arange(99, -101, -1)], ids=["ascending y", "descending y"])
def test_sample_returns_the_same_values_given_ascending_or_descending_y(no_rio_da, y):
    sampler = RegularGridXYSampler(Field.from_array(make_da(y)), "linear")
    z, _ = sampler.sample(xs=[10.0, -20.5], ys=[3.0, -7.25])
    np.testing.assert_allclose(z, [0.5 * 10 + 2 * 3, 0.5 * -20.5 + 2 * -7.25])


def test_sample_returns_the_same_values_given_the_dimensions_in_the_other_order(no_rio_da):
    field = Field.from_array(make_da(np.arange(-100, 100)).transpose("x", "y"))
    z, _ = RegularGridXYSampler(field, "linear").sample(xs=[10.0, -20.5], ys=[3.0, -7.25])
    np.testing.assert_allclose(z, [0.5 * 10 + 2 * 3, 0.5 * -20.5 + 2 * -7.25])


def test_sample_returns_values_given_custom_dimension_names(no_rio_da):
    field = Field.from_array(make_da(np.arange(-100, 100), x_name="lon", y_name="lat"), dims=("lon", "lat"))
    z, _ = RegularGridXYSampler(field, "linear").sample(xs=[10.0], ys=[3.0])
    np.testing.assert_allclose(z, [11.0])


@pytest.mark.parametrize(
    "xs, ys, expected_match",
    [
        ([0.0, 10.0, 20.0, 200.0], [0.0, 0.0, 0.0], "Shapes do not agree"),
        ([0.0, 10.0, 20.0, 200.0], [0.0], "Shapes do not agree"),
        ([[0.0, 10.0, 20.0, 200.0]], [0.0, 10.0, 20.0, 200.0], "1-dimensional"),
    ],
)
def test_sample_raises_given_coordinates_that_do_not_stack(no_rio_da, xs, ys, expected_match):
    sampler = RegularGridXYSampler(Field.from_array(no_rio_da), "linear")
    with pytest.raises(ValueError, match=expected_match):
        sampler.sample(xs=xs, ys=ys)


@pytest.mark.parametrize("dims", [("y", "z"), ("a", "b")])
def test_from_array_raises_given_missing_dimensions(dims):
    da = xarray.DataArray(data=np.zeros((3, 3)), coords={d: [0, 1, 2] for d in dims}, dims=dims)
    with pytest.raises(ValueError, match="must have"):
        Field.from_array(da)


def test_from_array_returns_an_empty_info_given_no_rio_accessor(no_rio_da):
    assert Field.from_array(no_rio_da).info == FieldInfo(crs=None, extent=None, resolution=None)


def test_crs_returns_none_given_no_rio_accessor(no_rio_da):
    assert RegularGridXYSampler(Field.from_array(no_rio_da), "linear").crs is None
