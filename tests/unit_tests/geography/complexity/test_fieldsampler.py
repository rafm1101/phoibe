import numpy as np
import pytest
import xarray

from phoibe.geography.complexity.rix.fieldsampler import FieldSampler, RegularGridXYSampler


def test_field_sampler_is_abstract():
    with pytest.raises(TypeError):
        FieldSampler()


class TestRegularGridXYSampler:
    @pytest.fixture
    def planar_field(self):
        x = np.arange(-100, 100)
        y = np.arange(-100, 100)
        xx, yy = np.meshgrid(x, y)
        z = 0.5 * xx
        return xarray.DataArray(data=z, coords={"x": x, "y": y}, dims=("y", "x"), name="field")

    @pytest.mark.parametrize(
        "method, xs, ys, expected_z, expected_nan",
        [
            ("linear", [0.0, 10.0, 20.0], [0.0, 0.0, 0.0], [0.0, 5.0, 10.0], 0),
            ("linear", [0.5, 0.5, 31.5], [0.0, 0.5, 0.5], [0.25, 0.25, 15.75], 0),
            ("linear", [105, -101], [0, 314], [np.nan, np.nan], 2),
            ("linear", [], [], [], 0),
            ("nearest", [0.0, 10.0, 20.0], [0.0, 0.0, 0.0], [0.0, 5.0, 10.0], 0),
            ("nearest", [0.5, 0.5, 0.6, 31.5], [0.0, 0.5, 0.0, 0.5], [0.0, 0.0, 0.5, 15.5], 0),
            ("nearest", [105, -101], [0, 314], [np.nan, np.nan], 2),
        ],
    )
    def test_regulargridxysampler_returns_expected_values(self, planar_field, method, xs, ys, expected_z, expected_nan):
        sampler = RegularGridXYSampler(da=planar_field, method=method)
        z, nan_count = sampler.sample(xs=xs, ys=ys)
        assert np.allclose(z, expected_z, equal_nan=True)
        assert nan_count == expected_nan

    @pytest.mark.parametrize(
        "method, xs, ys, expected_match",
        [
            ("linear", [0.0, 10.0, 20.0, 200.0], [0.0, 0.0, 0.0], "Shapes do not agree"),
            ("linear", [0.0, 10.0, 20.0, 200.0], [0.0], "Shapes do not agree"),
            ("linear", [[0.0, 10.0, 20.0, 200.0]], [0.0, 10.0, 20.0, 200.0], "1-dimensional"),
            ("linear", [["a", 1]], [0.0, 10.0, 20.0, 200.0], "could not convert string to float"),
        ],
    )
    def test_regulargridxysampler_rejects_unstackable_inputs(self, planar_field, method, xs, ys, expected_match):
        sampler = RegularGridXYSampler(da=planar_field, method=method)
        with pytest.raises(ValueError, match=expected_match):
            sampler.sample(xs=xs, ys=ys)

    @pytest.mark.parametrize(
        "method, xs, ys, expected_level, expected_message",
        [
            (
                "linear",
                [0.0, 10.0, 20.0, 200.0],
                [0.0, 0.0, 0.0, 0.0],
                "INFO",
                "NaNs encountered during sampling from field",
            ),
        ],
    )
    def test_regulargridxysampler_logs_on_nan(
        self, planar_field, caplog, method, xs, ys, expected_level, expected_message
    ):
        sampler = RegularGridXYSampler(da=planar_field, method=method)
        with caplog.at_level(expected_level):
            z, _ = sampler.sample(xs=xs, ys=ys)
        assert expected_message in caplog.text
        assert np.isnan(z[-1])


class TestRegularGridXYSamplerConstruction:
    """Unit tests for RegularGridXYSampler.__init__."""

    def test_rejects_dataarray_missing_x_dimension(self):
        da = xarray.DataArray(data=np.zeros((3, 3)), coords={"z": [0, 1, 2], "y": [0, 1, 2]}, dims=("y", "z"))
        with pytest.raises(ValueError, match="must have"):
            RegularGridXYSampler(da=da, method="linear")

    def test_rejects_dataarray_missing_both_x_and_y(self):
        da = xarray.DataArray(data=np.zeros((3, 3)), coords={"a": [0, 1, 2], "b": [0, 1, 2]}, dims=("a", "b"))
        with pytest.raises(ValueError, match="must have"):
            RegularGridXYSampler(da=da, method="linear")


class TestRegularGridXYSamplerCrsAndMeta:
    """Unit tests for the crs and meta properties without a rioxarray accessor.

    Notes
    -----
    1. rioxarray registers its `.rio` accessor globally for the whole process once imported.
    2. No rioxarray import here or any module, otherwise invalidates prerequisite of _no accessor_ of this test.
    3. Keep rioxarray-backed branch in separate module.
    """

    @pytest.fixture
    def planar_field(self):
        x = np.arange(-100, 100)
        y = np.arange(-100, 100)
        xx, yy = np.meshgrid(x, y)
        z = 0.5 * xx
        return xarray.DataArray(data=z, coords={"x": x, "y": y}, dims=("y", "x"), name="field")

    def test_crs_is_none_without_rio_accessor(self, planar_field):
        sampler = RegularGridXYSampler(da=planar_field, method="linear")
        assert sampler.crs is None

    def test_meta_is_empty_dem_record_without_rio_accessor(self, planar_field):
        sampler = RegularGridXYSampler(da=planar_field, method="linear")
        assert sampler.meta == {sampler.keys.dem: {}}

    def test_sample_returns_full_length_array_not_truncated_on_nan(self, planar_field):
        sampler = RegularGridXYSampler(da=planar_field, method="linear")
        xs = [0.0, 105.0, 10.0]
        ys = [0.0, 0.0, 0.0]
        z, nan_count = sampler.sample(xs=xs, ys=ys)
        assert len(z) == len(xs)
        assert nan_count == 1
