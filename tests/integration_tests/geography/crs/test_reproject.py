import numpy as np
import pyproj
import pytest
import rasterio.transform
import xarray

from phoibe.geography.crs import reproject_rasterdata


@pytest.fixture
def grid_shape_width_height():
    return 100, 80


@pytest.fixture
def bbox_wgs84_lon_min_lat_min_lon_max_lat_max():
    return (8.0, 48.0, 9.0, 49.0)


@pytest.fixture
def sample_int_array(grid_shape_width_height, bbox_wgs84_lon_min_lat_min_lon_max_lat_max):
    width, height = grid_shape_width_height
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84_lon_min_lat_min_lon_max_lat_max

    xs = np.linspace(lon_min, lon_max, width)
    ys = np.linspace(lat_max, lat_min, height)
    XX, YY = np.meshgrid(xs, ys)
    data = (XX + YY).astype(np.int16)

    da = xarray.DataArray(data, dims=("y", "x"), coords={"x": xs, "y": ys}, name="band1")
    da.rio.write_crs("EPSG:4326", inplace=True)
    da.rio.write_nodata(-32768, inplace=True)
    return da


@pytest.fixture
def grid_of_rearranged_sequence():
    width, height = 100, 80
    data = np.arange(width * height, dtype=float).reshape((height, width))

    transform = rasterio.transform.from_origin(0.0, 80.0, 1.0, 1.0)

    da = xarray.DataArray(data, dims=("y", "x"), coords={"x": np.arange(width), "y": np.arange(height)})
    da.rio.write_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    da.rio.write_transform(transform, inplace=True)
    da.rio.write_nodata(np.nan, inplace=True)
    return da


def test_reproject_returns(grid_of_rearranged_sequence):
    src = grid_of_rearranged_sequence
    dst_crs = pyproj.CRS.from_epsg(32632)
    da_out, summary = reproject_rasterdata(src, crs_to=dst_crs)

    assert "crs_from" in summary
    assert da_out.rio.crs.to_string() == dst_crs.to_string()
    assert isinstance(da_out, xarray.DataArray)

    assert np.isfinite(np.nanmean(da_out.values))
    assert summary["range_dst"]["width"] > 0
    assert summary["range_dst"]["height"] > 0


# def test_roundtrip_accuracy(sample_int_array):
#     src = sample_int_array
#     crs_src = src.rio.crs
#     crs_dst = pyproj.CRS.from_epsg(32632)

#     da_dst, _ = reproject_rasterdata(src, crs_to=crs_dst)
#     da_dst_src, _ = reproject_rasterdata(da_dst, crs_to=crs_src)

#     min_h = min(src.sizes["y"], da_dst_src.sizes["y"])
#     min_w = min(src.sizes["x"], da_dst_src.sizes["x"])
#     orig = np.asarray(src)[:min_h, :min_w].astype(float)
#     roundp = np.asarray(da_dst_src)[:min_h, :min_w].astype(float)

#     nodata = src.rio.nodata if hasattr(src, "rio") else None
#     valid_mask = ~np.isnan(orig) if nodata is None or np.isnan(nodata) else orig != nodata
#     orig_valid = orig[valid_mask]
#     roundp_valid = roundp[valid_mask]

#     assert np.allclose(orig, roundp, rtol=1e-1, atol=1e-1 * 8e3)


@pytest.mark.parametrize(
    "arr, expected_dtype, expected_nodata_bound",
    [
        (lambda: (np.arange(6).reshape((2, 3)).astype(np.int32), False), np.float64, 0),
        (lambda: (np.array([[np.nan, np.nan], [np.nan, np.nan]]), True), np.float64, 4),
    ],
)
def test_dtype_and_nodata_edgecases(arr, expected_dtype, expected_nodata_bound):
    a, _flag = arr()
    da = xarray.DataArray(a, dims=("y", "x"))
    da.rio.write_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    da.rio.write_transform(rasterio.transform.from_origin(0, a.shape[0], 1, 1), inplace=True)

    if _flag:
        da.rio.write_nodata(np.nan, inplace=True)
    out, summary = reproject_rasterdata(da, crs_to=pyproj.CRS.from_epsg(3857))

    assert out.dtype == expected_dtype
    assert isinstance(summary["nodata_dst"], int)
    assert summary["nodata_dst"] >= expected_nodata_bound
