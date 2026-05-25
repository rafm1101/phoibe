import logging

import affine
import numpy as np
import pyproj
import rasterio.transform
import rasterio.warp
import xarray

LOGGER = logging.getLogger(__name__)


def _get_crs(da: xarray.DataArray, crs: pyproj.CRS | int | str | None = None) -> pyproj.CRS | None:
    """Return CRS from either a Dataarray (priority), or external provision.

    Parameters
    ----------
    da
        Dataarray holding raster data in some CRS.
    crs
        CRS information in case not provided by `da`.

    Returns
    -------
    crs_result
        The `da`'s crs or the provided one. Values `None` if neither is given.
    """
    if hasattr(da, "rio") and getattr(da.rio, "crs", None) is not None:
        crs_result = da.rio.crs
    elif crs is not None:
        crs_result = pyproj.CRS.from_user_input(crs)
    else:
        crs_result = None
    return crs_result


def _compute_pixel_centers(transform: affine.Affine, width: int, height: int) -> tuple[list, list]:
    """Compute the pixel centers for the given transformation."""
    xs, _ = rasterio.transform.xy(transform=transform, rows=[0] * width, cols=list(range(width)), offset="center")
    _, ys = rasterio.transform.xy(transform=transform, rows=list(range(height)), cols=[0] * height, offset="center")
    return xs, ys


def _isnan(val) -> bool:
    try:
        return np.isnan(val)
    except Exception:
        return False


def reproject_rasterdata(
    da: xarray.DataArray,
    crs_to: pyproj.CRS | int | str,
    crs_from: pyproj.CRS | int | str | None = None,
    resampling=rasterio.warp.Resampling.bilinear,
    resolution: int | None = None,
) -> tuple[xarray.DataArray, dict]:
    """Reproject raster data to a different coordinate reference system.

    Parameters
    ----------
    da
        Dataarray holding raster data. Prefer it to bring its CRS information readable by `pyproj`.
    crs_to
        CRS to project to.
    crs_from
        CRS to project from, also accepted by its EPSG code. Use if `da` does not bring along one.
        If both are provided, `crs_from` is ignored.
    resampling
        Resampling method.
    resolution
        If `None`, the resulting resolution is determined by `rasterio`.
        Otherwise measured in the `crs_to` measurement system.

    Returns
    -------
    da_to
        Reprojected data.
    summary
        Gathered transformation information.

    Notes
    -----
    1. Computations in-memory. For rasters up to 10MP fine. For larger datasets, implement a chunked reprojection.
    2. Supports north-up destination crs. For the other case, implement `ys` flip on `destination_transform.e<0` only.
    3. Supports a single band. For multiple bands, extend functionality beforehand.
    """

    crs_to = pyproj.CRS.from_user_input(crs_to)
    crs_from = _get_crs(da=da, crs=crs_from)
    if crs_from is None:
        raise ValueError("No source CRS information given. As not part of `da`, provide as parameter `crs_from`.")

    source_bounds = da.rio.bounds()
    source_width, source_height = int(da.sizes["x"]), int(da.sizes["y"])
    source_transform = rasterio.transform.from_bounds(*source_bounds, source_width, source_height)
    LOGGER.debug(f"{source_bounds=}")
    LOGGER.debug(f"{source_width=}, {source_height=}")
    LOGGER.debug(f"{source_transform=}")

    if hasattr(da, "rio"):
        try:
            fill_value = int(da.rio.nodata)
        except ValueError:
            fill_value = da.rio.nodata
        except TypeError:
            fill_value = None
    else:
        fill_value = None

    destination_transform, destination_width, destination_height = rasterio.warp.calculate_default_transform(
        src_crs=crs_from,
        dst_crs=crs_to,
        width=source_width,
        height=source_height,
        left=source_bounds[0],
        bottom=source_bounds[1],
        right=source_bounds[2],
        top=source_bounds[3],
        resolution=resolution,
    )
    LOGGER.debug(f"{destination_width=}, {destination_height=}")
    LOGGER.debug(f"{destination_transform=}")
    destination = np.full(
        shape=(int(destination_height), int(destination_width)), fill_value=fill_value, dtype="float64"
    )

    _, remainder = rasterio.warp.reproject(
        source=np.asarray(da, dtype=float),
        destination=destination,
        src_transform=source_transform,
        src_crs=crs_from,
        src_nodata=fill_value,
        dst_transform=destination_transform,
        dst_crs=crs_to,
        dst_nodata=fill_value,
        resampling=resampling,
    )

    xs, ys = _compute_pixel_centers(
        transform=destination_transform, width=int(destination_width), height=int(destination_height)
    )
    LOGGER.debug(f"{xs=}")
    LOGGER.debug(f"{ys=}")
    if ys[0] < ys[-1]:
        ys = ys[::-1]
        destination = destination[::-1, :]

    da_to = xarray.DataArray(data=destination, dims=("y", "x"), coords={"x": xs, "y": ys}, attrs=da.attrs.copy())
    da_to.rio.write_crs(crs_to, inplace=True)
    da_to.rio.write_transform(destination_transform, inplace=True)
    da_to.rio.write_nodata(fill_value, inplace=True)

    if _isnan(fill_value):
        source_nodata_count = int(np.isnan(np.asarray(da)).sum())
        destination_nodata_count = int(np.isnan(destination).sum())
    else:
        source_nodata_count = int((np.asarray(da) == fill_value).sum())
        destination_nodata_count = int((destination == fill_value).sum())

    summary = {
        "crs_from": crs_from.to_string(),
        "bounds_src": dict(zip(("left", "bottom", "right", "top"), source_bounds, strict=True)),
        "range_src": {"width": source_width, "height": source_height},
        "range_dst": {"width": int(destination_width), "height": int(destination_height)},
        "transform": remainder,
        "nodata_src": source_nodata_count,
        "nodata_dst": destination_nodata_count,
    }

    LOGGER.debug("Reprojection: %s -> %s, dst shape=%s", crs_from.to_string(), crs_to.to_string(), destination.shape)

    return da_to, summary
