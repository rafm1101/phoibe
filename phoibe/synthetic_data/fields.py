import numpy as np
import pyproj
import rasterio
import xarray


def make_planar_field(nx: int, ny: int, dx: float, dy: float, slope_x: float, slope_y: float) -> xarray.DataArray:
    """Create a planar scalar field with constant gradients.

    The field is defined as
        z(x, y) = slope_x * x + slope_y * y
    on a regular Cartesian grid centered around the origin.

    Parameters
    ----------
    nx, ny
        Number of grid points in positive and negative x/y direction. Total size is (2*ny, 2*nx).
    dx, dy
        Grid spacing in x and y direction.
    slope_x, slope_y
        Constant gradients of the field in x and y direction.

    Returns
    -------
    xarray.DataArray
        2D field with coordinates ('x', 'y') and dimension order ('y', 'x').
    """
    x = np.arange(-nx, nx) * dx
    y = np.arange(-ny, ny) * dy
    xx, yy = np.meshgrid(x, y)

    z = slope_x * xx + slope_y * yy

    return xarray.DataArray(data=z, coords={"x": x, "y": y}, dims=("y", "x"), name="elevation")


def make_eggbox_field(nx: int, ny: int, dx: float, dy: float, freq_x: float, freq_y: float) -> xarray.DataArray:
    """Create a periodic 'eggbox' scalar field.

    The field is defined as
        z(x, y) = cos(freq_x * x) * cos(freq_y * y)
    on a regular Cartesian grid centered around the origin.

    Parameters
    ----------
    nx, ny
        Number of grid points in positive and negative x/y direction. Total size is (2*ny, 2*nx).
    dx, dy
        Grid spacing in x and y direction.
    freq_x, freq_y
        Spatial frequencies in x and y direction.

    Returns
    -------
    xarray.DataArray
        2D periodic field with coordinates ('x', 'y') and dimension order ('y', 'x').
    """
    x = np.arange(-nx, nx) * dx
    y = np.arange(-ny, ny) * dy
    xx, yy = np.meshgrid(x, y)

    z = np.cos(freq_x * xx) * np.cos(freq_y * yy)

    return xarray.DataArray(data=z, coords={"x": x, "y": y}, dims=("y", "x"), name="elevation")


def make_radial_wave_field(nx: int, ny: int, dx: float, dy: float, freq: float) -> xarray.DataArray:
    """Create a radial wave field depending on angular direction.

    The field is defined as
        z(x, y) = cos(freq * atan2(x, y))
    on a regular Cartesian grid centered around the origin.

    Parameters
    ----------
    nx, ny
        Number of grid points in positive and negative x/y direction. Total size is (2*ny, 2*nx).
    dx, dy
        Grid spacing in x and y direction.
    freq
        Angular frequency controlling the number of oscillations around the origin.

    Returns
    -------
    xarray.DataArray
        2D scalar field with angular periodicity and coordinates ('x', 'y').
    """
    x = np.arange(-nx, nx) * dx
    y = np.arange(-ny, ny) * dy
    xx, yy = np.meshgrid(x, y)

    z = np.cos(np.atan2(xx, yy) * freq)

    return xarray.DataArray(data=z, coords={"x": x, "y": y}, dims=("y", "x"), name="elevation")


def make_field_rio(
    da: xarray.DataArray,
    bounds: tuple[float, float, float, float],
    crs: pyproj.CRS | int | str,
    dtype: str | None = None,
    nodata: int | float | None = np.nan,
) -> xarray.DataArray:
    """Convert a vanilla dataarray to a raster dataarray.

    Parameters
    ----------
    da
        Dataarray holding raster data.
    bounds
        Bounds west, south, east, north in the desired CRS.
    crs
        CRS provided as CRS object or EPSG code or string.
    dtype
        Dtype to be set. If `None` use the input dtype.
    nodata
        Nodata identifier to be set, e.g. `None`, `np.nan`, `-32768`.
        Must be compatible to `dtype` (`np.nan` is no `int`).

    Returns
    -------
    dario
        Enriched field.

    Notes
    -----
    `rioxarray` is loaded silently if not yet done. Required for adding CRS-related functionality.
    """
    try:
        import rioxarray  # noqa: F401
    except ImportError as exception:
        raise ImportError("`make_field_rio` requires the package `rioxarray`. Please import.") from exception
    crs_to = pyproj.CRS.from_user_input(crs)

    width, height = da.sizes["x"], da.sizes["y"]
    dtype_to = dtype if dtype is not None else da.dtype

    west, south, east, north = bounds
    transform = rasterio.transform.from_bounds(
        west=west, south=south, east=east, north=north, width=width, height=height
    )

    field = np.asarray(da.values, dtype=dtype_to)
    dario = xarray.DataArray(data=field, dims=("y", "x"), name="band1")

    dario.rio.write_crs(crs_to, inplace=True)
    dario.rio.write_transform(transform, inplace=True)
    dario.rio.set_spatial_dims(x_dim="x", y_dim="y")

    if nodata is not None:
        dario.rio.write_nodata(nodata, inplace=True)

    return dario
