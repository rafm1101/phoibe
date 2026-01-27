import numpy as np
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
