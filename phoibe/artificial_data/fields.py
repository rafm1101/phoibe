import numpy as np
import xarray


def make_planar_field(nx: int, ny: int, dx: float, dy: float, slope_x: float, slope_y: float) -> xarray.DataArray:
    x = np.arange(-nx, nx) * dx
    y = np.arange(-ny, ny) * dy
    xx, yy = np.meshgrid(x, y)

    z = slope_x * xx + slope_y * yy

    return xarray.DataArray(data=z, coords={"x": x, "y": y}, dims=("y", "x"), name="elevation")


def make_eggbox_field(nx: int, ny: int, dx: float, dy: float, freq_x: float, freq_y: float) -> xarray.DataArray:
    x = np.arange(-nx, nx) * dx
    y = np.arange(-ny, ny) * dy
    xx, yy = np.meshgrid(x, y)

    z = np.cos(freq_x * xx) * np.cos(freq_y * yy)

    return xarray.DataArray(data=z, coords={"x": x, "y": y}, dims=("y", "x"), name="elevation")


def make_radial_wave_field(nx: int, ny: int, dx: float, dy: float, freq: float) -> xarray.DataArray:
    x = np.arange(-nx, nx) * dx
    y = np.arange(-ny, ny) * dy
    xx, yy = np.meshgrid(x, y)

    z = np.cos(np.atan2(xx, yy) * freq)

    return xarray.DataArray(data=z, coords={"x": x, "y": y}, dims=("y", "x"), name="elevation")
