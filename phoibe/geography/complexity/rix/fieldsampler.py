import logging
from typing import Literal, Protocol

import numpy as np
import pyproj
import scipy.interpolate
import xarray
from numpy.typing import NDArray

from .config import ColumnKeys

LOGGER = logging.getLogger(__name__)

COLUMN_KEYS = ColumnKeys()

INTERPOLATION_METHODS = Literal["linear", "nearest"]


class FieldSampler(Protocol):
    @property
    def crs(self) -> pyproj.CRS | None:
        """Return the sampler's CRS in case it has one."""
        raise NotImplementedError

    def sample(self, xs: NDArray[np.floating], ys: NDArray[np.floating]) -> tuple[NDArray[np.floating], int]:
        """Sample field values along a path.

        Parameters
        ----------
        xs
            Array of x coordinates.
        ys
            Array of y coordinates of same length.

        Returns
        -------
        NDArray[np.floating], int
            Sampled values and NaN count.

        Notes
        -----
        1. Implementation may truncate the result if NaN values are encountered.
        2. NaN counts shall be passed for diagnosis reasons.
           They may the user to check the map bounds or data gaps.
        """
        raise NotImplementedError


class RegularGridXYSampler:
    """Sample from a 2D field on a regular xy grid."""

    da: xarray.DataArray
    """Field to be sampled from. Assume that coordinates are 'x' and 'y'."""
    method: INTERPOLATION_METHODS
    """Interpolation method. One of 'linear' and 'nearest'."""
    keys: ColumnKeys
    """Keys identifying dimensions."""

    def __init__(self, da: xarray.DataArray, method: INTERPOLATION_METHODS, keys: ColumnKeys = COLUMN_KEYS):
        if not {keys.x, keys.y}.issubset(da.dims):
            raise ValueError(f"Field must have '{keys.x}' and '{keys.y}' coordinates.")
        self.da = da
        self.method = method

        x = da[keys.x].values
        y = da[keys.y].values
        self._interpolator = scipy.interpolate.RegularGridInterpolator(
            points=(y, x),
            values=da.transpose(keys.y, keys.x).values,
            method=method,
            bounds_error=False,
            fill_value=np.nan,
        )

    @property
    def crs(self) -> pyproj.CRS | None:
        if hasattr(self.da, "rio"):
            return self.da.rio.crs
        else:
            return None

    def sample(self, xs: NDArray[np.floating], ys: NDArray[np.floating]) -> tuple[NDArray[np.floating], int]:
        _xs = np.asarray(xs, dtype=float)
        _ys = np.asarray(ys, dtype=float)

        _validate_1D(_xs, _ys, prefix="`xs` and `ys`")
        _validate_shape_same(_xs, _ys, prefix="`xs` and `ys`")

        sample_points = np.column_stack([_ys, _xs])
        z = self._interpolator(xi=sample_points)

        if np.any(np.isnan(z)):
            LOGGER.info(
                "NaNs encountered during sampling from field, possibly outside domain: %s NaNs out of %s.",
                np.isnan(z).sum(),
                z.size,
            )
        return z, np.isnan(z).sum()


# TODO: tididi candidate.
def _validate_shape_same(*args, prefix: str):
    shapes = [array.shape for array in args]
    reference = shapes[0]
    for other in shapes[1:]:
        if reference != other:
            raise ValueError(f"{prefix}: Shapes do not agree.")


# TODO: tididi candidate.
def _validate_1D(*args, prefix: str):
    ndims = [array.ndim for array in args]
    for ndim in ndims:
        if ndim != 1:
            raise ValueError(f"{prefix} must be 1-dimensional arrays. Found {ndims}.")
