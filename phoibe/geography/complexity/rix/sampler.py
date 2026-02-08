import abc
import logging
from typing import Literal

import numpy as np
import scipy.interpolate
import xarray
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

INTERPOLATION_METHODS = Literal["linear", "nearest"]


class FieldSampler(abc.ABC):
    @abc.abstractmethod
    def sample(self, xs: NDArray[np.floating], ys: NDArray[np.floating]) -> NDArray[np.floating]:
        """Sample field values along a path.

        Parameters
        ----------
        xs
            Array of x coordinates.
        ys
            Array of y coordinates of same length.

        Notes
        -----
        1. Implementation may truncate the result if NaN values are encountered.
        """
        raise NotImplementedError


class RegularGridXYSampler(FieldSampler):
    """Sample from a 2D field on a regular xy grid."""

    da: xarray.DataArray
    """Field to be sampled from. Assume that coordinates are 'x' and 'y'."""
    method: INTERPOLATION_METHODS
    """Interpolation method. One of 'linear' and 'nearest'."""

    def __init__(self, da: xarray.DataArray, method: INTERPOLATION_METHODS):
        if not {"x", "y"}.issubset(da.coords):
            raise ValueError("Field must have 'x' and 'y' coordinates.")
        self.da = da
        self.method = method

        x = da["x"].values
        y = da["y"].values
        self._interpolator = scipy.interpolate.RegularGridInterpolator(
            points=(y, x), values=da.transpose("y", "x").values, method=method, bounds_error=False, fill_value=np.nan
        )

    def sample(self, xs: NDArray[np.floating], ys: NDArray[np.floating]) -> NDArray[np.floating]:
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
        return z


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
