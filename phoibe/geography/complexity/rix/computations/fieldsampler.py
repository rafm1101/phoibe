from __future__ import annotations

import dataclasses
import logging
import typing

import numpy as np
import pyproj
import scipy.interpolate
import xarray
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

INTERPOLATION_METHODS = typing.Literal["linear", "nearest"]


@dataclasses.dataclass(frozen=True)
class FieldInfo:
    """Metadata of the underlying raster data.

    This remains unpopulated if no rioxarray information is provided.
    """

    crs: str | None
    """ESPG code as a string."""
    extent: tuple[float, float, float, float] | None
    """Extend of window (left, bottom, right, top)."""
    resolution: tuple[float, float] | None
    """Resolution of the raster data."""


@dataclasses.dataclass(frozen=True)
class Field:
    """Field to be sampled from."""

    da: xarray.DataArray
    """Field data."""
    info: FieldInfo
    """Fields metadata. Filled from `rio` atttribute while calling `from_array`."""
    x: str
    """Name of the `da`'s zonal coordinate."""
    y: str
    """Name of the `da`'s meridional coordinate."""

    @classmethod
    def from_array(cls, da: xarray.DataArray, dims: tuple[str, str] = ("x", "y")) -> Field:
        """Generate the field from a rioxarray dataarray."""
        if not set(dims).issubset(da.dims):
            raise ValueError(f"Field must have '{dims[0]}' and '{dims[1]}' coordinates. Found '{da.dims}'.")
        if hasattr(da, "rio"):
            info = FieldInfo(
                crs=crs.to_string() if (crs := da.rio.crs) is not None else None,
                extent=da.rio.bounds(),
                resolution=da.rio.resolution(),
            )
        else:
            info = FieldInfo(crs=None, extent=None, resolution=None)
        return cls(da=da, info=info, x=dims[0], y=dims[1])


class FieldSampler(typing.Protocol):
    """Protocol for all samplers."""

    field: Field
    """The field sampled from."""

    @property
    def crs(self) -> pyproj.CRS | None:
        """Return the sampler's CRS in case it has one, otherwise `None`."""
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
        1. Implementation may truncate the result if NaN values are encountered. The sampler shall manage that.
        2. NaN counts shall be passed for diagnosis reasons.
           They may the user to check the map bounds or data gaps.
        """
        raise NotImplementedError


class RegularGridXYSampler:
    """Sample from a 2D field on a regular xy grid."""

    field: Field
    """Field to be sampled from."""
    method: INTERPOLATION_METHODS
    """Interpolation method. One of 'linear' and 'nearest'."""

    def __init__(self, field: Field, method: INTERPOLATION_METHODS):
        self.field = field
        self.method = method

        x = field.da[field.x].values
        y = field.da[field.y].values
        self._interpolator = scipy.interpolate.RegularGridInterpolator(
            points=(y, x),
            values=field.da.transpose(field.y, field.x).values,
            method=method,
            bounds_error=False,
            fill_value=np.nan,
        )

    @property
    def crs(self) -> pyproj.CRS | None:
        if hasattr(self.field.da, "rio"):
            return self.field.da.rio.crs
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
def _validate_shape_same(*args, prefix: str) -> None:
    shapes = [array.shape for array in args]
    reference = shapes[0]
    for other in shapes[1:]:
        if reference != other:
            raise ValueError(f"{prefix}: Shapes do not agree.")


# TODO: tididi candidate.
def _validate_1D(*args, prefix: str) -> None:
    ndims = [array.ndim for array in args]
    for ndim in ndims:
        if ndim != 1:
            raise ValueError(f"{prefix} must be 1-dimensional arrays. Found {ndims}.")
