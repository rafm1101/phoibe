"""Inbound adapter: Read a digital elevation model from a GeoTIFF into a `Field`.

The DEM is loaded completely: The domain must not hold lazy arrays (DP4).

Upcoming: Cutting a window out of the partitioned source data (1° x 1° patches) is a follow-up.
This adapter takes what it is given.

`rioxarray` is an optional dependency and imported on use (C4).
"""

from __future__ import annotations

import logging
import os

from ..computations.fieldsampler import Field
from ..pipeline.errors import PortError

LOGGER = logging.getLogger(__name__)


def read_field(path: str | os.PathLike[str], *, band: int = 1) -> Field:
    """Read a raster file into a `Field` with CRS, extent and resolution filled from the file.

    Nodata cells become NaN, so they are handled by the NaN policy like any other data gap.

    Parameters
    ----------
    path
        GeoTIFF or any raster format readable by rasterio.
    band
        Band holding the elevations (1-based).

    Raises
    ------
    PortError
        If rioxarray is not installed, or the file cannot be read, or the band does not exist.
    """
    try:
        import rioxarray
        import xarray
    except ImportError as error:
        raise PortError("Reading a DEM from file requires the optional dependency 'rioxarray'.") from error

    try:
        opened = rioxarray.open_rasterio(path, masked=True)
        if not isinstance(opened, xarray.DataArray):
            raise PortError(f"{path} holds several datasets (sub-datasets); pass a single-raster file.")
        data = opened
        if "band" in data.dims:
            data = data.sel(band=band, drop=True)
        data = data.load()
    except PortError:
        raise
    except (OSError, KeyError, ValueError) as error:
        raise PortError(f"Could not read a DEM from {path}: {error}") from error
    except Exception as error:
        # rasterio raises its own hierarchy (RasterioIOError, CRSError, ...)
        if type(error).__module__.startswith("rasterio"):
            raise PortError(f"Could not read a DEM from {path}: {error}") from error
        raise

    LOGGER.debug("Read DEM %s: shape %s, CRS %s", path, data.shape, data.rio.crs)
    return Field.from_array(data)
