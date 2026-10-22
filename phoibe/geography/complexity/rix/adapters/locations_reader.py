"""Inbound adapter: Read point locations (sites or reference locations) from a vector file.

Simple version: Whatever geopandas reads (GeoPackage, GeoJSON, Shapefile, ...). The identifier column and the CRS are
taken as they are; the use case normalises the identifier and reports CRS problems as findings.
"""

from __future__ import annotations

import logging
import os

import geopandas as gpd
import pyogrio

from ..pipeline.errors import InputRequirementError, PortError

LOGGER = logging.getLogger(__name__)


def read_locations(path: str | os.PathLike[str], *, layer: str | None = None) -> gpd.GeoDataFrame:
    """Read point locations from a vector file.

    Parameters
    ----------
    path
        Vector file.
    layer
        Layer to read if the file has several. Default: the first one.

    Raises
    ------
    PortError
        If the file or layer cannot be read.
    InputRequirementError
        If the file holds no locations, or anything but points.
    """
    try:
        locations = gpd.read_file(path, layer=layer)
    except (OSError, ValueError, pyogrio.errors.DataSourceError, pyogrio.errors.DataLayerError) as error:
        raise PortError(f"Could not read locations from {path}: {error}") from error

    if locations.empty:
        raise InputRequirementError(f"{path} contains no locations.")
    kinds = sorted(set(locations.geometry.geom_type.dropna()) - {"Point"})
    if kinds or locations.geometry.isna().any():
        raise InputRequirementError(f"{path} must contain point locations only, found: {kinds or 'missing geometry'}.")
    LOGGER.debug("Read %d locations from %s, CRS %s", len(locations), path, locations.crs)
    return locations
