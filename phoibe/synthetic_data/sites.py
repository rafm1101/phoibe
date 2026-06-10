import typing

import geopandas as gpd
import numpy as np
import pandas.core.dtypes.common
import pyproj


def make_sites(
    sites: int | list[typing.Any] | tuple[typing.Any],
    bounds: tuple[float, ...],
    buffer: float = 0,
    crs: pyproj.CRS = None,
    seed: int | None = None,
) -> gpd.GeoDataFrame:
    """Generate a given number of random sites within the given bounds.

    Parameters
    ----------
    sites
        Either number of sites to generate, or site names for which coordinates are generated.
    bounds
        Bounds west, south, east, north in the desired CRS.
    buffer
        Distance from the bounds that will not be populated. Measured in units of the CRS.
    crs
        CRS. If `None`, then remains unset.
    seed
        Seed for the random number generator.

    Returns
    -------
    gdf
        GeoDataFrame with enumerated locations.

    Examples
    --------
    1. Generate a given number of sites on a given raster map.

    > phoibe.synthetic_data.sites.make_sites(sites=11, bounds=da.rio.bounds(), buffer=3e4, crs=da.rio.crs, seed=23)
    """
    west, south, east, north = bounds
    n_sites = int(sites) if isinstance(sites, int) else len(sites)
    index = sites if pandas.core.dtypes.common.is_list_like(sites) else list(range(n_sites))
    random_generator = np.random.default_rng(seed=seed)

    x_locations = random_generator.uniform(low=west + buffer, high=east - buffer, size=n_sites)
    y_locations = random_generator.uniform(low=south + buffer, high=north - buffer, size=n_sites)
    gs = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(x=x_locations, y=y_locations), data=index, columns=["name"], crs=crs
    )
    return gs
