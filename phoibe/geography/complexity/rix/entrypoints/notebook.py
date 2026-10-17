"""Helpers for working with results in a notebook: views and plots of what the pipeline returns.

Presentation does not belong to the domain. Heavy optional libraries are imported on use.
"""

from __future__ import annotations

import typing
from collections.abc import Hashable

import numpy as np
import pandas as pd

from ..pipeline import tables
from ..pipeline.specification import ProductSpecification

if typing.TYPE_CHECKING:
    import geopandas as gpd

    from ..computations.results import RadialRuggedness


def ray_metrics(radial: RadialRuggedness) -> pd.DataFrame:
    """Metrics of every ray of a location, one ray per row. For inspection; not an output of the product."""
    return pd.DataFrame(
        [
            {
                "theta": ray.theta,
                "ruggedness": ray.ruggedness,
                "total_length_m": ray.total_length_m,
                "steep_length_m": ray.steep_length_m,
                "max_abs_slope": ray.max_abs_slope,
                "n_steep_segments": ray.n_steep_segments,
            }
            for ray in radial.rays
        ]
    )


def steep_segments_geodataframe(
    spec: ProductSpecification, radial: RadialRuggedness, location_id: Hashable = "location"
) -> gpd.GeoDataFrame:
    """Steep segments of one location, in the same form as the table of the product."""
    return tables.steep_segments(spec, {location_id: radial})


def plot_polar(radial: RadialRuggedness, ax=None):
    """Polar plot of the ruggedness over the ray directions.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes. If `ax` is given, its figure is returned.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("plot_polar() requires matplotlib.") from error

    angles = np.deg2rad(np.append(radial.angles, radial.angles[0]))
    values = np.append(radial.ruggednesses, radial.ruggednesses[0])
    if ax is None:
        figure, ax = plt.subplots(subplot_kw={"projection": "polar"})
    else:
        figure = ax.figure
    ax.plot(angles, values)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f"Directional RIX (threshold={radial.slope_critical:.3f})")
    ax.set_ylabel("RIX")
    return figure, ax
