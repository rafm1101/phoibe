import numpy as np

from ..computations.results import RadialRuggedness


def to_dataframe(radial_ruggedness: RadialRuggedness):
    """Generate a dataframe summarising the metrics of all rays.

    Returns
    -------
    pandas.DataFrame
        Dataframe holding all metrics as one ray per row.

    Raises
    ------
    ImportError
        If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError as exception:
        raise ImportError("RadialRixResult.to_dataframe() requires pandas.") from exception

    records = [ray.describe() for ray in radial_ruggedness.rays]
    return pd.DataFrame(records)


def steep_segments_geodataframe(radial_ruggedness: RadialRuggedness):
    """Generate a GeoDataFrame holding all steep segments.

    Returns
    -------
    geopandas.GeoDataFrame
        Geodataframe where each row contains the geometry of one steep segment.

    Raises
    ------
    ImportError
        If geopandas is not installed.
    """
    try:
        import geopandas as gpd
    except ImportError as exception:
        raise ImportError("RadialRixResult.steep_segments_geodataframe() requires geopandas.") from exception

    records = []
    for ray in radial_ruggedness.rays:
        for i, segment in enumerate(ray.steep_segments):
            records.append(
                {
                    radial_ruggedness.keys.theta: ray.theta,
                    radial_ruggedness.keys.segment_id: i,
                    radial_ruggedness.keys.geometry: segment,
                }
            )
    return gpd.GeoDataFrame(
        records,
        columns=[radial_ruggedness.keys.theta, radial_ruggedness.keys.segment_id, radial_ruggedness.keys.geometry],
        geometry=radial_ruggedness.keys.geometry,
        crs=radial_ruggedness.rays[0].profile.ray.crs,
    )


# def steep_segments_geodataframe(self):
#     """GeoDataFrame of steep segments for visualization.

#     Returns
#     -------
#     geopandas.GeoDataFrame
#         Geodataframe where each row contains the geometry of one steep segment.

#     Raises
#     ------
#     ImportError
#         If geopandas is not installed.
#     """
#     try:
#         import geopandas as gpd
#     except ImportError as exception:
#         raise ImportError("RayResult.steep_segments_geodataframe() requires geopandas.") from exception

#     records = [{self.keys.segment_id: i, self.keys.geometry: seg} for i, seg in enumerate(self.steep_segments)]

#     if records:
#         steep_segments = gpd.GeoDataFrame(records, geometry=self.keys.geometry, crs=self.profile.ray.crs)
#     else:
#         steep_segments = gpd.GeoDataFrame(
#             columns=[self.keys.segment_id, self.keys.geometry],
#             geometry=self.keys.geometry,
#             crs=self.profile.ray.crs,
#         )

#     return steep_segments


def plot_polar(radial_ruggedness: RadialRuggedness, ax=None):
    """Create polar plot of ruggedness values.

    Returns
    -------
    tuple[Figure, Axes]
        Matplotlib figure and axes.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exception:
        raise ImportError("RadialRixResult.plot_polar() requires matplotlib.") from exception

    angles_radians = np.deg2rad(np.append(radial_ruggedness.angles, radial_ruggedness.angles[0]))
    values = radial_ruggedness.ruggednesses
    values = np.append(values, values[0])

    if ax is None:
        figure, ax = plt.subplots(subplot_kw={"projection": "polar"})

    ax.plot(angles_radians, values)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f"Directional RIX (threshold={radial_ruggedness.slope_critical:.2f})")
    ax.set_ylabel("RIX")

    return figure, ax
