import typing

import cartopy.feature

SCALES = typing.Literal["10m", "50m", "110m"]


def plot_landmarks_to_map(ax, with_scale: SCALES = "10m"):
    """To a `GeoAxes`, add landmarks.

    Parameters
    ----------
    ax
        Ax object to add the landmarks to.
    with_scale
        Scale at which landmarks are added.

    Returns
    -------
    ax
        Ax object w/ added landmarks.
    """

    ax.add_feature(cartopy.feature.OCEAN, facecolor="blue")
    ax.add_feature(cartopy.feature.LAKES.with_scale(with_scale), facecolor="dodgerblue")
    ax.add_feature(cartopy.feature.RIVERS.with_scale(with_scale), edgecolor="dodgerblue")

    ax.add_feature(cartopy.feature.COASTLINE)

    ax.add_feature(cartopy.feature.BORDERS.with_scale(with_scale))
    ax.add_feature(cartopy.feature.LAND.with_scale(with_scale))

    return ax
