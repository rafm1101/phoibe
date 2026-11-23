import numpy as np
from numpy.typing import NDArray

from phoibe.ergaleiothiki.tididi.validate_numerics import (
    _validate_broadcastable,
    _validate_non_negative,
    _validate_notna_finite,
)


def compass_polar_to_cartesian(
    angle: float | NDArray[np.floating], radius: float | NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Convert compass polar coordinates to Cartesian coordinates with the angle measured in degrees.

    Find North, East, South and West mapped to 0°, 90°, 180° and 270° clockwise contrasting mathematical conventions.

    Parameters
    ----------
    angle
        Angle [°].
    radius
        Radius [any].

    Returns
    -------
    x
        Array for Cartesian coordinate value x of the converted angle and radius.
    y
        Array for Cartesian coordinate value y of the converted angle and radius.
    """
    _angle = np.asarray(angle, dtype=float)
    _radius = np.asarray(radius, dtype=float)

    _validate_broadcastable(_angle, _radius, "`angle` and `radius`")
    _validate_notna_finite(_angle, "`angle`")
    _validate_notna_finite(_radius, "`radius`")
    _validate_non_negative(_radius, "`radius`")

    _angle_rad = np.deg2rad(_angle)
    x = np.sin(_angle_rad) * _radius
    y = np.cos(_angle_rad) * _radius

    return x, y


def cartesian_to_compass_polar(
    x: float | NDArray[np.floating], y: float | NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Convert Cartesian coordinates to compass polar coordinates with the angle measured in degrees.

    Find North, East, South and West mapped to 0°, 90°, 180° and 270° clockwise contrasting mathematical conventions.

    Parameters
    ----------
    x
        Array coordinate value x for Cartesian coordinate.
    y
        Array coordinate value y for Cartesian coordinate.

    Returns
    -------
    angle
        Array for the angle in degrees of the converted Cartesian coordinates x and y.
    radius
        Array for the radius of the converted Cartesian coordinates x and y.

    Notes
    -----
    1. (0, 0) is mapped to angle 0.0.
    """
    _x = np.asarray(x, dtype=float)
    _y = np.asarray(y, dtype=float)

    _validate_broadcastable(_x, _y, "`x` and `y`")
    _validate_notna_finite(_x, "`x`")
    _validate_notna_finite(_y, "`y`")

    angle = (270 - np.rad2deg(np.arctan2(-_y, -_x))) % 360
    radius = (_x**2 + _y**2) ** 0.5
    angle = _set_zero_masked_elements(angle, radius == 0.0)

    return angle, radius


def _set_zero_masked_elements(a: NDArray[np.floating], mask: NDArray[np.bool]) -> NDArray[np.floating]:
    if isinstance(a, (float, int)):
        a = 0.0 if mask else a
    elif isinstance(a, np.ndarray):
        a[mask] = 0
    else:
        raise TypeError("Type %s not respected! %s", type(a))
    return a


def wind_uv_to_compass_polar(
    u: float | NDArray[np.floating], v: float | NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Convert meteorological wind components (u, v) to compass polar coordinates.

    Parameters
    ----------
    u, v
        Pair of wind-(u, v) coordinates.

    Returns
    -------
    direction, speed
        Tuple of arrays.
    """
    _u = np.asarray(u, dtype=float)
    _v = np.asarray(v, dtype=float)

    _validate_broadcastable(_u, _v, "`u` and `v`")
    _validate_notna_finite(_u, "`u`")
    _validate_notna_finite(_v, "`v`")

    direction, speed = cartesian_to_compass_polar(-_u, -_v)
    return direction, speed


def degrees_to_sector(
    angles: NDArray[np.floating], n_sectors: int = 12
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Sort angles into `n_sectors` sectors named after its midpoint.

    Parameters
    ----------
    angles
       Angles [°].
    n_sectors
       Number of sectors the angles are assigned to.

    Returns
    -------
    assigned_sector_centers
        Array of sector centers assigned to each angle.
    sector_centers
        Array of sector center values.


    Notes
    -----
    1. Angles below 0 or above 360 are mapped back to [0, 360) (angles % 360).
    2. The sectors are left-open and right-closed.
       For n=12, the first sector is 0 and includes angles from 345° to 15°.
    3. NaN and Inf are mapped to NaN.
    """
    _angles = np.asarray(angles, dtype=float)

    assigned_sector_centers = np.full_like(_angles, np.nan)
    mask = np.isfinite(_angles)

    if mask.any():
        sector_width = 360.0 / n_sectors
        _angles_normalized = np.mod(_angles[mask], 360)

        sector_index = np.mod(np.ceil((_angles_normalized - sector_width / 2) / sector_width).astype(int), n_sectors)
        assigned_sector_centers[mask] = sector_index * sector_width

    sector_centers = np.linspace(0, 360, n_sectors + 1)

    return assigned_sector_centers, sector_centers[:-1]
