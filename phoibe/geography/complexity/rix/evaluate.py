import logging
import typing

import numpy as np
import shapely.geometry
from numpy.typing import NDArray

from .fieldsampler import FieldSampler
from .geometry import RayGeometry
from .interface import Keys
from .profiles import NaNPolicy, RayProfile
from .results import RadialRuggedness, RayRuggedness

LOGGER = logging.getLogger(__name__)

KEYS = Keys()


def compute_regular_rix(
    location: shapely.Point,
    sampler: FieldSampler,
    n_angles: int,
    R_km: float,
    dr_km: float,
    crs: typing.Any,
    slope_critical: float,
    nan_policy="mask",
    keys: Keys = KEYS,
):
    """Compute the ruggedness index RIX of a location. The RIX assesses height profiles along
    rays originating at `location`.

    Parameters
    ----------
    location
        Coordinates of the location to be assessed.
    sampler
        A sampler of field values from a regular, metric grid.
    n_angles
        Number of rays to cover the entire circle.
    R_km
        Distance [km] to which the profiles are considered.
    dr_km
        Stepsize [km] to sample from the field.
    crs
        CRS of the location.
    slope_critical
        Threshold on the slope between two points for a segment to be considered steep.
    keys
        Column keys for the output.

    Returns
    -------
    RadialRixResult
        Gathered results of the evaluation.
    """
    angles = np.linspace(0, 360, n_angles, endpoint=False)
    results = []

    for theta in angles:
        ray = RayGeometry.from_compass_regular(location=location, theta=theta, R_km=R_km, dr_km=dr_km, crs=crs)
        ray_profile = RayProfile.create_regular(ray=ray, sampler=sampler, nan_policy=NaNPolicy(nan_policy), keys=keys)
        results.append(RayRuggedness(profile=ray_profile, slope_critical=slope_critical, keys=keys))

    return RadialRuggedness(rays=tuple(results))


def segment_lengths(profile: RayProfile) -> NDArray[np.floating]:
    """Physical length [m] of each profile segment.

    Parameters
    ----------
    profile
        Any ray profile.

    Returns
    -------
    np.array
        Array of the lengths of each segment, shorter by 1 that `profile.r_m`.
    """
    return np.diff(profile.r_m)


def total_length_m(profile: RayProfile) -> float:
    """Physical length [m] of the full profile ray.

    Parameters
    ----------
    profile
        Any ray profile.

    Returns
    -------
    float
        Total length of `profile.r_m`.
    """
    return float(np.sum(segment_lengths(profile)))


def slopes(profile: RayProfile) -> NDArray[np.floating]:
    """Slope of the profile between consecutive supporting points.

    Parameters
    ----------
    profile
        Any ray profile.

    Returns
    -------
    np.array
        Array of the slopes of each segment, shorter by 1 thatn `profile.r_m`.
    """
    if len(profile.z) < 2:
        return np.array([np.nan], dtype=float)
    dz = np.diff(profile.z)
    dr = segment_lengths(profile=profile)
    if not np.all(dr > 0):
        raise ValueError(f"Ray for theta={profile.ray.theta:.1f} requires strictly increasing distance from origin.")
    return dz / dr


def steep_mask(profile: RayProfile, slope_critical: float) -> NDArray[np.bool_]:
    """Boolean mask indicating segments steeper than a critical slope.

    Parameters
    ----------
    profile
        Any ray profile.
    slope_critical
        Slope [m/m] above which a segment is consideres as steep.

    Returns
    -------
    np.array
        Array of Booleans indicating if a given segment is to be considered steep, shorter by 1 thatn `profile.r_m`.
    """
    slope_values = slopes(profile=profile)
    if np.isnan(slope_values).all():
        return np.zeros_like(slope_values, dtype=bool)
    return np.abs(slope_values) > slope_critical


def ruggedness(profile: RayProfile, slope_critical: float) -> float:
    """Ruggedness along the ray.

    Parameters
    ----------
    profile
        Any ray profile.
    slope_critical
        Slope [m/m] above which a segment is consideres as steep.

    Returns
    -------
    float
        Ruggedness aka fraction of the length of steep segments among all segments along a ray.
    """
    dr = segment_lengths(profile)
    if len(dr) == 0:
        return np.nan

    mask = steep_mask(profile=profile, slope_critical=slope_critical)
    steep_length = float(np.sum(dr[mask]))
    length_m = total_length_m(profile=profile)

    if length_m <= 0:
        return 0.0
    else:
        return float(steep_length / length_m)


def _get_true_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Return the indices [start, stop) of intervals of contiguous `True` values."""
    changes = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    stops = np.where(changes == -1)[0]

    # runs = []
    # start = None

    # for index, value in enumerate(mask):
    #     if value and start is None:
    #         start = index
    #     elif not value and start is not None:
    #         runs.append((start, index))
    #         start = None

    # if start is not None:
    #     runs.append((start, len(mask)))
    return list(zip(starts, stops, strict=True))


def steep_segment_indices(profile: RayProfile, slope_critical: float) -> list[tuple[int, int]]:
    """Return the indices [start, stop) of intervals of contiguous steep segments.

    Parameters
    ----------
    profile
        Any ray profile.
    slope_critical
        Slope [m/m] above which a segment is consideres as steep.

    Returns
    -------
    list[tuple[int, int]]
        List containing all pairs of neighbouring indices of steep segments in `profile`.
    """
    mask = steep_mask(profile=profile, slope_critical=slope_critical)
    return _get_true_runs(mask=mask)


def steep_ray_segments(profile: RayProfile, slope_critical: float) -> list[shapely.geometry.LineString]:
    """Geometric representation of contiguous steep segments.

    Parameters
    ----------
    profile
        Any ray profile.
    slope_critical
        Slope [m/m] above which a segment is consideres as steep.

    Returns
    -------
    segments
        List containing all steep segments as geometric objects.
    """
    segments = []
    ray_segments = profile.ray
    for start, stop in steep_segment_indices(profile=profile, slope_critical=slope_critical):
        xs = ray_segments.xs[start : stop + 1]  # noqa: E203
        ys = ray_segments.ys[start : stop + 1]  # noqa: E203
        if len(xs) >= 2:
            segments.append(shapely.geometry.LineString(zip(xs, ys, strict=True)))
    return segments
