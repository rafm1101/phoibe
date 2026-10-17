import typing

import numpy as np
import shapely

from .fieldsampler import FieldSampler
from .geometry import RayGeometry
from .profiles import NaNPolicy, RayProfile
from .results import RadialRuggedness, RayRuggedness


def compute_regular_rix(
    location: shapely.Point,
    sampler: FieldSampler,
    n_angles: int,
    R_km: float,
    dr_km: float,
    crs: typing.Any,
    slope_critical: float,
    nan_policy: NaNPolicy | str = "mask",
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
    # keys
    #     Column keys for the output.

    Returns
    -------
    RadialRixResult
        Gathered results of the evaluation.
    """
    policy = NaNPolicy(nan_policy)
    angles = np.linspace(0, 360, n_angles, endpoint=False)
    rays = []

    for theta in angles:
        ray = RayGeometry.from_compass_regular(location=location, theta=theta, R_km=R_km, dr_km=dr_km, crs=crs)
        ray_profile = RayProfile.create_regular(ray=ray, sampler=sampler, nan_policy=policy)
        rays.append(RayRuggedness(profile=ray_profile, slope_critical=slope_critical))

    return RadialRuggedness(rays=tuple(rays))
