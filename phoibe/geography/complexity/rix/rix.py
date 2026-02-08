import logging

import numpy as np
from ergaleiothiki.perdix import LocationCCS

from .rays import NaNPolicy
from .rays import RayGeometry
from .rays import RegularRayProfile
from .sampler import FieldSampler

LOGGER = logging.getLogger(__name__)


def compute_radial_rix(
    location_ccs: LocationCCS, sampler: FieldSampler, n_angles: int, R_km, dr_km, slope_critical=0.3
):
    """Compute the ruggedness index RIX of a location. The RIX assesses height profiles along
    rays originating at `location_ccs`.

    Parameters
    ----------
    location_ccs
        Coordinates of the location to be assessed.
    sampler
        A sampler of field values from a regular, metric grid.
    n_angles
        Number of rays to cover the entire circle.
    R_km
        Distance [km] to which the profiles are considered.
    dr_km
        Stepsize [km] to sample from the field.
    slope_critical
        Threshold on the slope between two points for a segment to be considered steep.

    Returns
    -------
    rix_values
        RIX for each ray.
    ray_profiles
        Gathered ray profiles.
    steep_ray_segments
        Collection of steep segements.
    """
    angles = np.linspace(0, 360, n_angles, endpoint=False)
    ray_profiles = []
    rix_values = []
    steep_ray_segments = []

    for theta in angles:
        ray = RayGeometry.from_compass_regular(location=location_ccs, theta=theta, R_km=R_km, dr_km=dr_km)
        ray_profile = RegularRayProfile(ray=ray, sampler=sampler, nan_policy=NaNPolicy.ERROR)
        ray_profiles.append(ray_profile)
        rix_values.append(ray_profile.rix(slope_critical=slope_critical))
        steep_ray_segments.append(ray_profile.steep_ray_segments(slope_critical=slope_critical))
    return np.array(rix_values), ray_profiles, steep_ray_segments
