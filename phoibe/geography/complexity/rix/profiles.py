import dataclasses
import enum
import logging

import numpy as np
from ergaleiothiki.tididi.validate_numerics import _validate_positive
from numpy.typing import NDArray

from .fieldsampler import FieldSampler
from .geometry import RayGeometry

LOGGER = logging.getLogger(__name__)


class NaNPolicy(enum.Enum):
    """NaN handling policy in sampled profiles."""

    TRUNCATE = "truncate"
    """Cut profile at first occurring NaN."""
    MASK = "mask"
    """Keep NaN values as-is in the profile."""
    ERROR = "error"
    """Raise ValueError if any NaN encounters."""


@dataclasses.dataclass(frozen=True)
class RayProfile:
    """Immutable ray profile of a scalar field along a ray.

    A `RayProfile` represents values of a scalar field sampled along a geometric ray.

    Creation
    --------
    create_regular
        Sample at regular grid points on the given ray.
    create_levelcrossing
        Sample at level crossings along the given ray.
    """

    ray_: RayGeometry
    """Underlying geometric ray defining the spatial path of the profile. Expected to be strictly increasing."""
    r_m: NDArray[np.floating]
    """Distances [m] from the ray origin at which the field is evaluated. Must be strictly increasing."""
    z: NDArray[np.floating]
    """Sampled scalar field values along the ray. Each value corresponds to the same index in `r_m`."""

    def __post_init__(self):
        if len(self.r_m) != len(self.z):
            raise ValueError(f"r_m and z must have same length: {len(self.r_m)} vs {len(self.z)}")
        if len(self.r_m) > 1:
            dr = np.diff(self.r_m)
            _validate_positive(dr, "delta r_m")

    @classmethod
    def create_regular(cls, ray: RayGeometry, sampler: FieldSampler, nan_policy: NaNPolicy):
        """Generate `RayProfile` from a regular grid.

        Parameters
        ----------
        ray
            Representation of a ray embedded in the 2D world.
        sampler
            Sampler of the field to sample from.
        nan_policy
            NaN handling policy.

        Returns
        -------
        RayProfile
            Immutable representation of the sampled field along `ray`.
        """
        z = sampler.sample(xs=ray.xs, ys=ray.ys)
        r_m, z = _apply_nan_policy(r_m=ray.r_m, z=z, theta=ray.theta, policy=nan_policy)
        return cls(ray_=ray, r_m=r_m, z=z)

    @property
    def ray(self) -> RayGeometry:
        return self.ray_

    @classmethod
    def create_levelcrossing(
        cls, ray: RayGeometry, sampler: FieldSampler, levels: NDArray[np.floating], nan_policy: NaNPolicy
    ):
        """Generate `RayProfile` from a level crossings.

        Parameters
        ----------
        ray
            Representation of a ray embedded in the 2D world and used as intermediate sampler.
        sampler
            Sampler of the field to sample from.
        levels
            Levels at which crossings shall be recognised.
        nan_policy
            NaN handling policy.

        Returns
        -------
        RayProfile
            Immutable representation of the sampled field along `ray`.
        """
        z_regular = sampler.sample(xs=ray.xs, ys=ray.ys)
        r_regular, z_regular = _apply_nan_policy(r_m=ray.r_m, z=z_regular, theta=ray.theta, policy=nan_policy)

        levels = np.asarray(levels, dtype=float)
        r_crossings, z_crossings = _compute_level_crossings(r=r_regular, z=z_regular, levels=levels)

        ray_resampled = RayGeometry.from_compass(location=ray.location, theta=ray.theta, r_m=r_crossings)
        return cls(ray_=ray_resampled, r_m=r_crossings, z=z_crossings)


def _apply_nan_policy(
    r_m: NDArray, z: NDArray, theta: float, policy: NaNPolicy
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Apply NaN handling policy to the sampled profile.

    Notes
    -----
    1. ERROR: Raise if any NaNs are found.
    2. TRUNCATE: Cut profile at the first NaN.
    3. MASK: Keep NaNs as is.
    """
    if not np.isnan(z).any():
        return r_m.copy(), z.copy()

    if policy is NaNPolicy.ERROR:
        raise ValueError("NaNs encountered in ray profile for theta=%.1f.", theta)

    elif policy is NaNPolicy.TRUNCATE:
        first_nan = np.where(np.isnan(z))[0][0]
        if first_nan == 0:
            raise ValueError("Ray for theta=%.1f contains no valid numbers.", theta)
        return r_m[:first_nan].copy(), z[:first_nan].copy()

    elif policy is NaNPolicy.MASK:
        return r_m.copy(), z.copy()

    else:
        raise ValueError("Unknown NaN policy: %s.", policy)


def _compute_level_crossings(
    r: NDArray[np.floating], z: NDArray[np.floating], levels: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Determine level crossings from any sampled field."""
    levels = np.asarray(levels, dtype=float)
    r_crossings, z_crossings = [r[0]], [z[0]]

    for k in range(len(r) - 1):
        r_current, r_next = r[k], r[k + 1]
        z_current, z_next = z[k], z[k + 1]

        z_min, z_max = (z_current, z_next) if z_current < z_next else (z_next, z_current)
        levels_in_segment = levels[(levels >= z_min) & (levels <= z_max)]

        if len(levels_in_segment) == 0:
            continue

        if z_next == z_current:
            r_crossings.extend([r_current, r_next])
            z_crossings.extend([z_current, z_next])

        if z_next < z_current:
            levels_in_segment = levels_in_segment[::-1]

        for level in levels_in_segment:
            alpha = (level - z_current) / (z_next - z_current)
            r_crossing = r_current + alpha * (r_next - r_current)
            r_crossings.append(r_crossing)
            z_crossings.append(level)

    r_crossings.append(r[-1])
    z_crossings.append(z[-1])

    r_crossings_array = np.asarray(r_crossings, dtype=float)
    z_crossings_array = np.asarray(z_crossings, dtype=float)

    if r_crossings_array.size >= 2:
        eps = 1e-7
        mask_non_duplicates = np.concat([np.diff(r_crossings_array) > eps, [True]])
        r_crossings_array = r_crossings_array[mask_non_duplicates]
        z_crossings_array = z_crossings_array[mask_non_duplicates]

    return r_crossings_array, z_crossings_array
