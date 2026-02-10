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

    # @dataclasses.dataclass(frozen=True)
    # class RayProfile(RayProfile):
    #     """Ray profile sampled on a regular, ray-aligned grid.

    #     The profile is sampled at the regularly spaced supporting points w/o any additional resampling or
    # interpolation.
    #     """

    #     ray_: RayGeometry
    #     """Reference geometry of the ray. Used for profile discretization and geometric output."""
    #     r_m: NDArray[np.floating]
    #     """Distances [m] from the ray origin at which the field is evaluated. Equidistant."""
    #     z: NDArray[np.floating]
    #     """Sampled scalar field values along the ray. Each value corresponds to the same index in `r_m`."""

    @classmethod
    def create_regular(cls, ray: RayGeometry, sampler: FieldSampler, nan_policy: NaNPolicy):
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
        z_regular = sampler.sample(xs=ray.xs, ys=ray.ys)
        r_regular, z_regular = _apply_nan_policy(r_m=ray.r_m, z=z_regular, theta=ray.theta, policy=nan_policy)

        levels = np.asarray(levels, dtype=float)
        r_crossings, z_crossings = _compute_level_crossings(z_regular, r_regular, levels)

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
    z: NDArray[np.floating], r: NDArray[np.floating], levels: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
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


# @dataclasses.dataclass(frozen=True)
# class LevelCrossingRayProfile(RayProfile):
#     """Ray profile sampled at level crossings of the sampled field.

#     The profile is first sampled on the regular ray, and then resampled such that supporting points
#     occur at intersections of the profilewith specified level values.

#     Plateaus on specified levels are preserved by supporting point at the edges of the flat segment.
#     """

#     ray_geometry: RayGeometry
#     """Reference geometry of the ray. Used for profile discretization only."""
#     sampler: FieldSampler
#     """Sampler used to evaluate the field along the ray geometry."""
#     nan_policy: NaNPolicy
#     """Policy controlling how NaN values in the sampled profile are handled."""
#     levels: NDArray[np.floating]
#     """Scalar values at which level crossings are computed. Need to be ordered."""
#     z: NDArray[np.floating] = dataclasses.field(init=False)
#     r_m: NDArray[np.floating] = dataclasses.field(init=False)
#     _z: NDArray[np.floating] = dataclasses.field(init=False)
#     _r_m: NDArray[np.floating] = dataclasses.field(init=False)

#     @property
#     def ray(self) -> RayGeometry:
#         return RayGeometry.from_compass(
#             location=self.ray_geometry.location, theta=self.ray_geometry.theta, r_m=self.r_m
#         )

#     def _build_profile(self):
#         object.__setattr__(self, "_r_m", self.ray_geometry.r_m)
#         object.__setattr__(self, "_z", self.sampler.sample(xs=self.ray_geometry.xs, ys=self.ray_geometry.ys))
#         r_m_crossing, z_crossing = self._compute_level_crossings(self._z, self._r_m, self.levels)
#         object.__setattr__(self, "z", z_crossing)
#         object.__setattr__(self, "r_m", r_m_crossing)
#         self._apply_nan_policy()
#         if any(np.diff(self.r_m) <= 0):
#             raise ValueError("Points are too close to each other or not ordered: %s.", self.r_m)

#     def _apply_nan_policy(self):
#         """Apply NaN handling policy to the sampled profile.

#         Notes
#         -----
#         1. ERROR: Raise if any NaNs are found.
#         2. TRUNCATE: Cut profile at the first NaN.
#         3. MASK: Keep NaNs as is.
#         """
#         if not np.isnan(self.z).any():
#             return

#         if self.nan_policy is NaNPolicy.ERROR:
#             raise ValueError("NaNs encountered in ray profile for %1s.", self.ray_geometry.theta)

#         elif self.nan_policy is NaNPolicy.TRUNCATE:
#             nan_idx = np.where(np.isnan(self.z))[0]
#             if nan_idx.size == 0:
#                 return
#             first_nan = nan_idx[0]
#             if first_nan == 0:
#                 raise ValueError("Ray for theta=%.1f contains no valid numbers.", self.ray_geometry.theta)
#             self.z = self.z[:first_nan]
#             self.r_m = self.r_m[:first_nan]

#         elif self.nan_policy is NaNPolicy.MASK:
#             pass

#     @staticmethod
#     def _compute_level_crossings(z: NDArray[np.floating], r: NDArray[np.floating], levels: NDArray[np.floating]):
#         """Compute supporting points at level crossings along a ray profile.

#         Notes
#         -----
#         1. First and last point are always includes-
#         2. For monotone segments, one supporting point is inserted for each crossed level.
#         3. Zero-length segments are removed.
#         4. Plateaus are preserved by keeping their end points.
#         """
#         levels_ = np.asarray(levels, dtype=float)
#         r_crossings, z_crossings = [r[0]], [z[0]]
#         for index, r_current in enumerate(r[:-1]):
#             r_next = r[index + 1]
#             z_current, z_next = z[index], z[index + 1]
#             z_min, z_max = (z_current, z_next) if z_current < z_next else (z_next, z_current)
#             levels_touched = levels_[(levels_ >= z_min) & (levels_ <= z_max)]
#             if len(levels_touched) > 0:
#                 if z_next == z_current:
#                     r_crossings.extend([r_current, r_next])
#                     z_crossings.extend([z_current, z_next])
#                     continue
#                 if z_next < z_current:
#                     levels_touched = levels_touched[::-1]
#                 for level in levels_touched:
#                     alpha = (level - z_current) / (z_next - z_current)
#                     r_crossing = r_current + alpha * (r_next - r_current)
#                     z_crossings.append(level)
#                     r_crossings.append(r_crossing)
#         r_crossings.append(r[-1])
#         z_crossings.append(z[-1])
#         r_crossings_array = np.asarray(r_crossings, dtype=float)
#         z_crossings_array = np.asarray(z_crossings, dtype=float)
#         if r_crossings_array.size >= 2:
#             eps = 1e-7
#             mask_non_duplicates = np.concat([np.diff(r_crossings_array) > eps, [True]])
#             r_crossings_array = r_crossings_array[mask_non_duplicates]
#             z_crossings_array = z_crossings_array[mask_non_duplicates]
#         return r_crossings_array, z_crossings_array
