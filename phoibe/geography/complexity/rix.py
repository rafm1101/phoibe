import dataclasses
import enum
import functools
import logging

import ergaleiothiki.kiklos.circle
import numpy as np
import shapely
from ergaleiothiki.perdix import LocationCCS
from ergaleiothiki.tididi.validate_numerics import _validate_non_negative
from ergaleiothiki.tididi.validate_numerics import _validate_notna_finite
from numpy.typing import NDArray

from .sampler import FieldSampler
from .tididi import _validate_positive

LOGGER = logging.getLogger(__name__)


class NaNPolicy(enum.Enum):
    TRUNCATE = "truncate"
    MASK = "mask"
    ERROR = "error"


@dataclasses.dataclass(frozen=True)
class RayGeometry:
    """Ray with sampling grid."""

    location: LocationCCS
    """Root of the ray."""
    theta: float
    """Direction of the ray [°] with 0° facing North."""
    R_km: float
    """Length of the ray [km]."""
    dr_km: float
    """Stepsize of gridpoints [km]."""

    def __post_init__(self):
        _validate_notna_finite(self.R_km, "R_km")
        _validate_non_negative(self.R_km, "R_km")
        _validate_notna_finite(self.dr_km, "dr_km")
        _validate_positive(self.dr_km, "dr_km")
        if self.dr_km >= self.R_km:
            raise ValueError("dr_km must not exceed R_km.")

    @property
    def r_m(self) -> NDArray[np.floating]:
        """Ray grid points measured in [m] from the origin."""
        n_steps = int(np.floor(self.R_km / self.dr_km))
        r_m = np.arange(n_steps + 1, dtype=float) * self.dr_km * 1000

        remainder = self.R_km * 1000 - r_m[-1]
        if remainder > 1e-3:
            msg = "Ray for %.1f truncated as %.3f not multiple of %.3f. Last point at %.3fm."
            LOGGER.warning(msg, self.theta, self.R_km, self.dr_km, r_m[-1])
        return r_m

    @functools.cached_property
    def _dx_dy(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Ray grid points embedded in real-world coordinates."""
        return ergaleiothiki.kiklos.circle.compass_polar_to_cartesian(self.theta, self.r_m)

    @property
    def xs(self) -> NDArray[np.floating]:
        """Ray grid points x coordinates."""
        dx, _ = self._dx_dy
        return self.location.easting + dx

    @property
    def ys(self) -> NDArray[np.floating]:
        """Ray grid points y coordinates."""
        _, dy = self._dx_dy
        return self.location.northing + dy


@dataclasses.dataclass
class RayProfile:
    ray: RayGeometry
    sampler: FieldSampler
    nan_policy: NaNPolicy
    z: NDArray[np.floating] = dataclasses.field(init=False)
    r_m: NDArray[np.floating] = dataclasses.field(init=False)

    def __post_init__(self):
        self.r_m = self.ray.r_m
        self.z = self.sampler.sample(xs=self.ray.xs, ys=self.ray.ys)
        self._apply_nan_policy()

    def _apply_nan_policy(self):
        if not np.isnan(self.z).any():
            return

        if self.nan_policy is NaNPolicy.ERROR:
            raise ValueError("NaNs encountered in ray profile.")

        elif self.nan_policy is NaNPolicy.TRUNCATE:
            nan_idx = np.where(np.isnan(self.z))[0]
            if nan_idx.size == 0:
                return
            first_nan = nan_idx[0]
            if first_nan == 0:
                raise ValueError("Ray for theta=%.1f contains no valid numbers.", self.ray.theta)
            self.z = self.z[:first_nan]
            self.r_m = self.r_m[:first_nan]

        elif self.nan_policy is NaNPolicy.MASK:
            pass

    @property
    def slopes(self) -> NDArray[np.floating]:
        if len(self.z) < 2:
            return np.array([np.nan], dtype=float)
        dz = np.diff(self.z)
        dr = np.diff(self.r_m)
        if not np.all(dr > 0):
            raise ValueError("Ray for theta=%.1f requires strictly increasing distance from origin.", self.ray.theta)
        return dz / dr

    def steep_mask(self, slope_critical: float) -> NDArray[np.bool_]:
        if np.isnan(self.slopes).all():
            return np.zeros_like(self.slopes, dtype=bool)
        return np.abs(self.slopes) > slope_critical

    def segments(self, slope_critical: float) -> list[shapely.geometry.LineString]:
        mask = self.steep_mask(slope_critical=slope_critical)
        if not np.any(mask):
            return []

        segments: list[shapely.geometry.LineString] = []
        for current, next in self._get_contiguous_true_segments(mask=mask):
            coords = list(zip(self.ray.xs[current : next + 1], self.ray.ys[current : next + 1]))  # noqa: E203
            if len(coords) >= 2:
                segments.append(shapely.geometry.LineString(coords))
        return segments

    @staticmethod
    def _get_contiguous_true_segments(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
        segments = []
        start = None

        for index, value in enumerate(mask):
            if value and start is None:
                start = index
            elif not value and start is not None:
                segments.append((start, index))
                start = None

        if start is not None:
            segments.append((start, len(mask)))

        return segments

    def rix(self, slope_critical: float) -> float:
        mask = self.steep_mask(slope_critical=slope_critical)

        if mask.size == 0:
            return 0.0

        dr = np.diff(self.r_m)
        total_length = np.sum(dr)
        if total_length <= 0:
            return 0.0

        steep_length = 0.0
        # for left, right in self._get_contiguous_true_segments(mask=mask):
        #     steep_length += self.r_m[right] - self.r_m[left]
        steep_length = np.sum(dr[mask])

        return steep_length / total_length

    def steep_ray_segments(self, slope_critical: float):
        mask = self.steep_mask(slope_critical=slope_critical)
        segments = []
        for left, right in self._get_contiguous_true_segments(mask=mask):
            coords = list(zip(self.ray.xs[left : right + 1], self.ray.ys[left : right + 1]))  # noqa: E203
            if len(coords) >= 2:
                segments.append(shapely.geometry.LineString(coords))
        return segments


def compute_radial_rix(
    location_ccs: LocationCCS, sampler: FieldSampler, n_angles: int, R_km, dr_km, slope_critical=0.3
):
    angles = np.linspace(0, 360, n_angles, endpoint=False)
    ray_profiles = []
    rix_values = []
    steep_ray_segments = []

    for theta in angles:
        ray = RayGeometry(location=location_ccs, theta=theta, R_km=R_km, dr_km=dr_km)
        ray_profile = RayProfile(ray=ray, sampler=sampler, nan_policy=NaNPolicy.ERROR)
        ray_profiles.append(ray_profile)
        rix_values.append(ray_profile.rix(slope_critical=slope_critical))
        steep_ray_segments.append(ray_profile.steep_ray_segments(slope_critical=slope_critical))
    return np.array(rix_values), ray_profiles, steep_ray_segments


# def compute_ray_at_location(
#     location_ccs: LocationCCS, field: xarray.DataArray, theta, R, dr, slope_critical=0.3, return_segments=False
# ):
#     # Slope and its angle possible mixed.
#     r = np.arange(0, R + dr, dr) * 1000

#     dx, dy = ergaleiothiki.kiklos.circle.compass_polar_to_cartesian(theta, r)
#     xs = location_ccs.easting + dx
#     ys = location_ccs.northing + dy

#     z = sample_from_spatial_array(field, xs, ys)

#     dz = np.diff(z)

#     slopes = dz / (dr * 1000)
#     steep_mask = np.abs(slopes > slope_critical)

#     if not return_segments:
#         return float(np.mean(steep_mask))
#     else:
#         segments = build_steep_segments(xs, ys, steep_mask)
#         return {"theta": theta, "fraction_steep": float(np.mean(steep_mask)), "segments": segments, "r": r, "z": z}
