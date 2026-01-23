import dataclasses
import enum

import ergaleiothiki.kiklos.circle
import numpy as np
import shapely
from ergaleiothiki.perdix import LocationCCS
from ergaleiothiki.tididi.validate_numerics import _validate_non_negative
from ergaleiothiki.tididi.validate_numerics import _validate_notna_finite
from numpy.typing import NDArray

from .sampler import FieldSampler


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
        _validate_non_negative(self.dr_km, "dr_km")
        if self.dr_km >= self.R_km:
            raise ValueError("dr_km must not exceed R_km.")

    @property
    def r_m(self) -> NDArray[np.floating]:
        return np.arange(0, self.R_km + self.dr_km, self.dr_km) * 1000

    @property
    def _dx_dy(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        return ergaleiothiki.kiklos.circle.compass_polar_to_cartesian(self.theta, self.r_m)

    @property
    def xs(self) -> NDArray[np.floating]:
        dx, _ = self._dx_dy
        return self.location.easting + dx

    @property
    def ys(self) -> NDArray[np.floating]:
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
            notnan_mask = ~np.isnan(self.z)
            if not np.any(notnan_mask):
                raise ValueError("Ray contains no valid numbers.")
            last_notnan = np.where(notnan_mask)[0][-1] + 1
            self.z = self.z[:last_notnan]
            self.r_m = self.r_m[:last_notnan]

        elif self.nan_policy is NaNPolicy.MASK:
            pass

    @property
    def slopes(self) -> NDArray[np.floating]:
        dz = np.diff(self.z)
        dr = np.diff(self.r_m)
        return dz / dr

    def steep_mask(self, slope_critical: float) -> NDArray[np.floating]:
        return np.abs(self.slopes) > slope_critical

    def segments(self, slope_critical: float) -> list[shapely.geometry.LineString]:
        segments: list[shapely.geometry.LineString] = []
        for current, next in self._get_contiguous_true_segments(self.steep_mask(slope_critical)):
            coords = list(zip(self.ray.xs[current : next + 1], self.ray.ys[current : next + 1]))  # noqa: E203
            if len(coords) >= 2:
                segments.append(shapely.geometry.LineString(coords))
        return segments

    @staticmethod
    def _get_contiguous_true_segments(mask: NDArray[np.floating]) -> list[tuple[int, int]]:
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

        total_length = self.r_m[-1] - self.r_m[0]
        if total_length <= 0:
            return 0.0

        steep_length = 0.0
        for left, right in self._get_contiguous_true_segments(mask=mask):
            steep_length += self.r_m[right] - self.r_m[left]

        return steep_length / total_length

    def steep_segments_geometry(self, slope_critical: float):
        mask = self.steep_mask(slope_critical=slope_critical)
        segments = []
        for left, right in self._get_contiguous_true_segments(mask=mask):
            coords = list(zip(self.ray.xs[left : right + 1], self.ray.ys[left : right + 1]))  # noqa: E203
            if len(coords) >= 2:
                segments.append(shapely.geometry.LineString(coords))
        return segments


def compute_radial_rix(
    location_ccs: LocationCCS,
    sampler: FieldSampler,
    n_angles: int,
    R_km,
    dr_km,
    slope_critical=0.3,
    # return_segments=False,
):
    angles = np.linspace(0, 360, n_angles, endpoint=False)
    ray_profiles = []
    rix_values = []

    for theta in angles:
        ray = RayGeometry(location=location_ccs, theta=theta, R_km=R_km, dr_km=dr_km)
        ray_profile = RayProfile(ray=ray, sampler=sampler, nan_policy=NaNPolicy.ERROR)
        ray_profiles.append(ray_profile)
        rix_values.append(ray_profile.rix(slope_critical=slope_critical))
    return np.array(rix_values), ray_profiles


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
