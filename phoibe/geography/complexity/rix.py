import abc
import dataclasses
import enum
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
    """Ray with sampling grid translating internal ray geometry to the 2D world."""

    location: LocationCCS
    """Root of the ray."""
    theta: float
    """Direction of the ray [°] with 0° facing North."""
    r_m: NDArray[np.floating]
    xs: NDArray[np.floating]
    ys: NDArray[np.floating]
    # R_km: float
    # """Length of the ray [km]."""
    # dr_km: float
    # """Stepsize of gridpoints [km]."""

    def __post_init__(self):
        pass

    @classmethod
    def from_compass_regular(cls, location, theta, R_km, dr_km):
        _validate_notna_finite(R_km, "R_km")
        _validate_non_negative(R_km, "R_km")
        _validate_notna_finite(dr_km, "dr_km")
        _validate_positive(dr_km, "dr_km")
        if dr_km >= R_km:
            raise ValueError("dr_km must not exceed R_km.")

        n_steps = int(np.floor(R_km / dr_km))
        r_m = np.arange(n_steps + 1, dtype=float) * dr_km * 1000

        remainder = R_km * 1000 - r_m[-1]
        if remainder > 1e-3:
            msg = "Ray for %.1f truncated as %.3f not multiple of %.3f. Last point at %.3fm."
            LOGGER.warning(msg, theta, R_km, dr_km, r_m[-1])

        return cls._from_compass_r_m(location, theta, r_m)

    @classmethod
    def from_compass(cls, location, theta, r_m):
        return cls._from_compass_r_m(location, theta, r_m)

    @classmethod
    def _from_compass_r_m(cls, location, theta, r_m):
        r_m = np.asarray(r_m, dtype=float)
        _validate_positive(np.diff(r_m), "dr_m")

        dx, dy = ergaleiothiki.kiklos.circle.compass_polar_to_cartesian(theta, r_m)
        xs = location.easting + dx
        ys = location.northing + dy
        return cls(location=location, theta=theta, r_m=r_m, xs=xs, ys=ys)


class RayProfile(abc.ABC):
    ray: RayGeometry
    z: NDArray[np.floating]
    r_m: NDArray[np.floating]

    def __post_init__(self):
        self._build_profile()

    @abc.abstractmethod
    def _build_profile(self) -> None:
        """Populate `self.r_m` and `self.z` consistently."""

    @property
    @abc.abstractmethod
    def ray_segments(self) -> RayGeometry:
        """Ray that is used for filtering steep segments."""

    @property
    def segment_lengths(self) -> NDArray[np.floating]:
        return np.diff(self.r_m)

    @property
    def slopes(self) -> NDArray[np.floating]:
        if len(self.z) < 2:
            return np.array([np.nan], dtype=float)
        dz = np.diff(self.z)
        dr = self.segment_lengths
        if not np.all(dr > 0):
            raise ValueError("Ray for theta=%.1f requires strictly increasing distance from origin.", self.ray.theta)
        return dz / dr

    def rix(self, slope_critical: float) -> float:
        slopes = self.slopes
        if len(self.segment_lengths) == 0:
            return np.nan

        steep_mask = np.abs(slopes) > slope_critical
        total_length = np.sum(self.segment_lengths)
        steep_length = np.sum(self.segment_lengths[steep_mask])

        if total_length <= 0:
            return 0.0
        else:
            return steep_length / total_length

    def steep_mask(self, slope_critical: float) -> NDArray[np.bool_]:
        if np.isnan(self.slopes).all():
            return np.zeros_like(self.slopes, dtype=bool)
        return np.abs(self.slopes) > slope_critical

    @staticmethod
    def _get_true_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
        """Return the indices [start, stop) of intervals of contiguous `True` values."""
        runs = []
        start = None

        for index, value in enumerate(mask):
            if value and start is None:
                start = index
            elif not value and start is not None:
                runs.append((start, index))
                start = None

        if start is not None:
            runs.append((start, len(mask)))

        return runs

    def steep_segment_indices(self, slope_critical: float) -> list[tuple[int, int]]:
        mask = self.steep_mask(slope_critical=slope_critical)
        runs = self._get_true_runs(mask=mask)
        return runs

    def steep_ray_segments(self, slope_critical: float):
        segments = []
        for start, stop in self.steep_segment_indices(slope_critical=slope_critical):
            xs = self.ray_segments.xs[start : stop + 1]  # noqa: E203
            ys = self.ray_segments.ys[start : stop + 1]  # noqa: E203
            if len(xs) >= 2:
                segments.append(shapely.geometry.LineString(zip(xs, ys)))
        return segments


@dataclasses.dataclass
class RegularRayProfile(RayProfile):
    ray: RayGeometry
    sampler: FieldSampler
    nan_policy: NaNPolicy
    z: NDArray[np.floating] = dataclasses.field(init=False)
    r_m: NDArray[np.floating] = dataclasses.field(init=False)

    def _build_profile(self):
        self.r_m = self.ray.r_m
        self.z = self.sampler.sample(xs=self.ray.xs, ys=self.ray.ys)
        self._apply_nan_policy()

    def _apply_nan_policy(self):
        if not np.isnan(self.z).any():
            return

        if self.nan_policy is NaNPolicy.ERROR:
            raise ValueError("NaNs encountered in ray profile for theta=%.1f.", self.ray.theta)

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
    def ray_segments(self) -> RayGeometry:
        return self.ray


@dataclasses.dataclass(frozen=True)
class LevelCrossingRayProfile(RayProfile):
    ray: RayGeometry
    sampler: FieldSampler
    nan_policy: NaNPolicy
    levels: NDArray[np.floating]
    z: NDArray[np.floating] = dataclasses.field(init=False)
    r_m: NDArray[np.floating] = dataclasses.field(init=False)
    _z: NDArray[np.floating] = dataclasses.field(init=False)
    _r_m: NDArray[np.floating] = dataclasses.field(init=False)

    @property
    def ray_segments(self) -> RayGeometry:
        return RayGeometry.from_compass(location=self.ray.location, theta=self.ray.theta, r_m=self.r_m)

    def _build_profile(self):
        object.__setattr__(self, "_r_m", self.ray.r_m)
        object.__setattr__(self, "_z", self.sampler.sample(xs=self.ray.xs, ys=self.ray.ys))
        # self._apply_nan_policy()
        r_m_crossing, z_crossing = self._compute_level_crossings(self._z, self._r_m, self.levels)
        object.__setattr__(self, "z", z_crossing)
        object.__setattr__(self, "r_m", r_m_crossing)
        self._apply_nan_policy()
        if any(np.diff(self.r_m) <= 0):
            raise ValueError("Points are too close to each other or not ordered: %s.", self.r_m)

    def _apply_nan_policy(self):
        if not np.isnan(self.z).any():
            return

        if self.nan_policy is NaNPolicy.ERROR:
            raise ValueError("NaNs encountered in ray profile for %1s.", self.ray.theta)

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

    @staticmethod
    def _compute_level_crossings(z: NDArray[np.floating], r: NDArray[np.floating], levels: NDArray[np.floating]):
        levels_ = np.asarray(levels, dtype=float)
        r_crossings, z_crossings = [r[0]], [z[0]]
        for index, r_current in enumerate(r[:-1]):
            r_next = r[index + 1]
            z_current, z_next = z[index], z[index + 1]
            z_min, z_max = (z_current, z_next) if z_current < z_next else (z_next, z_current)
            levels_touched = levels_[(levels_ >= z_min) & (levels_ <= z_max)]
            if len(levels_touched) > 0:
                if z_next == z_current:
                    r_crossings.extend([r_current, r_next])
                    z_crossings.extend([z_current, z_next])
                    continue
                if z_next < z_current:
                    levels_touched = levels_touched[::-1]
                for level in levels_touched:
                    alpha = (level - z_current) / (z_next - z_current)
                    r_crossing = r_current + alpha * (r_next - r_current)
                    z_crossings.append(level)
                    r_crossings.append(r_crossing)
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


def compute_radial_rix(
    location_ccs: LocationCCS, sampler: FieldSampler, n_angles: int, R_km, dr_km, slope_critical=0.3
):
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
