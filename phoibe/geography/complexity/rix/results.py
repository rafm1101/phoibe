import dataclasses
import functools

import numpy as np
import shapely

from . import evaluate
from .interface import Keys, _get_parameter
from .profiles import RayProfile

KEYS = Keys()


@dataclasses.dataclass(frozen=True)
class RayProfileMeta:
    """Information about a RayProfile."""

    crs_ray: str | None
    """CRS for the ray coordinates."""
    crs_dem: str | None
    """CRS for the DEM coordinates."""
    resolution: tuple[float, float] | None
    """Resolution of the DEM (dx, dy), if known."""
    n_oob: int | None
    """Number of ray points that are NaN after sampling (out-of-bounds or data gaps)."""
    messages: str
    """Messages related to the ray."""


@dataclasses.dataclass(frozen=True)
class RayRuggedness:
    """Analysis result for a single ray direction.

    Combines the profile with computed metrics for inspection and debugging.

    Notes
    -----
    `.meta`: W/o rioxarray some DEM CRS metadata are missing. This need an additional graceful treatment.
    """

    profile: RayProfile
    """Ray profile of some field."""
    slope_critical: float
    """Slope [m/m] above which a segment is consideres as steep."""
    keys: Keys = dataclasses.field(repr=False, default=KEYS)

    @property
    def meta(self) -> RayProfileMeta:
        """Metadata relating to `RayProfile`. Includes CRS, (resolution), out-of-bound point count, messages."""
        dem_meta = _get_parameter(self.profile.meta, self.keys.dem, strict=False) or {}
        ray_profile_meta = RayProfileMeta(
            crs_ray=_get_parameter(self.profile.meta, self.keys.rays, self.keys.crs_ray, strict=False),
            crs_dem=_get_parameter(dem_meta, self.keys.crs_dem, strict=False),
            resolution=_get_parameter(dem_meta, self.keys.resolution_dem, strict=False),
            n_oob=_get_parameter(self.profile.meta, self.keys.rays, self.keys.nan_count, strict=False),
            messages=str(_get_parameter(self.profile.meta, self.keys.alignment, self.keys.message, strict=False)),
        )
        return ray_profile_meta

    @property
    def theta(self) -> float:
        """Ray direction [°]."""
        return self.profile.ray.theta

    @functools.cached_property
    def total_length_m(self) -> float:
        """Total physical length of the ray [m]."""
        return evaluate.total_length_m(self.profile)

    @functools.cached_property
    def steep_length_m(self) -> float:
        """Total length of steep segments [m]."""
        seg_lengths = evaluate.segment_lengths(self.profile)
        mask = evaluate.steep_mask(self.profile, self.slope_critical)
        return float(np.sum(seg_lengths[mask]))

    @functools.cached_property
    def steep_segments(self) -> tuple[shapely.geometry.LineString, ...]:
        """Geometric LineStrings of steep segments."""
        return tuple(evaluate.steep_ray_segments(self.profile, self.slope_critical))

    @functools.cached_property
    def max_abs_slope(self) -> float:
        """Maximum absolute slope along the ray."""
        slope_arr = evaluate.slopes(self.profile)
        if slope_arr.size == 0 or np.isnan(slope_arr).all():
            return np.nan
        return float(np.nanmax(np.abs(slope_arr)))

    @property
    def n_steep_segments(self) -> int:
        """Number of contiguous steep segments."""
        return len(self.steep_segments)

    @property
    def ruggedness(self) -> float:
        """Fraction of ray length that is steep (= RIX for this ray)."""
        return evaluate.ruggedness(self.profile, self.slope_critical)

    def describe(self) -> dict[str, float]:
        """Summary statistics for this ray."""
        return {
            self.keys.theta: self.theta,
            self.keys.ruggedness: self.ruggedness,
            self.keys.total_length_m: self.total_length_m,
            self.keys.steep_length_m: self.steep_length_m,
            self.keys.max_abs_slope: self.max_abs_slope,
            self.keys.n_steep_segments: self.n_steep_segments,
        }

    def steep_segments_geodataframe(self):
        """GeoDataFrame of steep segments for visualization.

        Returns
        -------
        geopandas.GeoDataFrame
            Geodataframe where each row contains the geometry of one steep segment.

        Raises
        ------
        ImportError
            If geopandas is not installed.
        """
        try:
            import geopandas as gpd
        except ImportError as exception:
            raise ImportError("RayResult.steep_segments_geodataframe() requires geopandas.") from exception

        records = [{self.keys.segment_id: i, self.keys.geometry: seg} for i, seg in enumerate(self.steep_segments)]

        if records:
            steep_segments = gpd.GeoDataFrame(records, geometry=self.keys.geometry, crs=self.profile.ray.crs)
        else:
            steep_segments = gpd.GeoDataFrame(
                columns=[self.keys.segment_id, self.keys.geometry],
                geometry=self.keys.geometry,
                crs=self.profile.ray.crs,
            )

        return steep_segments


@dataclasses.dataclass(frozen=True)
class RadialRuggedness:
    """Radial RIX analysis aggregating multiple ray directions.

    Provides access to individual rays, aggregate metrics, messages, and various views
    of the data for inspection and validation.
    """

    rays: tuple[RayRuggedness, ...] = dataclasses.field(repr=False)
    """Collection of rays being evaluated."""
    _ray_by_angle: dict[float, RayRuggedness] = dataclasses.field(init=False, repr=False, compare=False)
    """Access ray results by their angle."""
    keys: Keys = dataclasses.field(repr=False, default=KEYS)
    """Column keys to employ."""

    def __post_init__(self):
        """Build internal index for fast theta lookups."""
        if len(self.rays) == 0:
            raise ValueError("RadialRixResult requires at least one ray")

        slope_criticals = {ray.slope_critical for ray in self.rays}
        if len(slope_criticals) > 1:
            raise ValueError(f"All rays must use same slope_critical, got: {slope_criticals}")

        crss = {None if ray.profile.ray.crs is None else ray.profile.ray.crs.to_string() for ray in self.rays}
        if len(crss) > 1:
            raise ValueError(f"All rays must have the same crs, got: {crss}")

        rays_sorted = tuple(sorted(self.rays, key=lambda ray: ray.theta))
        object.__setattr__(self, "rays", rays_sorted)

    @property
    def z(self) -> tuple[float, float]:
        """Elevation of the centre point."""
        return float(np.mean([ray.profile.z[0] for ray in self.rays])), float(
            np.std([ray.profile.z[0] for ray in self.rays])
        )

    @property
    def rix(self) -> float:
        """RIX/mean ruggedness across all ray directions."""
        return float(np.mean([ray.ruggedness for ray in self.rays]))

    @property
    def slope_critical(self) -> float:
        """Slope [m/m] above which a segment is consideres as steep used for all rays."""
        return self.rays[0].slope_critical

    @property
    def n_rays(self) -> int:
        """Number of ray directions analyzed."""
        return len(self.rays)

    @property
    def angles(self) -> np.ndarray:
        """Array of ray angles [°] in sorted order."""
        return np.array([ray.theta for ray in self.rays], dtype=float)

    @property
    def ruggednesses(self) -> np.ndarray:
        """Array of ruggedness values aligned with angles.

        Returns
        -------
        np.ndarray
            Ruggedness for each angle in `self.angles`.
        """
        return np.array([ray.ruggedness for ray in self.rays], dtype=float)

    @property
    def meta(self) -> dict:
        crs_ray = list(
            {_get_parameter(ray.profile.meta, self.keys.rays, self.keys.crs_ray, strict=False) for ray in self.rays}
        )
        dem_metas = [_get_parameter(ray.profile.meta, self.keys.dem, strict=False) or {} for ray in self.rays]
        crs_dem = list({_get_parameter(dem_meta, self.keys.crs_dem, strict=False) for dem_meta in dem_metas})
        extent_dem = list({_get_parameter(dem_meta, self.keys.extent_dem, strict=False) for dem_meta in dem_metas})
        resolution_dem = list(
            {_get_parameter(dem_meta, self.keys.resolution_dem, strict=False) for dem_meta in dem_metas}
        )
        message = list(
            {
                _get_parameter(ray.profile.meta, self.keys.alignment, self.keys.message, strict=False)
                for ray in self.rays
            }
        )
        nan_count = int(
            np.sum(
                [
                    _get_parameter(ray.profile.meta, self.keys.rays, self.keys.nan_count, strict=False)
                    for ray in self.rays
                ],
                dtype=float,
            )
        )
        records = {
            self.keys.rays: {
                self.keys.crs_ray: crs_ray,
                self.keys.nan_count: nan_count,
            },
            self.keys.dem: {
                self.keys.crs_dem: crs_dem,
                self.keys.extent_dem: extent_dem,
                self.keys.resolution_dem: resolution_dem,
            },
            self.keys.alignment: {
                self.keys.message: message,
            },
        }
        return records

    def ray(self, theta: float, atol: float = 1e-5) -> RayRuggedness:
        """Get RayResult for a specific angle.

        Parameters
        ----------
        theta
            Ray direction [°].
        atol
            Tolerance for float comparison.

        Returns
        -------
        RayResult
            Analysis result for this direction.

        Raises
        ------
        KeyError
            If no ray exists for this angle.
        """
        idx = np.searchsorted(self.angles, theta)
        if idx < len(self.rays) and np.isclose(self.angles[idx], theta, atol=atol):
            return self.rays[idx]
        else:
            raise KeyError(f"No ray found for theta={theta:.1f}°. Available angles: {self.angles}")

    def to_dataframe(self):
        """Generate a dataframe summarising the metrics of all rays.

        Returns
        -------
        pandas.DataFrame
            Dataframe holding all metrics as one ray per row.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exception:
            raise ImportError("RadialRixResult.to_dataframe() requires pandas.") from exception

        records = [ray.describe() for ray in self.rays]
        return pd.DataFrame(records)

    def steep_segments_geodataframe(self):
        """Generate a GeoDataFrame holding all steep segments.

        Returns
        -------
        geopandas.GeoDataFrame
            Geodataframe where each row contains the geometry of one steep segment.

        Raises
        ------
        ImportError
            If geopandas is not installed.
        """
        try:
            import geopandas as gpd
        except ImportError as exception:
            raise ImportError("RadialRixResult.steep_segments_geodataframe() requires geopandas.") from exception

        records = []
        for ray in self.rays:
            for i, segment in enumerate(ray.steep_segments):
                records.append({self.keys.theta: ray.theta, self.keys.segment_id: i, self.keys.geometry: segment})
        return gpd.GeoDataFrame(
            records,
            columns=[self.keys.theta, self.keys.segment_id, self.keys.geometry],
            geometry=self.keys.geometry,
            crs=self.rays[0].profile.ray.crs,
        )

    def describe(self) -> dict[str, float]:
        """Summary statistics across all rays.

        Returns
        -------
        dict
            Summary statistics.
        """
        rix_values = [ray.ruggedness for ray in self.rays]

        return {
            self.keys.rix: float(np.mean(rix_values)),
            self.keys.rix_std: float(np.std(rix_values)),
            self.keys.rix_min: float(np.min(rix_values)),
            self.keys.rix_max: float(np.max(rix_values)),
            self.keys.n_rays: self.n_rays,
            self.keys.slope_critical: self.slope_critical,
        }

    def plot_polar(self, ax=None):
        """Create polar plot of ruggedness values.

        Returns
        -------
        tuple[Figure, Axes]
            Matplotlib figure and axes.

        Raises
        ------
        ImportError
            If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exception:
            raise ImportError("RadialRixResult.plot_polar() requires matplotlib.") from exception

        angles_radians = np.deg2rad(np.append(self.angles, self.angles[0]))
        values = self.ruggednesses
        values = np.append(values, values[0])

        if ax is None:
            figure, ax = plt.subplots(subplot_kw={"projection": "polar"})

        ax.plot(angles_radians, values)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"Directional RIX (threshold={self.slope_critical:.2f})")
        ax.set_ylabel("RIX")

        return figure, ax
