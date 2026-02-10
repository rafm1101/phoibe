import dataclasses
import functools

import numpy as np
import shapely

from . import analyse
from .profiles import RayProfile


@dataclasses.dataclass(frozen=True)
class RayResult:
    """Analysis result for a single ray direction.

    Combines the profile with computed metrics for inspection and debugging.
    """

    profile: RayProfile
    """Ray profile of some field."""
    slope_critical: float
    """Slope [m/m] above which a segment is consideres as steep."""

    @property
    def theta(self) -> float:
        """Ray direction [°]."""
        return self.profile.ray.theta

    @functools.cached_property
    def total_length_m(self) -> float:
        """Total physical length of the ray [m]."""
        return analyse.total_length_m(self.profile)

    @functools.cached_property
    def steep_length_m(self) -> float:
        """Total length of steep segments [m]."""
        seg_lengths = analyse.segment_lengths(self.profile)
        mask = analyse.steep_mask(self.profile, self.slope_critical)
        return float(np.sum(seg_lengths[mask]))

    @functools.cached_property
    def steep_segments(self) -> tuple[shapely.geometry.LineString, ...]:
        """Geometric LineStrings of steep segments."""
        return tuple(analyse.steep_ray_segments(self.profile, self.slope_critical))

    @functools.cached_property
    def max_abs_slope(self) -> float:
        """Maximum absolute slope along the ray."""
        slope_arr = analyse.slopes(self.profile)
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
        return analyse.ruggedness(self.profile, self.slope_critical)

    def describe(self) -> dict[str, float]:
        """Summary statistics for this ray."""
        return {
            "theta": self.theta,
            "ruggedness": self.ruggedness,
            "total_length_m": self.total_length_m,
            "steep_length_m": self.steep_length_m,
            "max_abs_slope": self.max_abs_slope,
            "n_steep_segments": self.n_steep_segments,
        }

    def steep_segments_geodataframe(self, *, crs):
        """GeoDataFrame of steep segments for visualization.

        Parameters
        ----------
        crs: `pyproj.CRS`
            Coordinate reference system for the GeoDataFrame.

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

        records = [{"segment_id": i, "geometry": seg} for i, seg in enumerate(self.steep_segments)]

        return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


@dataclasses.dataclass(frozen=True)
class RadialRixResult:
    """Radial RIX analysis aggregating multiple ray directions.

    Provides access to individual rays, aggregate metrics, and various views
    of the data for inspection and validation.
    """

    rays: tuple[RayResult, ...]
    """Collection of rays being evaluated."""
    _ray_by_angle: dict[float, RayResult] = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Build internal index for fast theta lookups."""
        if len(self.rays) == 0:
            raise ValueError("RadialRixResult requires at least one ray")

        slope_criticals = {ray.slope_critical for ray in self.rays}
        if len(slope_criticals) > 1:
            raise ValueError(f"All rays must use same slope_critical, got: {slope_criticals}")

        object.__setattr__(self, "_ray_by_angle", {ray.theta: ray for ray in self.rays})

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
        return np.array(sorted(self._ray_by_angle.keys()), dtype=float)

    def ray(self, theta: float) -> RayResult:
        """Get RayResult for a specific angle.

        Parameters
        ----------
        theta
            Ray direction [°].

        Returns
        -------
        RayResult
            Analysis result for this direction.

        Raises
        ------
        KeyError
            If no ray exists for this angle.
        """
        try:
            return self._ray_by_angle[theta]
        except KeyError:
            available = sorted(self._ray_by_angle.keys())
            raise KeyError(f"No ray found for theta={theta:.1f}°. " f"Available angles: {available}") from None

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

    def steep_segments_geodataframe(self, *, crs):
        """Generate a GeoDataFrame holding all steep segments.

        Parameters
        ----------
        crs: `pyproj.CRS`
            Coordinate reference system for the GeoDataFrame.
            Might be set to `None` in cases w/o real-world link.

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
                records.append({"theta": ray.theta, "segment_id": i, "geometry": segment})

        return gpd.GeoDataFrame(records, columns=["theta", "segment_id", "geometry"], geometry="geometry", crs=crs)

    def describe(self) -> dict[str, float]:
        """Summary statistics across all rays.

        Returns
        -------
        dict
            Summary statistics.
        """
        rix_values = [ray.ruggedness for ray in self.rays]

        return {
            "rix_mean": float(np.mean(rix_values)),
            "rix_std": float(np.std(rix_values)),
            "rix_min": float(np.min(rix_values)),
            "rix_max": float(np.max(rix_values)),
            "n_rays": self.n_rays,
            "slope_critical": self.slope_critical,
        }

    @property
    def ruggednesses(self) -> np.ndarray:
        """Array of ruggedness values aligned with angles.

        Returns
        -------
        np.ndarray
            Ruggedness for each angle in `self.angles`.
        """
        return np.array([self._ray_by_angle[theta].ruggedness for theta in self.angles], dtype=float)

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
