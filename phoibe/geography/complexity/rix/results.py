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
    slope_critical: float

    @property
    def theta(self) -> float:
        """Ray direction [°]."""
        return self.profile.ray.theta

    @functools.cached_property
    def steep_fraction(self) -> float:
        """Fraction of ray length that is steep (= RIX for this ray)."""
        return analyse.rix(self.profile, self.slope_critical)

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
        return analyse.rix(self.profile, self.slope_critical)

    def describe(self) -> dict[str, float]:
        """Summary statistics for this ray.

        Returns
        -------
        dict
            Dictionary with metric names and values

        Examples
        --------
        >>> ray_result.describe()
        {
            'theta': 45.0,
            'rix': 0.42,
            'total_length_m': 1000.0,
            'steep_length_m': 420.0,
            'max_abs_slope': 0.85,
            'n_steep_segments': 3
        }
        """
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
        crs : Any
            Coordinate reference system for the GeoDataFrame

        Returns
        -------
        geopandas.GeoDataFrame
            Each row is one steep segment with geometry

        Raises
        ------
        ImportError
            If geopandas is not installed

        Notes
        -----
        Requires optional dependency: pip install geopandas
        """
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError(
                "steep_segments_geodataframe() requires geopandas. " "Install with: pip install geopandas"
            ) from exc

        records = [{"segment_id": i, "geometry": seg} for i, seg in enumerate(self.steep_segments)]

        return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


@dataclasses.dataclass(frozen=True)
class RadialRixResult:
    """Radial RIX analysis aggregating multiple ray directions.

    Provides access to individual rays, aggregate metrics, and various views
    of the data for inspection and validation.
    """

    rays: tuple[RayResult, ...]
    _ray_by_angle: dict[float, RayResult] = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Build internal index for fast theta lookups."""
        if len(self.rays) == 0:
            raise ValueError("RadialRixResult requires at least one ray")

        # Verify all rays use same slope_critical
        slope_criticals = {ray.slope_critical for ray in self.rays}
        if len(slope_criticals) > 1:
            raise ValueError(f"All rays must use same slope_critical, got: {slope_criticals}")

        object.__setattr__(
            self,
            "_ray_by_angle",
            {ray.theta: ray for ray in self.rays},
        )

    @property
    def rix(self) -> float:
        """Mean RIX across all ray directions."""
        return float(np.mean([ray.steep_fraction for ray in self.rays]))

    @property
    def slope_critical(self) -> float:
        """Slope threshold used for all rays."""
        return self.rays[0].slope_critical

    @property
    def n_rays(self) -> int:
        """Number of ray directions analyzed."""
        return len(self.rays)

    # ========== Ray Access ==========

    @property
    def angles(self) -> np.ndarray:
        """Array of ray angles [°] in sorted order."""
        return np.array(sorted(self._ray_by_angle.keys()), dtype=float)

    def ray(self, theta: float) -> RayResult:
        """Get RayResult for a specific angle.

        Parameters
        ----------
        theta : float
            Ray direction [°]

        Returns
        -------
        RayResult
            Analysis result for this direction

        Raises
        ------
        KeyError
            If no ray exists for this angle
        """
        try:
            return self._ray_by_angle[theta]
        except KeyError:
            available = sorted(self._ray_by_angle.keys())
            raise KeyError(f"No ray found for theta={theta:.1f}°. " f"Available angles: {available}") from None

    # ========== Data Export ==========

    def to_dataframe(self):
        """Export all ray results as DataFrame.

        Returns
        -------
        pandas.DataFrame
            One row per ray with all metrics

        Raises
        ------
        ImportError
            If pandas is not installed

        Notes
        -----
        Requires optional dependency: pip install pandas

        Examples
        --------
        >>> df = result.to_dataframe()
        >>> df[['theta', 'steep_fraction', 'max_abs_slope']]
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("to_dataframe() requires pandas. " "Install with: pip install pandas") from exc

        records = [ray.describe() for ray in self.rays]
        return pd.DataFrame(records)

    def steep_segments_geodataframe(self, *, crs):
        """Export all steep segments as GeoDataFrame.

        Parameters
        ----------
        crs : Any
            Coordinate reference system

        Returns
        -------
        geopandas.GeoDataFrame
            All steep segments from all rays with theta labels

        Raises
        ------
        ImportError
            If geopandas is not installed

        Notes
        -----
        Requires optional dependency: pip install geopandas
        """
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError(
                "steep_segments_geodataframe() requires geopandas. " "Install with: pip install geopandas"
            ) from exc

        records = []
        for ray in self.rays:
            for i, segment in enumerate(ray.steep_segments):
                records.append(
                    {
                        "theta": ray.theta,
                        "segment_id": i,
                        "geometry": segment,
                    }
                )

        return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

    def describe(self) -> dict[str, float]:
        """Summary statistics across all rays.

        Returns
        -------
        dict
            Aggregate statistics

        Examples
        --------
        >>> result.describe()
        {
            'rix_mean': 0.42,
            'rix_std': 0.15,
            'rix_min': 0.10,
            'rix_max': 0.85,
            'n_rays': 72,
            'slope_critical': 0.3
        }
        """
        rix_values = [ray.steep_fraction for ray in self.rays]

        return {
            "rix_mean": float(np.mean(rix_values)),
            "rix_std": float(np.std(rix_values)),
            "rix_min": float(np.min(rix_values)),
            "rix_max": float(np.max(rix_values)),
            "n_rays": self.n_rays,
            "slope_critical": self.slope_critical,
        }

    def directional_stats(self) -> np.ndarray:
        """Array of RIX values aligned with angles.

        Returns
        -------
        np.ndarray
            RIX value for each angle in self.angles

        Examples
        --------
        >>> angles = result.angles
        >>> rix_vals = result.directional_stats()
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(angles, rix_vals)
        """
        return np.array(
            [self._ray_by_angle[theta].steep_fraction for theta in self.angles],
            dtype=float,
        )

    def plot_polar(self):
        """Create polar plot of directional RIX values.

        Returns
        -------
        tuple[Figure, Axes]
            Matplotlib figure and axes

        Raises
        ------
        ImportError
            If matplotlib is not installed

        Notes
        -----
        Requires optional dependency: pip install matplotlib

        Examples
        --------
        >>> fig, ax = result.plot_polar()
        >>> plt.show()
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("plot_polar() requires matplotlib. " "Install with: pip install matplotlib") from exc

        angles_rad = np.deg2rad(self.angles)
        values = self.directional_stats()

        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
        ax.plot(angles_rad, values)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"Directional RIX (threshold={self.slope_critical:.2f})")
        ax.set_ylabel("RIX")

        return fig, ax


# @dataclasses.dataclass(frozen=True)
# class RayResult:
#     """Keep results of the RIX-related analysis along a single ray."""

#     theta: float
#     """Direction of the ray [°] with 0° facing North and angles increasing clockwise."""
#     profile: RayProfile
#     """Ray profile that is evaluated."""
#     slope_critical: float
#     """Threshold on the absolute slope for segments to be considered steep."""
#     steep_fraction: float
#     """Relative total length of all steep segments along the ray."""
#     steep_segments: list[shapely.geometry.LineString]

#     @property
#     def total_length_m(self) -> float:
#         """Total physical length of the ray profile."""
#         return float(self.profile.total_length_m)

#     @property
#     def steep_length_m(self) -> float:
#         """Total length of steep segments along the ray."""
#         if len(self.profile.segment_lengths) == 0:
#             return 0.0
#         mask = self.profile.steep_mask(slope_critical=self.slope_critical)
#         return float(np.sum(self.profile.segment_lengths[mask]))

#     @property
#     def ruggedness(self) -> float:
#         return self.steep_length_m / self.total_length_m

#     @property
#     def max_abs_slope(self) -> float:
#         """Maximum absolute slope along the ray."""
#         slopes = self.profile.slopes
#         if slopes.size == 0 or np.isnan(slopes).all():
#             return np.nan
#         return float(np.nanmax(np.abs(slopes)))

#     @property
#     def n_steep_segments(self) -> int:
#         return len(self.steep_segments)


# @dataclasses.dataclass(frozen=True)
# class RadialRixResult:
#     rays: tuple[RayResult, ...]

#     def __post_init__(self):
#         object.__setattr__(
#             self,
#             "_ray_by_angle",
#             {ray.theta: ray for ray in self.rays},
#         )

#     @property
#     def rix(self) -> float:
#         return float(np.mean([ray.steep_fraction for ray in self.rays]))

#     @property
#     def center(self) -> float:
#         x = float(np.nanmean([ray.profile.ray.xs[0] for ray in self.rays]))
#         y = float(np.nanmean([ray.profile.ray.ys[0] for ray in self.rays]))
#         x_std = float(np.nanstd([ray.profile.ray.xs[0] for ray in self.rays]))
#         y_std = float(np.nanstd([ray.profile.ray.ys[0] for ray in self.rays]))
#         return x, y, float(np.hypot(x_std, y_std))

#     def ray(self, theta: float) -> RayResult:
#         """Return the ray result for a given angle [°]."""
#         try:
#             return self._ray_by_angle[theta]
#         except KeyError:
#             raise KeyError(f"No ray found for theta={theta:.1f}°")

#     @property
#     def angles(self) -> np.ndarray:
#         return np.array(sorted(self._ray_by_angle.keys()), dtype=float)

#     @property
#     def ray_steep_fraction(self) -> np.ndarray:
#         return np.array(
#             [self._ray_by_angle[theta].steep_fraction for theta in self.angles],
#             dtype=float,
#         )

#     def to_dataframe(self):
#         try:
#             import pandas as pd
#         except ImportError as exception:
#             raise ImportError("Call requires `pandas`. Please install in this environment.") from exception

#         df = pd.DataFrame(
#             {
#                 "theta": [ray.theta for ray in self.rays],
#                 "steep_fraction": [ray.steep_fraction for ray in self.rays],
#                 "total_length_m": [ray.total_length_m for ray in self.rays],
#                 "steep_length_m": [ray.steep_length_m for ray in self.rays],
#                 "max_abs_slope": [ray.max_abs_slope for ray in self.rays],
#                 "n_steep_segments": [ray.n_steep_segments for ray in self.rays],
#             }
#         )
#         return df

#     def steep_segments_geodataframe(self, *, crs):
#         try:
#             import geopandas as gpd
#         except ImportError as exception:
#             raise ImportError("Call requires `geopandas`. Please install in this environment.") from exception

#         records = []
#         for ray in self.rays:
#             for segment in ray.steep_segments:
#                 records.append({"theta": ray.theta, "geometry": segment})

#         return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

#     def plot_polar(self):
#         try:
#             import matplotlib.pyplot as plt
#         except ImportError as exc:
#             raise ImportError("Call requires `matlotlib`. Please install in this environment.") from exception

#         angles_rad = np.deg2rad(self.angles)
#         values = self.ray_steep_fraction

#         fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
#         ax.plot(angles_rad, values)
#         ax.set_theta_zero_location("N")
#         ax.set_theta_direction(-1)
#         ax.set_title("Directional steep fraction")
#         return fig, ax
