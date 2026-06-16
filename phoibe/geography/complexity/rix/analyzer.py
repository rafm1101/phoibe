from __future__ import annotations

import dataclasses
import datetime
import logging
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.spatial.distance
import xarray
import yaml

from . import trix
from .analyse import compute_regular_rix
from .config import ANALYZER_DEFAULTS, ColumnKeys
from .fieldsampler import RegularGridXYSampler
from .results import RadialRixResult

LOGGER = logging.getLogger(__name__)


_REQUIRED_KEYS = {"n_angles", "R_km", "dr_km", "slope_critical", "crs"}
COLUMN_KEYS = ColumnKeys()


@dataclasses.dataclass(frozen=True)
class ResultSummary:
    """Immutable container for a completed RIX analysis run.

    Attributes
    ----------
    locations_site
        Assessed sites.
    locations_reference
        Assessed reference locations, e.g. of a wind data base.
    summary
        Sites' RIX assessment including location_id, rix, rix_std, n_rays, slope_critical.
    roses
        RIX rose for all sites.
    trix
        Pairwise TRIX table (site x reference) if reference is provided. Otherwise `None`.
        Columns: id_a, id_b, rix_a, rix_b, trix_diff, trix_ratio.
    transferability
        Transferability matrix between site and reference.
        Values from 2 (below distance threshold A) to 0 (above limit threshold B).
    distance_A
        Distance treshold matrix A if reference is provided. Otherwise `None`.
    distance_B
        Distance treshold matrix B if reference is provided. Otherwise `None`.
    steep_segments
        Steep segments of sites as LineStrings. Columns: location_id, theta,
        segment_id, geometry.
    meta
        Meta information including config, timestamp.
    """

    locations_site: gpd.GeoDataFrame
    locations_reference: gpd.GeoDataFrame | None
    summary: pd.DataFrame
    roses: pd.DataFrame
    trix: pd.DataFrame | None
    transferability: pd.DataFrame | None
    distance_A: pd.DataFrame | None
    distance_B: pd.DataFrame | None
    steep_segments: gpd.GeoDataFrame
    meta: dict


class RIXAnalyzer:
    """Compute RIX (and optionally TRIX) for one or two GeoDataFrame collections.

    Parameters
    ----------
    config
        Validated configuration dictionary. Use `.from_config` for the
        normal entry point.

    Notes
    -----
    CRS responsibility lies with the caller: ``dem``, ``locations_a``, and
    ``locations_b`` must all share the projected metric CRS specified in
    ``config["crs"]``.
    """

    def __init__(self, config: dict) -> None:
        _validate_config(config=config)
        self._config = config

    @classmethod
    def from_config(cls, path: str | pathlib.Path) -> RIXAnalyzer:
        """Create an analyzer from a YAML config file.

        Parameters
        ----------
        path
            Path to ``config.yaml``.

        Returns
        -------
        RIXAnalyzer
            Ready-to-use analyzer instance.
        """
        return cls(config=_load_config(path))

    def run(
        self,
        dem: xarray.DataArray,
        locations_site: gpd.GeoDataFrame,
        locations_reference: gpd.GeoDataFrame | None = None,
        keys: ColumnKeys = COLUMN_KEYS,
    ) -> ResultSummary:
        """Run the full RIX analysis.

        Parameters
        ----------
        dem
            Digital elevation model. Must have 'x' and 'y' coordinates in the
            same projected CRS as the locations.
        locations_site
            Point locations. Must have a unique index used as location_id.
        locations_reference
            Optional second collection for pairwise TRIX computation.
        keys
            Column keys for output.

        Returns
        -------
        ResultSummary
            Frozen result object.

        Raises
        ------
        ValueError
            On CRS mismatch or invalid geometries.
        """
        # TODO: Add failed validations to messages, and add messages to meta.
        # self._validate_inputs(dem=dem, locations_site=locations_site, locations_wind=locations_wind)

        sampler = RegularGridXYSampler(da=dem, method="linear")

        radial_rix_site = self._compute_rix_results(sampler, locations_site, keys=keys)
        steep_segments_site = self._build_steep_segments(radial_rix_site, keys=keys)
        rix_roses = self._build_rix_rose(radial_rix_site)
        summary_site = self._build_summary(radial_rix_site, keys=keys)

        trix_values, transferability, A, B = None, None, None, None
        if locations_reference is not None:
            radial_rix_reference = self._compute_rix_results(sampler, locations_reference, keys=keys)
            summary_reference = self._build_summary(radial_rix_reference, keys=keys)
            trix_values, A, B = self._compute_trix(summary_site, summary_reference)
            distances = self._compute_pairwise_distances_m(locations_site.geometry, locations_reference.geometry) / 1000
            transferability_ = trix.evaluate_transferability_limits(distances=distances.values, A=A.values, B=B.values)
            transferability = pd.DataFrame(
                data=transferability_, index=locations_site[keys.site_id], columns=locations_reference[keys.site_id]
            )

        meta = self._build_meta(rix_results=radial_rix_site, keys=keys)

        return ResultSummary(
            locations_site=locations_site,
            locations_reference=locations_reference,
            summary=summary_site,
            roses=rix_roses,
            trix=trix_values,
            transferability=transferability,
            distance_A=A,
            distance_B=B,
            steep_segments=steep_segments_site,
            meta=meta,
        )

    def _validate_inputs(
        self,
        dem: xarray.DataArray,
        locations_site: gpd.GeoDataFrame,
        locations_wind: gpd.GeoDataFrame | None,
        keys: ColumnKeys,
    ) -> None:
        """Check CRS consistency and index uniqueness."""
        expected_crs = self._config["crs"]

        if locations_site.crs is None:
            raise ValueError("locations_wind has no CRS. Set it to match config['crs'].")
        if str(locations_site.crs) != expected_crs and locations_site.crs.to_epsg() != _epsg_int(expected_crs):
            raise ValueError(f"locations_wind CRS mismatch: expected {expected_crs}, got {locations_site.crs}.")
        if not locations_site.index.is_unique:
            raise ValueError("locations_wind index must be unique (used as location_id).")

        if locations_wind is not None:
            if locations_wind.crs is None:
                raise ValueError("locations_reference has no CRS.")
            if str(locations_wind.crs) != expected_crs and locations_wind.crs.to_epsg() != _epsg_int(expected_crs):
                raise ValueError(
                    f"locations_reference CRS mismatch: expected {expected_crs}, got {locations_wind.crs}."
                )
            if not locations_wind.index.is_unique:
                raise ValueError("locations_reference index must be unique.")

        if not {keys.x, keys.y}.issubset(dem.dims):
            raise ValueError("DEM must have 'x' and 'y' coordinates.")

    def _compute_rix_results(
        self, sampler: RegularGridXYSampler, locations: gpd.GeoDataFrame, keys: ColumnKeys
    ) -> dict[object, RadialRixResult]:
        """Run RIX for every location. Returns dict keyed by location_id."""
        cfg = self._config["parameters"]
        results = {}

        for location_id, row in locations.iterrows():
            LOGGER.debug("Computing RIX for location_id=%s", location_id)
            results[location_id] = compute_regular_rix(
                location=row.geometry,
                sampler=sampler,
                n_angles=cfg["n_angles"],
                R_km=cfg["R_km"],
                dr_km=cfg["dr_km"],
                slope_critical=cfg["slope_critical"],
                crs=locations.crs,
                keys=keys,
            )

        return results

    def _get_steep_segments(self, rix_results: dict[object, RadialRixResult]) -> dict[object, RadialRixResult]:
        """Run RIX for every location. Returns dict keyed by location_id."""
        steep_segments = {}

        for location_id, rix_result in rix_results.items():
            steep_segs = rix_result.steep_segments_geodataframe()
            steep_segments[location_id] = steep_segs

        return steep_segments

    def _build_rix_rose(self, radial_results: dict[object, RadialRixResult]) -> pd.DataFrame:
        """Build summary DataFrame and detail GeoDataFrame from radial results."""
        rix_rose_rows = []

        for location_id, radial_rix in radial_results.items():
            rix_rose = pd.Series(index=radial_rix.angles, data=radial_rix.ruggednesses, name=location_id)
            rix_rose_rows.append(rix_rose)

        rix_roses = pd.DataFrame(rix_rose_rows)

        return rix_roses

    def _build_summary(
        self, radial_results: dict[object, RadialRixResult], keys: ColumnKeys = COLUMN_KEYS
    ) -> pd.DataFrame:
        """Build summary DataFrame from radial results."""
        summary_rows = []

        for location_id, radial_rix in radial_results.items():
            description = radial_rix.describe()
            summary_rows.append(
                {
                    keys.site_id: location_id,
                    keys.elevation: radial_rix.z[0],
                    keys.elevation_std: radial_rix.z[1],
                    keys.rix: radial_rix.rix,
                    keys.rix_std: description[keys.rix_std],
                    keys.rix_min: description[keys.rix_min],
                    keys.rix_max: description[keys.rix_max],
                    keys.n_rays: radial_rix.n_rays,
                    keys.slope_critical: radial_rix.slope_critical,
                }
            )

        summary = pd.DataFrame(summary_rows)
        return summary

    def _build_meta(self, rix_results: dict[object, RadialRixResult], keys: ColumnKeys) -> dict:
        records = self._config.copy()
        records[keys.created_at] = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d %H:%M:%S %Z")

        crs_ray = list(set(crs for key, rix_result in rix_results.items() for crs in rix_result.meta[keys.crs_ray]))
        crs_dem = list(set(crs for key, rix_result in rix_results.items() for crs in rix_result.meta[keys.crs_dem]))
        message = list(
            set(message for key, rix_result in rix_results.items() for message in rix_result.meta[keys.message])
        )
        nan_count = sum([rix_result.meta[keys.nan_count] for key, rix_result in rix_results.items()])
        records[keys.data_sources] = dict(crs_ray=crs_ray, crs_dem=crs_dem, message=message, nan_count=nan_count)
        return records

    def _build_steep_segments(
        self, radial_results: dict[object, RadialRixResult], keys: ColumnKeys = COLUMN_KEYS
    ) -> gpd.GeoDataFrame:
        """Build summary DataFrame and detail GeoDataFrame from radial results."""
        crs = self._config["parameters"]["crs"]
        steep_segments_rows = []

        for location_id, radial_rix in radial_results.items():
            gdf_segments = radial_rix.steep_segments_geodataframe()
            crs = gdf_segments.crs
            gdf_segments.insert(0, keys.site_id, location_id)
            steep_segments_rows.append(gdf_segments)

        if steep_segments_rows:
            steep_segments = gpd.GeoDataFrame(
                pd.concat(steep_segments_rows, ignore_index=True), geometry="geometry", crs=crs
            )
        else:
            steep_segments = gpd.GeoDataFrame(
                columns=[keys.site_id, keys.theta, keys.segment_id, "geometry"], geometry="geometry", crs=crs
            )

        return steep_segments

    def _compute_trix(
        self, summary_site: pd.DataFrame, summary_reference: pd.DataFrame, keys: ColumnKeys = COLUMN_KEYS
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute pairwise TRIX as Cartesian product of summary_site × summary_reference.

        Returns
        -------
        trix_result, A, B
            Pairwise trix values and limit distance matrices A and B with sites in rows
            and reference locations in columns.
        """
        trix_records = trix.compute_trix(
            rix_site=np.array(summary_site[keys.rix]),
            elevation_site=np.array(summary_site[keys.elevation]),
            rix_wind=np.array(summary_reference[keys.rix]),
            elevation_wind=np.array(summary_reference[keys.elevation]),
        )
        trix_result = pd.DataFrame(
            data=trix_records, index=summary_site[keys.site_id], columns=summary_reference[keys.site_id]
        )

        _A, _B = trix.compute_trix_limit_distances(trix=trix_records, decimals=1)
        A = pd.DataFrame(data=_A, index=summary_site[keys.site_id], columns=summary_reference[keys.site_id])
        B = pd.DataFrame(data=_B, index=summary_site[keys.site_id], columns=summary_reference[keys.site_id])

        return trix_result, A, B

    def _compute_pairwise_distances_m(
        self, location_site: gpd.GeoSeries, location_reference: gpd.GeoSeries
    ) -> pd.DataFrame:
        """Compute pairwise Euclidean distances between site and wind locations."""
        coords_a = np.vstack([point.coords[0] for point in location_site])
        coords_b = np.vstack([point.coords[0] for point in location_reference])
        distances = pd.DataFrame(
            data=scipy.spatial.distance.cdist(coords_a, coords_b, metric="euclidean"),
            index=location_site.index,
            columns=location_reference.index,
        )
        return distances


def _epsg_int(crs_str: str) -> int | None:
    """Extract integer EPSG code from 'EPSG:XXXX' string, or None."""
    try:
        prefix, code = crs_str.upper().split(":")
        if prefix == "EPSG":
            return int(code)
    except (ValueError, AttributeError):
        pass
    return None


def _load_config(path: str | pathlib.Path) -> dict:
    """Load and validate analyzer configuration from a YAML file.

    Parameters
    ----------
    path
        Path to ``config.yaml``.

    Returns
    -------
    config
        Validated configuration dictionary.

    Raises
    ------
    ValueError
        If required keys are missing or values are out of range.

    Example config.yaml
    -------------------
    n_angles: 36
    R_km: 5.0
    dr_km: 0.05
    slope_critical: 0.3
    crs: "EPSG:2056"
    """
    path = pathlib.Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    config = {**ANALYZER_DEFAULTS, **raw}
    _validate_config(config)
    return config


def _validate_config(config: dict) -> None:
    parameters = config.get("parameters", {})
    missing_keys = _REQUIRED_KEYS - parameters.keys()
    if missing_keys:
        raise ValueError(f"config.yaml is missing required keys: {missing_keys}")
    # if config["crs"] is None:
    #     raise ValueError("config.yaml must specify 'crs' (e.g. 'EPSG:2056').")
    if parameters["dr_km"] >= parameters["R_km"]:
        raise ValueError("dr_km must be smaller than R_km.")
    if not (1 <= parameters["n_angles"] <= 360):
        raise ValueError(f"n_angles must be between 1 and 360, got {parameters['n_angles']}.")
    if parameters["slope_critical"] <= 0:
        raise ValueError(f"slope_critical must be positive, got {parameters['slope_critical']}.")
