from __future__ import annotations

import copy
import dataclasses
import datetime
import logging
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import scipy.spatial.distance
import xarray
import yaml

from . import evaluate, trix
from .config import ANALYZER_DEFAULTS, DEM_METADATA
from .fieldsampler import RegularGridXYSampler
from .keys import ColumnKeys, _get_parameter
from .results import RadialRuggedness

LOGGER = logging.getLogger(__name__)


_REQUIRED_KEYS = {"n_angles", "R_km", "dr_km", "slope_critical"}
COLUMN_KEYS = ColumnKeys()


@dataclasses.dataclass(frozen=True)
class ResultSummary:
    """Immutable container for a completed RIX analysis run."""

    locations_site: gpd.GeoDataFrame
    """Assessed sites."""
    locations_reference: gpd.GeoDataFrame | None
    """Assessed reference locations, e.g. of a wind data base."""
    summary: pd.DataFrame
    """Sites' RIX assessment including location_id, rix, rix_std, n_rays, slope_critical."""
    ruggedness_roses: pd.DataFrame
    """Ruggedness rose for all sites."""
    trix_table: pd.DataFrame | None
    """Table containing all pairwise metrics and threshold matrices A and B required for transferability.
       Columns: transferability, distance, trix, A, B."""
    steep_segments: gpd.GeoDataFrame
    """Steep segments of sites as LineStrings. Columns: location_id, theta, segment_id, geometry."""
    meta: dict
    """Meta information including config, timestamp and processing information."""


class TRIXAnalyzer:
    """Compute RIX (and optionally TRIX) for one or two GeoDataFrame collections.

    Parameters
    ----------
    config
        Validated configuration dictionary. Use `.from_config` for the
        normal entry point.

    Notes
    -----
    CRS responsibility lies with the caller: ``dem``, ``locations_site``, and
    ``locations_reference`` must all share the projected metric CRS specified in
    ``config["crs"]``.
    """

    def __init__(self, config: dict) -> None:
        _validate_config(config=config)
        self._config = config

    @classmethod
    def from_config(cls, path: str | pathlib.Path) -> TRIXAnalyzer:
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
        dem_metadata: dict = DEM_METADATA,
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
        dem_metadata
            Metadata of the `dem`.
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
        # self._validate_inputs(dem=dem, locations_site=locations_site, locations_reference=locations_reference)

        locations_site = locations_site.copy()
        if not keys.site_id == "index":
            locations_site = locations_site.set_index(keys.site_id)
            if locations_reference is not None:
                locations_reference = locations_reference.copy().set_index(keys.site_id)

        sampler = RegularGridXYSampler(da=dem, method=_get_parameter(self._config, "sampler", "interpolation_method"))

        radial_rix_site = self._compute_rix_results(sampler, locations_site, keys=keys)
        steep_segments_site = self._build_steep_segments(radial_rix_site, keys=keys)
        rix_roses = self._build_rix_rose(radial_rix_site)
        summary_site = self._build_summary(radial_rix_site, keys=keys)

        trix_values, transferability, A, B, trix_table = None, None, None, None, None
        if locations_reference is not None:
            radial_rix_reference = self._compute_rix_results(sampler, locations_reference, keys=keys)
            summary_reference = self._build_summary(radial_rix_reference, keys=keys)
            trix_values, A, B = self._compute_trix(summary_site, summary_reference)
            distances = self._compute_pairwise_distances_km(
                locations_site.geometry, locations_reference.geometry, keys=keys
            )
            transferability_ = trix.evaluate_transferability_limits(distances=distances.values, A=A.values, B=B.values)
            index = locations_site.index.rename(keys.site_id)
            columns = locations_reference.index.rename(keys.reference_id)
            transferability = pd.DataFrame(data=transferability_, index=index, columns=columns)
            trix_table = self._build_trix_results(
                trix=trix_values, A=A, B=B, distances=distances, transferability=transferability, keys=keys
            )

        meta = self._build_meta(rix_results=radial_rix_site, dem_metadata=dem_metadata, keys=keys)

        return ResultSummary(
            locations_site=locations_site,
            locations_reference=locations_reference,
            summary=summary_site,
            ruggedness_roses=rix_roses,
            trix_table=trix_table,
            steep_segments=steep_segments_site,
            meta=meta,
        )

    def _validate_inputs(
        self,
        dem: xarray.DataArray,
        locations_site: gpd.GeoDataFrame,
        locations_reference: gpd.GeoDataFrame | None,
        keys: ColumnKeys,
    ) -> None:
        """Check CRS consistency and index uniqueness."""
        expected_crs = self._config["crs"]

        if locations_reference is not None:
            if locations_site.crs != locations_reference.crs:
                raise ValueError(
                    f"CRS mismatch between locations_site {locations_site.crs} and "
                    f"locations_reference {locations_reference.crs}."
                )
        if expected_crs is not None and locations_site.crs is not None:
            if locations_site.crs != pyproj.CRS.from_user_input(expected_crs):
                LOGGER.warning("CRS of locations_site %s differs from config CRS %s,", locations_site.crs, expected_crs)

        if not locations_site.index.is_unique:
            raise ValueError("locations_site index must be unique (as being used as identifier).")
        if locations_reference is not None:
            if not locations_reference.index.is_unique:
                raise ValueError("locations_reference index must be unique (as being used as identifier).")

        # Check in RegularGridXYSampler.sample possibly early enough.
        # if not {keys.x, keys.y}.issubset(dem.dims):
        #     raise ValueError("DEM must have 'x' and 'y' coordinates.")

    def _compute_rix_results(
        self, sampler: RegularGridXYSampler, locations: gpd.GeoDataFrame, keys: ColumnKeys
    ) -> dict[object, RadialRuggedness]:
        """Run RIX for every location. Returns dict keyed by location_id."""
        cfg = self._config["parameters"]
        results = {}

        for location_id, row in locations.iterrows():
            LOGGER.debug("Computing RIX for location_id=%s", location_id)
            results[location_id] = evaluate.compute_regular_rix(
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

    def _get_steep_segments(self, rix_results: dict[object, RadialRuggedness]) -> dict[object, RadialRuggedness]:
        """Run RIX for every location. Returns dict keyed by location_id."""
        steep_segments = {}

        for location_id, rix_result in rix_results.items():
            steep_segs = rix_result.steep_segments_geodataframe()
            steep_segments[location_id] = steep_segs

        return steep_segments

    def _build_rix_rose(self, radial_results: dict[object, RadialRuggedness]) -> pd.DataFrame:
        """Build summary DataFrame and detail GeoDataFrame from radial results."""
        rix_rose_rows = []

        for location_id, radial_rix in radial_results.items():
            rix_rose = pd.Series(index=radial_rix.angles, data=radial_rix.ruggednesses, name=location_id)
            rix_rose_rows.append(rix_rose)

        rix_roses = pd.DataFrame(rix_rose_rows)

        return rix_roses

    def _build_summary(
        self, radial_results: dict[object, RadialRuggedness], keys: ColumnKeys = COLUMN_KEYS
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

        summary = pd.DataFrame(summary_rows).set_index(keys.site_id)
        return summary

    def _build_meta(
        self, rix_results: dict[object, RadialRuggedness], dem_metadata: dict[str, str], keys: ColumnKeys
    ) -> dict:
        records: dict = {
            "meta": {key: self._config[key] for key in ["name", "version", "description"]}
            | {
                keys.created_at: datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d %H:%M:%S %Z"),
            }
        }
        records["parameters"] = {
            "ray": {key: _get_parameter(self._config, "parameters", key) for key in ["n_angles", "R_km", "dr_km"]},
            "slope": {key: _get_parameter(self._config, "parameters", key) for key in ["slope_critical"]},
            "sampler": _get_parameter(self._config, "sampler").copy(),
        }

        crs_ray = list(
            set(
                crs
                for _, rix_result in rix_results.items()
                for crs in _get_parameter(rix_result.meta, "rays", keys.crs_ray)
            )
        )
        crs_dem = list(
            set(
                crs
                for _, rix_result in rix_results.items()
                for crs in _get_parameter(rix_result.meta, "dem", keys.crs_dem)
            )
        )
        unique_extends_dem = set(
            extent
            for _, rix_result in rix_results.items()
            for extent in _get_parameter(rix_result.meta, "dem", keys.extent_dem)
        )
        extent_dem = list(
            dict(zip(["west", "south", "east", "north"], extent, strict=True)) for extent in unique_extends_dem
        )
        unique_resolution_dem = set(
            res
            for _, rix_result in rix_results.items()
            for res in _get_parameter(rix_result.meta, "dem", keys.resolution_dem)
        )
        resolution_dem = list(dict(zip(["dx", "dy"], resolution, strict=True)) for resolution in unique_resolution_dem)
        message = list(
            set(
                message
                for _, rix_result in rix_results.items()
                for message in _get_parameter(rix_result.meta, "alignment", keys.message)
            )
        )
        nan_count = sum(
            [_get_parameter(rix_result.meta, "rays", keys.nan_count) for _, rix_result in rix_results.items()]
        )
        records[keys.spatial_context] = {
            keys.source_dem: dem_metadata.copy() | dict(crs=crs_dem, extent=extent_dem, resolution=resolution_dem),
            keys.source_ray: dict(crs=crs_ray, nan_count=nan_count),
            keys.alignment: dict(messages=message),
        }
        return records

    def _build_steep_segments(
        self, radial_results: dict[object, RadialRuggedness], keys: ColumnKeys = COLUMN_KEYS
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
                pd.concat(steep_segments_rows, ignore_index=True), geometry=keys.geometry, crs=crs
            )
        else:
            steep_segments = gpd.GeoDataFrame(
                columns=[keys.site_id, keys.theta, keys.segment_id, keys.geometry], geometry=keys.geometry, crs=crs
            ).set_index(keys.site_id)

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
        index = summary_site.index.rename(keys.site_id)
        columns = summary_reference.index.rename(keys.reference_id)

        trix_result = pd.DataFrame(data=trix_records, index=index, columns=columns)

        _A, _B = trix.compute_trix_limit_distances(trix=trix_records, decimals=1)
        A = pd.DataFrame(data=_A, index=index, columns=columns)
        B = pd.DataFrame(data=_B, index=index, columns=columns)

        return trix_result, A, B

    def _compute_pairwise_distances_km(
        self, location_site: gpd.GeoSeries, location_reference: gpd.GeoSeries, keys: ColumnKeys
    ) -> pd.DataFrame:
        """Compute pairwise Euclidean distances [km] between site and wind locations."""
        coords_a = np.vstack([point.coords[0] for point in location_site])
        coords_b = np.vstack([point.coords[0] for point in location_reference])
        distances = pd.DataFrame(
            data=scipy.spatial.distance.cdist(coords_a, coords_b, metric="euclidean") / 1000,
            index=location_site.index.rename(keys.site_id),
            columns=location_reference.index.rename(keys.reference_id),
        )
        return distances

    def _build_trix_results(self, trix, A, B, distances, transferability, keys: ColumnKeys):
        """Build the T-RIX table containing the transferability, distances between wind data base and site, T-RIX
        and threshold distances A and B.
        """
        trix_ = trix.stack().rename(keys.trix)
        A_ = A.stack().rename(keys.A)
        B_ = B.stack().rename(keys.B)
        distances_ = distances.stack().rename(keys.distance)
        transferability_ = transferability.stack().rename(keys.transferability)

        table = pd.concat([transferability_, distances_, trix_, A_, B_], axis=1)

        return table


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
    parameters:
      n_angles: 36
      R_km: 5.0
      dr_km: 0.05
      slope_critical: 0.3
      crs: "EPSG:2056"
    """
    path = pathlib.Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    raw_parameters = raw.get("parameters", {})
    config = copy.deepcopy(ANALYZER_DEFAULTS)

    for key in raw_parameters:
        if key in config["parameters"]:
            config["parameters"][key] = raw_parameters[key]
        else:
            raise ValueError(f"Configuration file {str(path)} contains unknown parameter {key}.")

    _validate_config(config)
    return config


def _validate_config(config: dict) -> None:
    parameters = config.get("parameters", {})
    missing_keys = _REQUIRED_KEYS - parameters.keys()
    if missing_keys:
        raise ValueError(f"config.yaml is missing required keys: {missing_keys}")
    if parameters["dr_km"] >= parameters["R_km"]:
        raise ValueError("dr_km must be smaller than R_km.")
    if not (1 <= parameters["n_angles"] <= 360):
        raise ValueError(f"n_angles must be between 1 and 360, got {parameters['n_angles']}.")
    if parameters["slope_critical"] <= 0:
        raise ValueError(f"slope_critical must be positive, got {parameters['slope_critical']}.")
