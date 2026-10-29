from __future__ import annotations

import dataclasses
import datetime
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import scipy.spatial.distance
import xarray

from ..computations.fieldsampler import RegularGridXYSampler
from ..computations.results import RadialRuggedness
from ..computations.rix import compute_regular_rix
from ..definitions.interface import Keys
from .product.schema import PRODUCT_SPECIFICATION
from .product.specification import ProductSpecification

# from .results.tables import _build_rix_rose, _build_summary, _compute_trix

LOGGER = logging.getLogger(__name__)


KEYS = Keys()


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


# def compute_regular_rix(
#     location: shapely.Point,
#     sampler: FieldSampler,
#     n_angles: int,
#     R_km: float,
#     dr_km: float,
#     crs: typing.Any,
#     slope_critical: float,
#     nan_policy="mask",
#     keys: Keys = KEYS,
# ):
#     """Compute the ruggedness index RIX of a location. The RIX assesses height profiles along
#     rays originating at `location`.

#     Parameters
#     ----------
#     location
#         Coordinates of the location to be assessed.
#     sampler
#         A sampler of field values from a regular, metric grid.
#     n_angles
#         Number of rays to cover the entire circle.
#     R_km
#         Distance [km] to which the profiles are considered.
#     dr_km
#         Stepsize [km] to sample from the field.
#     crs
#         CRS of the location.
#     slope_critical
#         Threshold on the slope between two points for a segment to be considered steep.
#     keys
#         Column keys for the output.

#     Returns
#     -------
#     RadialRixResult
#         Gathered results of the evaluation.
#     """
#     angles = np.linspace(0, 360, n_angles, endpoint=False)
#     results = []

#     for theta in angles:
#         ray = RayGeometry.from_compass_regular(location=location, theta=theta, R_km=R_km, dr_km=dr_km, crs=crs)
#         ray_profile = RayProfile.create_regular(ray=ray, sampler=sampler, nan_policy=NaNPolicy(nan_policy), keys=keys)
#         results.append(RayRuggedness(profile=ray_profile, slope_critical=slope_critical, keys=keys))

#     return RadialRuggedness(rays=tuple(results))


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

    def __init__(self, config: ProductSpecification = PRODUCT_SPECIFICATION) -> None:
        # _validate_config(config=config)
        self._config = config

    # TODO: Remove as loading and validation of configs moved.
    # @classmethod
    # def from_config(cls, path: str | pathlib.Path) -> TRIXAnalyzer:
    #     """Create an analyzer from a YAML config file.

    #     Parameters
    #     ----------
    #     path
    #         Path to ``config.yaml``.
    #     keys
    #         Keys for the config.

    #     Returns
    #     -------
    #     RIXAnalyzer
    #         Ready-to-use analyzer instance.
    #     """
    #     return cls(config=_load_config(path))

    # def run(
    #     self,
    #     dem: xarray.DataArray,
    #     locations_site: gpd.GeoDataFrame,
    #     locations_reference: gpd.GeoDataFrame | None = None,
    #     dem_metadata: dict | None = None,
    #     keys: Keys = KEYS,
    # ) -> ResultSummary:
    #     """Run the full RIX analysis.

    #     Parameters
    #     ----------
    #     dem
    #         Digital elevation model. Must have 'x' and 'y' coordinates in the
    #         same projected CRS as the locations.
    #     locations_site
    #         Point locations. Must have a unique index used as location_id.
    #     locations_reference
    #         Optional second collection for pairwise TRIX computation.
    #     dem_metadata
    #         Metadata of the `dem`.
    #     keys
    #         Column keys for output.

    #     Returns
    #     -------
    #     ResultSummary
    #         Frozen result object.

    #     Raises
    #     ------
    #     ValueError
    #         On CRS mismatch or invalid geometries.
    #     """
    #     locations_site = locations_site.copy()
    #     if keys.site_id in locations_site.columns:
    #         locations_site = locations_site.set_index(keys.site_id)
    #     if locations_reference is not None:
    #         locations_reference = locations_reference.copy()
    #         if keys.site_id in locations_reference.columns:
    #             locations_reference = locations_reference.set_index(keys.site_id)

    #     self._validate_inputs(
    #         dem=dem, locations_site=locations_site, locations_reference=locations_reference, keys=keys
    #     )
    #     field = Field.from_array(da=dem, dims=("x", "y"))
    #     # interpolation_method = InterpolationMethods(self._config.parameter("sampler", "interpolation_method").value)
    #     interpolation_method = self._config.parameter("sampler", "interpolation_method").value
    #     sampler = RegularGridXYSampler(field=field, method=interpolation_method)

    #     radial_rix_site = self._compute_rix_results(sampler, locations_site)
    #     # steep_segments_site = self._build_steep_segments(radial_rix_site, keys=keys)
    #     steep_segments_site = None
    # rix_roses = _build_rix_rose(radial_rix_site)
    # summary_site = _build_summary(
    #     radial_rix_site, spec=self._config.tables["rix_table"], index_name=locations_site.index.name
    # )

    # trix_values, transferability, A, B, trix_table = None, None, None, None, None
    # radial_rix_reference = None
    # if locations_reference is not None:
    #     radial_rix_reference = self._compute_rix_results(sampler, locations_reference)
    #     summary_reference = _build_summary(
    #         radial_rix_reference, spec=self._config.tables["rix_table"], index_name=locations_reference.index.name
    #     )
    #     trix_values, A, B = _compute_trix(summary_site, summary_reference, spec=self._config.tables["trix_table"])
    #     distances = self._compute_pairwise_distances_km(
    #         locations_site.geometry, locations_reference.geometry, keys=keys
    #     )
    #     transferability_ = trix.evaluate_transferability_limits(distances=distances.values, A=A.values, B=B.values)
    #     index = locations_site.index.rename(keys.site_id)
    #     columns = locations_reference.index.rename(keys.reference_id)
    #     transferability = pd.DataFrame(data=transferability_, index=index, columns=columns)
    #     trix_table = self._build_trix_results(
    #         trix=trix_values, A=A, B=B, distances=distances, transferability=transferability, keys=keys
    #     )

    # meta = self._build_meta(
    #     results_site=radial_rix_site,
    #     results_reference=radial_rix_reference,
    #     trix_table=trix_table,
    #     dem_metadata=dem_metadata,
    #     keys=keys,
    # )

    # return ResultSummary(
    #     locations_site=locations_site,
    #     locations_reference=locations_reference,
    #     summary=summary_site,
    #     ruggedness_roses=rix_roses,
    #     trix_table=trix_table,
    #     steep_segments=steep_segments_site,
    #     meta=meta,
    # )

    def _validate_inputs(
        self,
        dem: xarray.DataArray,
        locations_site: gpd.GeoDataFrame,
        locations_reference: gpd.GeoDataFrame | None,
        keys: Keys,
    ) -> None:
        # Checks go to compute/checks.py.
        """Check CRS consistency and index uniqueness."""
        expected_crs = self._config.parameter("ray", "crs").value

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
        self, sampler: RegularGridXYSampler, locations: gpd.GeoDataFrame
    ) -> dict[object, RadialRuggedness]:
        """Run RIX for every location. Returns dict keyed by location_id."""
        results = {}

        for location_id, row in locations.iterrows():
            LOGGER.debug("Computing RIX for location_id=%s", location_id)
            results[location_id] = compute_regular_rix(
                location=row.geometry,
                sampler=sampler,
                n_angles=self._config.parameter("ray", "n_angles").value,
                R_km=self._config.parameter("ray", "R_km").value,
                dr_km=self._config.parameter("ray", "dr_km").value,
                slope_critical=self._config.parameter("slope", "slope_critical").value,
                crs=locations.crs,
                # keys=keys,
            )

        return results

    # def _get_steep_segments(self, rix_results: dict[object, RadialRuggedness]) -> dict[object, RadialRuggedness]:
    #     """Run RIX for every location. Returns dict keyed by location_id."""
    #     steep_segments = {}

    #     for location_id, rix_result in rix_results.items():
    #         steep_segs = rix_result.steep_segments_geodataframe()
    #         steep_segments[location_id] = steep_segs

    #     return steep_segments

    # def _build_rix_rose(self, radial_results: dict[object, RadialRuggedness]) -> pd.DataFrame:
    #     """Build summary DataFrame and detail GeoDataFrame from radial results."""
    #     rix_rose_rows = []

    #     for location_id, radial_rix in radial_results.items():
    #         rix_rose = pd.Series(index=radial_rix.angles, data=radial_rix.ruggednesses, name=location_id)
    #         rix_rose_rows.append(rix_rose)

    #     rix_roses = pd.DataFrame(rix_rose_rows)

    #     return rix_roses

    # def _build_summary(
    #     self, radial_results: dict[object, RadialRuggedness], keys: Keys = KEYS, index_name: object = None
    # ) -> pd.DataFrame:
    #     """Build summary DataFrame from radial results."""
    #     summary_rows = []

    #     for location_id, radial_rix in radial_results.items():
    #         description = radial_rix.describe()
    #         summary_rows.append(
    #             {
    #                 keys.site_id: location_id,
    #                 keys.elevation: radial_rix.z[0],
    #                 keys.elevation_std: radial_rix.z[1],
    #                 keys.rix: radial_rix.rix,
    #                 keys.rix_std: description[keys.rix_std],
    #                 keys.rix_min: description[keys.rix_min],
    #                 keys.rix_max: description[keys.rix_max],
    #                 keys.n_rays: radial_rix.n_rays,
    #                 keys.slope_critical: radial_rix.slope_critical,
    #             }
    #         )
    #     index = pd.Index(radial_results.keys(), name=index_name)
    #     summary = pd.DataFrame(summary_rows, index=index)
    #     return summary

    def _build_meta(
        self,
        results_site: dict[object, RadialRuggedness],
        results_reference: dict[object, RadialRuggedness] | None,
        trix_table: pd.DataFrame | None,
        dem_metadata: dict[str, str] | None,
        keys: Keys,
    ) -> dict:
        records: dict = {
            keys.meta: {key: getattr(self._config, key) for key in ["name", "version", "description"]}
            | {
                keys.created_at: datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d %H:%M:%S %Z"),
            }
        }
        records[keys.parameters] = {
            "ray": {
                self._config.parameter("ray", key).name: self._config.parameter("ray", key).value
                for key in ["n_angles", "R_km", "dr_km", "crs"]
            },
            "slope": {
                self._config.parameter("slope", key).name: self._config.parameter("slope", key).value
                for key in ["slope_critical"]
            },
            "sampler": {
                self._config.parameter("sampler", key).name: self._config.parameter("sampler", key).value
                for key in ["interpolation_method"]
            },
        }

        # crs_ray = list(
        #     set(
        #         crs
        #         for _, rix_result in results_site.items()
        #         for crs in _get_parameter(rix_result.meta, keys.rays, keys.crs_ray)
        #     )
        # )
        # crs_dem = list(
        #     set(
        #         crs
        #         for _, rix_result in results_site.items()
        #         for crs in _get_parameter(rix_result.meta, keys.dem, keys.crs_dem)
        #     )
        # )
        # unique_extends_dem = set(
        #     extent
        #     for _, rix_result in results_site.items()
        #     for extent in _get_parameter(rix_result.meta, keys.dem, keys.extent_dem)
        # )
        # extent_dem = [
        #     dict(zip(["west", "south", "east", "north"], extent, strict=True))
        #     for extent in unique_extends_dem
        #     if extent is not None
        # ]
        # unique_resolution_dem = set(
        #     res
        #     for _, rix_result in results_site.items()
        #     for res in _get_parameter(rix_result.meta, keys.dem, keys.resolution_dem)
        # )
        # resolution_dem = [
        #     dict(zip(["dx", "dy"], resolution, strict=True))
        #     for resolution in unique_resolution_dem
        #     if resolution is not None
        # ]
        # message = list(
        #     set(
        #         message
        #         for _, rix_result in results_site.items()
        #         for message in _get_parameter(rix_result.meta, keys.alignment, keys.message)
        #     )
        # )
        # n_nans = sum(
        #     [_get_parameter(rix_result.meta, keys.rays, keys.nan_count) for _, rix_result in results_site.items()]
        # )
        # dem_metadata = copy.deepcopy(dem_metadata) if dem_metadata is not None else {}
        # records[keys.spatial_context] = {
        #     keys.source_dem: dem_metadata.copy() | dict(crs=crs_dem, extent=extent_dem, resolution=resolution_dem),
        #     keys.source_ray: dict(crs=crs_ray, nan_count=n_nans),
        #     keys.alignment: dict(messages=message),
        # }
        # n_sites_w_nans = sum(
        #     [_get_parameter(rix_result.meta, keys.rays, keys.nan_count) > 0 for _, rix_result in results_site.items()]
        # )
        computed = ["rix_site"]
        transferability_counts = {}
        if results_reference is not None:
            computed.append("rix_reference")
        if trix_table is not None:
            computed.append("trix")
            transferability_counts = trix_table[keys.transferability].value_counts().sort_index().to_dict()
        records[keys.run] = {
            keys.n_sites: len(results_site),
            keys.n_references: len(results_reference) if results_reference is not None else 0,
            keys.computed: computed,
            keys.diagnostics: {
                keys.n_sites_with_nans: 0,  # n_sites_w_nans,
                keys.transferability_counts: transferability_counts,
            },
        }
        return records

    # def _build_steep_segments(
    #     self, radial_results: dict[object, RadialRuggedness], keys: Keys = KEYS
    # ) -> gpd.GeoDataFrame:
    #     """Build summary DataFrame and detail GeoDataFrame from radial results."""
    #     steep_segments_rows = []

    #     for location_id, radial_rix in radial_results.items():
    #         gdf_segments = radial_rix.steep_segments_geodataframe()
    #         gdf_segments.insert(0, keys.site_id, location_id)
    #         steep_segments_rows.append(gdf_segments)

    #     if steep_segments_rows:
    #         steep_segments = gpd.GeoDataFrame(
    #             pd.concat(steep_segments_rows, ignore_index=True), geometry=keys.geometry
    #         ).set_index(keys.site_id)
    #     else:
    #         steep_segments = gpd.GeoDataFrame(
    #             columns=[keys.site_id, keys.theta, keys.segment_id, keys.geometry], geometry=keys.geometry, crs=None
    #         ).set_index(keys.site_id)

    #     return steep_segments

    # def _compute_trix(
    #     self, summary_site: pd.DataFrame, summary_reference: pd.DataFrame, keys: Keys = KEYS
    # ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #     # Move to compute/trix.py?
    #     """Compute pairwise TRIX as Cartesian product of summary_site × summary_reference.

    #     Returns
    #     -------
    #     trix_result, A, B
    #         Pairwise trix values and limit distance matrices A and B with sites in rows
    #         and reference locations in columns.
    #     """
    #     trix_records = trix.compute_trix(
    #         rix_site=np.array(summary_site[keys.rix]),
    #         elevation_site=np.array(summary_site[keys.elevation]),
    #         rix_reference=np.array(summary_reference[keys.rix]),
    #         elevation_reference=np.array(summary_reference[keys.elevation]),
    #     )
    #     index = summary_site.index.rename(keys.site_id)
    #     columns = summary_reference.index.rename(keys.reference_id)

    #     trix_result = pd.DataFrame(data=trix_records, index=index, columns=columns)

    #     _A, _B = trix.compute_trix_limit_distances(trix=trix_records, decimals=1)
    #     A = pd.DataFrame(data=_A, index=index, columns=columns)
    #     B = pd.DataFrame(data=_B, index=index, columns=columns)

    #     return trix_result, A, B

    def _compute_pairwise_distances_km(
        self, location_site: gpd.GeoSeries, location_reference: gpd.GeoSeries, keys: Keys
    ) -> pd.DataFrame:
        # Move to compute/trix.py.
        """Compute pairwise Euclidean distances [km] between site and wind locations."""
        coords_a = np.vstack([point.coords[0] for point in location_site])
        coords_b = np.vstack([point.coords[0] for point in location_reference])
        distances = pd.DataFrame(
            data=scipy.spatial.distance.cdist(coords_a, coords_b, metric="euclidean") / 1000,
            index=location_site.index.rename(keys.site_id),
            columns=location_reference.index.rename(keys.reference_id),
        )
        return distances

    def _build_trix_results(self, trix, A, B, distances, transferability, keys: Keys):
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
