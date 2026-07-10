from __future__ import annotations

import datetime
import enum
import logging
import pathlib
import typing

import geopandas as gpd
import pandas as pd
import yaml

from .analyzer import ResultSummary
from .config import WRITER_DEFAULTS
from .interface import Keys

LOGGER = logging.getLogger(__name__)


class WriterProfile(enum.StrEnum):
    """Controls which artifacts are written."""

    SUMMARY = "summary"
    """Lightweight output: Manifest plus output tables."""
    FULL = "full"
    """Detailed output: Summary plus additional detailed packaged for GIS inspection."""


_FILENAMES = WRITER_DEFAULTS["filenames"]
_GPKG_LAYERS = WRITER_DEFAULTS["gpkg_layers"]
KEYS = Keys()


class RIXWriter:
    """Serialize a `ResultSummary` to disk.

    Parameters
    ----------
    result
        Completed analysis result.
    profile
        Controls which artifacts are written. See :class:`WriterProfile`.
    locations_site
        Original point GeoDataFrame for collection A (required for FULL profile).
    locations_reference
        Original point GeoDataFrame for collection B (required for FULL profile
        when TRIX was computed).
    filenames
        Filenames for the results.
    gpkg_layers
        Layer names for detailed results.

    Notes
    -----
    1. Profiles:
       1. Profile SUMMARY writes:
            summary.yaml      - manifest: metadata, config, artifact references
            rix_summary.csv   - RIX per location + per-angle ruggednesses
            trix.csv          - pairwise TRIX table (omitted if result.trix_table is None)
       1. Profile FULL writes everything in SUMMARY plus:
            rix_details.gpkg  - GeoPackage with four layers:
                                locations_site                 (Point)
                                locations_reference            (Point)
                                ruggedness                (LineString)
                                trix    (attribute table, no geometry)

    Examples
    --------
    >>> writer = RIXWriter(result, profile=WriterProfile.SUMMARY)
    >>> writer.write(directory="output/")
    """

    def __init__(
        self,
        result: ResultSummary,
        profile: WriterProfile = WriterProfile.SUMMARY,
        locations_site: gpd.GeoDataFrame | None = None,
        locations_reference: gpd.GeoDataFrame | None = None,
        filenames: dict = _FILENAMES,
        gpkg_layers: dict = _GPKG_LAYERS,
        keys: Keys = KEYS,
    ) -> None:
        self._result = result
        self._profile = WriterProfile(profile)
        self._locations_site = locations_site
        self._locations_reference = locations_reference
        self._filenames = filenames
        self._gpkg_layers = gpkg_layers
        self._keys = keys

        # TODO: Validate configuration arguments for completeness depending on the profile.

        # if self._profile is WriterProfile.FULL:
        #     if locations_site is None:
        #         raise ValueError("WriterProfile.FULL requires `locations_site`.")
        #     if result.trix_table is not None and locations_reference is None:
        #         raise ValueError("WriterProfile.FULL with TRIX requires `locations_reference`.")

    def write(self, directory: str | pathlib.Path, project_name: str = "") -> None:
        """Write all artifacts for the configured profile to `directory`.

        Parameters
        ----------
        directory
            Target directory. Created if it does not exist.

        Raises
        ------
        ValueError
            If FULL profile is requested but required inputs are missing.
        """
        out = pathlib.Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        self._write_rix_summary(out=out, summary=self._result.summary)

        if (trix := self._result.trix_table) is not None:
            self._write_trix(out=out, trix=trix)

        gpkg_records = None
        if self._profile is WriterProfile.FULL:
            gpkg_records = self._write_geopackage(out)

        self._write_manifest(out=out, result=self._result, gpkg_records=gpkg_records, project_name=project_name)

        LOGGER.info("RIXWriter(%s): wrote artifacts to %s", self._profile, out)

    def _write_rix_summary(self, out: pathlib.Path, summary: pd.DataFrame) -> None:
        """Write summary of rix assessment."""
        path = out / self._filenames[self._keys.rix_summary]
        summary.reset_index().to_csv(path, index=False)
        LOGGER.debug("Wrote %s", path)

    def _write_trix(self, out: pathlib.Path, trix: pd.DataFrame) -> None:
        """Write pairwise trix-results."""
        path = out / self._filenames[self._keys.trix_table]
        trix.reset_index().to_csv(path, index=False)
        LOGGER.debug("Wrote %s", path)

    def _write_manifest(
        self, out: pathlib.Path, result: ResultSummary, gpkg_records: dict | None, project_name: str
    ) -> None:
        """Write result summary and manifest including metadata and config."""

        artifacts: dict = {
            "profile": str(self._profile),
            "files": {
                self._keys.manifest: self._filenames[self._keys.manifest],
                self._keys.rix_summary: self._filenames[self._keys.rix_summary],
            },
        }
        if result.trix_table is not None:
            artifacts["files"][self._keys.trix_table] = self._filenames[self._keys.trix_table]
        if self._profile is WriterProfile.FULL:
            artifacts["files"][self._keys.geopackage] = self._filenames[self._keys.geopackage]
            artifacts["geopackage_records_per_layer"] = gpkg_records

        manifest = {"project_name": project_name} | result.meta | {"artifacts": artifacts}

        path = out / self._filenames[self._keys.manifest]
        with path.open("w") as filestream:
            yaml.safe_dump(manifest, filestream, sort_keys=False, allow_unicode=True)
        LOGGER.debug("Wrote %s", path)

    def _write_geopackage(self, out: pathlib.Path) -> dict:
        """Write all spatial layers to a single GeoPackage."""
        filepath = out / self._filenames[self._keys.geopackage]
        records = {}

        if self._locations_site is not None:
            self._locations_site.to_file(
                filepath, layer=self._gpkg_layers[self._keys.locations_site_layer], driver="GPKG"
            )
            records[self._keys.locations_site_layer] = len(self._locations_site)
            LOGGER.debug("Wrote layer '%s'", self._gpkg_layers[self._keys.locations_site_layer])

        if self._locations_reference is not None:
            self._locations_reference.to_file(
                filepath, layer=self._gpkg_layers[self._keys.locations_reference_layer], driver="GPKG"
            )
            records[self._keys.locations_reference_layer] = len(self._locations_reference)
            LOGGER.debug("Wrote layer '%s'", self._gpkg_layers[self._keys.locations_reference_layer])

        self._result.steep_segments.to_file(
            filepath, layer=self._gpkg_layers[self._keys.ruggedness_layer], driver="GPKG"
        )
        records[self._keys.ruggedness_layer] = len(self._result.steep_segments)
        LOGGER.debug("Wrote layer '%s'", self._gpkg_layers[self._keys.ruggedness_layer])

        if self._result.trix_table is not None:
            _write_dataframe_to_gpkg(self._result.trix_table, filepath, layer=self._gpkg_layers[self._keys.trix_layer])
            records[self._keys.trix_layer] = len(self._result.trix_table)
            LOGGER.debug("Wrote layer '%s'", self._gpkg_layers[self._keys.trix_layer])

        LOGGER.debug("Wrote GeoPackage %s", filepath)
        return records


def _write_dataframe_to_gpkg(df: pd.DataFrame, path: pathlib.Path, layer: str) -> None:
    """Write a vanilla DataFrame w/o geometry as a table layer in a GeoPackage.

    Parameters
    ----------
    df
        DataFrame to write. Must not contain a 'geometry' column.
    path
        Path to the target GeoPackage (appended to if it exists).
    layer
        Layer name inside the GeoPackage.

    Notes
    -----
    1. GeoPackage supports attribute-only tables via fiona/pyogrio when a geometry column is absent.
       Employ a minimal GeoDataFrame with null geometries as the portable fallback.
    """
    gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.GeoSeries([None] * len(df)))
    gdf.to_file(path, layer=layer, driver="GPKG")


def _is_datetime(obj: typing.Any) -> typing.TypeGuard[datetime.datetime]:
    return isinstance(obj, datetime.datetime)
