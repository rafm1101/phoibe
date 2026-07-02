from __future__ import annotations

import datetime
import enum
import logging
import typing
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

from .analyzer import ResultSummary

LOGGER = logging.getLogger(__name__)


class WriterProfile(enum.StrEnum):
    """Controls which artifacts are written."""

    SUMMARY = "summary"
    """Lightweight output: Manifest plus output tables."""
    FULL = "full"
    """Detailed output: Summary plus additional detailed packaged for GIS inspection."""


_FILENAMES = {
    "manifest": "summary.yaml",
    "rix_summary": "rix_summary.csv",
    "trix_table": "trix.csv",
    "geopackage": "rix_results.gpkg",
}

_GPKG_LAYERS = {
    "locations_site": "locations_site",
    "locations_reference": "locations_reference",
    "ruggedness": "ruggedness",
    "trix": "trix",
}


class RIXWriter:
    """Serialize a :class:`ResultSummary` to disk.

    Parameters
    ----------
    result
        Completed analysis result.
    profile
        Controls which artifacts are written. See :class:`WriterProfile`.
    locations_a
        Original point GeoDataFrame for collection A (required for FULL profile).
    locations_b
        Original point GeoDataFrame for collection B (required for FULL profile
        when TRIX was computed).
    filenames
        Filenames for the results.
    gpkg_layers
        Layer names for detailed results.

    Notes
    -----
    1. Profiles:
       1. SUMMARY profile writes:
            summary.yaml      - manifest: metadata, config, artifact references
            rix_summary.csv   - RIX per location + per-angle ruggednesses
            trix.csv          - pairwise TRIX table (omitted if result.trix is None)
       1. FULL profile writes everything in SUMMARY plus:
            rix_results.gpkg  - GeoPackage with four layers:
                                standorte          (Point)
                                windmessungen      (Point)
                                rix_detail         (LineString)
                                trix_entscheidungen (attribute table, no geometry)
    ``locations_a`` and ``locations_b`` are kept separate from
    :class:`ResultSummary` intentionally: the result contains derived data
    only. Raw input geometries and their metadata columns are passed here
    when spatial output is needed.

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
    ) -> None:
        self._result = result
        self._profile = WriterProfile(profile)
        self._locations_site = locations_site
        self._locations_reference = locations_reference
        self._filenames = filenames
        self._gpkg_layers = gpkg_layers

        # TODO: Validate configuration arguments for completeness depending on the profile.

        if self._profile is WriterProfile.FULL:
            if locations_site is None:
                raise ValueError("WriterProfile.FULL requires locations_a.")
            if result.trix is not None and locations_reference is None:
                raise ValueError("WriterProfile.FULL with TRIX requires locations_b.")

    def write(self, directory: str | Path) -> None:
        """Write all artifacts for the configured profile to ``directory``.

        Parameters
        ----------
        directory
            Target directory. Created if it does not exist.

        Raises
        ------
        ValueError
            If FULL profile is requested but required inputs are missing.
        """
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        self._write_rix_summary(out=out, summary=self._result.summary)

        if (trix := self._result.trix_table) is not None:
            print("summary")
            self._write_trix(out=out, trix=trix)

        if self._profile is WriterProfile.FULL:
            self._write_geopackage(out)

        self._write_manifest(out=out, result=self._result)

        LOGGER.info("RIXWriter(%s): wrote artifacts to %s", self._profile, out)

    def _write_rix_summary(self, out: Path, summary: pd.DataFrame) -> None:
        """Write summary of rix assessment."""
        path = out / self._filenames["rix_summary"]
        summary.to_csv(path, index=False)
        LOGGER.debug("Wrote %s", path)

    def _write_trix(self, out: Path, trix: pd.DataFrame) -> None:
        """Write pairwise trix-results."""
        print("summary")
        path = out / self._filenames["trix_table"]
        print("summary")
        trix.reset_index().to_csv(path, index=False)
        LOGGER.debug("Wrote %s", path)

    def _write_manifest(self, out: Path, result: ResultSummary) -> None:
        """Write result summary and manifest including metadata and config."""

        artifacts: dict = {
            "rix_summary": self._filenames["rix_summary"],
        }
        if result.trix is not None:
            artifacts["trix"] = self._filenames["trix_table"]
        if self._profile is WriterProfile.FULL:
            artifacts["geopackage"] = self._filenames["geopackage"]

        manifest = {
            "profile": str(self._profile),
            "timestamp": timestamp.isoformat() if _is_datetime(timestamp := result.meta.get("timestamp")) else None,
            "meta": result.meta,
            "artifacts": artifacts,
        }

        path = out / self._filenames["manifest"]
        with path.open("w") as filestream:
            yaml.safe_dump(manifest, filestream, sort_keys=False, allow_unicode=True)
        LOGGER.debug("Wrote %s", path)

    def _write_geopackage(self, out: Path) -> None:
        """Write all spatial layers to a single GeoPackage."""
        filepath = out / self._filenames["geopackage"]

        if self._locations_site is not None:
            self._locations_site.to_file(filepath, layer=self._gpkg_layers["locations_site"], driver="GPKG")
            LOGGER.debug("Wrote layer '%s'", self._gpkg_layers["locations_site"])

        if self._locations_reference is not None:
            self._locations_reference.to_file(filepath, layer=self._gpkg_layers["locations_reference"], driver="GPKG")
            LOGGER.debug("Wrote layer '%s'", self._gpkg_layers["locations_reference"])

        self._result.steep_segments.to_file(filepath, layer=self._gpkg_layers["ruggedness"], driver="GPKG")
        LOGGER.debug("Wrote layer '%s'", self._gpkg_layers["ruggedness"])

        if self._result.trix is not None:
            _write_dataframe_to_gpkg(self._result.trix, filepath, layer=self._gpkg_layers["trix"])
            LOGGER.debug("Wrote layer '%s'", self._gpkg_layers["trix"])

        LOGGER.debug("Wrote GeoPackage %s", filepath)


def _write_dataframe_to_gpkg(df: pd.DataFrame, path: Path, layer: str) -> None:
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
