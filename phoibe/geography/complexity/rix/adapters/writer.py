from __future__ import annotations

import datetime
import enum
import logging
import pathlib
import typing
import warnings
from collections.abc import Mapping

import geopandas as gpd
import pandas as pd
import pyogrio
import yaml

from ..definitions.config import WRITER_DEFAULTS
from ..definitions.interface import Keys
from ..pipeline.errors import ContractViolation, PortError
from ..pipeline.ports import Target, WrittenArtifacts
from ..pipeline.product.specification import ArtifactSpecification, ProductSpecification
from ..pipeline.results.result_summary import ResultSummary

LOGGER = logging.getLogger(__name__)

_MANIFEST_ARTIFACT = "manifest"


class FileResultWriter:
    """Implements `ResultWriter` for a local directory.

    Parameters
    ----------
    spec
        Product specification. It provides filenames, layer names, and the artifacts of each profile.
    """

    def __init__(self, spec: ProductSpecification) -> None:
        self._spec = spec

    def write_artifacts(self, result: ResultSummary, target: Target, profile: str) -> WrittenArtifacts:
        """Write the data files of `profile` into the directory `target` and read them back.

        Raises
        ------
        ContractViolation
            If the result lacks a table the profile needs. Nothing is written then.
        PortError
            If a file cannot be written or read back.
        """
        present = result.record.present_inputs
        artifacts = [a for a in self._spec.artifacts_for(profile, present) if a.key != _MANIFEST_ARTIFACT]
        self._require_tables(result, artifacts, present)

        directory = pathlib.Path(target)
        files: dict[str, str] = {}
        table_columns: dict[str, tuple[str, ...]] = {}
        table_records: dict[str, int] = {}
        layer_columns: dict[str, tuple[str, ...]] = {}
        layer_geometry_types: dict[str, str | None] = {}
        layer_records: dict[str, int] = {}

        try:
            directory.mkdir(parents=True, exist_ok=True)
            for artifact in artifacts:
                path = directory / artifact.filename
                if artifact.table is not None:
                    columns, records = self._write_csv(path, result.tables[artifact.table])
                    table_columns[artifact.key], table_records[artifact.key] = columns, records
                else:
                    layers = self._write_geopackage(path, artifact, result, present)
                    for key, (columns, geometry_type, records) in layers.items():
                        layer_columns[key] = columns
                        layer_geometry_types[key] = geometry_type
                        layer_records[key] = records
                files[artifact.key] = artifact.filename
                LOGGER.debug("Wrote %s", path)
        except (OSError, pyogrio.errors.DataSourceError, pyogrio.errors.DataLayerError, pd.errors.ParserError) as error:
            raise PortError(f"Could not write artifacts to {directory}: {error}") from error

        return WrittenArtifacts(
            profile=profile,
            files=files,
            table_columns=table_columns,
            table_records=table_records,
            layer_columns=layer_columns,
            layer_geometry_types=layer_geometry_types,
            layer_records=layer_records,
        )

    def write_manifest(self, manifest: Mapping[str, typing.Any], target: Target) -> None:
        """Write the manifest as YAML. Call it last: it marks the artifact as complete."""
        path = pathlib.Path(target) / self._spec.filename(_MANIFEST_ARTIFACT)
        try:
            with path.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(dict(manifest), stream, sort_keys=False, allow_unicode=True)
        except OSError as error:
            raise PortError(f"Could not write the manifest {path}: {error}") from error
        LOGGER.debug("Wrote %s", path)

    def _require_tables(
        self, result: ResultSummary, artifacts: list[ArtifactSpecification], present: frozenset[str]
    ) -> None:
        needed = [artifact.table for artifact in artifacts if artifact.table is not None]
        for artifact in artifacts:
            needed += [layer.table for layer in self._spec.layers_for(artifact.key, present)]
        missing = sorted({name for name in needed if name not in result.tables})
        if missing:
            raise ContractViolation(
                [f"the result has no table {name!r}, which the profile requires." for name in missing]
            )

    @staticmethod
    def _write_csv(path: pathlib.Path, df: pd.DataFrame) -> tuple[tuple[str, ...], int]:
        df.reset_index().to_csv(path, index=False)
        written = pd.read_csv(path)
        return tuple(written.columns), len(written)

    def _write_geopackage(
        self, path: pathlib.Path, artifact: ArtifactSpecification, result: ResultSummary, present: frozenset[str]
    ) -> dict[str, tuple[tuple[str, ...], str | None, int]]:
        layers = self._spec.layers_for(artifact.key, present)
        if not layers:
            raise PortError(f"Artifact {artifact.key!r} has neither a table nor layers; nothing to serialise.")
        path.unlink(missing_ok=True)  # no stale layers from an earlier run
        for layer in layers:
            df = result.tables[layer.table].reset_index()
            if layer.geometry_type is None:
                pyogrio.write_dataframe(pd.DataFrame(df), path, layer=layer.name, driver="GPKG")
            else:
                if not isinstance(df, gpd.GeoDataFrame):
                    raise ContractViolation(
                        [f"Table {layer.table} has no geometry, but layer {layer.key!r} is {layer.geometry_type}."]
                    )
                df.to_file(path, layer=layer.name, driver="GPKG", index=False, geometry_type=layer.geometry_type)  # type: ignore [operator]

        read_back = {}
        for layer in layers:
            info = pyogrio.read_info(path, layer=layer.name)
            columns = tuple(str(name) for name in info["fields"])
            read_back[layer.key] = (columns, info["geometry_type"], int(info["features"]))
        return read_back


@warnings.deprecated("`WriterProfile` is deprecated and shall be removed.")
class WriterProfile(enum.StrEnum):
    """Controls which artifacts are written."""

    SUMMARY = "summary"
    """Lightweight output: Manifest plus output tables."""
    FULL = "full"
    """Detailed output: Summary plus additional detailed packaged for GIS inspection."""


_FILENAMES = WRITER_DEFAULTS["filenames"]
_GPKG_LAYERS = WRITER_DEFAULTS["gpkg_layers"]
KEYS = Keys()


@warnings.deprecated("`RIXWriter` is deprecated and shall be removed.")
class RIXWriter:
    """Serialize a `ResultSummary` to disk.

    Parameters
    ----------
    result
        Completed analysis result.
    profile
        Controls which artifacts are written. See :class:`WriterProfile`.
    filenames
        Filenames for the results.
    gpkg_layers
        Layer names for detailed results.

    Notes
    -----
    1. Profiles:
       1. Profile SUMMARY writes:
            summary.yaml      - manifest: metadata, config, artifact references
            rix_table.csv     - RIX per location + per-angle ruggednesses
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
        # locations_site: gpd.GeoDataFrame | None = None,
        # locations_reference: gpd.GeoDataFrame | None = None,
        filenames: dict = _FILENAMES,
        gpkg_layers: dict = _GPKG_LAYERS,
        keys: Keys = KEYS,
    ) -> None:
        self._result = result
        self._profile = WriterProfile(profile)
        # self._locations_site = locations_site
        # self._locations_reference = locations_reference
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

        self._write_rix_table(out=out, summary=self._result.summary)

        if (trix := self._result.trix_table) is not None:
            self._write_trix(out=out, trix=trix)

        gpkg_records = None
        if self._profile is WriterProfile.FULL:
            gpkg_records = self._write_geopackage(out)

        self._write_manifest(out=out, result=self._result, gpkg_records=gpkg_records, project_name=project_name)

        LOGGER.info("RIXWriter(%s): wrote artifacts to %s", self._profile, out)

    def _write_rix_table(self, out: pathlib.Path, summary: pd.DataFrame) -> None:
        """Write summary of rix assessment."""
        path = out / self._filenames[self._keys.rix_table]
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
                self._keys.rix_table: self._filenames[self._keys.rix_table],
            },
        }
        if result.trix_table is not None:
            artifacts["files"][self._keys.trix_table] = self._filenames[self._keys.trix_table]
        if self._profile is WriterProfile.FULL:
            artifacts["files"][self._keys.geopackage] = self._filenames[self._keys.geopackage]
            artifacts["geopackage_records_per_layer"] = gpkg_records

        manifest = {"project_name": project_name} | {"artifacts": artifacts}  # | result.meta middleware

        path = out / self._filenames[self._keys.manifest]
        with path.open("w") as filestream:
            yaml.safe_dump(manifest, filestream, sort_keys=False, allow_unicode=True)
        LOGGER.debug("Wrote %s", path)

    def _write_geopackage(self, out: pathlib.Path) -> dict:
        """Write all spatial layers to a single GeoPackage."""
        filepath = out / self._filenames[self._keys.geopackage]
        records = {}

        if self._result.locations_site is not None:
            self._result.locations_site.to_file(
                filepath, layer=self._gpkg_layers[self._keys.locations_site_layer], driver="GPKG"
            )
            records[self._keys.locations_site_layer] = len(self._result.locations_site)
            LOGGER.debug("Wrote layer '%s'", self._gpkg_layers[self._keys.locations_site_layer])

        if self._result.locations_reference is not None:
            self._result.locations_reference.to_file(
                filepath, layer=self._gpkg_layers[self._keys.locations_reference_layer], driver="GPKG"
            )
            records[self._keys.locations_reference_layer] = len(self._result.locations_reference)
            LOGGER.debug("Wrote layer '%s'", self._gpkg_layers[self._keys.locations_reference_layer])

        if self._result.steep_segments is not None:
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
