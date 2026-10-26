"""Ports: what the use case needs from the outside world.

Formulated from the point of view of the application (P2). Implementations live in `adapters`.
"""

from __future__ import annotations

import dataclasses
import os
import typing
from collections.abc import Mapping

from . import results

Target = str | os.PathLike[str]
"""Where artifacts are written, e.g. a directory."""


@dataclasses.dataclass(frozen=True)
class WrittenArtifacts:
    """What a writer has written, as read back from its own output. Compared with the specification afterwards."""

    profile: str
    """Output profile."""
    files: Mapping[str, str]
    """Artifact key -> filename, for the data files written. The manifest is not among them."""
    table_columns: Mapping[str, tuple[str, ...]]
    """Artifact key -> column names in the file, in order, for artifacts that carry a table."""
    table_records: Mapping[str, int]
    """Artifact key -> number of rows, for artifacts that carry a table."""
    layer_columns: Mapping[str, tuple[str, ...]]
    """Layer key -> attribute column names in the layer, in order. Geometry columns not listed."""
    layer_geometry_types: Mapping[str, str | None]
    """Layer key -> geometry type of the layer, `None` for attribute-only layers."""
    layer_records: Mapping[str, int]
    """Layer key -> number of records, for layers inside container artifacts."""


@typing.runtime_checkable
class ResultWriter(typing.Protocol):
    """Publishes a result. Two steps, so the use case controls the order (DA3, DA4).

    Obligations of every implementation, to be verified by the port test suite (T4):

    - `write_artifacts` writes exactly the artifacts and layers the specification lists for `profile`, given the inputs
      the result was computed from, and never writes the manifest.
    - Filenames and layer names are taken from the specification.
    - Technical failures are translated into a port error (E2).
    - `write_manifest` is called last. It writes the given content and makes no decisions about it.
    """

    def write_artifacts(self, result: results.ResultSummary, target: Target, profile: str) -> WrittenArtifacts:
        """Write all data files of `profile`. Return what was written."""
        ...

    def write_manifest(self, manifest: Mapping[str, typing.Any], target: Target) -> None:
        """Write the manifest. It marks the artifact as complete."""
        ...
