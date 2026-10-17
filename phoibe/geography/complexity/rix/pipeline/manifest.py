"""Assembly of the run manifest from the specification, the run record and what was written.

A pure function: no clock, no file access. The result only contains plain Python values, so it serialises with
``yaml.safe_dump`` (DA5, D-A5).

Shape of the manifest
---------------------
::

    contract                      generated from the specification, identical for all runs of a version
    meta                          name, version, description, created_at (UTC, ISO 8601)
    parameters                    {group: {name: value}} as used in this run
    spatial_context
        source_dem                free-form provenance + crs, extent {west, south, east, north}, resolution {dx, dy}
        source_ray                crs: distinct CRS of the rays
        alignment                 messages: distinct messages from aligning rays and field
    run
        project_name, n_sites, n_references, computed
        diagnostics
            nan_count                {collection: n}
            n_locations_with_nans    {collection: n}
            transferability_counts   {state: n}
            finding_counts           {check: n}
            findings_omitted         n
        findings                  [{check, severity, collection, location_id, theta, detail}], capped per check
    artifacts
        profile, files {key: filename}, table_records {key: n}, layer_records {key: n}
"""

from __future__ import annotations

import copy
import datetime
import typing
from collections.abc import Mapping

import numpy as np

from .contract import render_contract
from .inputs import LOCATIONS_REFERENCE, LOCATIONS_SITE
from .ports import WrittenArtifacts
from .records import RunRecord
from .specification import ProductSpecification

_MANIFEST_ARTIFACT = "manifest"
_EXTENT_KEYS = ("west", "south", "east", "north")
_RESOLUTION_KEYS = ("dx", "dy")


def build_manifest(spec: ProductSpecification, record: RunRecord, written: WrittenArtifacts) -> dict[str, typing.Any]:
    """Build the manifest of a run.

    Parameters
    ----------
    spec
        Specification of the product. It provides the contract and the manifest's own filename.
    record
        Record of the run.
    written
        What the writer wrote. The manifest file is added, as it is written last.

    Raises
    ------
    ValueError
        If the parameters of the run and the specification do not describe the same set of parameters.
    """
    field_info = record.field_info
    source_dem = copy.deepcopy(dict(record.dem_provenance)) | {
        "crs": field_info.crs,
        "extent": _labelled(_EXTENT_KEYS, field_info.extent),
        "resolution": _labelled(_RESOLUTION_KEYS, field_info.resolution),
    }
    records = record.collections
    site = records[LOCATIONS_SITE]
    reference = records.get(LOCATIONS_REFERENCE)

    manifest = {
        "contract": render_contract(spec),
        "meta": {
            "name": spec.name,
            "version": spec.version,
            "description": spec.description,
            "created_at": record.created_at.astimezone(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "parameters": _parameters_used(spec, record.parameters),
        "spatial_context": {
            "source_dem": source_dem,
            "source_ray": {"crs": _union(item.ray_crs for item in records.values())},
            "alignment": {"messages": _union(item.messages for item in records.values())},
        },
        "run": {
            "project_name": record.project_name,
            "n_sites": site.n_locations,
            "n_references": 0 if reference is None else reference.n_locations,
            "computed": list(record.computed),
            "diagnostics": {
                "nan_count": {name: item.nan_count for name, item in records.items()},
                "n_locations_with_nans": {name: item.n_locations_with_nans for name, item in records.items()},
                "transferability_counts": dict(record.transferability_counts or {}),
                "finding_counts": dict(record.finding_counts),
                "findings_omitted": record.n_findings_omitted,
            },
            "findings": [
                {
                    "check": item.finding.check,
                    "severity": item.severity,
                    "collection": item.collection,
                    "location_id": item.location_id,
                    "theta": item.finding.theta,
                    "detail": item.finding.detail,
                }
                for item in record.findings
            ],
        },
        "artifacts": {
            "profile": written.profile,
            "files": {**written.files, _MANIFEST_ARTIFACT: spec.filename(_MANIFEST_ARTIFACT)},
            "table_records": dict(written.table_records),
            "layer_records": dict(written.layer_records),
        },
    }
    return _plain(manifest)


def _parameters_used(spec: ProductSpecification, values: Mapping[str, typing.Any]) -> dict[str, dict[str, typing.Any]]:
    defined = {parameter.name for parameter in spec.parameters}
    missing = sorted(defined - values.keys())
    undefined = sorted(values.keys() - defined)
    if missing or undefined:
        raise ValueError(
            "Parameters of the run and of the specification differ: "
            f"missing in the run: {missing or 'none'}; not in the specification: {undefined or 'none'}."
        )
    grouped: dict[str, dict[str, typing.Any]] = {}
    for parameter in spec.parameters:
        grouped.setdefault(parameter.group, {})[parameter.name] = values[parameter.name]
    return grouped


def _labelled(labels: tuple[str, ...], values: typing.Sequence[float] | None) -> dict[str, float] | None:
    return None if values is None else dict(zip(labels, values, strict=True))


def _union(groups: typing.Iterable[typing.Iterable[typing.Any]]) -> list[typing.Any]:
    """Distinct values over all groups, in order of appearance."""
    return list(dict.fromkeys(value for group in groups for value in group))


def _plain(value: typing.Any) -> typing.Any:
    """Convert to values that ``yaml.safe_dump`` writes and ``safe_load`` reads back unchanged."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {_plain_key(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _plain_key(key: typing.Any) -> typing.Any:
    if isinstance(key, np.generic):
        return key.item()
    return key if isinstance(key, (str, int, float, bool)) else str(key)
