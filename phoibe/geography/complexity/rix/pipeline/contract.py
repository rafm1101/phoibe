"""The contract of the product, generated from the specification (DV2).

The contract is identical for all runs of one product version (DA1). It states what the product needs and delivers:
inputs, parameters, checks and outputs. It contains only plain Python values (dict, list, str, int, float, bool, None),
so it serialises to YAML with ``safe_dump`` without loss.

Parameters: a locked parameter carries its fixed value (DV6). For all others the contract only states that they
exist and which values are allowed. Their value in a run belongs to the run section of the manifest.
"""

from __future__ import annotations

import typing

from .specification import ColumnSpecification, LayerSpecification, ProductSpecification


def render_contract(spec: ProductSpecification) -> dict[str, typing.Any]:
    """Render the contract of the product described by `spec`."""
    return {
        "product": spec.name,
        "version": spec.version,
        "description": spec.description,
        "inputs": {name: _input(item) for name, item in spec.inputs.items()},
        "parameters": _parameters(spec),
        "checks": [_check(item) for item in spec.checks.values()],
        "outputs": {key: _output(spec, key) for key in spec.artifacts},
        "profiles": {profile: list(keys) for profile, keys in spec.profiles.items()},
        "references": {key: dict(entry) for key, entry in spec.references.items()},
    }


def _input(item) -> dict[str, typing.Any]:
    rendered: dict[str, typing.Any] = {"required": item.required, "description": item.description}
    if item.type is not None:
        rendered["type"] = item.type
    if item.notes:
        rendered["notes"] = list(item.notes)
    if item.conditions:
        rendered["conditions"] = list(item.conditions)
    return rendered


def _parameters(spec: ProductSpecification) -> dict[str, dict[str, typing.Any]]:
    grouped: dict[str, dict[str, typing.Any]] = {}
    for parameter in spec.parameters:
        rendered: dict[str, typing.Any] = {
            "description": parameter.description,
            "dtype": parameter.dtype,
            "unit": parameter.unit,
            "locked": parameter.locked,
        }
        if parameter.locked:
            rendered["value"] = parameter.value
        if parameter.valid_range is not None:
            rendered["range"] = list(parameter.valid_range)
        if parameter.source is not None:
            rendered["source"] = parameter.source
        grouped.setdefault(parameter.group, {})[parameter.name] = rendered
    return grouped


def _check(item) -> dict[str, typing.Any]:
    rendered: dict[str, typing.Any] = {
        "name": item.name,
        "description": item.description,
        "on_failure": item.severity,
    }
    if item.parameters:
        rendered["parameters"] = dict(item.parameters)
    return rendered


def _column(column: ColumnSpecification) -> dict[str, typing.Any]:
    rendered: dict[str, typing.Any] = {
        "name": column.name,
        "type": column.dtype,
        "unit": column.unit,
        "description": column.description,
    }
    if column.valid_range is not None:
        rendered["range"] = list(column.valid_range)
    if column.values is not None:
        rendered["values"] = dict(column.values)
    return rendered


def _layer(spec: ProductSpecification, layer: LayerSpecification) -> dict[str, typing.Any]:
    rendered: dict[str, typing.Any] = {
        "name": layer.name,
        "geometry_type": layer.geometry_type,
        "columns": [_column(column) for column in spec.table(layer.table).columns],
    }
    if layer.requires_input is not None:
        rendered["requires_input"] = layer.requires_input
    return rendered


def _output(spec: ProductSpecification, key: str) -> dict[str, typing.Any]:
    artifact = spec.artifact(key)
    columns = [] if artifact.table is None else [_column(column) for column in spec.table(artifact.table).columns]
    rendered: dict[str, typing.Any] = {"file": artifact.filename, "columns": columns}
    if artifact.requires_input is not None:
        rendered["requires_input"] = artifact.requires_input
    layers = {layer.key: _layer(spec, layer) for layer in spec.layers.values() if layer.artifact == key}
    if layers:
        rendered["layers"] = layers
    return rendered
