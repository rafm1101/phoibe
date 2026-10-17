"""Post-condition of publishing: what was written must match the specification (D-R2, DA4).

Pure functions. They compare the specification with `WrittenArtifacts`, which the writer fills by reading its own
files back. The same functions serve the run-time check and the drift test D-A2.
"""

from __future__ import annotations

from collections.abc import Collection

from .errors import ContractViolation
from .ports import WrittenArtifacts
from .specification import ProductSpecification
from .tables import GEOMETRY

_MANIFEST_ARTIFACT = "manifest"


def tables_missing_for(
    spec: ProductSpecification, available: Collection[str], profile: str, present_inputs: Collection[str]
) -> list[str]:
    """Tables a profile needs but the result does not have. Checked before anything is written."""
    needed: list[str] = []
    for artifact in spec.artifacts_for(profile, present_inputs):
        if artifact.table is not None:
            needed.append(artifact.table)
        needed += [layer.table for layer in spec.layers_for(artifact.key, present_inputs)]
    return sorted({name for name in needed if name not in available})


def written_artifact_problems(
    spec: ProductSpecification, written: WrittenArtifacts, present_inputs: Collection[str]
) -> list[str]:
    """Deviations of the written files from the specification. Empty if they match."""
    problems: list[str] = []
    expected = [a for a in spec.artifacts_for(written.profile, present_inputs) if a.key != _MANIFEST_ARTIFACT]

    expected_keys = {a.key for a in expected}
    for key in sorted(expected_keys - written.files.keys()):
        problems.append(f"artifact {key!r} was not written.")
    for key in sorted(written.files.keys() - expected_keys):
        problems.append(f"artifact {key!r} was written but is not expected for profile {written.profile!r}.")

    expected_layer_keys: set[str] = set()
    for artifact in expected:
        if artifact.key not in written.files:
            continue
        if written.files[artifact.key] != artifact.filename:
            actual = written.files[artifact.key]
            problems.append(f"artifact {artifact.key!r}: file {actual!r}, expected {artifact.filename!r}.")
        if artifact.table is not None:
            problems += _table_problems(spec, artifact.key, artifact.table, written)
        for layer in spec.layers_for(artifact.key, present_inputs):
            expected_layer_keys.add(layer.key)
            problems += _layer_problems(spec, layer.key, written)

    for key in sorted(written.layer_columns.keys() - expected_layer_keys):
        problems.append(f"layer {key!r} was written but is not expected.")
    return problems


def check_written_artifacts(
    spec: ProductSpecification, written: WrittenArtifacts, present_inputs: Collection[str]
) -> None:
    """Raise `ContractViolation` if the written files deviate from the specification.

    Raises
    ------
    ContractViolation
        Listing all deviations. Nothing may be published then, in particular no manifest.
    """
    problems = written_artifact_problems(spec, written, present_inputs)
    if problems:
        raise ContractViolation(problems)


def _table_problems(spec: ProductSpecification, key: str, table: str, written: WrittenArtifacts) -> list[str]:
    expected = spec.table(table).column_names
    actual = written.table_columns.get(key)
    if actual is None:
        return [f"artifact {key!r}: columns were not reported."]
    if tuple(actual) != expected:
        return [f"artifact {key!r}: columns {list(actual)}, expected {list(expected)}."]
    if key not in written.table_records:
        return [f"artifact {key!r}: number of rows was not reported."]
    return []


def _layer_problems(spec: ProductSpecification, key: str, written: WrittenArtifacts) -> list[str]:
    layer = spec.layer(key)
    if key not in written.layer_columns:
        return [f"layer {key!r} was not written."]
    problems = []
    columns = spec.table(layer.table).column_names
    expected = tuple(name for name in columns if layer.geometry_type is None or name != GEOMETRY)
    if tuple(written.layer_columns[key]) != expected:
        problems.append(f"layer {key!r}: columns {list(written.layer_columns[key])}, expected {list(expected)}.")
    if written.layer_geometry_types.get(key) != layer.geometry_type:
        problems.append(
            f"layer {key!r}: geometry type {written.layer_geometry_types.get(key)!r}, expected {layer.geometry_type!r}."
        )
    if key not in written.layer_records:
        problems.append(f"layer {key!r}: number of records was not reported.")
    return problems
