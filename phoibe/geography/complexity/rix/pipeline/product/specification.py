from __future__ import annotations

import dataclasses
import re
import types
import typing
from collections.abc import Collection, Iterator, Mapping, Sequence

Severity = typing.Literal["abort", "warn"]

_SEVERITIES = ("abort", "warn")
_DTYPES = ("int", "float", "str", "bool")
_GEOMETRY_TYPES = ("Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon")
_VERSION_PATTERN = re.compile(r"\d+(\.\d+)*")
_MANIFEST_ARTIFACT = "manifest"
"""Key of the artifact that documents a run. Convention: it is part of every profile."""
_TYPE_NAMES = {int: "int", float: "float", str: "str", bool: "bool"}


class SpecificationError(ValueError):
    """The product definition is invalid. `problems` lists every finding with its path."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        super().__init__("Invalid product specification:\n" + "\n".join(f"  - {problem}" for problem in problems))


@dataclasses.dataclass(frozen=True)
class ParameterSpecification:
    """A parameter of the assessment w/ `name` and given `group`."""

    group: str
    name: str
    description: str
    value: typing.Any
    """Fixed value if `locked`, otherwise the default. None means: to be set by the user."""
    dtype: str
    unit: str | None = None
    valid_range: tuple[float, float] | None = None
    source: str | None = None
    """Key into the references."""
    locked: bool = False
    """Locked parameters define conformity to the source. Deviating from them has to be made visible."""


@dataclasses.dataclass(frozen=True)
class InputSpecification:
    """An input of the assessment. Conditions are documentation, not yet enforced."""

    name: str
    description: str
    required: bool
    type: str | None = None
    notes: tuple[str, ...] = ()
    conditions: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class CheckSpecification:
    """A check whose findings are reported and, depending on the severity, stop the run."""

    name: str
    description: str
    severity: Severity
    parameters: Mapping[str, typing.Any] = dataclasses.field(default_factory=lambda: types.MappingProxyType({}))


@dataclasses.dataclass(frozen=True)
class ColumnSpecification:
    """A column of a table."""

    name: str
    description: str
    unit: str | None = None
    dtype: str | None = None
    valid_range: tuple[float, float] | None = None
    values: Mapping[typing.Any, str] | None = None
    """Meaning of coded values."""


@dataclasses.dataclass(frozen=True)
class TableSpecification:
    """A table, defined once and used by every artifact or layer that carries it."""

    name: str
    description: str
    columns: tuple[ColumnSpecification, ...]

    @property
    def column_names(self) -> tuple[str, ...]:
        """Column names in their defined order."""
        return tuple(column.name for column in self.columns)

    def column(self, name: str) -> ColumnSpecification:
        """Column by name. Raises KeyError listing the available names."""
        return _lookup({column.name: column for column in self.columns}, name, f"column of table '{self.name}'")


@dataclasses.dataclass(frozen=True)
class ArtifactSpecification:
    """An output file."""

    key: str
    filename: str
    table: str | None = None
    """Table written to this file, if any."""
    requires_input: str | None = None
    """The artifact is only produced if this input is present."""


@dataclasses.dataclass(frozen=True)
class LayerSpecification:
    """A layer inside a container artifact, e.g. a GeoPackage."""

    key: str
    artifact: str
    name: str
    """Layer name inside the file."""
    table: str
    geometry_type: str | None = None
    """None for attribute-only layers."""
    requires_input: str | None = None


@dataclasses.dataclass(frozen=True)
class ManifestKeySpecification:
    """A key of the run manifest, with nested keys."""

    name: str
    description: str
    keys: tuple[ManifestKeySpecification, ...] = ()
    values: Mapping[str, str] | None = None
    """Meaning of enumerated values."""
    free_form: bool = False
    """Arbitrary sub-keys are allowed, e.g. provenance supplied by the user."""

    def paths(self, *, leaves_only: bool = False, _prefix: str = "") -> Iterator[str]:
        """Dotted paths of all nested keys, e.g. ``run.diagnostics.n_sites_with_nans``."""
        for child in self.keys:
            path = f"{_prefix}{child.name}"
            if not child.keys or not leaves_only:
                yield path
            yield from child.paths(leaves_only=leaves_only, _prefix=f"{path}.")


@dataclasses.dataclass(frozen=True)
class ProductSpecification:
    """Complete specification of the data product, from inputs to outputs.

    Build it with `from_dict`, which validates. Look things up by id; an unknown id raises a KeyError
    that lists the available ones.
    """

    name: str
    description: str
    version: str
    references: Mapping[str, Mapping[str, str]]
    parameters: tuple[ParameterSpecification, ...]
    inputs: Mapping[str, InputSpecification]
    checks: Mapping[str, CheckSpecification]
    tables: Mapping[str, TableSpecification]
    artifacts: Mapping[str, ArtifactSpecification]
    layers: Mapping[str, LayerSpecification]
    profiles: Mapping[str, tuple[str, ...]]
    """Artifact keys per output profile."""
    manifest: ManifestKeySpecification

    @classmethod
    def from_dict(cls, raw: Mapping[str, typing.Any]) -> ProductSpecification:
        """Parse and validate a product definition.

        Raises
        ------
        SpecificationError
            Listing all problems found, each with its path.
        """
        problems = _Problems()
        specification = _parse_product(raw, problems)
        if specification is None or problems.items:
            raise SpecificationError(problems.items)
        return specification

    def __repr__(self):
        return " ".join([self.name, self.version, "\n", self.description])

    def parameter(self, group: str, name: str) -> ParameterSpecification:
        """Parameter by group and name."""
        return _lookup({(param.group, param.name): param for param in self.parameters}, (group, name), "parameter")

    def table(self, name: str) -> TableSpecification:
        """Table by name."""
        return _lookup(self.tables, name, "table")

    def artifact(self, key: str) -> ArtifactSpecification:
        """Artifact by key."""
        return _lookup(self.artifacts, key, "artifact")

    def layer(self, key: str) -> LayerSpecification:
        """Layer by key."""
        return _lookup(self.layers, key, "layer")

    def check(self, name: str) -> CheckSpecification:
        """Check by name."""
        return _lookup(self.checks, name, "check")

    def filename(self, key: str) -> str:
        """Filename of an artifact."""
        return self.artifact(key).filename

    def artifacts_for(self, profile: str, present_inputs: Collection[str]) -> tuple[ArtifactSpecification, ...]:
        """Artifacts a run must produce for a profile, given which inputs were supplied."""
        selected = (self.artifacts[key] for key in _lookup(self.profiles, profile, "profile"))
        return tuple(a for a in selected if a.requires_input is None or a.requires_input in present_inputs)

    def layers_for(self, artifact: str, present_inputs: Collection[str]) -> tuple[LayerSpecification, ...]:
        """Layers a container artifact must hold, given which inputs were supplied."""
        self.artifact(artifact)
        return tuple(
            layer
            for layer in self.layers.values()
            if layer.artifact == artifact and (layer.requires_input is None or layer.requires_input in present_inputs)
        )


def _lookup(mapping: Mapping, key: typing.Any, kind: str):
    try:
        return mapping[key]
    except KeyError:
        available = ", ".join(str(k) for k in mapping)
        raise KeyError(f"Unknown {kind} {key!r}. Available: {available}") from None


###########
# Parsing
###########


class _Problems:
    def __init__(self) -> None:
        self.items: list[str] = []

    def add(self, path: str, message: str) -> None:
        self.items.append(f"{path}: {message}")


def _frozen(mapping: Mapping) -> Mapping:
    return types.MappingProxyType(dict(mapping))


def _as_mapping(raw: typing.Any, path: str, problems: _Problems) -> Mapping:
    if isinstance(raw, Mapping):
        return raw
    problems.add(path, f"expected a mapping, got {type(raw).__name__}")
    return {}


def _check_keys(raw: Mapping, allowed: Collection[str], path: str, problems: _Problems) -> None:
    for key in raw:
        if key not in allowed:
            problems.add(path, f"unknown key {key!r} (allowed: {', '.join(sorted(allowed))})")


def _text(raw: Mapping, key: str, path: str, problems: _Problems, *, required: bool = True) -> str | None:
    value = raw.get(key)
    if value is None:
        if required:
            problems.add(path, f"missing {key!r}")
        return None
    if not isinstance(value, str) or not value.strip():
        problems.add(f"{path}.{key}", "expected non-empty text")
        return None
    return value


def _texts(raw: Mapping, key: str, path: str, problems: _Problems) -> tuple[str, ...]:
    value = raw.get(key, ())
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        problems.add(f"{path}.{key}", "expected a list of text")
        return ()
    return tuple(value)


def _flag(raw: Mapping, key: str, path: str, problems: _Problems, *, default: bool | None) -> bool:
    if key not in raw:
        if default is None:
            problems.add(path, f"missing {key!r}")
            return False
        return default
    value = raw[key]
    if not isinstance(value, bool):
        problems.add(f"{path}.{key}", "expected true or false")
        return bool(default)
    return value


def _is_number(value: typing.Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _matches_dtype(value: typing.Any, dtype: str) -> bool:
    if dtype == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if dtype == "float":
        return _is_number(value)
    if dtype == "str":
        return isinstance(value, str)
    return isinstance(value, bool)


def _range(raw: Mapping, path: str, problems: _Problems) -> tuple[float, float] | None:
    value = raw.get("range")
    if value is None:
        return None
    if not (isinstance(value, (list, tuple)) and len(value) == 2 and all(_is_number(v) for v in value)):
        problems.add(f"{path}.range", "expected [lower, upper]")
        return None
    lower, upper = value
    if not lower < upper:
        problems.add(f"{path}.range", f"lower bound must be smaller than upper bound, got {value}")
        return None
    return (lower, upper)


def _optional_unit(raw: Mapping, path: str, problems: _Problems) -> str | None:
    return _text(raw, "unit", path, problems, required=False)


def _parse_parameters(raw: typing.Any, problems: _Problems) -> tuple[ParameterSpecification, ...]:
    result = []
    for group, entries in _as_mapping(raw, "parameters", problems).items():
        for name, entry in _as_mapping(entries, f"parameters.{group}", problems).items():
            result.append(_parse_parameter(group, name, entry, f"parameters.{group}.{name}", problems))
    return tuple(result)


def _parse_parameter(group: str, name: str, raw: typing.Any, path: str, problems: _Problems) -> ParameterSpecification:
    raw = _as_mapping(raw, path, problems)
    _check_keys(raw, {"description", "unit", "value", "dtype", "range", "source", "locked"}, path, problems)
    description = _text(raw, "description", path, problems) or ""
    value = raw.get("value")
    dtype = raw.get("dtype")

    if dtype is None:
        if value is None:
            problems.add(path, "'dtype' is required when 'value' is None")
        else:
            dtype = _TYPE_NAMES.get(type(value))
            if dtype is None:
                problems.add(path, f"cannot infer dtype from {type(value).__name__}")
    elif dtype not in _DTYPES:
        problems.add(f"{path}.dtype", f"must be one of {_DTYPES}, got {dtype!r}")
    elif value is not None and not _matches_dtype(value, dtype):
        problems.add(f"{path}.value", f"{value!r} does not match dtype {dtype!r}")

    valid_range = _range(raw, path, problems)
    if valid_range is not None and _is_number(value) and not valid_range[0] <= value <= valid_range[1]:
        problems.add(f"{path}.value", f"{value} outside range {list(valid_range)}")

    locked = _flag(raw, "locked", path, problems, default=False)
    source = _text(raw, "source", path, problems, required=False)
    if locked:
        if value is None:
            problems.add(path, "a locked parameter needs a value")
        if source is None:
            problems.add(path, "a locked parameter needs a 'source'")

    return ParameterSpecification(
        group=group,
        name=name,
        description=description,
        value=value,
        dtype=dtype or "",
        unit=_optional_unit(raw, path, problems),
        valid_range=valid_range,
        source=source,
        locked=locked,
    )


def _parse_inputs(raw: typing.Any, problems: _Problems) -> dict[str, InputSpecification]:
    result = {}
    for name, entry in _as_mapping(raw, "inputs", problems).items():
        path = f"inputs.{name}"
        entry = _as_mapping(entry, path, problems)
        _check_keys(entry, {"description", "required", "type", "notes", "conditions"}, path, problems)
        result[name] = InputSpecification(
            name=name,
            description=_text(entry, "description", path, problems) or "",
            required=_flag(entry, "required", path, problems, default=None),
            type=_text(entry, "type", path, problems, required=False),
            notes=_texts(entry, "notes", path, problems),
            conditions=_texts(entry, "conditions", path, problems),
        )
    return result


def _parse_checks(raw: typing.Any, problems: _Problems) -> dict[str, CheckSpecification]:
    result = {}
    for name, entry in _as_mapping(raw, "checks", problems).items():
        path = f"checks.{name}"
        entry = _as_mapping(entry, path, problems)
        _check_keys(entry, {"description", "severity", "parameters"}, path, problems)
        severity = entry.get("severity")
        if severity not in _SEVERITIES:
            problems.add(f"{path}.severity", f"must be one of {_SEVERITIES}, got {severity!r}")
        result[name] = CheckSpecification(
            name=name,
            description=_text(entry, "description", path, problems) or "",
            severity=typing.cast(Severity, severity),
            parameters=_frozen(_as_mapping(entry.get("parameters", {}), f"{path}.parameters", problems)),
        )
    return result


def _parse_column(key: str, raw: typing.Any, path: str, problems: _Problems) -> ColumnSpecification:
    raw = _as_mapping(raw, path, problems)
    _check_keys(raw, {"name", "description", "unit", "dtype", "range", "values"}, path, problems)
    if raw.get("name", key) != key:
        problems.add(path, f"'name' {raw['name']!r} differs from its key {key!r}")
    dtype = raw.get("dtype")
    if dtype is not None and dtype not in _DTYPES:
        problems.add(f"{path}.dtype", f"must be one of {_DTYPES}, got {dtype!r}")
    values = raw.get("values")
    if values is not None:
        values = _frozen(_as_mapping(values, f"{path}.values", problems))
    return ColumnSpecification(
        name=key,
        description=_text(raw, "description", path, problems) or "",
        unit=_optional_unit(raw, path, problems),
        dtype=dtype,
        valid_range=_range(raw, path, problems),
        values=values,
    )


def _parse_tables(raw: typing.Any, problems: _Problems) -> dict[str, TableSpecification]:
    result = {}
    for name, entry in _as_mapping(raw, "tables", problems).items():
        path = f"tables.{name}"
        entry = _as_mapping(entry, path, problems)
        _check_keys(entry, {"description", "columns"}, path, problems)
        columns_raw = _as_mapping(entry.get("columns"), f"{path}.columns", problems)
        if not columns_raw:
            problems.add(path, "needs at least one column")
        columns = tuple(_parse_column(key, col, f"{path}.columns.{key}", problems) for key, col in columns_raw.items())
        result[name] = TableSpecification(
            name=name, description=_text(entry, "description", path, problems) or "", columns=columns
        )
    return result


def _parse_artifacts(
    raw: typing.Any, problems: _Problems
) -> tuple[dict[str, ArtifactSpecification], dict[str, LayerSpecification], dict[str, tuple[str, ...]]]:
    raw = _as_mapping(raw, "artifacts", problems)
    _check_keys(raw, {"files", "profiles", "layers"}, "artifacts", problems)

    files = {}
    for key, entry in _as_mapping(raw.get("files"), "artifacts.files", problems).items():
        path = f"artifacts.files.{key}"
        entry = _as_mapping(entry, path, problems)
        _check_keys(entry, {"filename", "table", "requires_input"}, path, problems)
        files[key] = ArtifactSpecification(
            key=key,
            filename=_text(entry, "filename", path, problems) or "",
            table=_text(entry, "table", path, problems, required=False),
            requires_input=_text(entry, "requires_input", path, problems, required=False),
        )

    layers = {}
    for key, entry in _as_mapping(raw.get("layers", {}), "artifacts.layers", problems).items():
        path = f"artifacts.layers.{key}"
        entry = _as_mapping(entry, path, problems)
        _check_keys(entry, {"artifact", "name", "table", "geometry_type", "requires_input"}, path, problems)
        geometry_type = entry.get("geometry_type")
        if geometry_type is not None and geometry_type not in _GEOMETRY_TYPES:
            problems.add(f"{path}.geometry_type", f"must be None or one of {_GEOMETRY_TYPES}, got {geometry_type!r}")
        layers[key] = LayerSpecification(
            key=key,
            artifact=_text(entry, "artifact", path, problems) or "",
            name=_text(entry, "name", path, problems) or "",
            table=_text(entry, "table", path, problems) or "",
            geometry_type=geometry_type,
            requires_input=_text(entry, "requires_input", path, problems, required=False),
        )

    profiles = {}
    for name, keys in _as_mapping(raw.get("profiles"), "artifacts.profiles", problems).items():
        if not isinstance(keys, (list, tuple)) or not all(isinstance(k, str) for k in keys) or not keys:
            problems.add(f"artifacts.profiles.{name}", "expected a non-empty list of artifact keys")
            continue
        profiles[name] = tuple(keys)

    return files, layers, profiles


def _parse_manifest_key(key: str, raw: typing.Any, path: str, problems: _Problems) -> ManifestKeySpecification:
    raw = _as_mapping(raw, path, problems)
    _check_keys(raw, {"name", "description", "keys", "values", "free_form"}, path, problems)
    if raw.get("name", key) != key:
        problems.add(path, f"'name' {raw['name']!r} differs from its key {key!r}")
    children = tuple(
        _parse_manifest_key(child_key, child, f"{path}.keys.{child_key}", problems)
        for child_key, child in _as_mapping(raw.get("keys", {}), f"{path}.keys", problems).items()
    )
    free_form = _flag(raw, "free_form", path, problems, default=False)
    if free_form and children:
        problems.add(path, "a free-form key cannot declare nested keys")
    values = raw.get("values")
    if values is not None:
        values = _frozen(_as_mapping(values, f"{path}.values", problems))
    return ManifestKeySpecification(
        name=key,
        description=_text(raw, "description", path, problems) or "",
        keys=children,
        values=values,
        free_form=free_form,
    )


def _parse_references(raw: typing.Any, problems: _Problems) -> Mapping[str, Mapping[str, str]]:
    result = {}
    for key, entry in _as_mapping(raw, "references", problems).items():
        entry = _as_mapping(entry, f"references.{key}", problems)
        result[key] = _frozen({field: str(value) for field, value in entry.items()})
    return _frozen(result)


def _parse_product(raw: Mapping[str, typing.Any], problems: _Problems) -> ProductSpecification | None:
    raw = _as_mapping(raw, "definition", problems)
    _check_keys(
        raw,
        {
            "name",
            "description",
            "version",
            "references",
            "parameters",
            "inputs",
            "checks",
            "tables",
            "artifacts",
            "manifest",
        },
        "definition",
        problems,
    )

    name = _text(raw, "name", "definition", problems) or ""
    description = _text(raw, "description", "definition", problems) or ""
    version = _text(raw, "version", "definition", problems) or ""
    if version and not _VERSION_PATTERN.fullmatch(version):
        problems.add("definition.version", f"expected dotted numbers like '1.1', got {version!r}")

    references = _parse_references(raw.get("references", {}), problems)
    parameters = _parse_parameters(raw.get("parameters"), problems)
    inputs = _parse_inputs(raw.get("inputs"), problems)
    checks = _parse_checks(raw.get("checks", {}), problems)
    tables = _parse_tables(raw.get("tables"), problems)
    files, layers, profiles = _parse_artifacts(raw.get("artifacts"), problems)
    if "manifest" not in raw:
        problems.add("definition", "missing 'manifest'")
    manifest = _parse_manifest_key("manifest", raw.get("manifest", {}), "manifest", problems)

    _cross_check(references, parameters, inputs, tables, files, layers, profiles, problems)

    if problems.items:
        return None
    return ProductSpecification(
        name=name,
        description=description,
        version=version,
        references=references,
        parameters=parameters,
        inputs=_frozen(inputs),
        checks=_frozen(checks),
        tables=_frozen(tables),
        artifacts=_frozen(files),
        layers=_frozen(layers),
        profiles=_frozen(profiles),
        manifest=manifest,
    )


def _cross_check(
    references: Mapping,
    parameters: Sequence[ParameterSpecification],
    inputs: Mapping[str, InputSpecification],
    tables: Mapping[str, TableSpecification],
    files: Mapping[str, ArtifactSpecification],
    layers: Mapping[str, LayerSpecification],
    profiles: Mapping[str, tuple[str, ...]],
    problems: _Problems,
) -> None:
    """References between sections: every id used must be defined."""
    for parameter in parameters:
        if parameter.source is not None and parameter.source not in references:
            problems.add(
                f"parameters.{parameter.group}.{parameter.name}.source", f"unknown reference {parameter.source!r}"
            )

    seen_filenames: dict[str, str] = {}
    for key, artifact in files.items():
        path = f"artifacts.files.{key}"
        if artifact.table is not None and artifact.table not in tables:
            problems.add(f"{path}.table", f"unknown table {artifact.table!r}")
        if artifact.requires_input is not None and artifact.requires_input not in inputs:
            problems.add(f"{path}.requires_input", f"unknown input {artifact.requires_input!r}")
        if artifact.filename in seen_filenames:
            problems.add(
                f"{path}.filename", f"{artifact.filename!r} is already used by {seen_filenames[artifact.filename]!r}"
            )
        seen_filenames[artifact.filename] = key

    seen_layers: set[tuple[str, str]] = set()
    for key, layer in layers.items():
        path = f"artifacts.layers.{key}"
        if layer.artifact not in files:
            problems.add(f"{path}.artifact", f"unknown artifact {layer.artifact!r}")
        if layer.table not in tables:
            problems.add(f"{path}.table", f"unknown table {layer.table!r}")
        if layer.requires_input is not None and layer.requires_input not in inputs:
            problems.add(f"{path}.requires_input", f"unknown input {layer.requires_input!r}")
        if (layer.artifact, layer.name) in seen_layers:
            problems.add(f"{path}.name", f"layer name {layer.name!r} is used twice in artifact {layer.artifact!r}")
        seen_layers.add((layer.artifact, layer.name))

    if files and _MANIFEST_ARTIFACT not in files:
        problems.add("artifacts.files", f"missing the {_MANIFEST_ARTIFACT!r} artifact")
    for profile, keys in profiles.items():
        for key in keys:
            if key not in files:
                problems.add(f"artifacts.profiles.{profile}", f"unknown artifact {key!r}")
        if _MANIFEST_ARTIFACT in files and _MANIFEST_ARTIFACT not in keys:
            problems.add(f"artifacts.profiles.{profile}", f"every profile must contain {_MANIFEST_ARTIFACT!r}")
