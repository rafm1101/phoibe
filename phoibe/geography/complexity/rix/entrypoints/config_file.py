"""Inbound adapter: read the configuration of a run from a YAML file or a mapping.

Shape::

    project_name: demo            # optional label of the run
    parameters:                   # optional, flat: parameter name -> value
      dr_km: 0.1
      crs: EPSG:2056

Every parameter that is not given takes the default of the product specification. Which parameters exist, their types
and their ranges also come from the specification, so nothing is defined twice (DV1).

- Parameters the product fixes and the caller cannot set (e.g. `decimals`) are rejected.
- Locked parameters that can be set (e.g. `n_angles`) are accepted. A deviation is not an error here; the run reports it
  as a finding and the manifest records it.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import typing
import warnings
from collections.abc import Mapping

import yaml

from .. import pipeline
from ..definitions.config import ANALYZER_DEFAULTS, AnalyzerConfig
from ..pipeline.errors import ConfigurationError
from ..pipeline.product.specification import ParameterSpecification, ProductSpecification

_TOP_LEVEL_KEYS = ("project_name", "parameters")


@dataclasses.dataclass(frozen=True)
class RunConfiguration:
    """Everything a caller configures for one run."""

    parameters: pipeline.RunParametersRunParameters
    sampler_parameters: pipeline.SampleParameters
    project_name: str = ""


def default_configuration(spec: ProductSpecification) -> RunConfiguration:
    """The configuration made of the defaults of the specification only."""
    return load_configuration({}, spec)


def load_configuration(
    source: str | os.PathLike[str] | Mapping[str, typing.Any], spec: ProductSpecification
) -> RunConfiguration:
    """Read a configuration and complete it with the defaults of the specification.

    Parameters
    ----------
    source
        Path of a YAML file, or an already parsed mapping.
    spec
        Product specification: defines the parameters, their defaults, types and ranges.

    Raises
    ------
    ConfigurationError
        If the file cannot be read, or contains unknown or fixed parameters, wrong types or values out of range.
    """
    raw = _read(source)
    unknown_keys = sorted(set(raw) - set(_TOP_LEVEL_KEYS))
    if unknown_keys:
        raise ConfigurationError(f"Unknown key(s) {unknown_keys}; allowed: {list(_TOP_LEVEL_KEYS)}.")

    project_name = raw.get("project_name", "")
    if not isinstance(project_name, str):
        raise ConfigurationError("project_name must be text.")

    supplied = raw.get("parameters") or {}
    if not isinstance(supplied, Mapping):
        raise ConfigurationError("parameters must be a mapping of parameter name to value.")

    by_name = {parameter.name: parameter for parameter in spec.parameters}
    settable = [
        field.name for cls in (pipeline.RunParameters, pipeline.SampleParameters) for field in dataclasses.fields(cls)
    ]

    problems = []
    for name in supplied:
        if name not in by_name:
            problems.append(f"unknown parameter {name!r}")
        elif name not in settable:
            problems.append(f"parameter {name!r} is fixed by the product and cannot be configured")
    if problems:
        raise ConfigurationError(f"{'; '.join(problems)}. Configurable: {', '.join(settable)}.")

    values = {
        name: _checked(by_name[name], supplied[name] if name in supplied else by_name[name].value) for name in settable
    }
    try:
        parameters = pipeline.RunParameters(
            **{field.name: values[field.name] for field in dataclasses.fields(pipeline.RunParameters)}
        )
        sampler = pipeline.SampleParameters(
            **{field.name: values[field.name] for field in dataclasses.fields(pipeline.SampleParameters)}
        )
    except ValueError as error:
        raise ConfigurationError(str(error)) from error
    return RunConfiguration(parameters=parameters, sampler_parameters=sampler, project_name=project_name)


def _read(source: str | os.PathLike[str] | Mapping[str, typing.Any]) -> Mapping[str, typing.Any]:
    if isinstance(source, Mapping):
        return source
    path = pathlib.Path(source)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"Could not read the configuration {path}: {error}") from error
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigurationError(f"The configuration {path} must be a mapping.")
    return raw


def _checked(parameter: ParameterSpecification, value: typing.Any) -> typing.Any:
    """The value if it matches dtype and range of the parameter. None stands for 'not set' where the default is None."""
    if value is None:
        if parameter.value is None:
            return None
        raise ConfigurationError(f"Parameter {parameter.name!r} must not be empty.")
    accepted_types: dict[str, type | tuple[type, ...]] = {"int": int, "float": (int, float), "str": str, "bool": bool}
    accepted = accepted_types[parameter.dtype]
    if isinstance(value, bool) != (parameter.dtype == "bool") or not isinstance(value, accepted):
        raise ConfigurationError(f"Parameter {parameter.name!r} must be of type {parameter.dtype}, got {value!r}.")
    if parameter.valid_range is not None and isinstance(value, (int, float)):
        lower, upper = parameter.valid_range
        if not lower <= value <= upper:
            raise ConfigurationError(f"Parameter {parameter.name!r} is {value}, allowed range is {[lower, upper]}.")
    return value


@warnings.deprecated("`config_file.load_config` possibly outdated and will be removed at some point.")
def load_config(path: str | pathlib.Path) -> AnalyzerConfig:
    """Load and validate analyzer configuration from a YAML file.

    Parameters
    ----------
    path
        Path to ``config.yaml``.
    keys
        Key holding parameter key.

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
    default_parameters = dataclasses.asdict(ANALYZER_DEFAULTS.parameters)

    for key in raw_parameters:
        if key not in default_parameters.keys():
            raise ValueError(f"Configuration file {str(path)} contains unknown parameter {key}.")

    merged_parameters = {**default_parameters, **raw_parameters}

    config = AnalyzerConfig(
        name=ANALYZER_DEFAULTS.name,
        version=ANALYZER_DEFAULTS.version,
        description=ANALYZER_DEFAULTS.description,
        parameters=pipeline.RunParameters(**merged_parameters),
        sampler=ANALYZER_DEFAULTS.sampler,
    )
    _validate_config(config)
    return config


@warnings.deprecated("`config_file._validate_config` possibly outdated and will be removed at some point.")
def _validate_config(config: AnalyzerConfig) -> None:
    parameters = config.parameters
    # missing_keys = _REQUIRED_KEYS - parameters.keys()
    # if missing_keys:
    #     raise ValueError(f"config.yaml is missing required keys: {missing_keys}")
    if parameters.dr_km >= parameters.R_km:
        raise ValueError("dr_km must be smaller than R_km.")
    if not (1 <= parameters.n_angles <= 360):
        raise ValueError(f"n_angles must be between 1 and 360, got {parameters.n_angles}.")
    if parameters.slope_critical <= 0:
        raise ValueError(f"slope_critical must be positive, got {parameters.slope_critical}.")
