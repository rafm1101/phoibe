import dataclasses
import pathlib

import yaml

from ..definitions.config import ANALYZER_DEFAULTS, AnalyzerConfig, RunParameters


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
        parameters=RunParameters(**merged_parameters),
        sampler=ANALYZER_DEFAULTS.sampler,
    )
    _validate_config(config)
    return config


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
