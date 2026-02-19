from __future__ import annotations

import dataclasses
import pathlib
import typing

import yaml

from phoibe.layered.logging.logging import get_logger

logger = get_logger(__name__)


@dataclasses.dataclass
class ValidationConfig:
    """Value object of a layer validation configuration.

    The configuration defines: Layer identification, variable detection patterns and active rules with parameters.

    Example
    -------
    1. Definition of a set and configuration of rules: [{'name': 'rule_name', 'params': {...}}]
       > [
       >     {'name': 'temporal_resolution', 'params': {'expected_minutes': 10, 'points': 10}},
       >     {'name': 'data_gaps', 'params': {'threshold': 0.05, 'points': 10}}
       > ]
    """

    layer_name: str
    """Unique name of the validation layer ('raw', 'bronze', 'silver', 'gold')."""
    variable_patterns: dict[str, list[str]]
    """Mapping of variable names to detection patterns: {variable_name: [regex_patterns]}."""
    rules: list[dict[str, typing.Any]] = dataclasses.field(default_factory=list)
    """Rule names as passed to the registry and their respective configurations."""

    @classmethod
    def from_yaml(cls, config_path: str | pathlib.Path) -> ValidationConfig:
        """Load configuration from YAML file.

        Parameters
        ----------
        config_path
            Path to YAML configuration file.

        Returns
        -------
        ValidationConfig
            Loaded validation configuration.

        Raises
        ------
        FileNotFoundError
            If config file does not exist.
        ValueError
            If config format is invalid.
        """
        config_path = pathlib.Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logger.debug(f"Loading configuration from: {config_path}.")
        with open(config_path) as filestream:
            data = yaml.safe_load(filestream)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid config format in {config_path}: expected dict, got {type(data)}")

        layer_name = data.get("layer_name")
        if not layer_name:
            raise ValueError(f"Missing 'layer_name' in config: {config_path}")

        variable_patterns = data.get("variable_patterns", {})
        if not isinstance(variable_patterns, dict):
            raise ValueError(f"'variable_patterns' must be dict, got {type(variable_patterns)}")

        rules = data.get("rules", [])
        if not isinstance(rules, list):
            raise ValueError(f"'rules' must be list, got {type(rules)}")

        logger.info(f"Loaded config for layer '{layer_name}': " f"{len(variable_patterns)} signals, {len(rules)} rules")

        return cls(layer_name=layer_name, variable_patterns=variable_patterns, rules=rules)


__all__ = ["ValidationConfig"]
