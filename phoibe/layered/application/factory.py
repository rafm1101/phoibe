import typing

import pandas as pd

from phoibe.layered.application.config import ValidationConfig
from phoibe.layered.application.validator import LayerValidator
from phoibe.layered.infrastructure.detector import RegexVariableDetector
from phoibe.layered.infrastructure.io import InMemoryDataLoader
from phoibe.layered.infrastructure.io import PandasDataLoader
from phoibe.layered.logging.logging import get_logger
from phoibe.layered.rules.rule import ValidationRule

logger = get_logger(__name__)


class RuleRegistry:
    """Central registry for validation rules.

    Rules self-register using the @RuleRegistry.register decorator.

    Example
    -------
    1. Register some rule:

       > @RuleRegistry.register("temporal_resolution")
       > class TemporalResolutionRule(ValidationRule):
       >     ...
    """

    _rules: dict[str, typing.Type[ValidationRule]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a validation rule."""

        def wrapper(rule_class: typing.Type[ValidationRule]):
            if name in cls._rules:
                logger.warning(f"Rule '{name}' already registered, overwriting.")
            cls._rules[name] = rule_class
            logger.debug(f"Registered rule: {name} -> {rule_class.__name__}")
            return rule_class

        return wrapper

    @classmethod
    def get(cls, name: str):  # -> Type[ValidationRule]:
        """Get rule class by name.

        Parameters
        ----------
        name
            Rule identifier.

        Return
        ------
        ValidationRule
            Rule class.

        Raises
        ------
        KeyError
            If rule not found.
        """
        if name not in cls._rules:
            available = ", ".join(sorted(cls._rules.keys()))
            raise KeyError(f"Rule '{name}' not found in registry. " f"Available rules: {available}.")
        return cls._rules[name]

    @classmethod
    def list_rules(cls) -> list[str]:
        """Get list of all registered rule names."""
        return sorted(cls._rules.keys())

    @classmethod
    def clear(cls):
        """Clear all registered rules (for testing)."""
        cls._rules.clear()


class ValidatorFactory:
    """Factory for creating layer validators.

    Provides convenient methods for creating validators with different
    data sources (files, memory) and configurations.
    """

    @staticmethod
    def create_from_config(config: ValidationConfig) -> LayerValidator:
        """Create validator from configuration.

        Parameters
        ----------
        config
            Validation configuration.

        Returns
        -------
        LayerValidator.
            Configured LayerValidator.
        """
        data_loader = PandasDataLoader()
        variable_detector = RegexVariableDetector(config.variable_patterns)

        rules = []
        for rule_config in config.rules:
            rule_name = rule_config["name"]
            rule_params = rule_config.get("params", {})

            try:
                rule_class = RuleRegistry.get(rule_name)
                rule = rule_class(**rule_params)
                rules.append(rule)
                logger.debug(f"Created rule: {rule_name}")
            except KeyError as e:
                logger.error(f"Failed to create rule '{rule_name}': {e}")
                raise

        logger.info(f"Created validator for layer '{config.layer_name}' with {len(rules)} rules.")

        return LayerValidator(
            layer_name=config.layer_name, data_loader=data_loader, variable_detector=variable_detector, rules=rules
        )

    @staticmethod
    def create_from_memory(
        config: ValidationConfig, data: pd.DataFrame, filename: str = "in_memory_data"
    ) -> LayerValidator:
        """Create validator for in-memory data.

        Parameters
        ----------
        config: Validation configuration
        data: DataFrame to validate
        filename: Virtual filename for reporting

        Returns
        -------
            Configured LayerValidator with InMemoryDataLoader
        """
        data_loader = InMemoryDataLoader(data, filename=filename)
        variable_detector = RegexVariableDetector(config.variable_patterns)

        rules = []
        for rule_config in config.rules:
            rule_name = rule_config["name"]
            rule_params = rule_config.get("params", {})

            try:
                rule_class = RuleRegistry.get(rule_name)
                rule = rule_class(**rule_params)
                rules.append(rule)
                logger.debug(f"Created rule: {rule_name}")
            except KeyError as e:
                logger.error(f"Failed to create rule '{rule_name}': {e}")
                raise

        logger.info(f"Created in-memory validator for layer '{config.layer_name}' with {len(rules)} rules.")

        return LayerValidator(
            layer_name=config.layer_name, data_loader=data_loader, variable_detector=variable_detector, rules=rules
        )


__all__ = ["RuleRegistry", "ValidatorFactory"]
