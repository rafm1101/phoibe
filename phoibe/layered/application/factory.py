import warnings

import pandas as pd

from phoibe.layered.application.config import ValidationConfig
from phoibe.layered.application.registry import RuleRegistry
from phoibe.layered.application.validator import LayerValidator
from phoibe.layered.core.entities import ValidationMode
from phoibe.layered.core.interfaces import DataLoader
from phoibe.layered.core.interfaces import VariableDetector
from phoibe.layered.infrastructure.detector import RegexVariableDetector
from phoibe.layered.infrastructure.io import InMemoryDataLoader
from phoibe.layered.infrastructure.io import PandasDataLoader
from phoibe.layered.logging.logging import get_logger
from phoibe.layered.rules.rule import ValidationRule

logger = get_logger(__name__)


class ValidatorFactory:
    """Factory for creating layer validators.

    Provides convenient methods for creating validators with different
    data sources (files, memory) and configurations.
    """

    @classmethod
    def profiling(
        cls,
        config: ValidationConfig,
        *,
        data: pd.DataFrame | None = None,
        filename: str = "in_memory",
        **kwargs,
    ) -> LayerValidator:
        """Create validator in PROFILING mode.

        Data source:
            - If data=None: File-based (default, uses PandasDataLoader)
            - If data=DataFrame: Memory-based (uses InMemoryDataLoader)

        Parameters
        ----------
        config
            Validation configuration.
        data
            Optional DataFrame for in-memory validation.
        filename
            Filename for metadata when using in-memory data.
        **kwargs
            Optional data_loader, variable_detector (advanced).

        Return
        ------
        LayerValidator
            Validator in PROFILING mode.

        Example
        -------
        1. File-based profiling:

           > validator = ValidatorFactory.profiling(config)
           > report = validator.validate("data.csv", "WEA_01")

        2. Memory-based profiling:

           > df = pd.read_csv("data.csv")
           > validator = ValidatorFactory.profiling(config, data=df)
           > report = validator.validate("", "WEA_01")
        """
        if data is not None and "data_loader" not in kwargs:
            kwargs["data_loader"] = InMemoryDataLoader(data, filename=filename)

        return cls.create(config, mode=ValidationMode.PROFILING, **kwargs)

    @classmethod
    def contract(
        cls,
        config: ValidationConfig,
        *,
        data: pd.DataFrame | None = None,
        filename: str = "in_memory",
        **kwargs,
    ) -> LayerValidator:
        """Create validator in CONTRACT mode.

        Data source:
            - If data=None: File-based (default, uses PandasDataLoader)
            - If data=DataFrame: Memory-based (uses InMemoryDataLoader)

        Parameters
        ----------
        config
            Validation configuration.
        data
            Optional DataFrame for in-memory validation.
        filename
            Filename for metadata when using in-memory data.
        **kwargs
            Optional data_loader, variable_detector (advanced).

        Return
        ------
        LayerValidator
            Validator in CONTRACT mode.

        Example
        -------
        1. File-based contract validation:

           > validator = ValidatorFactory.contract(config)
           > try:
           >     report = validator.validate("data.csv", "WEA_01")
           > except GateFailureError as e:
           >     print(f"Gate failed: {e}")

        2. Memory-based contract validation:

           > df = pd.read_csv("data.csv")
           > validator = ValidatorFactory.contract(config, data=df)
           > report = validator.validate("", "WEA_01")
        """
        if data is not None and "data_loader" not in kwargs:
            kwargs["data_loader"] = InMemoryDataLoader(data, filename=filename)

        return cls.create(config, mode=ValidationMode.CONTRACT, **kwargs)

    @classmethod
    def create(
        cls,
        config: ValidationConfig,
        mode: ValidationMode = ValidationMode.PROFILING,
        *,
        data_loader: DataLoader | None = None,
        variable_detector: VariableDetector | None = None,
    ) -> LayerValidator:
        """Create validator with optional dependency injection.

        Advanced method for full control over dependencies.

        Parameters
        ----------
        config
            Validation configuration.
        mode
            PROFILING (default) or CONTRACT.
        data_loader
            Optional custom data loader (default: PandasDataLoader).
        variable_detector
            Optional custom detector (default: RegexVariableDetector).

        Return
        ------
        LayerValidator
            Configured validator.

        Example
        -------
        1. Custom data loader:

           > validator = ValidatorFactory.create(
           >     config,
           >     mode=ValidationMode.CONTRACT,
           >     data_loader=CustomDataLoader()
           > )
        """
        loader = data_loader or PandasDataLoader()
        detector = variable_detector or RegexVariableDetector(config.variable_patterns)

        rules = cls._create_rules(config.rules)

        logger.info(f"Created validator: layer={config.layer_name}, mode={mode.value}, rules={len(rules)}.")

        return LayerValidator(
            layer_name=config.layer_name,
            data_loader=loader,
            variable_detector=detector,
            rules=rules,
            mode=mode,
        )

    @classmethod
    def _create_rules(cls, rule_configs: list[dict]) -> list[ValidationRule]:
        """Instantiate rules from configurations.

        Parameters
        ----------
        rule_configs
            List of {name: str, params: dict}.

        Return
        ------
        list[ValidationRule]
            Instantiated rules.

        Raises
        ------
        KeyError
            If rule not in registry.
        TypeError
            If rule parameters invalid, e.g. missing required, unexpected, mistyped.
        """
        rules = []

        for config in rule_configs:
            rule_name = config["name"]
            rule_params = config.get("params", {})

            try:
                rule_class = RuleRegistry.get(rule_name)
                rule = rule_class(**rule_params)
                rules.append(rule)
                logger.debug(f"Created rule: {rule_name}")

            except KeyError as exception:
                logger.error(f"Failed to create unidentifiable rule '{rule_name}': {exception}")
                raise

            except TypeError as exception:
                logger.error(f"Failed to instantiate rule '{rule_name}': {exception}")
                raise TypeError(
                    f"Failed to instantiate rule '{rule_name}': {exception}. " f"Provided params: {rule_params}"
                ) from exception

        return rules

    @staticmethod
    @warnings.deprecated("`create_from_config` is deprecated. Please use `create`, `contract` or `profiling`.")
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
        logger.warning("`create_from_config` is deprecated. Please use `Validatorfactory.profiling(config)`.")
        return ValidatorFactory.profiling(config=config)

    @staticmethod
    @warnings.deprecated("`create_from_memory` is deprecated. Please use `create`, `contract` or `profiling`.")
    def create_from_memory(
        config: ValidationConfig, data: pd.DataFrame, filename: str = "in_memory_data"
    ) -> LayerValidator:
        """Create validator for in-memory data.

        Parameters
        ----------
        config
            Validation configuration
        data
            DataFrame to validate
        filename
            Virtual filename for reporting

        Return
        ------
            Configured LayerValidator with InMemoryDataLoader
        """
        logger.warning(
            "`create_from_memory` is deprecated. Please use `Validatorfactory.profiling(config, data=data)`."
        )

        return ValidatorFactory.profiling(config=config, data=data, filename=filename)


create_contract_validator = ValidatorFactory.contract
create_profiling_validator = ValidatorFactory.profiling

__all__ = ["ValidatorFactory", "create_contract_validator", "create_profiling_validator"]
