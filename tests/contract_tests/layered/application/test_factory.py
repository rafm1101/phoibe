from pathlib import Path

import pandas as pd
import pytest

from phoibe.layered.application.config import ValidationConfig
from phoibe.layered.application.context import ValidationContext
from phoibe.layered.application.factory import RuleRegistry
from phoibe.layered.application.factory import ValidatorFactory
from phoibe.layered.application.validator import LayerValidator
from phoibe.layered.core.entities import RuleExecutionResult
from phoibe.layered.core.entities import Severity
from phoibe.layered.core.entities import Status
from phoibe.layered.rules.rule import ValidationRule


class TestValidatorFactoryContract:

    def setup_method(self):
        RuleRegistry.clear()

        @RuleRegistry.register("mock_rule")
        class MockRule(ValidationRule):
            def __init__(self, points: int = 10):
                self.points = points

            @property
            def name(self):
                return "mock_rule"

            def execute(self, df: pd.DataFrame, context: ValidationContext):
                return RuleExecutionResult(
                    "mock_rule", Status.PASSED, Severity.INFO, True, True, self.points, self.points
                )

    def teardown_method(self):
        RuleRegistry.clear()

    @pytest.fixture
    def config_file(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            """
layer_name: raw
variable_patterns:
  timestamp:
    - zeitstempel
    - timestamp
  wind_speed:
    - ws.*gondel
    - wind.*speed
rules:
  - name: mock_rule
    params:
      points: 10
"""
        )
        return config

    @pytest.fixture
    def sample_data(self, tmp_path):
        csv = tmp_path / "data.csv"
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=5), "ws_gondel": [5, 6, 7, 8, 9]})
        df.to_csv(csv, index=False)
        return csv

    def test_create_from_config_returns_validator(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        validator = ValidatorFactory.create_from_config(config)
        assert isinstance(validator, LayerValidator)

    def test_validator_has_correct_layer_name(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        validator = ValidatorFactory.create_from_config(config)
        assert validator.layer_name == "raw"

    def test_validator_has_rules_from_config(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        validator = ValidatorFactory.create_from_config(config)
        assert len(validator.rules) == 1
        assert validator.rules[0].name == "mock_rule"

    def test_validator_works_end_to_end(self, config_file, sample_data):
        config = ValidationConfig.from_yaml(config_file)
        validator = ValidatorFactory.create_from_config(config)
        report = validator.validate(sample_data, "WEA_01")
        assert report.turbine_id == "WEA_01"
        assert len(report.rule_execution_results) == 1

    def test_raises_for_unknown_rule(self, tmp_path):
        config = tmp_path / "bad_config.yaml"
        config.write_text(
            """
layer_name: raw
variable_patterns: {}
rules:
  - name: nonexistent_rule
    params: {}
"""
        )

        cfg = ValidationConfig.from_yaml(config)
        with pytest.raises(KeyError, match="not found in registry"):
            ValidatorFactory.create_from_config(cfg)

    def test_create_from_memory_returns_validator(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        df = pd.DataFrame({"Zeitstempel": [1, 2, 3]})
        validator = ValidatorFactory.create_from_memory(config, df)
        assert isinstance(validator, LayerValidator)

    def test_memory_validator_uses_dataframe(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=5), "ws_gondel": [5, 6, 7, 8, 9]})
        validator = ValidatorFactory.create_from_memory(config, df, filename="test.csv")
        report = validator.validate("", "WEA_01")
        assert report.file_metadata.filename == "test.csv"
        assert report.file_metadata.format == "in_memory"

    def test_memory_validator_default_filename(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        df = pd.DataFrame({"col": [1, 2]})
        validator = ValidatorFactory.create_from_memory(config, df)
        report = validator.validate("", "WEA_01")
        assert report.file_metadata.filename == "in_memory_data"


class TestValidationConfigContract:

    @pytest.fixture
    def config_file(self, tmp_path):
        """Create sample config"""
        config = tmp_path / "config.yaml"
        config.write_text(
            """
layer_name: raw
variable_patterns:
  timestamp:
    - zeitstempel
  power:
    - leistung
rules:
  - name: rule1
    params:
      param1: value1
"""
        )
        return config

    def test_from_yaml_returns_config(self, config_file):
        config = ValidationConfig.from_yaml(config_file)

        assert isinstance(config, ValidationConfig)

    def test_from_yaml_accepts_string_path(self, config_file):
        config = ValidationConfig.from_yaml(str(config_file))

        assert isinstance(config, ValidationConfig)

    def test_from_yaml_accepts_path_object(self, config_file):
        config = ValidationConfig.from_yaml(Path(config_file))

        assert isinstance(config, ValidationConfig)

    def test_loads_layer_name(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        assert config.layer_name == "raw"

    def test_loads_variable_patterns(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        assert "timestamp" in config.variable_patterns
        assert config.variable_patterns["timestamp"] == ["zeitstempel"]

    def test_loads_rules(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        assert len(config.rules) == 1
        assert config.rules[0]["name"] == "rule1"
        assert config.rules[0]["params"]["param1"] == "value1"

    def test_raises_for_missing_file(self, tmp_path):
        missing = tmp_path / "missing.yaml"
        with pytest.raises(FileNotFoundError):
            ValidationConfig.from_yaml(missing)

    def test_raises_for_missing_layer_name(self, tmp_path):
        config = tmp_path / "bad.yaml"
        config.write_text("variable_patterns: {}")
        with pytest.raises(ValueError, match="layer_name"):
            ValidationConfig.from_yaml(config)

    def test_handles_empty_variable_patterns(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            """
layer_name: raw
variable_patterns: {}
rules: []
"""
        )
        validation_config = ValidationConfig.from_yaml(config)
        assert validation_config.variable_patterns == {}

    def test_handles_empty_rules(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            """
layer_name: raw
variable_patterns: {}
rules: []
"""
        )
        validation_config = ValidationConfig.from_yaml(config)
        assert validation_config.rules == []
