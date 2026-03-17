import pandas as pd
import pytest

from phoibe.layered.application.config import ValidationConfig
from phoibe.layered.application.context import ValidationContext
from phoibe.layered.application.factory import ValidatorFactory
from phoibe.layered.application.registry import RuleRegistry
from phoibe.layered.application.validator import LayerValidator
from phoibe.layered.core.entities import RuleExecutionResult
from phoibe.layered.core.entities import Severity
from phoibe.layered.core.entities import Status
from phoibe.layered.core.entities import ValidationMode
from phoibe.layered.rules.rule import ValidationRule


class TestValidatorFactory:

    def setup_method(self):
        RuleRegistry.clear()

        @RuleRegistry.register("mock_rule")
        class MockRule(ValidationRule):
            def __init__(self, points: int = 10):
                self.points = points

            @property
            def name(self):
                return "mock_rule"

            @property
            def severity(self):
                return Severity.INFO

            def execute(self, df: pd.DataFrame, context: ValidationContext):
                return RuleExecutionResult(
                    rule_name="mock_rule",
                    status=Status.PASSED,
                    severity=Severity.INFO,
                    required=True,
                    actual=True,
                    points_max=self.points,
                    points_achieved=self.points,
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

    def test_profiling_returns_validator(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        validator = ValidatorFactory.profiling(config)
        assert isinstance(validator, LayerValidator)

    def test_profiling_creates_profiling_mode(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        validator = ValidatorFactory.profiling(config)
        assert validator.mode == ValidationMode.PROFILING

    def test_profiling_validator_has_correct_layer_name(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        validator = ValidatorFactory.profiling(config)
        assert validator.layer_name == "raw"

    def test_profiling_validator_has_rules_from_config(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        validator = ValidatorFactory.profiling(config)
        assert len(validator.rules) == 1
        assert validator.rules[0].name == "mock_rule"

    def test_profiling_works_end_to_end(self, config_file, sample_data):
        config = ValidationConfig.from_yaml(config_file)
        validator = ValidatorFactory.profiling(config)
        report = validator.validate(sample_data, "WEA_01")
        assert report.turbine_id == "WEA_01"
        assert len(report.rule_execution_results) == 1

    def test_contract_returns_validator(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        validator = ValidatorFactory.contract(config)
        assert isinstance(validator, LayerValidator)

    def test_contract_creates_contract_mode(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        validator = ValidatorFactory.contract(config)
        assert validator.mode == ValidationMode.CONTRACT

    def test_contract_validator_works_end_to_end(self, config_file, sample_data):
        config = ValidationConfig.from_yaml(config_file)
        validator = ValidatorFactory.contract(config)
        report = validator.validate(sample_data, "WEA_01")
        assert report.turbine_id == "WEA_01"

    def test_profiling_with_data_returns_validator(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        df = pd.DataFrame({"Zeitstempel": [1, 2, 3]})
        validator = ValidatorFactory.profiling(config, data=df)
        assert isinstance(validator, LayerValidator)

    def test_profiling_with_data_uses_memory_loader(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        df = pd.DataFrame({"Zeitstempel": [1, 2, 3]})
        validator = ValidatorFactory.profiling(config, data=df)

        from phoibe.layered.infrastructure.io import InMemoryDataLoader

        assert isinstance(validator.data_loader, InMemoryDataLoader)

    def test_profiling_memory_validator_uses_dataframe(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=5), "ws_gondel": [5, 6, 7, 8, 9]})
        validator = ValidatorFactory.profiling(config, data=df, filename="test.csv")
        report = validator.validate("", "WEA_01")
        assert report.file_metadata.filename == "test.csv"
        assert report.file_metadata.format == "in_memory"

    def test_profiling_memory_default_filename(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        df = pd.DataFrame({"col": [1, 2]})
        validator = ValidatorFactory.profiling(config, data=df)
        report = validator.validate("", "WEA_01")
        assert report.file_metadata.filename == "in_memory"

    def test_contract_with_data_uses_memory_loader(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        df = pd.DataFrame({"Zeitstempel": [1, 2, 3]})
        validator = ValidatorFactory.contract(config, data=df)
        from phoibe.layered.infrastructure.io import InMemoryDataLoader

        assert isinstance(validator.data_loader, InMemoryDataLoader)

    def test_contract_memory_custom_filename(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        df = pd.DataFrame({"col": [1, 2]})
        validator = ValidatorFactory.contract(config, data=df, filename="custom.csv")
        report = validator.validate("", "WEA_01")
        assert report.file_metadata.filename == "custom.csv"

    def test_raises_for_unknown_rule(self, tmp_path):
        """Error: Raises KeyError for unknown rule"""
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
            ValidatorFactory.profiling(cfg)

    def test_raises_for_invalid_rule_params(self, tmp_path):
        """Error: Raises TypeError for invalid rule parameters"""
        config = tmp_path / "bad_params.yaml"
        config.write_text(
            """
layer_name: raw
variable_patterns: {}
rules:
  - name: mock_rule
    params:
      invalid_param: 999
"""
        )

        cfg = ValidationConfig.from_yaml(config)
        with pytest.raises(TypeError, match="mock_rule"):
            ValidatorFactory.profiling(cfg)

    def test_create_from_config_still_works(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        validator = ValidatorFactory.create_from_config(config)
        assert isinstance(validator, LayerValidator)
        assert validator.mode == ValidationMode.PROFILING

    def test_create_from_memory_still_works(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        df = pd.DataFrame({"col": [1, 2, 3]})
        validator = ValidatorFactory.create_from_memory(config, df)
        assert isinstance(validator, LayerValidator)
        report = validator.validate("", "WEA_01")
        assert report.file_metadata.filename == "in_memory_data"

    def test_create_with_explicit_mode(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        profiling = ValidatorFactory.create(config, mode=ValidationMode.PROFILING)
        contract = ValidatorFactory.create(config, mode=ValidationMode.CONTRACT)

        assert profiling.mode == ValidationMode.PROFILING
        assert contract.mode == ValidationMode.CONTRACT

    def test_create_with_custom_data_loader(self, config_file):
        config = ValidationConfig.from_yaml(config_file)

        from phoibe.layered.infrastructure.io import PandasDataLoader

        custom_loader = PandasDataLoader()
        validator = ValidatorFactory.create(config, mode=ValidationMode.PROFILING, data_loader=custom_loader)

        assert validator.data_loader is custom_loader

    def test_create_with_custom_variable_detector(self, config_file):
        config = ValidationConfig.from_yaml(config_file)

        from phoibe.layered.infrastructure.detector import RegexVariableDetector

        custom_detector = RegexVariableDetector({})
        validator = ValidatorFactory.create(config, mode=ValidationMode.PROFILING, variable_detector=custom_detector)

        assert validator.variable_detector is custom_detector
