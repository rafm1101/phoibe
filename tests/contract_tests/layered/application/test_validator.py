import datetime
import pathlib

import pandas as pd
import pytest

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.application.validator import LayerValidator
from phoibe.layered.core.entities import LayerReport
from phoibe.layered.core.entities import RuleExecutionResult
from phoibe.layered.core.entities import Severity
from phoibe.layered.core.entities import Status
from phoibe.layered.infrastructure.detector import RegexVariableDetector
from phoibe.layered.infrastructure.io import InMemoryDataLoader
from phoibe.layered.infrastructure.io import PandasDataLoader
from phoibe.layered.rules.rule import ValidationRule


class MockRule(ValidationRule):

    def __init__(self, name: str = "mock_rule", should_pass: bool = True, points: int = 10):
        self._name = name
        self.should_pass = should_pass
        self.points = points

    @property
    def name(self) -> str:
        return self._name

    def execute(self, df: pd.DataFrame, context: ValidationContext) -> RuleExecutionResult:
        status = Status.PASSED if self.should_pass else Status.FAILED
        points_achieved = self.points if self.should_pass else 0

        return RuleExecutionResult(
            rule_name=self.name,
            status=status,
            severity=Severity.CRITICAL,
            required=True,
            actual=self.should_pass,
            points_max=self.points,
            points_achieved=points_achieved,
            message="Mock rule result",
        )


class TestLayerValidatorContract:

    @pytest.fixture
    def sample_data_file(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame(
            {
                "Zeitstempel": pd.date_range("2024-01-01", periods=10, freq="10min"),
                "ws_gondel": [5.2, 6.1, 5.8, 7.2, 6.5, 5.9, 6.8, 7.1, 6.4, 5.7],
                "Leistung": [1200, 1500, 1300, 1800, 1400, 1250, 1600, 1750, 1450, 1280],
            }
        )
        df.to_csv(csv_file, index=False)
        return csv_file

    @pytest.fixture
    def validator(self):
        data_loader = PandasDataLoader()
        variable_detector = RegexVariableDetector(
            {"timestamp": [r"zeitstempel"], "wind_speed": [r"ws.*gondel"], "power": [r"leistung"]}
        )
        rules = [MockRule("rule1", should_pass=True, points=10), MockRule("rule2", should_pass=True, points=20)]

        return LayerValidator(
            layer_name="raw", data_loader=data_loader, variable_detector=variable_detector, rules=rules
        )

    def test_validate_returns_layer_report(self, validator, sample_data_file):
        result = validator.validate(sample_data_file, "WEA 01")
        assert isinstance(result, LayerReport)

    def test_validate_accepts_string_path(self, validator, sample_data_file):
        result = validator.validate(str(sample_data_file), "WEA 01")
        assert isinstance(result, LayerReport)

    def test_validate_accepts_path_object(self, validator, sample_data_file):
        result = validator.validate(pathlib.Path(sample_data_file), "WEA 01")
        assert isinstance(result, LayerReport)

    def test_validate_sets_turbine_id(self, validator, sample_data_file):
        result = validator.validate(sample_data_file, "WEA 01")
        assert result.turbine_id == "WEA 01"

    def test_validate_sets_layer_name(self, validator, sample_data_file):
        result = validator.validate(sample_data_file, "WEA 01")
        assert result.layer_name == "raw"

    def test_validate_includes_file_metadata(self, validator, sample_data_file):
        result = validator.validate(sample_data_file, "WEA 01")
        assert result.file_metadata is not None
        assert result.file_metadata.filename == "test.csv"
        assert result.file_metadata.size_bytes > 0

    def test_validate_includes_detected_variables(self, validator, sample_data_file):
        result = validator.validate(sample_data_file, "WEA 01")
        assert result.detected_variables is not None
        assert isinstance(result.detected_variables, dict)
        assert result.detected_variables["timestamp"] == "Zeitstempel"

    def test_validate_includes_timestamp(self, validator, sample_data_file):
        before = datetime.datetime.now()
        result = validator.validate(sample_data_file, "WEA 01")
        after = datetime.datetime.now()
        assert before <= result.timestamp <= after

    def test_executes_all_rules(self, validator, sample_data_file):
        result = validator.validate(sample_data_file, "WEA 01")
        assert len(result.rule_execution_results) == 2
        rule_names = {record.rule_name for record in result.rule_execution_results}
        assert rule_names == {"rule1", "rule2"}

    def test_rules_receive_dataframe(self, sample_data_file):
        received_df = None

        class CaptureRule(ValidationRule):
            @property
            def name(self):
                return "capture"

            def execute(self, df, context):
                nonlocal received_df
                received_df = df
                return RuleExecutionResult("capture", Status.PASSED, Severity.INFO, True, True, 0, 0)

        data_loader = PandasDataLoader()
        variable_detector = RegexVariableDetector({})
        validator = LayerValidator("raw", data_loader, variable_detector, [CaptureRule(points=13)])
        validator.validate(sample_data_file, "WEA 01")
        assert received_df is not None
        assert isinstance(received_df, pd.DataFrame)
        assert len(received_df) == 10

    def test_rules_receive_validation_context(self, sample_data_file):
        received_context = None

        class CaptureRule(ValidationRule):
            @property
            def name(self):
                return "capture"

            def execute(self, df, context):
                nonlocal received_context
                received_context = context
                return RuleExecutionResult("capture", Status.PASSED, Severity.INFO, True, True, 0, 0)

        data_loader = PandasDataLoader()
        variable_detector = RegexVariableDetector({"timestamp": [r"zeit"]})
        validator = LayerValidator("raw", data_loader, variable_detector, [CaptureRule(points=13)])
        validator.validate(sample_data_file, "WEA 01")
        assert received_context is not None
        assert isinstance(received_context, ValidationContext)
        assert received_context.turbine_id == "WEA 01"
        assert received_context.layer_name == "raw"

    def test_handles_missing_file_gracefully(self, validator, tmp_path):
        nonexistent = tmp_path / "missing.csv"
        with pytest.raises(FileNotFoundError):
            validator.validate(nonexistent, "WEA 01")

    def test_handles_rule_exception_gracefully(self, sample_data_file):
        class CrashingRule(ValidationRule):
            @property
            def name(self):
                return "crasher"

            def execute(self, df, context):
                raise ValueError("Intentional crash")

        data_loader = PandasDataLoader()
        variable_detector = RegexVariableDetector({})
        validator = LayerValidator(
            "raw", data_loader, variable_detector, [CrashingRule(points=13), MockRule("survivor")]
        )
        result = validator.validate(sample_data_file, "WEA 01")
        assert len(result.rule_execution_results) == 2
        crasher_result = next(record for record in result.rule_execution_results if record.rule_name == "crasher")
        assert crasher_result.status == Status.ERROR

    def test_continues_after_rule_failure(self, sample_data_file):
        data_loader = PandasDataLoader()
        variable_detector = RegexVariableDetector({})
        validator = LayerValidator(
            "raw",
            data_loader,
            variable_detector,
            [
                MockRule("rule1", should_pass=False),
                MockRule("rule2", should_pass=True),
                MockRule("rule3", should_pass=True),
            ],
        )
        result = validator.validate(sample_data_file, "WEA 01")
        assert len(result.rule_execution_results) == 3

    def test_works_with_in_memory_loader(self):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=5), "ws_gondel": [5, 6, 7, 8, 9]})
        data_loader = InMemoryDataLoader(df, filename="memory_data")
        variable_detector = RegexVariableDetector({"timestamp": [r"zeit"]})
        validator = LayerValidator("raw", data_loader, variable_detector, [MockRule()])
        result = validator.validate("", "WEA 01")

        assert isinstance(result, LayerReport)
        assert result.file_metadata.filename == "memory_data"

    def test_works_with_no_rules(self, sample_data_file):
        data_loader = PandasDataLoader()
        variable_detector = RegexVariableDetector({})
        validator = LayerValidator("raw", data_loader, variable_detector, rules=[])
        result = validator.validate(sample_data_file, "WEA 01")

        assert isinstance(result, LayerReport)
        assert len(result.rule_execution_results) == 0
        assert result.score_max == 0
        assert result.score_achieved == 0

    def test_works_with_no_detected_variables(self, sample_data_file):
        data_loader = PandasDataLoader()
        variable_detector = RegexVariableDetector({"nonexistent": [r"will_not_match"]})
        validator = LayerValidator("raw", data_loader, variable_detector, [])
        result = validator.validate(sample_data_file, "WEA 01")

        assert result.detected_variables["nonexistent"] is None
