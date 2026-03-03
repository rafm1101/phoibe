import logging

import pandas as pd
import pytest

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.core.entities import RuleExecutionResult
from phoibe.layered.core.entities import Severity
from phoibe.layered.core.entities import Status
from phoibe.layered.rules.rule import RuleExecutionResultBuilder
from phoibe.layered.rules.rule import ValidationRule


class MockValidationRule(ValidationRule):

    def __init__(
        self,
        rule_name: str = "mock_rule",
        points: int = 10,
        severity: Severity = Severity.CRITICAL,
        logger: logging.Logger | None = None,
    ):
        self._rule_name = rule_name
        super().__init__(points, severity, logger=logger)

    @property
    def name(self) -> str:
        return self._rule_name

    def execute(self, df: pd.DataFrame, context: ValidationContext) -> RuleExecutionResult:
        return self.result_builder.passed(required=True, actual=True, message="Mock execution")


class TestValidationRuleContract:

    def test_is_abstract_base_class(self):
        with pytest.raises(TypeError):
            ValidationRule(points=10)

    def test_accepts_points_parameter(self):
        rule = MockValidationRule(points=15)
        assert rule.points == 15

    @pytest.mark.parametrize("severity", [Severity.INFO, Severity.WARNING, Severity.CRITICAL])
    def test_accepts_severity_parameter(self, severity):
        rule = MockValidationRule(severity=severity)
        assert rule.severity == severity

    def test_severity_defaults_to_critical(self):
        rule = MockValidationRule()
        assert rule.severity == Severity.CRITICAL

    def test_accepts_optional_logger(self):
        logger = logging.getLogger("test")
        rule = MockValidationRule()
        assert rule._logger is None
        rule_with_logger = MockValidationRule(logger=logger)
        assert rule_with_logger._logger is logger

    def test_has_name_property(self):
        rule = MockValidationRule(rule_name="test_rule")
        assert hasattr(rule, "name")
        assert rule.name == "test_rule"

    def test_name_is_string(self):
        rule = MockValidationRule(rule_name="test_rule")
        assert isinstance(rule.name, str)

    def test_name_is_accessible_after_init(self):
        rule = MockValidationRule(rule_name="accessible")
        _ = rule.name

    def test_has_result_builder_property(self):
        rule = MockValidationRule()
        assert hasattr(rule, "result_builder")

    def test_result_builder_is_lazy_initialized(self):
        rule = MockValidationRule()
        assert rule._result_builder is None
        _ = rule.result_builder
        assert rule._result_builder is not None

    def test_result_builder_returns_builder_instance(self):
        rule = MockValidationRule()
        builder = rule.result_builder
        assert isinstance(builder, RuleExecutionResultBuilder)

    def test_result_builder_uses_rule_name(self):
        rule = MockValidationRule(rule_name="custom_name")
        builder = rule.result_builder
        assert builder.rule_name == "custom_name"

    def test_result_builder_uses_rule_points(self):
        rule = MockValidationRule(points=25)
        builder = rule.result_builder
        assert builder.points == 25

    def test_result_builder_uses_rule_severity(self):
        rule = MockValidationRule(severity=Severity.WARNING)
        builder = rule.result_builder
        assert builder.severity == Severity.WARNING

    def test_result_builder_cached_after_first_access(self):
        rule = MockValidationRule()
        builder1 = rule.result_builder
        builder2 = rule.result_builder
        assert builder1 is builder2

    def test_has_execute_method(self):
        rule = MockValidationRule()
        assert hasattr(rule, "execute")
        assert callable(rule.execute)

    def test_execute_accepts_dataframe_and_context(self):
        rule = MockValidationRule()
        df = pd.DataFrame({"col": [1, 2, 3]})
        context = ValidationContext(layer_name="raw", detected_variables={}, turbine_id="WEA 01")
        _ = rule.execute(df, context)

    def test_execute_returns_rule_execution_result(self):
        rule = MockValidationRule()
        df = pd.DataFrame({"col": [1, 2, 3]})
        context = ValidationContext(layer_name="raw", detected_variables={}, turbine_id="WEA 01")
        result = rule.execute(df, context)
        assert isinstance(result, RuleExecutionResult)


class TestRuleExecutionResultBuilderContract:

    @pytest.fixture
    def builder(self):
        return RuleExecutionResultBuilder(rule_name="test_rule", points=10, severity=Severity.CRITICAL)

    def test_accepts_rule_name(self):
        builder = RuleExecutionResultBuilder("my_rule", 10)
        assert builder.rule_name == "my_rule"

    def test_accepts_points(self):
        builder = RuleExecutionResultBuilder("rule", 25)
        assert builder.points == 25

    def test_accepts_severity(self):
        builder = RuleExecutionResultBuilder("rule", 10, Severity.WARNING)
        assert builder.severity == Severity.WARNING

    def test_severity_defaults_to_critical(self):
        builder = RuleExecutionResultBuilder("rule", 10)
        assert builder.severity == Severity.CRITICAL

    def test_passed_returns_rule_execution_result(self, builder):
        result = builder.passed(required=True, actual=True)
        assert isinstance(result, RuleExecutionResult)

    def test_passed_sets_status_passed(self, builder):
        result = builder.passed(required=True, actual=True)
        assert result.status == Status.PASSED

    def test_passed_awards_full_points(self, builder):
        result = builder.passed(required=True, actual=True)
        assert result.points_achieved == builder.points
        assert result.points_max == builder.points

    def test_passed_includes_required_and_actual(self, builder):
        result = builder.passed(required="expected", actual="got")
        assert result.required == "expected"
        assert result.actual == "got"

    def test_passed_includes_message(self, builder):
        result = builder.passed(required=True, actual=True, message="Success!")
        assert result.message == "Success!"

    def test_passed_includes_details(self, builder):
        result = builder.passed(required=True, actual=True, details={"key": "value"})
        assert result.details == {"key": "value"}

    def test_passed_details_defaults_to_empty_dict(self, builder):
        result = builder.passed(required=True, actual=True)
        assert result.details == {}

    def test_warning_returns_rule_execution_result(self, builder):
        result = builder.warning(required=True, actual=True)
        assert isinstance(result, RuleExecutionResult)

    def test_warning_sets_status_passed(self, builder):
        result = builder.warning(required=True, actual=True)
        assert result.status == Status.WARNING

    @pytest.mark.parametrize("points_achieved, expected", [(5, 5), (None, 10)])
    def test_warning_awards_full_points(self, builder, points_achieved, expected):
        result = builder.warning(required=True, actual=True, points=points_achieved)
        assert result.points_achieved == expected
        assert result.points_max == builder.points

    def test_warning_includes_required_and_actual(self, builder):
        result = builder.warning(required="expected", actual="got")
        assert result.required == "expected"
        assert result.actual == "got"

    def test_warning_includes_message(self, builder):
        result = builder.warning(required=True, actual=True, message="Success!")
        assert result.message == "Success!"

    def test_warning_includes_details(self, builder):
        result = builder.warning(required=True, actual=True, details={"key": "value"})
        assert result.details == {"key": "value"}

    def test_warning_details_defaults_to_empty_dict(self, builder):
        result = builder.warning(required=True, actual=True)
        assert result.details == {}

    def test_failed_returns_rule_execution_result(self, builder):
        result = builder.failed(required=True, actual=False)
        assert isinstance(result, RuleExecutionResult)

    def test_failed_sets_status_failed(self, builder):
        result = builder.failed(required=True, actual=False)
        assert result.status == Status.FAILED

    def test_failed_awards_zero_points(self, builder):
        result = builder.failed(required=True, actual=False)
        assert result.points_achieved == 0
        assert result.points_max == builder.points

    def test_failed_includes_required_and_actual(self, builder):
        result = builder.failed(required="expected", actual="got_wrong")
        assert result.required == "expected"
        assert result.actual == "got_wrong"

    def test_not_checked_returns_rule_execution_result(self, builder):
        result = builder.not_checked("Reason")
        assert isinstance(result, RuleExecutionResult)

    def test_not_checked_sets_status_not_checked(self, builder):
        result = builder.not_checked("Reason")
        assert result.status == Status.NOT_CHECKED

    def test_not_checked_awards_zero_points(self, builder):
        result = builder.not_checked("Reason")
        assert result.points_achieved == 0

    def test_not_checked_sets_required_to_na(self, builder):
        result = builder.not_checked("Reason")
        assert result.required == "N/A"

    def test_not_checked_sets_actual_to_none(self, builder):
        result = builder.not_checked("Reason")
        assert result.actual is None

    def test_not_checked_includes_message(self, builder):
        result = builder.not_checked("Variable not detected")
        assert result.message == "Variable not detected"

    def test_error_returns_rule_execution_result(self, builder):
        result = builder.error("Error occurred")
        assert isinstance(result, RuleExecutionResult)

    def test_error_sets_status_error(self, builder):
        result = builder.error("Error occurred")
        assert result.status == Status.ERROR

    def test_error_awards_zero_points(self, builder):
        result = builder.error("Error occurred")
        assert result.points_achieved == 0

    def test_error_includes_message(self, builder):
        result = builder.error("Something went wrong")
        assert result.message == "Something went wrong"

    def test_error_appends_exception_to_message(self, builder):
        exception = ValueError("Invalid value")
        result = builder.error("Failed", exception=exception)
        assert "Failed" in result.message
        assert "Invalid value" in result.message

    def test_error_works_without_exception(self, builder):
        result = builder.error("Error message only")
        assert result.message == "Error message only"

    def test_all_methods_use_builder_rule_name(self):
        builder = RuleExecutionResultBuilder("my_rule", 10)
        assert builder.passed(True, True).rule_name == "my_rule"
        assert builder.failed(True, False).rule_name == "my_rule"
        assert builder.not_checked("msg").rule_name == "my_rule"
        assert builder.error("msg").rule_name == "my_rule"

    def test_all_methods_use_builder_severity(self):
        builder = RuleExecutionResultBuilder("rule", 10, Severity.INFO)
        assert builder.passed(True, True).severity == Severity.INFO
        assert builder.failed(True, False).severity == Severity.INFO
        assert builder.not_checked("msg").severity == Severity.INFO
        assert builder.error("msg").severity == Severity.INFO

    def test_all_methods_set_points_max(self):
        builder = RuleExecutionResultBuilder("rule", 15)
        assert builder.passed(True, True).points_max == 15
        assert builder.failed(True, False).points_max == 15
        assert builder.not_checked("msg").points_max == 15
        assert builder.error("msg").points_max == 15
