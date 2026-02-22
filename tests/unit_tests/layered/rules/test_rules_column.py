import pandas as pd
import pytest

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.core.entities import Severity
from phoibe.layered.core.entities import Status
from phoibe.layered.rules.rules_columns import RequiredVariableRule


class TestRequiredVariableRuleUnit:

    @pytest.fixture
    def rule(self):
        return RequiredVariableRule(variable_name="timestamp", points=10, severity=Severity.CRITICAL)

    def test_passes_when_variable_detected(self, rule):
        context = ValidationContext(
            layer_name="raw", detected_variables={"timestamp": "Zeitstempel"}, turbine_id="WEA 01"
        )
        df = pd.DataFrame({"Zeitstempel": [1, 2, 3]})
        result = rule.execute(df, context)
        assert result.status == Status.PASSED
        assert result.points_achieved == 10

    def test_passed_result_includes_column_name(self, rule):
        context = ValidationContext(
            layer_name="raw", detected_variables={"timestamp": "actual_column"}, turbine_id="WEA 01"
        )
        df = pd.DataFrame({"actual_column": [1, 2, 3]})
        result = rule.execute(df, context)
        assert "actual_column" in result.actual
        assert result.details["column_name"] == "actual_column"

    def test_fails_when_variable_not_detected(self, rule):
        context = ValidationContext(layer_name="raw", detected_variables={"timestamp": None}, turbine_id="WEA_ 1")
        df = pd.DataFrame({"other_column": [1, 2, 3]})
        result = rule.execute(df, context)
        assert result.status == Status.FAILED
        assert result.points_achieved == 0

    def test_fails_when_variable_missing_from_context(self, rule):
        context = ValidationContext(layer_name="raw", detected_variables={}, turbine_id="WEA_ 1")
        df = pd.DataFrame({"col": [1, 2, 3]})
        result = rule.execute(df, context)
        assert result.status == Status.FAILED
        assert result.points_achieved == 0

    def test_failed_result_includes_variable_name(self, rule):
        context = ValidationContext(layer_name="raw", detected_variables={"timestamp": None}, turbine_id="WEA_ 1")
        df = pd.DataFrame()
        result = rule.execute(df, context)
        assert "timestamp" in result.message.lower()

    def test_rule_name_includes_variable_name(self):
        rule = RequiredVariableRule(variable_name="wind_speed")
        assert "wind_speed" in rule.name

    def test_different_variables_have_different_names(self):
        rule1 = RequiredVariableRule(variable_name="timestamp")
        rule2 = RequiredVariableRule(variable_name="power")
        assert rule1.name != rule2.name

    def test_accepts_custom_points(self):
        rule = RequiredVariableRule(variable_name="timestamp", points=25)
        assert rule.points == 25

    def test_accepts_custom_severity(self):
        rule = RequiredVariableRule(variable_name="timestamp", severity=Severity.WARNING)
        assert rule.severity == Severity.WARNING

    def test_works_with_empty_dataframe(self, rule):
        context = ValidationContext(layer_name="raw", detected_variables={"timestamp": "col"}, turbine_id="WEA_ 1")
        df = pd.DataFrame()
        result = rule.execute(df, context)
        assert result.status == Status.PASSED

    def test_works_with_multiple_detected_variables(self, rule):
        context = ValidationContext(
            layer_name="raw",
            detected_variables={"timestamp": "Zeit", "power": "Leistung", "wind_speed": "ws"},
            turbine_id="WEA 01",
        )
        df = pd.DataFrame({"Zeit": [1, 2]})
        result = rule.execute(df, context)
        assert result.status == Status.PASSED

    def test_variable_name_case_sensitive(self):
        rule = RequiredVariableRule(variable_name="Timestamp")
        context = ValidationContext(
            layer_name="raw", detected_variables={"timestamp": "col"}, turbine_id="WEA_ 1"
        )  # lowercase
        df = pd.DataFrame({"col": [1, 2]})
        result = rule.execute(df, context)
        assert result.status == Status.FAILED
