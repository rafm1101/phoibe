import numpy as np
import pandas as pd
import pytest

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.core.entities import Severity
from phoibe.layered.core.entities import Status
from phoibe.layered.rules.rules_values import RangeRule


class TestRangeRule:

    @pytest.fixture
    def rule(self):
        return RangeRule(
            variable_ranges={"power_kw": [0, 3000], "wind_speed": [0, 40], "temperature": [-40, 60]}, points=10
        )

    @pytest.fixture
    def context_with_variables(self):
        return ValidationContext(
            layer_name="test",
            detected_variables={"power_kw": "Leistung", "wind_speed": "ws_gondel", "temperature": "Temp_Gondel"},
            turbine_id="WEA 01",
        )

    def test_detects_no_violations_all_in_range(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [1000, 1500, 2000], "ws_gondel": [5, 10, 15], "Temp_Gondel": [10, 15, 20]})
        result = rule.execute(df, context_with_variables)

        assert result.status == Status.PASSED
        assert result.details["n_out_of_range"]["power_kw"] == 0
        assert result.details["n_out_of_range"]["wind_speed"] == 0
        assert result.details["n_out_of_range"]["temperature"] == 0

    def test_counts_violations_below_min(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [-100, 1500, 2000], "ws_gondel": [5, 10, 15], "Temp_Gondel": [10, 15, 20]})
        result = rule.execute(df, context_with_variables)

        assert result.details["n_out_of_range"]["power_kw"] == 1

    def test_counts_violations_above_max(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [1000, 1500, 2000], "ws_gondel": [5, 50, 15], "Temp_Gondel": [10, 15, 20]})
        result = rule.execute(df, context_with_variables)

        assert result.details["n_out_of_range"]["wind_speed"] == 1
        assert result.status == Status.PASSED

    def test_counts_multiple_violations_same_variable(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [-100, 3500, 2000], "ws_gondel": [5, 10, 15], "Temp_Gondel": [10, 15, 20]})
        result = rule.execute(df, context_with_variables)

        assert result.details["n_out_of_range"]["power_kw"] == 2

    def test_counts_violations_across_variables(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [-100, 1500, 2000], "ws_gondel": [5, 50, 15], "Temp_Gondel": [-50, 15, 70]})
        result = rule.execute(df, context_with_variables)

        assert result.details["n_out_of_range"]["power_kw"] == 1
        assert result.details["n_out_of_range"]["wind_speed"] == 1
        assert result.details["n_out_of_range"]["temperature"] == 2

    def test_values_at_exact_boundaries_valid(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [0, 3000], "ws_gondel": [0, 40], "Temp_Gondel": [-40, 60]})
        result = rule.execute(df, context_with_variables)

        assert result.details["n_out_of_range"]["power_kw"] == 0
        assert result.details["n_out_of_range"]["wind_speed"] == 0
        assert result.details["n_out_of_range"]["temperature"] == 0

    def test_values_just_outside_boundaries_invalid(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [-0.1, 3000.1], "ws_gondel": [5, 10], "Temp_Gondel": [10, 15]})
        result = rule.execute(df, context_with_variables)

        assert result.details["n_out_of_range"]["power_kw"] == 2

    def test_skips_undetected_variables(self, rule):
        context = ValidationContext(
            layer_name="test",
            detected_variables={"power_kw": "Leistung", "wind_speed": None, "temperature": None},
            turbine_id="WEA 01",
        )
        df = pd.DataFrame({"Leistung": [1000, 1500, 2000]})
        result = rule.execute(df, context)

        assert "wind_speed" in result.details["skipped"]
        assert "temperature" in result.details["skipped"]
        assert "power_kw" in result.details["n_out_of_range"].keys()
        assert len(result.details["n_out_of_range"]) == 1

    def test_warning_when_some_variables_skipped(self, rule):
        context = ValidationContext(
            layer_name="test",
            detected_variables={"power_kw": "Leistung", "wind_speed": None, "temperature": None},
            turbine_id="WEA 01",
        )
        df = pd.DataFrame({"Leistung": [1000, 1500, 2000]})
        result = rule.execute(df, context)

        assert result.status == Status.WARNING

    def test_warning_awards_reduced_points(self, rule):
        context = ValidationContext(
            layer_name="test",
            detected_variables={"power_kw": "Leistung", "wind_speed": None, "temperature": None},
            turbine_id="WEA 01",
        )
        df = pd.DataFrame({"Leistung": [1000, 1500, 2000]})
        result = rule.execute(df, context)

        assert result.points_achieved == rule.points // 3

    def test_returns_passed_when_no_variables_skipped(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [1000, 1500, 2000], "ws_gondel": [5, 10, 15], "Temp_Gondel": [10, 15, 20]})
        result = rule.execute(df, context_with_variables)

        assert result.status == Status.PASSED
        assert len(result.details["skipped"]) == 0

    def test_passed_awards_full_points(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [1000, 1500, 2000], "ws_gondel": [5, 10, 15], "Temp_Gondel": [10, 15, 20]})
        result = rule.execute(df, context_with_variables)

        assert result.points_achieved == rule.points

    def test_result_includes_n_out_of_range_dict(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [1000], "ws_gondel": [5], "Temp_Gondel": [10]})
        result = rule.execute(df, context_with_variables)

        assert "n_out_of_range" in result.details
        assert isinstance(result.details["n_out_of_range"], dict)

    def test_result_includes_skipped_list(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [1000], "ws_gondel": [5], "Temp_Gondel": [10]})
        result = rule.execute(df, context_with_variables)

        assert "skipped" in result.details
        assert isinstance(result.details["skipped"], list)

    def test_n_out_of_range_keys_are_actual_variable_names(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [1000], "ws_gondel": [5], "Temp_Gondel": [10]})
        result = rule.execute(df, context_with_variables)

        assert "power_kw" in result.details["n_out_of_range"]
        assert "wind_speed" in result.details["n_out_of_range"]
        assert "temperature" in result.details["n_out_of_range"]

    def test_skipped_uses_variable_names(self, rule):
        context = ValidationContext(
            layer_name="test",
            detected_variables={"power_kw": None, "wind_speed": None, "temperature": "Temp"},
            turbine_id="WEA 01",
        )
        df = pd.DataFrame({"Temp": [10]})
        result = rule.execute(df, context)

        assert "power_kw" in result.details["skipped"]
        assert "wind_speed" in result.details["skipped"]

    def test_handles_empty_dataframe(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [], "ws_gondel": [], "Temp_Gondel": []})
        result = rule.execute(df, context_with_variables)

        assert result.details["n_out_of_range"]["power_kw"] == 0

    def test_handles_single_row(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [5000], "ws_gondel": [5], "Temp_Gondel": [10]})
        result = rule.execute(df, context_with_variables)

        assert result.details["n_out_of_range"]["power_kw"] == 1

    def test_handles_nan_values(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [1000, np.nan, 2000], "ws_gondel": [5, 10, 15], "Temp_Gondel": [10, 15, 20]})
        result = rule.execute(df, context_with_variables)

        assert isinstance(result.details["n_out_of_range"]["power_kw"], int)

    def test_handles_all_values_out_of_range(self, rule, context_with_variables):
        df = pd.DataFrame(
            {"Leistung": [-100, -200, -300], "ws_gondel": [5, 10, 15], "Temp_Gondel": [10, 15, 20]}  # All below min
        )
        result = rule.execute(df, context_with_variables)

        assert result.details["n_out_of_range"]["power_kw"] == 3

    def test_accepts_custom_ranges(self):
        rule = RangeRule(variable_ranges={"custom_var": [-1000, 1000]})

        assert "custom_var" in rule.variable_ranges
        assert rule.variable_ranges["custom_var"] == [-1000, 1000]

    def test_accepts_single_variable(self):
        rule = RangeRule(variable_ranges={"power_kw": [0, 3000]})
        context = ValidationContext(layer_name="test", detected_variables={"power_kw": "Power"}, turbine_id="WEA 01")
        df = pd.DataFrame({"Power": [1000, 1500, 2000]})
        result = rule.execute(df, context)

        assert result.status == Status.PASSED
        assert len(result.details["n_out_of_range"]) == 1

    def test_accepts_many_variables(self):
        rule = RangeRule(variable_ranges={f"var_{i}": [0, 100] for i in range(10)})

        assert len(rule.variable_ranges) == 10

    def test_accepts_asymmetric_ranges(self):
        rule = RangeRule(variable_ranges={"var": [-1000, 100]})
        context = ValidationContext(layer_name="test", detected_variables={"var": "V"}, turbine_id="WEA 01")
        df = pd.DataFrame({"V": [-999, 99]})
        result = rule.execute(df, context)

        assert result.details["n_out_of_range"]["var"] == 0

    def test_severity_is_info(self, rule):
        assert rule.severity == Severity.INFO

    def test_accepts_custom_points(self):
        rule = RangeRule(variable_ranges={"power_kw": [0, 3000]}, points=20)

        assert rule.points == 20

    def test_handles_empty_variable_ranges(self):
        rule = RangeRule(variable_ranges={})
        context = ValidationContext(layer_name="test", detected_variables={}, turbine_id="WEA 01")
        df = pd.DataFrame({"col": [1, 2, 3]})
        result = rule.execute(df, context)

        assert result.status == Status.PASSED
        assert len(result.details["n_out_of_range"]) == 0
        assert len(result.details["skipped"]) == 0
