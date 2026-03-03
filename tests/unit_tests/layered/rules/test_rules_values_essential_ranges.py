import numpy as np
import pandas as pd
import pytest

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.core.entities import Severity
from phoibe.layered.core.entities import Status
from phoibe.layered.rules.rules_values import EssentialRanges


class TestEssentialRanges:

    @pytest.fixture
    def rule(self):
        return EssentialRanges(variable_names=["power_kw", "wind_speed", "temperature"], proportion=0.995, points=10)

    @pytest.fixture
    def context_with_variables(self):
        return ValidationContext(
            layer_name="test",
            detected_variables={"power_kw": "Leistung", "wind_speed": "ws_gondel", "temperature": "Temp_Gondel"},
            turbine_id="WEA_01",
        )

    def test_computes_hdi_for_all_variables(self, rule, context_with_variables):
        df = pd.DataFrame(
            {
                "Leistung": np.random.uniform(1000, 2000, 100),
                "ws_gondel": np.random.uniform(5, 15, 100),
                "Temp_Gondel": np.random.uniform(10, 20, 100),
            }
        )
        result = rule.execute(df, context_with_variables)

        assert result.status == Status.PASSED
        assert "power_kw" in result.details["checked"]
        assert "wind_speed" in result.details["checked"]
        assert "temperature" in result.details["checked"]

    def test_hdi_is_tuple_of_two_floats(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [1000, 1500, 2000], "ws_gondel": [5, 10, 15], "Temp_Gondel": [10, 15, 20]})

        result = rule.execute(df, context_with_variables)

        hdi = result.details["checked"]["power_kw"]
        assert isinstance(hdi, (list, tuple))
        assert len(hdi) == 2
        assert isinstance(hdi[0], float)
        assert isinstance(hdi[1], float)

    def test_hdi_lower_bound_less_than_upper(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [1000, 1500, 2000], "ws_gondel": [5, 10, 15], "Temp_Gondel": [10, 15, 20]})
        result = rule.execute(df, context_with_variables)
        hdi = result.details["checked"]["power_kw"]

        assert hdi[0] <= hdi[1]

    def test_hdi_contains_most_data(self, rule, context_with_variables):
        data = list(range(0, 100))
        df = pd.DataFrame({"Leistung": data, "ws_gondel": [10] * 100, "Temp_Gondel": [15] * 100})
        result = rule.execute(df, context_with_variables)
        hdi = result.details["checked"]["power_kw"]

        assert hdi[0] == 0
        assert hdi[1] == 99

    def test_hdi_given_regular_grid(self, rule):
        data = np.linspace(0, 100, 1000)
        hdi = rule._compute_high_density_interval(data, proportion=0.95)

        assert 0 <= hdi[0] <= 5
        assert 95 <= hdi[1] <= 100

    def test_hdi_given_normal_distribution(self, rule):
        np.random.seed(42)
        data = np.random.normal(50, 10, 1000)
        hdi = rule._compute_high_density_interval(data, proportion=0.95)

        assert 30 <= hdi[0] <= 32
        assert 68 <= hdi[1] <= 70

    def test_hdi_given_skewed_distribution(self, rule):
        np.random.seed(42)
        data = np.random.exponential(10, 1000)
        hdi = rule._compute_high_density_interval(data, proportion=0.95)

        assert hdi[0] < 1
        assert hdi[1] > hdi[0]

    def test_hdi_given_bimodal_distribution(self, rule):
        data = np.concatenate([np.random.normal(20, 2, 500), np.random.normal(80, 2, 500)])
        hdi = rule._compute_high_density_interval(data, proportion=0.95)

        assert isinstance(hdi, list)
        assert len(hdi) == 2
        assert hdi[0] < 20
        assert hdi[1] > 80

    def test_hdi_handles_empty_array(self, rule):
        data = np.array([])
        hdi = rule._compute_high_density_interval(data, proportion=0.95)

        assert len(hdi) == 2
        assert np.isnan(hdi[0]) or hdi[0] is not None
        assert np.isnan(hdi[1]) or hdi[1] is not None

    def test_hdi_handles_all_nan(self, rule):
        data = np.array([np.nan, np.nan, np.nan])
        hdi = rule._compute_high_density_interval(data, proportion=0.95)

        assert np.isnan(hdi).all()

    def test_hdi_handles_mixed_nan(self, rule):
        data = np.array([1, 2, np.nan, 3, 4, np.nan, 5])
        hdi = rule._compute_high_density_interval(data, proportion=0.9)

        assert not np.isnan(hdi[0])
        assert not np.isnan(hdi[1])
        assert 1 <= hdi[0] <= 1
        assert 5 <= hdi[1] <= 5

    def test_hdi_handles_single_value(self, rule):
        data = np.array([42.0])
        hdi = rule._compute_high_density_interval(data, proportion=0.95)

        assert hdi[0] == 42.0
        assert hdi[1] == 42.0

    def test_hdi_handles_two_values(self, rule):
        data = np.array([10.0, 20.0])
        hdi = rule._compute_high_density_interval(data, proportion=0.95)

        assert hdi[0] == 10.0
        assert hdi[1] == 20.0

    def test_hdi_handles_identical_values(self, rule):
        data = np.array([42.0] * 100)
        hdi = rule._compute_high_density_interval(data, proportion=0.95)

        assert hdi[0] == 42.0
        assert hdi[1] == 42.0

    def test_hdi_given_zero_proportion(self, rule):
        data = np.array([1, 2, 3, 4, 5])
        hdi = rule._compute_high_density_interval(data, proportion=0.0)

        assert isinstance(hdi, list)
        assert len(hdi) == 2

    def test_hdi_given_one_proportion(self, rule):
        data = np.array([1, 2, 3, 4, 5])
        hdi = rule._compute_high_density_interval(data, proportion=1.0)

        assert hdi[0] == 1.0
        assert hdi[1] == 5.0

    def test_skips_undetected_variables(self, rule):
        context = ValidationContext(
            layer_name="test",
            detected_variables={"power_kw": "Leistung", "wind_speed": None, "temperature": None},
            turbine_id="WEA_01",
        )
        df = pd.DataFrame({"Leistung": [1000, 1500, 2000]})
        result = rule.execute(df, context)

        assert "wind_speed" in result.details["skipped"]
        assert "temperature" in result.details["skipped"]
        assert "power_kw" in result.details["checked"].keys()

    def test_returns_warning_when_some_variables_skipped(self, rule):
        context = ValidationContext(
            layer_name="test",
            detected_variables={"power_kw": "Leistung", "wind_speed": None, "temperature": None},
            turbine_id="WEA_01",
        )
        df = pd.DataFrame({"Leistung": [1000, 1500, 2000]})
        result = rule.execute(df, context)

        assert result.status == Status.WARNING

    def test_returns_passed_when_no_variables_skipped(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [1000, 1500, 2000], "ws_gondel": [5, 10, 15], "Temp_Gondel": [10, 15, 20]})
        result = rule.execute(df, context_with_variables)

        assert result.status == Status.PASSED
        assert len(result.details["skipped"]) == 0

    def test_returns_full_points_when_all_checked(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [1000, 1500, 2000], "ws_gondel": [5, 10, 15], "Temp_Gondel": [10, 15, 20]})
        result = rule.execute(df, context_with_variables)

        assert result.points_achieved == rule.points

    def test_returns_reduced_points_when_some_skipped(self, rule):
        context = ValidationContext(
            layer_name="test",
            detected_variables={"power_kw": "Leistung", "wind_speed": None, "temperature": None},
            turbine_id="WEA_01",
        )
        df = pd.DataFrame({"Leistung": [1000, 1500, 2000]})
        result = rule.execute(df, context)

        expected_points = int(rule.points * 1 / 3)
        assert result.points_achieved == expected_points

    def test_returns_reduced_points_when_multiple_checked(self, rule):
        context = ValidationContext(
            layer_name="test",
            detected_variables={"power_kw": "Leistung", "wind_speed": "ws", "temperature": None},
            turbine_id="WEA_01",
        )
        df = pd.DataFrame({"Leistung": [1000, 1500, 2000], "ws": [5, 10, 15]})
        result = rule.execute(df, context)

        expected_points = int(rule.points * 2 / 3)
        assert result.points_achieved == expected_points

    def test_message_when_all_checked(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [1000, 1500, 2000], "ws_gondel": [5, 10, 15], "Temp_Gondel": [10, 15, 20]})
        result = rule.execute(df, context_with_variables)

        assert "all" in result.message.lower()
        assert "essential ranges" in result.message.lower()

    def test_message_when_some_skipped(self, rule):
        context = ValidationContext(
            layer_name="test",
            detected_variables={"power_kw": "Leistung", "wind_speed": None, "temperature": None},
            turbine_id="WEA_01",
        )
        df = pd.DataFrame({"Leistung": [1000, 1500, 2000]})
        result = rule.execute(df, context)

        assert "1" in result.message
        assert "3" in result.message

    def test_result_includes_checked_dict(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [1000], "ws_gondel": [5], "Temp_Gondel": [10]})
        result = rule.execute(df, context_with_variables)

        assert "checked" in result.details
        assert isinstance(result.details["checked"], dict)

    def test_result_includes_skipped_list(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [1000], "ws_gondel": [5], "Temp_Gondel": [10]})
        result = rule.execute(df, context_with_variables)

        assert "skipped" in result.details
        assert isinstance(result.details["skipped"], list)

    def test_checked_keys_are_variable_names(self, rule, context_with_variables):
        df = pd.DataFrame({"Leistung": [1000], "ws_gondel": [5], "Temp_Gondel": [10]})
        result = rule.execute(df, context_with_variables)

        assert "power_kw" in result.details["checked"]
        assert "wind_speed" in result.details["checked"]
        assert "temperature" in result.details["checked"]

    def test_accepts_custom_proportion(self):
        rule = EssentialRanges(variable_names=["power_kw"], proportion=0.99)

        assert rule.proportion == 0.99

    def test_accepts_single_variable(self):
        rule = EssentialRanges(variable_names=["power_kw"])
        context = ValidationContext(layer_name="test", detected_variables={"power_kw": "Power"}, turbine_id="WEA_01")
        df = pd.DataFrame({"Power": [1000, 1500, 2000]})
        result = rule.execute(df, context)

        assert result.status == Status.PASSED
        assert len(result.details["checked"]) == 1

    def test_accepts_many_variables(self):
        rule = EssentialRanges(variable_names=[f"var_{i}" for i in range(10)])

        assert len(rule.variable_names) == 10

    def test_handles_empty_variable_names(self):
        rule = EssentialRanges(variable_names=[])
        context = ValidationContext(layer_name="test", detected_variables={}, turbine_id="WEA_01")
        df = pd.DataFrame({"col": [1, 2, 3]})
        result = rule.execute(df, context)

        assert result.status == Status.PASSED

    def test_severity_is_info(self, rule):
        assert rule.severity == Severity.INFO

    def test_accepts_custom_points(self):
        rule = EssentialRanges(variable_names=["power_kw"], points=20)

        assert rule.points == 20
