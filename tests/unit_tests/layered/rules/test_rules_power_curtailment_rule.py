import pandas as pd
import pytest

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.core.entities import Severity
from phoibe.layered.core.entities import Status
from phoibe.layered.rules.rules_power import CurtailmentRule


class TestCurtailmentRule:

    @pytest.fixture
    def rule(self):
        return CurtailmentRule(wind_speed_threshold=14.0, prominence_threshold=1e-7, points=10)

    @pytest.fixture
    def context_with_power_windspeed(self):
        return ValidationContext(
            layer_name="test",
            detected_variables={"power_kw": "Leistung", "wind_speed": "ws_gondel"},
            turbine_id="WEA 01",
        )

    def test_detects_single_curtailment_given_uniform_power(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [2000] * 100 + [1500] * 50, "ws_gondel": [15] * 100 + [10] * 50})
        result = rule.execute(df, context_with_power_windspeed)

        assert result.status == Status.PASSED
        assert "n_curtailments" in result.details
        assert result.details["n_curtailments"] == 1

    def test_detects_curtailments_given_single_restriction(self, rule, context_with_power_windspeed):
        df = pd.DataFrame(
            {"Leistung": [2000] * 50 + [1500] * 50 + [1000] * 50, "ws_gondel": [15] * 50 + [15] * 50 + [10] * 50}
        )
        result = rule.execute(df, context_with_power_windspeed)

        assert result.status == Status.PASSED
        assert result.details["n_curtailments"] == 2
        assert set(result.details["power"]) == set([1500, 2000])

    def test_detects_curtailments_given_multiple_restricutions(self, rule, context_with_power_windspeed):
        df = pd.DataFrame(
            {"Leistung": [2000] * 40 + [1800] * 40 + [1600] * 40 + [1000] * 30, "ws_gondel": [15] * 120 + [10] * 30}
        )
        result = rule.execute(df, context_with_power_windspeed)

        assert result.details["n_curtailments"] >= 2
        assert set(result.details["power"]) == set([1600, 1800, 2000])

    def test_result_includes_n_curtailments(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [2000] * 100, "ws_gondel": [15] * 100})
        result = rule.execute(df, context_with_power_windspeed)

        assert "n_curtailments" in result.details
        assert isinstance(result.details["n_curtailments"], int)

    def test_result_includes_n_candidates(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [2000] * 100, "ws_gondel": [15] * 100})
        result = rule.execute(df, context_with_power_windspeed)

        assert "n_candidates_detected" in result.details
        assert isinstance(result.details["n_candidates_detected"], int)

    def test_result_includes_power_levels(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [2000] * 100, "ws_gondel": [15] * 100})
        result = rule.execute(df, context_with_power_windspeed)

        assert "power" in result.details
        assert isinstance(result.details["power"], list)

    def test_result_includes_density_heights(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [2000] * 100, "ws_gondel": [15] * 100})
        result = rule.execute(df, context_with_power_windspeed)

        assert "height" in result.details
        assert isinstance(result.details["height"], list)

    def test_result_includes_ignored_density(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [2000] * 100, "ws_gondel": [15] * 100})
        result = rule.execute(df, context_with_power_windspeed)

        assert "ignored_below" in result.details
        assert isinstance(result.details["ignored_below"], float)

    def test_power_and_height_lists_same_length(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [2000] * 50 + [1500] * 50, "ws_gondel": [15] * 100})
        result = rule.execute(df, context_with_power_windspeed)

        assert len(result.details["power"]) == len(result.details["height"])

    def test_power_list_length_matches_n_curtailments(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [2000] * 50 + [1500] * 50, "ws_gondel": [15] * 100})
        result = rule.execute(df, context_with_power_windspeed)

        assert len(result.details["power"]) == result.details["n_curtailments"]

    def test_power_values_rounded_to_10kw_multiples(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [2005] * 100, "ws_gondel": [15] * 100})
        result = rule.execute(df, context_with_power_windspeed)

        for power in result.details["power"]:
            assert power % 10 == 0

    def test_filters_low_prominence_peaks(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [2000] * 95 + list(range(1000, 1005)), "ws_gondel": [15] * 100})
        result = rule.execute(df, context_with_power_windspeed)

        assert result.details["n_curtailments"] <= result.details["n_candidates_detected"]
        assert result.details["n_curtailments"] == 2

    def test_ignores_data_below_wind_speed_threshold(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [500] * 50 + [2000] * 100, "ws_gondel": [10] * 50 + [15] * 100})

        result = rule.execute(df, context_with_power_windspeed)
        assert result.status == Status.PASSED
        assert result.details["n_curtailments"] == 1
        assert result.details["power"] == [2000]

    def test_custom_wind_speed_threshold(self):
        rule = CurtailmentRule(wind_speed_threshold=12.0)
        context = ValidationContext(
            layer_name="test", detected_variables={"power_kw": "Power", "wind_speed": "WS"}, turbine_id="WEA 01"
        )
        df = pd.DataFrame({"Power": [2000] * 100, "WS": [13] * 100})
        result = rule.execute(df, context)

        assert result.status == Status.PASSED

    def test_returns_not_checked_when_power_not_detected(self, rule):
        context = ValidationContext(
            layer_name="test", detected_variables={"power_kw": None, "wind_speed": "ws"}, turbine_id="WEA 01"
        )
        df = pd.DataFrame({"ws": [15] * 100})
        result = rule.execute(df, context)

        assert result.status == Status.NOT_CHECKED
        assert "power" in result.message.lower()

    def test_returns_not_checked_when_wind_speed_not_detected(self, rule):
        context = ValidationContext(
            layer_name="test", detected_variables={"power_kw": "power", "wind_speed": None}, turbine_id="WEA 01"
        )
        df = pd.DataFrame({"power": [2000] * 100})
        result = rule.execute(df, context)

        assert result.status == Status.NOT_CHECKED
        assert "wind speed" in result.message.lower()

    def test_returns_not_checked_when_insufficient_high_wind_data(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [2000] + [500] * 99, "ws_gondel": [15] + [10] * 99})
        result = rule.execute(df, context_with_power_windspeed)

        assert result.status == Status.NOT_CHECKED
        assert "sufficient data" in result.message.lower()

    def test_returns_not_checked_message_suggests_lower_threshold(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [2000], "ws_gondel": [15]})
        result = rule.execute(df, context_with_power_windspeed)

        assert "lower threshold" in result.message.lower()

    def test_handles_exactly_2_high_wind_points(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [2000, 2500], "ws_gondel": [15, 15]})
        result = rule.execute(df, context_with_power_windspeed)

        assert result.status == Status.PASSED

    def test_handles_wide_power_range(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": list(range(0, 3000, 10)), "ws_gondel": [15] * 300})
        result = rule.execute(df, context_with_power_windspeed)

        assert result.status == Status.PASSED

    def test_handles_narrow_power_range(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [2000 + i * 0.1 for i in range(100)], "ws_gondel": [15] * 100})
        result = rule.execute(df, context_with_power_windspeed)

        assert result.status == Status.PASSED

    def test_handles_all_identical_power_values(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [2000] * 100, "ws_gondel": [15] * 100})
        result = rule.execute(df, context_with_power_windspeed)

        assert result.status == Status.PASSED
        assert result.details["n_curtailments"] == 1

    def test_always_returns_passed(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [2000] * 100, "ws_gondel": [15] * 100})
        result = rule.execute(df, context_with_power_windspeed)

        assert result.status == Status.PASSED

    def test_severity_is_info(self, rule):
        assert rule.severity == Severity.INFO

    def test_always_awards_full_points(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [2000] * 100, "ws_gondel": [15] * 100})
        result = rule.execute(df, context_with_power_windspeed)

        assert result.points_achieved == rule.points

    def test_message_includes_curtailment_count(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [2000] * 100, "ws_gondel": [15] * 100})
        result = rule.execute(df, context_with_power_windspeed)

        assert str(result.details["n_curtailments"]) in result.message
        assert "full load" in result.message.lower()

    def test_accepts_custom_prominence_threshold(self):
        rule = CurtailmentRule(prominence_threshold=1e-6)

        assert rule.prominence_threshold == 1e-6

    def test_accepts_custom_points(self):
        rule = CurtailmentRule(points=20)

        assert rule.points == 20

    def test_handles_negative_power_values(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [-50] * 50 + [2000] * 50, "ws_gondel": [15] * 100})
        result = rule.execute(df, context_with_power_windspeed)

        assert result.status in [Status.PASSED, Status.NOT_CHECKED]
        assert set(result.details["power"]) == set([-50, 2000])

    def test_handles_zero_power_values(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [0] * 50 + [2000] * 50, "ws_gondel": [15] * 100})
        result = rule.execute(df, context_with_power_windspeed)

        assert result.status == Status.PASSED
        assert set(result.details["power"]) == set([0, 2000])

    def test_handles_float_power_values(self, rule, context_with_power_windspeed):
        df = pd.DataFrame({"Leistung": [2000.5, 2000.7, 2000.3] * 33 + [2000.0], "ws_gondel": [15] * 100})
        result = rule.execute(df, context_with_power_windspeed)

        assert result.status == Status.PASSED
        assert set(result.details["power"]) == set([2000])
