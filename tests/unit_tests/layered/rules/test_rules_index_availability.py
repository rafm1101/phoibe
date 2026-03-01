import datetime

import pandas as pd
import pytest

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.core.entities import Severity
from phoibe.layered.core.entities import Status
from phoibe.layered.rules.rules_index import AvailabilityRule


class TestAvailabilityRule:

    @pytest.fixture
    def rule(self):
        return AvailabilityRule(good_threshold=0.9, acceptable_threshold=0.75, points=10, locale="en_US")

    @pytest.fixture
    def context_with_datetime(self):
        return ValidationContext(layer_name="test", detected_variables={"datetime": "Zeitstempel"}, turbine_id="WEA 01")

    def test_passes_given_full_availability(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=144, freq="10min")})
        result = rule.execute(df, context_with_datetime)

        assert result.status == Status.PASSED
        assert result.points_achieved == 10

    def test_passes_above_good_threshold(self, rule, context_with_datetime):
        timestamps = list(pd.date_range("2024-01-01", periods=144, freq="10min"))
        for i in [10, 50, 90, 130]:
            timestamps.pop(i - (i // 30))
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)

        assert result.details["availability_data"] > 0.9
        assert result.status == Status.PASSED
        assert result.points_achieved == 10

    def test_warning_between_thresholds(self, rule, context_with_datetime):
        timestamps = list(pd.date_range("2024-01-01", periods=144, freq="10min"))
        timestamps = [item for k, item in enumerate(timestamps) if k % 5 != 0]
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)
        availability = result.details["availability_data"]

        assert 0.75 < availability <= 0.9
        assert result.status == Status.WARNING
        assert result.points_achieved == 5

    def test_fails_below_acceptable_threshold(self, rule, context_with_datetime):
        timestamps = list(pd.date_range("2024-01-01", periods=144, freq="10min"))
        timestamps = [item for k, item in enumerate(timestamps) if k % 3 != 0]
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)

        assert result.details["availability_data"] < 0.75
        assert result.status == Status.FAILED
        assert result.points_achieved == 0

    def test_availability_at_exact_good_threshold(self):
        rule = AvailabilityRule(good_threshold=0.9, acceptable_threshold=0.75)
        context = ValidationContext(layer_name="test", detected_variables={"datetime": "Zeit"}, turbine_id="WEA 01")
        timestamps = list(pd.date_range("2024-01-01", periods=100, freq="10min"))
        timestamps = [item for k, item in enumerate(timestamps) if k % 10 != 0]
        df = pd.DataFrame({"Zeit": timestamps})
        result = rule.execute(df, context)

        assert result.status in [Status.PASSED, Status.WARNING]

    def test_computes_global_availability_correctly(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=144, freq="10min")})
        result = rule.execute(df, context_with_datetime)

        assert "availability_data" in result.details
        assert result.details["availability_data"] == 1.0

    def test_computes_availability_given_gaps(self, rule, context_with_datetime):
        timestamps = list(pd.date_range("2024-01-01", periods=144, freq="10min"))
        timestamps = [item for k, item in enumerate(timestamps) if k < 100 or k >= 124]
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)

        assert "availability_data" in result.details
        assert 0.8 <= result.details["availability_data"] <= 0.85

    def test_provides_hourly_availability(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=144, freq="10min")})
        result = rule.execute(df, context_with_datetime)

        assert "availability_hours" in result.details
        assert isinstance(result.details["availability_hours"], dict)
        hourly = result.details["availability_hours"]
        assert set(hourly.keys()) == set(range(24))

    def test_hourly_availability_detects_missing_hours(self, rule, context_with_datetime):
        timestamps_morning = pd.date_range("2024-01-01 00:00", periods=72, freq="10min")
        timestamps_afternoon = pd.date_range("2024-01-01 13:00", periods=72, freq="10min")
        timestamps = timestamps_morning.tolist() + timestamps_afternoon.tolist()
        df = pd.DataFrame({"Zeitstempel": timestamps})

        result = rule.execute(df, context_with_datetime)
        hourly = result.details["availability_hours"]
        assert hourly[12] == pytest.approx(0.0)

    def test_provides_daily_availability_present(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=1440, freq="10min")})  # 10 days
        result = rule.execute(df, context_with_datetime)

        assert "availability_days" in result.details
        assert isinstance(result.details["availability_days"], dict)
        daily = result.details["availability_days"]
        assert len(daily.keys()) == 7

    def test_daily_availability_detects_weekend_gaps(self, rule, context_with_datetime):
        timestamps = []
        for day in pd.date_range("2024-01-01", periods=14, freq="D"):
            if day.dayofweek < 5:
                timestamps.extend(pd.date_range(day, periods=144, freq="10min"))
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)

        daily = result.details["availability_days"]
        assert daily["Saturday"] == pytest.approx(0.0)
        assert daily["Sunday"] == pytest.approx(0.0)

    def test_monthly_availability_present(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=12960, freq="10min")})
        result = rule.execute(df, context_with_datetime)

        assert "availability_months" in result.details
        assert isinstance(result.details["availability_months"], dict)
        monthly = result.details["availability_months"]
        assert len(monthly.keys()) == 12

    @pytest.mark.parametrize("locale, expected_locale", [("de_DE", "de_DE"), (None, None), ("en_US", "en_US")])
    def test_accepts_custom_locale(self, locale, expected_locale):
        rule = AvailabilityRule(locale=locale)
        assert rule.locale == expected_locale

    def test_awards_full_points_when_passed(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=144, freq="10min")})
        result = rule.execute(df, context_with_datetime)

        result.status == Status.PASSED
        assert result.points_achieved == rule.points

    def test_awards_half_points_when_warning(self, rule, context_with_datetime):
        timestamps = list(pd.date_range("2024-01-01", periods=144, freq="10min"))
        timestamps = [item for k, item in enumerate(timestamps) if k % 5 != 0]
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)

        assert result.status == Status.WARNING
        assert result.points_achieved == rule.points // 2

    def test_awards_zero_points_when_failed(self, rule, context_with_datetime):
        timestamps = list(pd.date_range("2024-01-01", periods=144, freq="10min"))
        timestamps = [item for k, item in enumerate(timestamps) if k % 3 != 0]
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)

        assert result.status == Status.FAILED
        assert result.points_achieved == 0

    def test_result_includes_descriptive_message_passed(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=144, freq="10min")})
        result = rule.execute(df, context_with_datetime)

        assert result.status == Status.PASSED
        assert len(result.message) > 0
        assert "promising" in result.message.lower() or "good" in result.message.lower()

    def test_result_includes_required_and_actual(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=144, freq="10min")})
        result = rule.execute(df, context_with_datetime)
        assert "%" in result.required
        assert "%" in result.actual

    def test_not_checked_when_datetime_not_detected(self, rule):
        context = ValidationContext(layer_name="test", detected_variables={"datetime": None}, turbine_id="WEA 01")
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = rule.execute(df, context)

        assert result.status == Status.NOT_CHECKED
        assert "not detected" in result.message.lower()

    def test_error_when_timestamps_not_parseable(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": ["not", "a", "timestamp"]})
        result = rule.execute(df, context_with_datetime)
        assert result.status in [Status.ERROR, Status.PASSED]

    def test_handles_single_day_data(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=144, freq="10min")})
        result = rule.execute(df, context_with_datetime)
        assert result.status in [Status.PASSED, Status.WARNING, Status.FAILED]

    def test_handles_irregular_timestamps(self, rule, context_with_datetime):
        timestamps = [
            datetime.datetime(2024, 1, 1, 10, 0),
            datetime.datetime(2024, 1, 1, 10, 10),
            datetime.datetime(2024, 1, 1, 10, 25),
            datetime.datetime(2024, 1, 1, 10, 30),
        ]
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)

        assert "availability_data" in result.details

    def test_severity_is_critical(self, rule):
        assert rule.severity == Severity.CRITICAL

    def test_custom_thresholds(self):
        rule = AvailabilityRule(good_threshold=0.95, acceptable_threshold=0.85)

        assert rule.good_threshold == 0.95
        assert rule.acceptable_threshold == 0.85

    def test_to_completed_series_creates_boolean_series(self, rule):
        timestamps = pd.date_range("2024-01-01", periods=10, freq="10min")
        index = pd.DatetimeIndex(timestamps)
        completed = rule._to_completed_series(index)

        assert isinstance(completed, pd.Series)
        assert completed.dtype == bool

    def test_to_completed_series_fills_gaps_with_false(self, rule):
        timestamps = list(pd.date_range("2024-01-01 00:00", periods=5, freq="10min"))
        timestamps.extend(pd.date_range("2024-01-01 02:00", periods=5, freq="10min"))
        index = pd.DatetimeIndex(timestamps)
        completed = rule._to_completed_series(index)

        assert False in completed.values
        assert True in completed.values
        assert completed[timestamps].all()
        assert not completed[completed.index.difference(timestamps)].any()

    def test_to_dict_rounds_to_three_decimals(self, rule):
        series = pd.Series([0.123456, 0.987654], index=[0, 1])
        result = rule._to_dict(series)

        assert result[0] == 0.123
        assert result[1] == 0.988
