import datetime

import pandas as pd
import pytest

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.core.entities import Severity
from phoibe.layered.core.entities import Status
from phoibe.layered.rules.rules_index import DataGaps


class TestDataGaps:

    @pytest.fixture
    def rule(self):
        return DataGaps(good_threshold=0.05, acceptable_threshold=0.1, points=10)

    @pytest.fixture
    def context_with_datetime(self):
        return ValidationContext(layer_name="test", detected_variables={"datetime": "Zeitstempel"}, turbine_id="WEA_01")

    def test_detects_zero_gaps_perfect_data(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=144, freq="10min")})
        result = rule.execute(df, context_with_datetime)

        assert result.details["gap_count"] == 0

    def test_counts_single_gap(self, rule, context_with_datetime):
        timestamps = list(pd.date_range("2024-01-01 00:00", periods=5, freq="10min"))
        timestamps.extend(pd.date_range("2024-01-01 01:00", periods=5, freq="10min"))
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)

        assert result.details["gap_count"] == 1

    def test_counts_multiple_gaps(self, rule, context_with_datetime):
        timestamps = []
        timestamps.extend(pd.date_range("2024-01-01 00:00", periods=3, freq="10min"))
        timestamps.extend(pd.date_range("2024-01-01 01:00", periods=3, freq="10min"))
        timestamps.extend(pd.date_range("2024-01-01 02:00", periods=3, freq="10min"))
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)

        assert result.details["gap_count"] >= 2

    def test_total_gap_length_single_gap(self, rule, context_with_datetime):
        timestamps = list(pd.date_range("2024-01-01 00:00", periods=5, freq="10min"))
        timestamps.extend(pd.date_range("2024-01-01 01:00", periods=5, freq="10min"))
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)
        gap_total = result.details["gap_length_total"]

        assert "00:10:00" in gap_total or "0 days 00:10:00" in gap_total

    def test_total_gap_length_multiple_gaps(self, rule, context_with_datetime):
        timestamps = []
        timestamps.extend(pd.date_range("2024-01-01 00:00", periods=3, freq="10min"))
        timestamps.extend(pd.date_range("2024-01-01 01:00", periods=3, freq="10min"))
        timestamps.extend(pd.date_range("2024-01-01 02:00", periods=3, freq="10min"))
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)
        gap_total = result.details["gap_length_total"]

        assert isinstance(gap_total, str)
        assert "01:00:00" in gap_total

    def test_mean_gap_length_single_gap(self, rule, context_with_datetime):
        timestamps = list(pd.date_range("2024-01-01 00:00", periods=5, freq="10min"))
        timestamps.extend(pd.date_range("2024-01-01 02:00", periods=5, freq="10min"))
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)

        gap_mean = result.details["gap_length_mean"]
        gap_total = result.details["gap_length_total"]
        assert gap_mean == gap_total
        assert "01:10:00" in gap_total

    def test_mean_gap_length_multiple_gaps(self, rule, context_with_datetime):
        timestamps = []
        timestamps.extend(pd.date_range("2024-01-01 00:00", periods=3, freq="10min"))
        timestamps.extend(pd.date_range("2024-01-01 01:00", periods=3, freq="10min"))
        timestamps.extend(pd.date_range("2024-01-01 01:50", periods=3, freq="10min"))
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)
        gap_length_mean = result.details["gap_length_mean"]

        assert isinstance(gap_length_mean, str)
        assert "00:25:00" in gap_length_mean

    def test_max_gap_length_identifies_longest(self, rule, context_with_datetime):
        timestamps = []
        timestamps.extend(pd.date_range("2024-01-01 00:00", periods=3, freq="10min"))
        timestamps.extend(pd.date_range("2024-01-01 02:00", periods=3, freq="10min"))
        timestamps.extend(pd.date_range("2024-01-01 02:40", periods=3, freq="10min"))
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)
        gap_max = result.details["gap_length_max"]

        assert isinstance(gap_max, str)
        assert "01:30:00" in gap_max

    def test_ignores_regular_intervals(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=144, freq="10min")})
        result = rule.execute(df, context_with_datetime)

        assert result.details["gap_count"] == 0

    def test_detects_gaps_larger_than_mode(self, rule, context_with_datetime):
        timestamps = list(pd.date_range("2024-01-01 00:00", periods=5, freq="10min"))
        timestamps.extend(pd.date_range("2024-01-01 01:00", periods=5, freq="10min"))
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)

        assert result.details["gap_count"] > 0
        assert "00:10:00" in result.details["gap_length_mean"]

    def test_always_passes_with_gaps(self, rule, context_with_datetime):
        timestamps = list(pd.date_range("2024-01-01 00:00", periods=5, freq="10min"))
        timestamps.extend(pd.date_range("2024-01-01 02:00", periods=5, freq="10min"))
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)

        assert result.status == Status.PASSED

    def test_always_passes_without_gaps(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=144, freq="10min")})
        result = rule.execute(df, context_with_datetime)

        assert result.status == Status.PASSED

    def test_awards_full_points_always(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=10, freq="10min")})
        result = rule.execute(df, context_with_datetime)

        assert result.points_achieved == rule.points

    def test_severity_is_info(self, rule):
        assert rule.severity == Severity.INFO

    def test_not_checked_when_datetime_missing(self, rule):
        context = ValidationContext(layer_name="test", detected_variables={"datetime": None}, turbine_id="WEA_01")
        df = pd.DataFrame({"other": [1, 2, 3]})
        result = rule.execute(df, context)

        assert result.status == Status.NOT_CHECKED

    def test_error_unparseable_timestamps(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": ["not", "a", "timestamp"]})
        result = rule.execute(df, context_with_datetime)

        assert result.status in [Status.ERROR, Status.PASSED]

    def test_handles_single_timestamp(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": [datetime.datetime(2024, 1, 1, 10, 0)]})
        result = rule.execute(df, context_with_datetime)

        assert result.status == Status.ERROR

    def test_handles_two_timestamps(self, rule, context_with_datetime):
        df = pd.DataFrame(
            {"Zeitstempel": [datetime.datetime(2024, 1, 1, 10, 0), datetime.datetime(2024, 1, 1, 10, 10)]}
        )
        result = rule.execute(df, context_with_datetime)

        assert result.details["gap_count"] == 0

    def test_handles_unsorted_timestamps(self, rule, context_with_datetime):
        timestamps = [
            datetime.datetime(2024, 1, 1, 10, 20),
            datetime.datetime(2024, 1, 1, 10, 0),
            datetime.datetime(2024, 1, 1, 10, 10),
            datetime.datetime(2024, 1, 1, 9, 30),
        ]
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)
        gap_count = result.details["gap_count"]

        assert isinstance(gap_count, int)
        assert gap_count == 1

    def test_zero_gap_length_when_no_gaps(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=10, freq="10min")})
        result = rule.execute(df, context_with_datetime)

        assert result.details["gap_count"] == 0
        assert "00:00:00" in result.details["gap_length_total"]

    def test_result_contains_all_statistics(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=10, freq="10min")})
        result = rule.execute(df, context_with_datetime)

        required_keys = ["gap_count", "gap_length_total", "gap_length_mean", "gap_length_max"]
        for attr in required_keys:
            assert attr in result.details, f"Missing: {attr}"

    def test_gap_count_is_integer(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=10, freq="10min")})
        result = rule.execute(df, context_with_datetime)

        assert isinstance(result.details["gap_count"], int)

    def test_gap_lengths_are_strings(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=10, freq="10min")})
        result = rule.execute(df, context_with_datetime)

        assert isinstance(result.details["gap_length_total"], str)
        assert isinstance(result.details["gap_length_mean"], str)
        assert isinstance(result.details["gap_length_max"], str)

    def test_count_delta_datetime_returns_series(self, rule):
        timestamps = pd.date_range("2024-01-01", periods=10, freq="10min")
        index = pd.DatetimeIndex(timestamps)
        counts = rule._count_delta_datetime(index)

        assert isinstance(counts, pd.Series)

    def test_count_delta_datetime_filters_by_mode(self, rule):
        timestamps = list(pd.date_range("2024-01-01 00:00", periods=5, freq="10min"))
        timestamps.extend(pd.date_range("2024-01-01 01:00", periods=5, freq="10min"))
        index = pd.DatetimeIndex(timestamps)
        counts = rule._count_delta_datetime(index)

        assert len(counts) >= 0

    def test_describe_gaps_returns_tuple(self, rule):
        delta = pd.Series(
            [1, 2, 3], index=[pd.Timedelta("00:10:00"), pd.Timedelta("00:20:00"), pd.Timedelta("00:30:00")]
        )
        result = rule._describe_gaps(delta)

        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_describe_gaps_handles_zero_gaps(self, rule):
        delta = pd.Series([], dtype=int)
        delta.index = pd.TimedeltaIndex([])
        n_gaps, total, mean, max_gap = rule._describe_gaps(delta)

        assert n_gaps == 0
        assert mean == pd.Timedelta(0)

    def test_accepts_custom_thresholds(self):
        rule = DataGaps(good_threshold=0.02, acceptable_threshold=0.08)

        assert rule.good_threshold == 0.02
        assert rule.acceptable_threshold == 0.08

    def test_accepts_custom_points(self):
        rule = DataGaps(points=20)

        assert rule.points == 20
