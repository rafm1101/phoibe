import datetime

import pandas as pd
import pytest

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.core.entities import Severity
from phoibe.layered.core.entities import Status
from phoibe.layered.rules.rules_index import TemporalAttributes


class TestTemporalAttributes:

    @pytest.fixture
    def rule(self):
        return TemporalAttributes(points=10, severity=Severity.INFO)

    @pytest.fixture
    def context_with_datetime(self):
        return ValidationContext(layer_name="raw", detected_variables={"datetime": "Zeitstempel"}, turbine_id="WEA 01")

    def test_extracts_start_datetime(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=5, freq="10min")})
        result = rule.execute(df, context_with_datetime)

        assert result.status == Status.WARNING
        assert "start" in result.details
        assert "2024-01-01T00:00:00" in result.details["start"]
        assert result.points_achieved == 8

    def test_extracts_end_datetime(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=5, freq="10min")})
        result = rule.execute(df, context_with_datetime)
        assert "end" in result.details
        assert "2024-01-01T00:40:00" in result.details["end"]

    def test_extracts_timezone_info(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=5, freq="10min")})
        result = rule.execute(df, context_with_datetime)
        assert "tzinfo" in result.details
        assert result.details["tzinfo"] == "None"

    def test_extracts_timezone_aware_info(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=5, freq="10min", tz="UTC")})
        result = rule.execute(df, context_with_datetime)
        assert "UTC" in result.details["tzinfo"]

    def test_detects_no_duplicates(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=5, freq="10min")})
        result = rule.execute(df, context_with_datetime)
        assert "has_duplicates" in result.details
        assert result.details["has_duplicates"] is False

    def test_detects_duplicates(self, rule, context_with_datetime):
        timestamps = [
            datetime.datetime(2024, 1, 1, 10, 0),
            datetime.datetime(2024, 1, 1, 10, 10),
            datetime.datetime(2024, 1, 1, 10, 10),
            datetime.datetime(2024, 1, 1, 10, 20),
        ]
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)
        assert result.details["has_duplicates"] is True

    def test_detects_multiple_duplicates(self, rule, context_with_datetime):
        timestamps = [
            datetime.datetime(2024, 1, 1, 10, 0),
            datetime.datetime(2024, 1, 1, 10, 0),  # Duplicate
            datetime.datetime(2024, 1, 1, 10, 10),
            datetime.datetime(2024, 1, 1, 10, 10),  # Duplicate
        ]
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)
        assert result.details["has_duplicates"] is True

    def test_detects_sorted_timestamps(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=5, freq="10min")})
        result = rule.execute(df, context_with_datetime)
        assert "is_sorted" in result.details
        assert result.details["is_sorted"] is True

    def test_detects_unsorted_timestamps(self, rule, context_with_datetime):
        timestamps = [
            datetime.datetime(2024, 1, 1, 10, 20),
            datetime.datetime(2024, 1, 1, 10, 0),
            datetime.datetime(2024, 1, 1, 10, 30),
        ]
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)
        assert result.details["is_sorted"] is False

    def test_detects_reverse_sorted(self, rule, context_with_datetime):
        timestamps = [
            datetime.datetime(2024, 1, 1, 10, 30),
            datetime.datetime(2024, 1, 1, 10, 20),
            datetime.datetime(2024, 1, 1, 10, 10),
            datetime.datetime(2024, 1, 1, 10, 0),
        ]
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)
        assert result.details["is_sorted"] is False

    def test_sorted_with_duplicates_considered_sorted(self, rule, context_with_datetime):
        timestamps = [
            datetime.datetime(2024, 1, 1, 10, 0),
            datetime.datetime(2024, 1, 1, 10, 10),
            datetime.datetime(2024, 1, 1, 10, 10),
            datetime.datetime(2024, 1, 1, 10, 20),
        ]
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)
        assert result.details["is_sorted"] is True

    def test_not_checked_when_datetime_not_detected(self, rule):
        context = ValidationContext(layer_name="raw", detected_variables={"datetime": None}, turbine_id="WEA 01")
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = rule.execute(df, context)
        assert result.status == Status.NOT_CHECKED
        assert "not detected" in result.message.lower()

    def test_return_warning_given_column_not_parseable(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": ["not", "a", "datetime"]})
        result = rule.execute(df, context_with_datetime)
        assert result.status == Status.WARNING

    def test_return_warning_given_all_nat(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.to_datetime(["invalid"] * 5, errors="coerce")})
        result = rule.execute(df, context_with_datetime)
        assert result.status == Status.WARNING

    def test_handles_single_timestamp(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": [datetime.datetime(2024, 1, 1, 10, 0)]})
        result = rule.execute(df, context_with_datetime)
        assert result.status == Status.WARNING
        assert result.details["start"] == result.details["end"]
        assert result.details["has_duplicates"] is False
        assert result.details["is_sorted"] is True

    def test_handles_two_timestamps(self, rule, context_with_datetime):
        df = pd.DataFrame(
            {
                "Zeitstempel": [
                    datetime.datetime(2024, 1, 1, 10, 0, tzinfo=datetime.timezone.utc),
                    datetime.datetime(2024, 1, 1, 10, 10, tzinfo=datetime.timezone.utc),
                ]
            }
        )
        result = rule.execute(df, context_with_datetime)

        assert result.status == Status.PASSED
        assert result.details["is_sorted"] is True
        assert result.details["has_duplicates"] is False

    def test_handles_empty_dataframe(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": []})
        result = rule.execute(df, context_with_datetime)
        assert result.status == Status.WARNING

    def test_handles_nat_values_mixed_with_valid(self, rule, context_with_datetime):
        df = pd.DataFrame(
            {
                "Zeitstempel": pd.to_datetime(
                    [
                        "2024-01-01 10:00",
                        "invalid",
                        "2024-01-01 10:20",
                        "also invalid",
                        "2024-01-01 10:30",
                        "2024-01-01 10:40",
                    ],
                    errors="coerce",
                    utc=True,
                )
            }
        )
        result = rule.execute(df, context_with_datetime)
        assert result.status == Status.PASSED

    def test_handles_different_datetime_formats(self, rule, context_with_datetime):
        df = pd.DataFrame(
            {
                "Zeitstempel": [
                    "2024-01-01+00:00",
                    "2024-01-01 10:00:00+00:00",
                    "01/01/2024+00:00",
                    "2024-01-01T10:30:00+00:00",
                ]
            }
        )
        result = rule.execute(df, context_with_datetime)
        assert result.status == Status.PASSED

    def test_handles_mixed_timezones(self, rule, context_with_datetime):
        utc = datetime.timezone.utc
        timestamps = [
            datetime.datetime(2024, 1, 1, 10, 0, tzinfo=utc),
            datetime.datetime(2024, 1, 1, 11, 0, tzinfo=utc),
        ]
        df = pd.DataFrame({"Zeitstempel": timestamps})
        result = rule.execute(df, context_with_datetime)
        assert result.status == Status.PASSED
        assert "UTC" in result.details["tzinfo"]

    def test_naive_datetime_has_none_tzinfo(self, rule, context_with_datetime):
        df = pd.DataFrame(
            {
                "Zeitstempel": [
                    datetime.datetime(2024, 1, 1, 10, 0),
                    datetime.datetime(2024, 1, 1, 10, 10),
                ]
            }
        )
        result = rule.execute(df, context_with_datetime)
        assert result.details["tzinfo"] == "None"

    def test_start_end_in_iso_format(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-15 14:30:45", periods=3, freq="1h")})
        result = rule.execute(df, context_with_datetime)
        assert "T" in result.details["start"]
        assert "T" in result.details["end"]
        assert result.details["start"].startswith("2024-01-15T14:30:45")

    def test_always_passes_with_valid_data(self, rule, context_with_datetime):
        df1 = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=5, freq="10min", tz="utc")})
        utc = datetime.timezone.utc
        assert rule.execute(df1, context_with_datetime).status == Status.PASSED
        df2 = pd.DataFrame(
            {
                "Zeitstempel": [
                    datetime.datetime(2024, 1, 1, 10, 20, tzinfo=utc),
                    datetime.datetime(2024, 1, 1, 10, 0, tzinfo=utc),
                ]
            }
        )
        assert rule.execute(df2, context_with_datetime).status == Status.WARNING
        df3 = pd.DataFrame(
            {
                "Zeitstempel": [
                    datetime.datetime(2024, 1, 1, 10, 0, tzinfo=utc),
                    datetime.datetime(2024, 1, 1, 10, 0, tzinfo=utc),
                ]
            }
        )
        assert rule.execute(df3, context_with_datetime).status == Status.WARNING

    def test_severity_is_info(self, rule):
        assert rule.severity == Severity.INFO

    def test_always_awards_full_points(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=3, freq="10min", tz="utc")})
        result = rule.execute(df, context_with_datetime)
        assert result.points_achieved == rule.points

    def test_result_contains_all_required_attributes(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=5, freq="10min")})
        result = rule.execute(df, context_with_datetime)
        required_attrs = ["start", "end", "tzinfo", "has_duplicates", "is_sorted"]
        for attr in required_attrs:
            assert attr in result.details, f"Missing attribute: {attr}"

    def test_all_attributes_have_correct_types(self, rule, context_with_datetime):
        df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=5, freq="10min")})
        result = rule.execute(df, context_with_datetime)
        assert isinstance(result.details["start"], str)
        assert isinstance(result.details["end"], str)
        assert isinstance(result.details["tzinfo"], str)
        assert isinstance(result.details["has_duplicates"], bool)
        assert isinstance(result.details["is_sorted"], bool)
