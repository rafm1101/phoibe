import logging

import pandas as pd

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.application.factory import RuleRegistry
from phoibe.layered.core.entities import RuleExecutionResult
from phoibe.layered.core.entities import Severity
from phoibe.layered.rules.rule import ValidationRule

# TODO: Document properly, refactor `TemporalResolutionRule`, `DataGapRule`.


@RuleRegistry.register("temporal_attributes")
class TemporalAttributes(ValidationRule):
    """Extract various information from timestamps."""

    def __init__(
        self,
        points: int = 10,
        severity: Severity = Severity.INFO,
        logger: logging.Logger | None = None,
    ):
        super().__init__(points, severity, logger)

    @property
    def name(self):
        return "temporal_attributes"

    def execute(self, df: pd.DataFrame, context: ValidationContext):
        """Verify the temporal properties.

        Parameters
        ----------
        df
            Dataframe with a pandas datetime index.
        context
            Validation context.

        Returns
        -------
        RuleExecutionResult
            PASSED.
        """
        datetime_key = context.get_column_key("datetime")
        if datetime_key is None:
            return self.result_builder.not_checked("Datetime variable not detected.")
        try:
            datetimes = pd.to_datetime(df[datetime_key], errors="coerce")
            datetime_column = pd.DatetimeIndex(datetimes)
        except Exception as e:
            return self.result_builder.error("Failed to parse timestamps", exception=e)

        start = str(datetime_column.min().isoformat())
        end = str(datetime_column.max().isoformat())
        tzinfo = str(datetime_column.tzinfo)

        has_duplicates = datetime_column.has_duplicates
        delta_datetime_column = datetime_column.to_series().diff()
        is_sorted = not bool((delta_datetime_column.dropna() < pd.to_timedelta(0)).any())

        result = {
            "start": start,
            "end": end,
            "tzinfo": tzinfo,
            "has_duplicates": has_duplicates,
            "is_sorted": is_sorted,
        }
        return self.result_builder.passed(required="", actual="", message="", details=result)


class TemporalResolutionRule(ValidationRule):
    """Validate a temporal resolution of time series data.

    Heuristic: Verify that the mode of the delta timestep counts is the expected resolution.
    """

    def __init__(
        self,
        expected_minutes: int,
        tolerance_seconds: int = 60,
        points: int = 10,
        severity: Severity = Severity.CRITICAL,
        logger: logging.Logger | None = None,
    ):
        super().__init__(points, severity, logger)
        self.expected_minutes = expected_minutes
        self.tolerance_seconds = tolerance_seconds

    @property
    def name(self) -> str:
        return "temporal_resolution"

    def execute(self, df: pd.DataFrame, context: ValidationContext) -> RuleExecutionResult:
        """Verify the temporal resolution.

        Parameters
        ----------
        df
            Dataframe with a pandas datetime index.
        context
            Validation context.

        Returns
        -------
        RuleExecutionResult
            PASSED if resolution matches, FAILED if not, NOT_CHECKED if timestamp missing
        """
        # 1. Check if timestamp detected
        timestamp_col = context.get_column_key("timestamp")
        if timestamp_col is None:
            return self.result_builder.not_checked("Timestamp signal not detected")

        # 2. Try to parse timestamps
        try:
            df_time = pd.to_datetime(df[timestamp_col], errors="coerce")
        except Exception as e:
            return self.result_builder.error("Failed to parse timestamps", exception=e)

        # 3. Calculate time differences
        time_diffs = df_time.diff().dropna()

        if len(time_diffs) == 0:
            return self.result_builder.error("No time differences found (single row or all NaT)")

        # 4. Core validation logic (pure function)
        median_seconds, is_valid = self._check_resolution(time_diffs, self.expected_minutes, self.tolerance_seconds)

        # 5. Build result
        required = f"{self.expected_minutes}min"
        actual = f"{median_seconds / 60:.1f}min"

        if is_valid:
            return self.result_builder.passed(
                required=required,
                actual=actual,
                message=f"Resolution matches expected {required}",
                details={
                    "median_seconds": median_seconds,
                    "expected_seconds": self.expected_minutes * 60,
                    "tolerance_seconds": self.tolerance_seconds,
                },
            )
        else:
            return self.result_builder.failed(
                required=required,
                actual=actual,
                message=f"Resolution {actual} outside tolerance of {required}",
                details={
                    "median_seconds": median_seconds,
                    "expected_seconds": self.expected_minutes * 60,
                    "tolerance_seconds": self.tolerance_seconds,
                    "deviation_seconds": abs(median_seconds - self.expected_minutes * 60),
                },
            )

    def _check_resolution(
        self, time_diffs: pd.Series, expected_minutes: int, tolerance_seconds: int
    ) -> tuple[float, bool]:
        """
        Pure function: Check if temporal resolution matches expected.

        Parameters
        ----------
            time_diffs: Series of time differences (timedelta)
            expected_minutes: Expected resolution in minutes
            tolerance_seconds: Tolerance in seconds

        Returns
        -------
            Tuple of (median_seconds, is_valid)
        """
        median_diff = time_diffs.median()
        median_seconds = median_diff.total_seconds()
        expected_seconds = expected_minutes * 60

        deviation = abs(median_seconds - expected_seconds)
        is_valid = deviation <= tolerance_seconds

        return median_seconds, is_valid


class DataGapRule(ValidationRule):
    """Validate data completeness by addressing gaps.

    Calculates the percentage of missing data points based on
    expected temporal resolution.

    Thresholds:
    - < good_threshold: PASSED (full points)
    - < acceptable_threshold: WARNING (partial points)
    - >= acceptable_threshold: FAILED (zero points)
    """

    def __init__(
        self,
        expected_resolution_minutes: int,
        good_threshold: float = 0.01,
        acceptable_threshold: float = 0.05,
        points: int = 10,
        logger: logging.Logger | None = None,
    ):
        super().__init__(points, Severity.CRITICAL, logger)
        self.expected_resolution_minutes = expected_resolution_minutes
        self.good_threshold = good_threshold
        self.acceptable_threshold = acceptable_threshold

    @property
    def name(self) -> str:
        return "data_gaps"

    def execute(self, df: pd.DataFrame, context: ValidationContext) -> RuleExecutionResult:
        """Assess data completeness.

        Parameters
        ----------
        df
            DataFrame with timestamp column
        context
            Validation context

        Returns
        -------
        RuleExecutionResult
            PASSED/WARNING/FAILED based on gap percentage
        """
        # 1. Check if timestamp detected
        timestamp_col = context.get_column_key("timestamp")
        if timestamp_col is None:
            return self.result_builder.not_checked("Timestamp signal not detected")

        # 2. Try to parse timestamps
        try:
            df_time = pd.to_datetime(df[timestamp_col], errors="coerce")
        except Exception as e:
            return self.result_builder.error("Failed to parse timestamps", exception=e)

        # 3. Check if we have data
        if len(df_time) < 2:
            return self.result_builder.error("Insufficient data (less than 2 rows)")

        # 4. Core gap calculation (pure function)
        gap_percentage, expected_count, actual_count = self._calculate_gaps(df_time, self.expected_resolution_minutes)

        # 5. Build result based on threshold
        required = f"<{self.good_threshold * 100:.1f}%"
        actual = f"{gap_percentage * 100:.2f}%"
        details = {
            "gap_percentage": gap_percentage,
            "expected_count": expected_count,
            "actual_count": actual_count,
            "missing_count": expected_count - actual_count,
        }

        if gap_percentage < self.good_threshold:
            # Excellent - full points
            return self.result_builder.passed(
                required=required,
                actual=actual,
                message=f"Data completeness excellent ({actual} gaps)",
                details=details,
            )
        elif gap_percentage < self.acceptable_threshold:
            # Acceptable - partial points (WARNING)
            from phoibe.layered.core.entities import Status

            partial_points = int(self.points * 0.5)

            return RuleExecutionResult(
                rule_name=self.name,
                status=Status.WARNING,
                severity=Severity.WARNING,
                required=required,
                actual=actual,
                points_max=self.points,
                points_achieved=partial_points,
                message=f"Data completeness acceptable ({actual} gaps)",
                details=details,
            )
        else:
            # Too many gaps - failed
            return self.result_builder.failed(
                required=required, actual=actual, message=f"Too many data gaps ({actual})", details=details
            )

    def _calculate_gaps(self, df_time: pd.Series, expected_resolution_minutes: int) -> tuple[float, int, int]:
        """
        Pure function: Calculate data gap percentage.

        Parameters
        ----------
            df_time: Series of timestamps
            expected_resolution_minutes: Expected sampling interval

        Returns
        -------
            Tuple of (gap_percentage, expected_count, actual_count)
        """
        # Remove NaT values
        df_time = df_time.dropna()

        if len(df_time) < 2:
            return 1.0, 0, 0

        # Calculate expected number of samples
        start_time = df_time.min()
        end_time = df_time.max()
        time_span = (end_time - start_time).total_seconds()
        expected_count = int(time_span / (expected_resolution_minutes * 60)) + 1

        # Actual count
        actual_count = len(df_time)

        # Gap percentage
        missing_count = expected_count - actual_count
        gap_percentage = missing_count / expected_count if expected_count > 0 else 0

        return max(0, gap_percentage), expected_count, actual_count


__all__ = ["TemporalResolutionRule", "DataGapRule"]
