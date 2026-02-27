import logging

import pandas as pd

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.application.factory import RuleRegistry
from phoibe.layered.core.entities import RuleExecutionResult
from phoibe.layered.core.entities import Severity
from phoibe.layered.rules.rule import ValidationRule

# TODO: Document properly, refactor `TemporalResolutionRule`.


@RuleRegistry.register("temporal_attributes")
class TemporalAttributes(ValidationRule):
    """Extract various information from timestamps.

    Extracted information includes:
    - Period: Start and end.
    - Timezone.
    - Duplicates.
    - Sorted.
    - Frequency (as most common timedelta).
    - Timedeltas shorter than the frequency.
    """

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

        points = self.points
        start = str(datetime_column.min().isoformat())
        end = str(datetime_column.max().isoformat())
        tzinfo = datetime_column.tzinfo
        n_dates = len(df)
        points = points - 2 if tzinfo is None else points

        has_duplicates = datetime_column.has_duplicates
        delta_datetime_column = datetime_column.to_series().diff()
        is_sorted = not bool((delta_datetime_column.dropna() < pd.to_timedelta(0)).any())
        points = points - 2 if has_duplicates or not is_sorted else points

        delta_datetime_counts = self._count_delta_datetime(datetime_column)
        mode = delta_datetime_counts.index[int(delta_datetime_counts.argmax())]
        outlier = delta_datetime_counts.index[delta_datetime_counts.index < mode]
        points = points - 2 if len(outlier) > 0 else points

        details = {
            "start": start,
            "end": end,
            "tzinfo": str(tzinfo),
            "n_dates": n_dates,
            "has_duplicates": has_duplicates,
            "is_sorted": is_sorted,
            "frequency": str(mode),
            "oversampling": [str(item) for item in outlier],
        }
        if points == self.points:
            return self.result_builder.passed(required="", actual="", message="", details=details)
        else:
            return self.result_builder.warning(required="", actual="", points=points, message="", details=details)

    def _count_delta_datetime(self, index: pd.DatetimeIndex) -> pd.Series:
        """Determine count statistics of delta datetimes."""
        index_sorted = index.sort_values(ascending=True)
        delta_index = index_sorted.to_series().diff()
        delta_index_counts = delta_index.value_counts()
        return delta_index_counts


@RuleRegistry.register("data_gaps")
class DataGaps(ValidationRule):
    """Validate data completeness by addressing gaps.

    Determine:
    - Number of gaps.
    - Total length of all gaps.
    - Mean length of all gaps.
    - Length of the longest gap.
    """

    def __init__(
        self,
        good_threshold: float = 0.05,
        acceptable_threshold: float = 0.1,
        points: int = 10,
        logger: logging.Logger | None = None,
    ):
        super().__init__(points, Severity.INFO, logger)
        self.good_threshold = good_threshold
        self.acceptable_threshold = acceptable_threshold

    @property
    def name(self) -> str:
        return "data_gaps"

    def execute(self, df: pd.DataFrame, context: ValidationContext) -> RuleExecutionResult:
        """Assess data gap statistics.

        Parameters
        ----------
        df
            DataFrame with timestamp column.
        context
            Validation context including the datetime column identification.

        Returns
        -------
        RuleExecutionResult
            PASSED/NOT_CHECKED/FAILED.
        """
        datetime_key = context.get_column_key("datetime")
        if datetime_key is None:
            return self.result_builder.not_checked("Datetime variable not detected.")
        try:
            datetimes = pd.to_datetime(df[datetime_key], errors="coerce")
            datetime_column = pd.DatetimeIndex(datetimes)
        except Exception as e:
            return self.result_builder.error("Failed to parse timestamps", exception=e)

        delta_datetime_counts = self._count_delta_datetime(datetime_column)
        n_gaps, gap_length_total, gap_length_mean, gap_length_max = self._describe_gaps(delta_datetime_counts)

        details = {
            "gap_count": int(n_gaps),
            "gap_length_total": str(gap_length_total),
            "gap_length_mean": str(gap_length_mean),
            "gap_length_max": str(gap_length_max),
        }
        return self.result_builder.passed(required="", actual="", message="", details=details)

    def _count_delta_datetime(self, index: pd.DatetimeIndex) -> pd.Series:
        """Determine count statistics of delta datetimes that are larger than the estimated frequency."""
        index_sorted = index.sort_values(ascending=True)
        delta_index = index_sorted.to_series().diff()
        delta_index_counts = delta_index.value_counts()
        delta_index_counts = delta_index_counts.sort_index()
        mode = delta_index_counts.index[int(delta_index_counts.argmax())]
        delta_index_counts_conditioned = delta_index_counts[delta_index_counts.index > mode]
        return delta_index_counts_conditioned

    def _describe_gaps(self, delta: pd.Series) -> tuple[int, pd.Timedelta, pd.Timedelta, pd.Timedelta]:
        """Determine count statistics of delta datetimes that are larger than the estimated frequency."""
        n_gaps: int = delta.sum()
        gap_length_total: pd.Timedelta = (delta * delta.index).sum()
        gap_length_mean: pd.Timedelta = gap_length_total / n_gaps if n_gaps > 0 else pd.Timedelta(0)
        gap_length_max: pd.Timedelta = delta.index[-1]
        return n_gaps, gap_length_total, gap_length_mean, gap_length_max


@RuleRegistry.register("availability")
class AvailabilityRule(ValidationRule):
    """Assess the availability of timestamps globally and hourly, daily, monthly."""

    def __init__(
        self,
        good_threshold: float = 0.9,
        acceptable_threshold: float = 0.75,
        locale: str = "en_US",
        points: int = 10,
        logger: logging.Logger | None = None,
    ):
        super().__init__(points, Severity.INFO, logger)
        self.good_threshold = good_threshold
        self.acceptable_threshold = acceptable_threshold
        self.locale = locale

    @property
    def name(self) -> str:
        return "availability"

    def execute(self, df: pd.DataFrame, context: ValidationContext) -> RuleExecutionResult:
        """Assess data gap statistics.

        Parameters
        ----------
        df
            DataFrame with timestamp column.
        context
            Validation context including the datetime column identification.

        Returns
        -------
        RuleExecutionResult
            PASSED/NOT_CHECKED/FAILED.
        """
        datetime_key = context.get_column_key("datetime")
        if datetime_key is None:
            return self.result_builder.not_checked("Datetime variable not detected.")
        try:
            datetimes = pd.to_datetime(df[datetime_key], errors="coerce")
            datetime_column = pd.DatetimeIndex(datetimes)
        except Exception as e:
            return self.result_builder.error("Failed to parse timestamps", exception=e)

        availability_df = self._to_completed_series(index=datetime_column)
        availability_global = availability_df.mean()

        availability_hours, availability_days, availability_months = self._estimate_conditional_availabilities(
            df=availability_df
        )

        details = {
            "availability_data": float(round(availability_global, 3)),
            "availability_hours": self._to_dict(availability_hours),
            "availability_days": self._to_dict(availability_days),
            "availability_months": self._to_dict(availability_months),
        }
        required = f"{self.good_threshold*100:.2f}%"
        actual = f"{availability_global*100:.2f}%"

        if availability_global > self.good_threshold:
            return self.result_builder.passed(
                required=required, actual=actual, message="Data availability promising.", details=details
            )
        elif availability_global > self.acceptable_threshold:
            return self.result_builder.warning(
                required=required,
                actual=actual,
                points=int(self.points / 2),
                message="Data availability acceptable.",
                details=details,
            )
        else:
            return self.result_builder.failed(
                required=required, actual=actual, message="Data availability critical.", details=details
            )

    def _to_completed_series(self, index: pd.DatetimeIndex) -> pd.Series:
        """Generate Boolean series w/ regularised index."""
        index_sorted = index.sort_values(ascending=True)
        delta_index = index_sorted.to_series().diff()
        delta_index_counts = delta_index.value_counts()
        delta_index_counts = delta_index_counts.sort_index()

        mode = delta_index_counts.index[int(delta_index_counts.argmax())]
        start, end = index.min(), index.max()

        df = pd.Series(data=True, index=index, name="available", dtype=bool)
        df = df.reindex(index=pd.date_range(start=start, end=end, freq=mode), fill_value=False)
        return df

    def _estimate_conditional_availabilities(self, df: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Estimate the availabilities given hour, day and month."""
        assert isinstance(df.index, pd.DatetimeIndex)
        df_mean_hours = df.groupby(df.index.hour).agg("mean").sort_index()

        week_order = pd.date_range("2024-01-01", periods=7, freq="D").day_name(locale=self.locale)
        day_of_week = pd.Categorical(df.index.day_name(locale=self.locale), categories=week_order, ordered=True)
        df_mean_days = df.groupby(day_of_week.codes).agg("mean").sort_index()

        month_order = pd.date_range("2024-01-01", periods=12, freq="ME").month_name(locale=self.locale)
        month_of_year = pd.Categorical(df.index.month_name(locale=self.locale), categories=month_order, ordered=True)
        df_mean_months = df.groupby(month_of_year.codes).agg("mean").sort_index()

        return df_mean_hours, df_mean_days, df_mean_months

    def _to_dict(self, series: pd.Series) -> dict:
        return {key: round(value, 3) for key, value in series.items()}


__all__ = ["TemporalAttributes", "DataGaps", "AvailabilityRule"]
