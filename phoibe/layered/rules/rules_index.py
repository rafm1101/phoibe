import locale
import logging
import typing

import numpy as np
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

    Notes
    -----
    1. Extracted information includes:
        - Period: Start and end.
        - Timezone.
        - Duplicates.
        - Sorted.
        - Frequency (as most common timedelta).
        - Timedeltas shorter than the frequency.
    2. Datetime conversion:
        - Currently, timestamps that are not parseable are converted to NaT, and this requires that
          all timestamps follow the same format extracted from the first timestamp.
          Timestamps in a different format are set to NaT.
        - An alternative might be to call `pd.to_datetime` w/ argument `format="mixed"`. Then all formats
          are inferred individually.
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
        except Exception as exception:
            return self.result_builder.error("Failed to parse timestamps", exception=exception)

        points = self.points
        start = str(datetime_column.min().isoformat())
        end = str(datetime_column.max().isoformat())
        tzinfo = datetime_column.tzinfo
        n_dates = len(df)
        points = points - 2 if tzinfo is None else points

        has_duplicates = datetime_column.dropna().has_duplicates
        delta_datetime_column = datetime_column.to_series().diff()
        is_sorted = not bool((delta_datetime_column.dropna() < pd.to_timedelta(0)).any())
        points = points - 2 if has_duplicates or not is_sorted else points

        delta_datetime_counts = self._count_delta_datetime(datetime_column)
        try:
            mode = delta_datetime_counts.index[int(delta_datetime_counts.argmax())]
            message = ""
            outlier = delta_datetime_counts.index[delta_datetime_counts.index < mode]
            points = points - 2 if len(outlier) > 0 else points
        except ValueError:
            mode = np.nan
            message = "Failed to determine a reasonable frequency. Check the amount of parseable timestamps."
            outlier = []
            points = points - 2

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
            return self.result_builder.passed(required="", actual="", message=message, details=details)
        else:
            return self.result_builder.warning(required="", actual="", points=points, message=message, details=details)

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
        except Exception as exception:
            return self.result_builder.error("Failed to parse timestamps", exception=exception)

        delta_datetime_counts = self._count_delta_datetime(datetime_column)
        if delta_datetime_counts.empty:
            return self.result_builder.error("Failed to determine timedeltas.")
        else:
            gap_statistics = self._condition_on_actual_gaps(delta_datetime_counts)
            n_gaps, gap_length_total, gap_length_mean, gap_length_max = self._describe_gaps(gap_statistics)

        details = {
            "gap_count": int(n_gaps),
            "gap_length_total": str(gap_length_total),
            "gap_length_mean": str(gap_length_mean),
            "gap_length_max": str(gap_length_max),
        }
        return self.result_builder.passed(required="", actual="", message="", details=details)

    def _count_delta_datetime(self, index: pd.DatetimeIndex) -> pd.Series:
        """Determine count statistics of delta datetimes."""
        index_sorted = index.sort_values(ascending=True)
        delta_index = index_sorted.to_series().diff()
        delta_index_counts = delta_index.value_counts()
        delta_index_counts = delta_index_counts.sort_index()
        return delta_index_counts

    def _condition_on_actual_gaps(self, series: pd.Series) -> pd.Series:
        """Determine conditioned series conditioned on values larger than the estimated frequency."""
        mode = series.index[int(series.argmax())]
        delta_index_counts_conditioned = series[series.index > mode]
        delta_index_counts_conditioned.index = delta_index_counts_conditioned.index - mode
        return delta_index_counts_conditioned

    def _describe_gaps(self, delta: pd.Series) -> tuple[int, pd.Timedelta, pd.Timedelta, pd.Timedelta]:
        """Determine count statistics of delta datetimes that are larger than the estimated frequency."""
        n_gaps: int = delta.sum()
        gap_length_total: pd.Timedelta = (delta * delta.index).sum()
        gap_length_mean: pd.Timedelta = gap_length_total / n_gaps if n_gaps > 0 else pd.Timedelta(0)
        gap_length_max: pd.Timedelta = delta.index[-1] if n_gaps > 0 else pd.Timedelta(0)
        return n_gaps, gap_length_total, gap_length_mean, gap_length_max


@RuleRegistry.register("availability")
class AvailabilityRule(ValidationRule):
    """Assess the availability of timestamps globally and hourly, daily, monthly."""

    def __init__(
        self,
        good_threshold: float = 0.9,
        acceptable_threshold: float = 0.75,
        locale: str | None = None,
        points: int = 10,
        logger: logging.Logger | None = None,
    ):
        super().__init__(points, Severity.CRITICAL, logger)
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
        except Exception as exception:
            return self.result_builder.error("Failed to parse timestamps", exception=exception)

        try:
            availability_df = self._to_completed_series(index=datetime_column)
        except ValueError as exception:
            return self.result_builder.error(
                "Failed to fill gaps of timestamps. Please check for the amount of NaT.", exception=exception
            )
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
        df_mean_hours = df.groupby(df.index.hour).agg("mean").sort_index().reindex(range(24), fill_value=np.nan)

        day_of_week, day_names = self._get_day_of_week(df.index)
        df_mean_days = df.groupby(day_of_week).agg("mean").sort_index().reindex(day_names, fill_value=np.nan)

        month_of_year, month_names = self._get_month_names(df.index)
        df_mean_months = df.groupby(month_of_year).agg("mean").sort_index().reindex(month_names, fill_value=np.nan)

        return df_mean_hours, df_mean_days, df_mean_months

    def _to_dict(self, series: pd.Series) -> dict[typing.Hashable, float]:
        return {key: round(value, 3) for key, value in series.items()}

    def _get_day_of_week(self, index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Index]:
        dates = pd.date_range("2024-01-01", periods=7, freq="D")
        try:
            if self.locale is not None:
                day_names = dates.day_name(locale=self.locale)
                day_of_week = pd.Categorical(index.day_name(locale=self.locale), categories=day_names, ordered=True)
        except locale.Error:
            pass
        day_names = dates.day_name()
        day_of_week = pd.Categorical(index.day_name(), categories=day_names, ordered=True)
        day_of_week_series = pd.Series(day_of_week, index=index)
        return day_of_week_series, day_names

    def _get_month_names(self, index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Index]:
        dates = pd.date_range("2024-01-01", periods=12, freq="ME")
        try:
            if self.locale is not None:
                month_names = dates.month_name(locale=self.locale)
                month_of_year = pd.Categorical(
                    index.month_name(locale=self.locale), categories=month_names, ordered=True
                )
        except locale.Error:
            pass
        month_names = dates.month_name()
        month_of_year = pd.Categorical(index.month_name(), categories=month_names, ordered=True)
        month_of_year_series = pd.Series(month_of_year, index=index)
        return month_of_year_series, month_names


__all__ = ["TemporalAttributes", "DataGaps", "AvailabilityRule"]
