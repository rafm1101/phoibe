"""Generic measures of the time axis."""

import dataclasses
import typing

import numpy as np
import pandas as pd

from phoibe.dokimasia import core
from phoibe.dokimasia.rules import _runs


@dataclasses.dataclass(frozen=True, slots=True)
class GapMeasure:
    """Find the intervals of the time axis in which whole time steps are missing.

    The measure works on the distinct, valid timestamps in time order. A difference between neighbors of `m` steps
    (rounded) is a gap of `m - 1` missing steps; the finding spans the missing timestamps, so `start` and `end` are
    not themselves present. Rows that are missing at the very start or end of the data have no neighbor and cannot
    be seen. Timestamps that are missing (NaT), repeated or out of order are counted, not repaired.

    Metrics: `step_s`, `n_gaps`, `n_missing_steps`, `longest_gap_steps`, `share_missing_steps`,
    `n_missing_timestamps`, `n_duplicate_timestamps` and `n_decreasing_steps`.
    """

    ASPECT: typing.ClassVar[str] = "temporal"
    METRICS: typing.ClassVar[tuple[str, ...]] = (
        "step_s",
        "n_gaps",
        "n_missing_steps",
        "longest_gap_steps",
        "share_missing_steps",
        "n_missing_timestamps",
        "n_duplicate_timestamps",
        "n_decreasing_steps",
    )

    expected_step: pd.Timedelta | None = None
    """Length of one time step. If `None`, the median difference of the data is used and reported as `step_s`."""
    measure_id: str = "time_gaps"
    """Identifier of this measure in a profile; the contract key when built from a contract."""

    def __post_init__(self) -> None:
        """Reject a step that is not positive."""
        if self.expected_step is not None and self.expected_step <= pd.Timedelta(0):
            raise ValueError(f"expected_step must be positive, got {self.expected_step}")

    @property
    def id(self) -> str:
        """Identifier of the measure."""
        return self.measure_id

    @property
    def aspect(self) -> str:
        """Aspect of data quality."""
        return self.ASPECT

    def measure(self, dataset: core.Dataset) -> core.Measured | core.NotEvaluable:
        """Measure the gaps of the time axis.

        Parameters
        ----------
        dataset
            The data; only its time axis is read.

        Returns
        -------
        Measured | NotEvaluable
            Metrics and one `gap` finding per interval of missing steps; not evaluable if fewer than two distinct
            valid timestamps exist.

        """
        axis = dataset.time_axis
        valid = axis[axis.notna()]
        distinct = valid.unique().sort_values()
        if len(distinct) < 2:
            return core.NotEvaluable(
                measure_id=self.id,
                reason=core.NotEvaluableReason.INSUFFICIENT_DATA,
                detail=f"{len(distinct)} distinct valid timestamps; at least 2 are needed",
            )

        differences = pd.Series(distinct).diff().iloc[1:]
        step = self.expected_step if self.expected_step is not None else differences.median()
        multiples = np.rint((differences / step).to_numpy(dtype=float)).astype(np.int64)
        is_gap = multiples >= 2
        previous = distinct[:-1][is_gap]
        following = distinct[1:][is_gap]
        gap_steps = multiples[is_gap] - 1
        findings = tuple(
            core.Finding(
                kind="gap",
                signals=(),
                start=pd.Timestamp(first + step),
                end=pd.Timestamp(last - step),
                n_steps=int(n_steps),
                magnitude=float(n_steps) * step.total_seconds(),
                magnitude_unit="s",
            )
            for first, last, n_steps in zip(previous, following, gap_steps, strict=True)
        )

        n_missing_steps = int(gap_steps.sum())
        n_present = len(distinct)
        metrics = (
            core.Metric("step_s", step.total_seconds(), unit="s"),
            core.Metric("n_gaps", float(len(findings))),
            core.Metric("n_missing_steps", float(n_missing_steps)),
            core.Metric("longest_gap_steps", float(gap_steps.max()) if len(gap_steps) else 0.0),
            core.Metric("share_missing_steps", n_missing_steps / (n_present + n_missing_steps)),
            core.Metric("n_missing_timestamps", float(len(axis) - len(valid))),
            core.Metric("n_duplicate_timestamps", float(len(valid) - n_present)),
            core.Metric("n_decreasing_steps", float(np.count_nonzero(valid[1:] < valid[:-1]))),
        )
        return core.Measured(measure_id=self.id, metrics=metrics, findings=findings)


@dataclasses.dataclass(frozen=True, slots=True)
class TimeSpanMeasure:
    """Report the period of the time axis, its time zone, and steps that are shorter than they should be.

    The measure works on the distinct, valid timestamps in time order. There is one finding of kind `time_span` from
    the first to the last timestamp, so that start and end are in the profile with their time zone. A step of less than
    half the expected step is an oversampled step; consecutive ones are one finding of kind
    `short_step` from the first timestamp to the last.

    Metrics: `n_timestamps` (distinct and valid), `span_s`, `start_epoch_s` and `end_epoch_s` (seconds since 1970 in
    UTC; a naive time axis is read as UTC), `is_timezone_aware` (1 or 0), `n_utc_offsets` (distinct offsets from UTC, 1
    for a time axis in UTC, more where daylight saving time shifts it; 1 for a naive one) and `n_short_steps`.
    """

    ASPECT: typing.ClassVar[str] = "temporal"
    METRICS: typing.ClassVar[tuple[str, ...]] = (
        "n_timestamps",
        "span_s",
        "start_epoch_s",
        "end_epoch_s",
        "is_timezone_aware",
        "n_utc_offsets",
        "n_short_steps",
    )

    expected_step: pd.Timedelta | None = None
    """Length of one time step. If `None`, the median difference of the data is used, as in `GapMeasure`."""
    measure_id: str = "time_span"
    """Identifier of this measure in a profile; the contract key when built from a contract."""

    def __post_init__(self) -> None:
        """Reject a step that is not positive."""
        if self.expected_step is not None and self.expected_step <= pd.Timedelta(0):
            raise ValueError(f"expected_step must be positive, got {self.expected_step}")

    @property
    def id(self) -> str:
        """Identifier of the measure."""
        return self.measure_id

    @property
    def aspect(self) -> str:
        """Aspect of data quality."""
        return self.ASPECT

    def measure(self, dataset: core.Dataset) -> core.Measured | core.NotEvaluable:
        """Measure the period, the time zone and the oversampled steps.

        Parameters
        ----------
        dataset
            The data; only its time axis is read.

        Returns
        -------
        Measured | NotEvaluable
            Metrics and findings; not evaluable if no timestamp is valid.
        """
        axis = dataset.time_axis
        distinct = axis[axis.notna()].unique().sort_values()
        if len(distinct) == 0:
            return core.NotEvaluable(
                self.id, core.NotEvaluableReason.INSUFFICIENT_DATA, "no row with a valid timestamp"
            )

        aware = distinct.tz is not None
        utc = distinct.tz_convert("UTC").tz_localize(None) if aware else distinct
        n_offsets = len(pd.unique((distinct.tz_localize(None) - utc).to_numpy())) if aware else 1
        start, end = distinct[0], distinct[-1]
        epoch = pd.Timestamp("1970-01-01")
        span = end - start

        step = None
        n_short = 0
        short_findings: tuple[core.Finding, ...] = ()
        if len(distinct) >= 2:
            differences = pd.Series(distinct).diff().iloc[1:]
            step = self.expected_step if self.expected_step is not None else differences.median()
            is_short = (differences < step / 2).to_numpy(dtype=bool)
            n_short = int(is_short.sum())
            short_findings = tuple(
                core.Finding(
                    "short_step",
                    (),
                    distinct[first],
                    distinct[last],
                    last - first + 1,
                )
                for first, last in _runs.find_runs(is_short)
            )
        n_steps = 1 if step is None else round(span / step) + 1
        span_finding = core.Finding(
            "time_span", (), start, end, n_steps, magnitude=span.total_seconds(), magnitude_unit="s"
        )
        metrics = (
            core.Metric("n_timestamps", float(len(distinct))),
            core.Metric("span_s", span.total_seconds(), unit="s"),
            core.Metric("start_epoch_s", (utc[0] - epoch).total_seconds(), unit="s"),
            core.Metric("end_epoch_s", (utc[-1] - epoch).total_seconds(), unit="s"),
            core.Metric("is_timezone_aware", 1.0 if aware else 0.0),
            core.Metric("n_utc_offsets", float(n_offsets)),
            core.Metric("n_short_steps", float(n_short)),
        )
        return core.Measured(self.id, metrics, (span_finding, *short_findings))
