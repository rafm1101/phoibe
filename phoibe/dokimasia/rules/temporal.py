"""Generic measures of the time axis."""

import dataclasses
import typing

import numpy as np
import pandas as pd

from phoibe.dokimasia import core


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
