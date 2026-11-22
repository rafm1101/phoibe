"""Generic measures of validity."""

import dataclasses
import typing

import numpy as np
import pandas as pd

from phoibe.dokimasia import core
from phoibe.dokimasia.rules import _runs


def _read_numbers(raw: pd.Series) -> pd.Series:
    """Read values as numbers. Cells that are neither numbers nor text, and text that is no number, become NaN."""
    readable = raw.map(lambda cell: isinstance(cell, int | float | str | np.number)).astype(bool)
    return pd.to_numeric(raw.where(readable), errors="coerce").astype(float)


@dataclasses.dataclass(frozen=True, slots=True)
class RangesMeasure:
    """Count the values of one signal that lie outside `[min, max]`, and locate the runs.

    Values that are text are read as numbers where possible; what cannot be read is counted as `n_unreadable` and not
    checked. Missing values are not checked either. Infinite values lie outside any finite limit.

    Metrics: `n_out_of_range`, `share_out_of_range` (of the checked values), `n_checked`, `n_unreadable`, and the
    limits that were applied as `limit_min` and `limit_max`, so that the profile records them.
    """

    ASPECT: typing.ClassVar[str] = "validity"
    METRICS: typing.ClassVar[tuple[str, ...]] = (
        "n_out_of_range",
        "share_out_of_range",
        "n_checked",
        "n_unreadable",
        "limit_min",
        "limit_max",
    )

    signal: str
    """Catalog name of the signal."""
    min: float | None = None
    """Lower limit, inclusive, or `None`."""
    max: float | None = None
    """Upper limit, inclusive, or `None`."""
    measure_id: str = "ranges"
    """Identifier of this measure in a profile; the contract key when built from a contract."""

    def __post_init__(self) -> None:
        """Require a limit, and the lower limit not above the upper."""
        if self.min is None and self.max is None:
            raise ValueError("ranges needs at least one of min and max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min {self.min:g} is above max {self.max:g}")

    @property
    def id(self) -> str:
        """Identifier of the measure."""
        return self.measure_id

    @property
    def aspect(self) -> str:
        """Aspect of data quality."""
        return self.ASPECT

    def measure(self, dataset: core.Dataset) -> core.Measured | core.NotEvaluable:
        """Measure the values outside the limits.

        Parameters
        ----------
        dataset
            The data.

        Returns
        -------
        Measured | NotEvaluable
            Metrics and `out_of_range` findings; not evaluable if the signal is missing or ambiguous or has no value
            that can be checked.
        """
        positions = [i for i, name in enumerate(dataset.signal_names) if name == self.signal]
        if not positions:
            return core.NotEvaluable(
                self.id, core.NotEvaluableReason.MISSING_SIGNAL, f"signal '{self.signal}' is not in the dataset"
            )
        if len(positions) > 1:
            return core.NotEvaluable(
                self.id, core.NotEvaluableReason.MISSING_STRUCTURE, f"signal '{self.signal}' has duplicate columns"
            )
        raw = dataset.frame.iloc[:, positions[0]]
        numbers = _read_numbers(raw)
        n_unreadable = int((numbers.isna() & ~raw.isna()).sum())
        n_checked = int(numbers.notna().sum())
        if n_checked == 0:
            return core.NotEvaluable(
                self.id, core.NotEvaluableReason.INSUFFICIENT_DATA, f"no value of '{self.signal}' can be checked"
            )

        below = (numbers < self.min).to_numpy() if self.min is not None else np.zeros(len(numbers), dtype=bool)
        above = (numbers > self.max).to_numpy() if self.max is not None else np.zeros(len(numbers), dtype=bool)
        outside_all = below | above
        n_outside = int(outside_all.sum())

        located = pd.Series(outside_all, index=dataset.time_axis)
        ordered = located[located.index.notna()].sort_index(kind="stable")
        index = ordered.index
        findings = tuple(
            core.Finding("out_of_range", (self.signal,), index[start], index[end - 1], end - start)
            for start, end in _runs.find_runs(np.asarray(ordered, dtype=bool))
        )
        metrics = [
            core.Metric("n_out_of_range", float(n_outside), signal=self.signal),
            core.Metric("share_out_of_range", n_outside / n_checked, signal=self.signal),
            core.Metric("n_checked", float(n_checked), signal=self.signal),
            core.Metric("n_unreadable", float(n_unreadable), signal=self.signal),
        ]
        if self.min is not None:
            metrics.append(core.Metric("limit_min", self.min, signal=self.signal))
        if self.max is not None:
            metrics.append(core.Metric("limit_max", self.max, signal=self.signal))
        return core.Measured(self.id, tuple(metrics), findings)
