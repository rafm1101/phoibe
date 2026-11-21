"""Generic measures of completeness."""

import dataclasses
import typing

import numpy as np

from phoibe.dokimasia import core
from phoibe.dokimasia.rules import _runs


@dataclasses.dataclass(frozen=True, slots=True)
class MissingRunsMeasure:
    """Find the runs of missing values (NaN, None, NaT) in each signal.

    Rows are taken in time order; rows without a valid timestamp are left out. A run is one finding of kind
    `missing` with the signal and the first and last timestamp. Infinite values and text are not missing; other
    measures look at them.

    Metrics per signal: `n_missing`, `share_missing`, `n_runs` and `longest_run_steps`.
    """

    ASPECT: typing.ClassVar[str] = "completeness"
    METRICS: typing.ClassVar[tuple[str, ...]] = ("n_missing", "share_missing", "n_runs", "longest_run_steps")

    signals: tuple[str, ...] | None = None
    """Signals to measure; `None` for all."""
    measure_id: str = "missing_runs"
    """Identifier of this measure in a profile; the contract key when built from a contract."""

    @property
    def id(self) -> str:
        """Identifier of the measure."""
        return self.measure_id

    @property
    def aspect(self) -> str:
        """Aspect of data quality."""
        return self.ASPECT

    def measure(self, dataset: core.Dataset) -> core.Measured | core.NotEvaluable:
        """Measure the missing values of every signal.

        Parameters
        ----------
        dataset
            The data.

        Returns
        -------
        Measured | NotEvaluable
            Metrics and `missing` findings; not evaluable if there is no signal or no row with a valid timestamp.

        """
        absent = [name for name in (self.signals or ()) if name not in dataset.signal_names]
        if absent:
            return core.NotEvaluable(
                measure_id=self.id,
                reason=core.NotEvaluableReason.MISSING_SIGNAL,
                detail=f"signals not in the dataset: {', '.join(absent)}",
            )
        if not dataset.signal_names:
            return core.NotEvaluable(
                measure_id=self.id, reason=core.NotEvaluableReason.MISSING_SIGNAL, detail="the dataset has no signal"
            )
        frame = dataset.frame[dataset.time_axis.notna()].sort_index(kind="stable")
        if frame.empty:
            return core.NotEvaluable(
                measure_id=self.id,
                reason=core.NotEvaluableReason.INSUFFICIENT_DATA,
                detail="no row with a valid timestamp",
            )

        index = frame.index
        metrics: list[core.Metric] = []
        findings: list[core.Finding] = []
        for position, name in enumerate(dataset.signal_names):
            if self.signals is not None and name not in self.signals:
                continue
            mask = frame.iloc[:, position].isna().to_numpy(dtype=bool)
            runs = _runs.find_runs(mask)
            lengths = [end - start for start, end in runs]
            metrics.extend(
                [
                    core.Metric("n_missing", float(np.count_nonzero(mask)), signal=name),
                    core.Metric("share_missing", float(np.count_nonzero(mask)) / len(mask), signal=name),
                    core.Metric("n_runs", float(len(runs)), signal=name),
                    core.Metric("longest_run_steps", float(max(lengths, default=0)), signal=name),
                ]
            )
            findings.extend(
                core.Finding(
                    kind="missing", signals=(name,), start=index[start], end=index[end - 1], n_steps=end - start
                )
                for start, end in runs
            )
        return core.Measured(measure_id=self.id, metrics=tuple(metrics), findings=tuple(findings))
