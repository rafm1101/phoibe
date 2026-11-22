"""Generic measures of consistency between signals."""

import dataclasses
import typing

import numpy as np
import pandas as pd

from phoibe.dokimasia import core
from phoibe.dokimasia.rules import _runs


def _find_unique_positions(
    dataset: core.Dataset, names: tuple[str, ...], measure_id: str
) -> list[int] | core.NotEvaluable:
    """Find the column of each name, or say why a name cannot be used."""
    absent = [name for name in names if name not in dataset.signal_names]
    if absent:
        return core.NotEvaluable(
            measure_id, core.NotEvaluableReason.MISSING_SIGNAL, f"signals not in the dataset: {', '.join(absent)}"
        )
    repeated = [name for name in names if dataset.signal_names.count(name) > 1]
    if repeated:
        return core.NotEvaluable(
            measure_id,
            core.NotEvaluableReason.MISSING_STRUCTURE,
            f"signals with duplicate columns: {', '.join(repeated)}",
        )
    return [dataset.signal_names.index(name) for name in names]


def _read_ordered_numbers(dataset: core.Dataset, positions: list[int]) -> pd.DataFrame:
    """Read the columns as numbers, in time order; rows without a valid timestamp are left out."""
    numbers = dataset.frame.iloc[:, positions].apply(pd.to_numeric, errors="coerce").astype(float)
    numbers.index = dataset.time_axis
    ordered: pd.DataFrame = numbers[numbers.index.notna()].sort_index(kind="stable")
    return ordered


def list_statistic_triples(catalog: core.Catalog) -> tuple[tuple[str, str, str], ...]:
    """List the (min, mean, max) signal names of every quantity that the catalog has all three of.

    Parameters
    ----------
    catalog
        The signals of a domain pack.

    Returns
    -------
    tuple
        One `(lower, middle, upper)` triple per quantity, in catalog order of the mean signal.
    """
    by_quantity: dict[tuple[str, str | None], dict[core.Statistic, str]] = {}
    for signal in catalog.signals:
        by_quantity.setdefault((signal.quantity, signal.measurement_point), {})[signal.statistic] = signal.name
    return tuple(
        (found[core.Statistic.MIN], found[core.Statistic.MEAN], found[core.Statistic.MAX])
        for found in by_quantity.values()
        if {core.Statistic.MIN, core.Statistic.MEAN, core.Statistic.MAX} <= found.keys()
    )


@dataclasses.dataclass(frozen=True, slots=True)
class StatisticOrderMeasure:
    """Find the rows where a mean lies outside its minimum and maximum, or the minimum lies above the maximum.

    The check is arithmetic on three signals of one quantity. A row is checked if all three values are numbers; a row
    with a missing value is not checked. A row is a violation if `lower - tolerance > middle` or
    `middle > upper + tolerance`; a minimum above the maximum is caught by one of the two. Consecutive violating rows
    are one finding of kind `statistic_order` that names the three signals.

    Metrics, on the middle signal: `n_violations`, `share_violations` (of the checked rows), `n_checked`, `n_runs`
    and the applied `tolerance`.
    """

    ASPECT: typing.ClassVar[str] = "consistency"
    METRICS: typing.ClassVar[tuple[str, ...]] = ("n_violations", "share_violations", "n_checked", "n_runs", "tolerance")

    lower: str
    """Catalog name of the minimum signal."""
    middle: str
    """Catalog name of the mean signal."""
    upper: str
    """Catalog name of the maximum signal."""
    tolerance: float = 0.0
    """Absolute slack in the unit of the signals, for rounding in the source."""
    measure_id: str = "statistic_order"
    """Identifier of this measure in a profile; the contract key when built from a contract."""

    def __post_init__(self) -> None:
        """Require three different signals and a tolerance of zero or more."""
        if len({self.lower, self.middle, self.upper}) != 3:
            raise ValueError(
                f"statistic order needs three different signals, got {self.lower}, {self.middle}, {self.upper}"
            )
        if not self.tolerance >= 0.0:
            raise ValueError(f"tolerance must be zero or more, got {self.tolerance:g}")

    @property
    def id(self) -> str:
        """Identifier of the measure."""
        return self.measure_id

    @property
    def aspect(self) -> str:
        """Aspect of data quality."""
        return self.ASPECT

    def measure(self, dataset: core.Dataset) -> core.Measured | core.NotEvaluable:
        """Measure the order of minimum, mean and maximum.

        Parameters
        ----------
        dataset
            The data.

        Returns
        -------
        Measured | NotEvaluable
            Metrics and `statistic_order` findings; not evaluable if a signal is missing or ambiguous or no row has
            all three values.
        """
        positions = _find_unique_positions(dataset, (self.lower, self.middle, self.upper), self.id)
        if isinstance(positions, core.NotEvaluable):
            return positions
        numbers = _read_ordered_numbers(dataset, positions)
        low, mid, high = (numbers.iloc[:, i].to_numpy() for i in range(3))
        checked = ~(np.isnan(low) | np.isnan(mid) | np.isnan(high))
        n_checked = int(checked.sum())
        if n_checked == 0:
            return core.NotEvaluable(
                self.id,
                core.NotEvaluableReason.INSUFFICIENT_DATA,
                f"no row has all of '{self.lower}', '{self.middle}' and '{self.upper}'",
            )

        with np.errstate(invalid="ignore"):
            violation = checked & ((low - self.tolerance > mid) | (mid > high + self.tolerance))
        runs = _runs.find_runs(violation)
        index = numbers.index
        signals = (self.lower, self.middle, self.upper)
        n_violations = int(violation.sum())
        metrics = (
            core.Metric("n_violations", float(n_violations), signal=self.middle),
            core.Metric("share_violations", n_violations / n_checked, signal=self.middle),
            core.Metric("n_checked", float(n_checked), signal=self.middle),
            core.Metric("n_runs", float(len(runs)), signal=self.middle),
            core.Metric("tolerance", self.tolerance, signal=self.middle),
        )
        findings = tuple(
            core.Finding("statistic_order", signals, index[start], index[end - 1], end - start) for start, end in runs
        )
        return core.Measured(self.id, metrics, findings)


@dataclasses.dataclass(frozen=True, slots=True)
class FrozenRunsMeasure:
    """Find the runs in which all given signals hold the same values step after step.

    A turbine that logs the state of its detectors instead of tracking current values holds several signals constant at
    once; one signal alone staying constant says little, because a stopped turbine does that anyway. A row repeats the
    previous one if every signal equals its previous value; a missing value never repeats. Rows are taken in time order,
    and a missing row between two rows is not looked at. The state is held by the rows from the first to the last of
    a run, so a run of `n` repeating rows is one finding of `n + 1` steps of kind `frozen`. Only runs of at least
    `min_steps` steps are findings.

    Which signals and how many steps are the user's decision; there is no default.

    Metrics: `n_frozen_steps` (steps in findings), `share_frozen_steps` (of the rows), `n_runs`, `longest_run_steps`
    and the applied `min_steps`.
    """

    ASPECT: typing.ClassVar[str] = "validity"
    METRICS: typing.ClassVar[tuple[str, ...]] = (
        "n_frozen_steps",
        "share_frozen_steps",
        "n_runs",
        "longest_run_steps",
        "min_steps",
    )

    signals: tuple[str, ...]
    """Catalog names of the signals that must all repeat; at least two."""
    min_steps: int
    """Fewest steps that hold the same state to count as a finding; at least two."""
    measure_id: str = "frozen_runs"
    """Identifier of this measure in a profile; the contract key when built from a contract."""

    def __post_init__(self) -> None:
        """Require at least two different signals and two steps."""
        if len(self.signals) < 2 or len(set(self.signals)) != len(self.signals):
            raise ValueError(f"frozen runs needs at least two different signals, got {list(self.signals)}")
        if self.min_steps < 2:
            raise ValueError(f"min_steps must be at least 2, got {self.min_steps}")

    @property
    def id(self) -> str:
        """Identifier of the measure."""
        return self.measure_id

    @property
    def aspect(self) -> str:
        """Aspect of data quality."""
        return self.ASPECT

    def measure(self, dataset: core.Dataset) -> core.Measured | core.NotEvaluable:
        """Measure the runs of an unchanged state.

        Parameters
        ----------
        dataset
            The data.

        Returns
        -------
        Measured | NotEvaluable
            Metrics and `frozen` findings; not evaluable if a signal is missing or ambiguous or fewer than two rows
            have a valid timestamp.
        """
        positions = _find_unique_positions(dataset, self.signals, self.id)
        if isinstance(positions, core.NotEvaluable):
            return positions
        numbers = _read_ordered_numbers(dataset, positions)
        if len(numbers) < 2:
            return core.NotEvaluable(
                self.id, core.NotEvaluableReason.INSUFFICIENT_DATA, "fewer than two rows with a valid timestamp"
            )

        values = numbers.to_numpy()
        repeats = (values[1:] == values[:-1]).all(axis=1)  # NaN equals nothing
        runs = [
            (start, end + 1)  # the state is held from the row before the first repeat
            for start, end in _runs.find_runs(repeats)
            if end - start + 1 >= self.min_steps
        ]
        index = numbers.index
        lengths = [end - start for start, end in runs]
        n_frozen = sum(lengths)
        metrics = (
            core.Metric("n_frozen_steps", float(n_frozen)),
            core.Metric("share_frozen_steps", n_frozen / len(numbers)),
            core.Metric("n_runs", float(len(runs))),
            core.Metric("longest_run_steps", float(max(lengths, default=0))),
            core.Metric("min_steps", float(self.min_steps)),
        )
        findings = tuple(
            core.Finding("frozen", self.signals, index[start], index[end - 1], end - start) for start, end in runs
        )
        return core.Measured(self.id, metrics, findings)
