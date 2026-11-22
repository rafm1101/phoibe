"""Generic measures of the distribution of values."""

import dataclasses
import typing

import numpy as np

from phoibe.dokimasia import core
from phoibe.dokimasia.rules import signals as signals_module


def find_shortest_interval(values: np.ndarray, proportion: float) -> tuple[float, float]:
    """Find the shortest interval that contains a proportion of the values.

    The interval holds `ceil(proportion * n) + 1` of the `n` values, or all of them, as the essential range of
    `layered` does; so it holds slightly more than the proportion.

    Parameters
    ----------
    values
        Finite values, at least one.
    proportion
        Share of the values that the interval must contain, above 0 and up to 1.

    Returns
    -------
    tuple
        Lower and upper end of the interval; both are values. If several intervals are equally short, the lowest wins.
    """
    ordered = np.sort(values)
    n = len(ordered)
    k = int(min(np.ceil(proportion * n), n - 1))
    widths = ordered[k:] - ordered[: n - k]
    lowest = int(np.argmin(widths))
    return float(ordered[lowest]), float(ordered[lowest + k])


@dataclasses.dataclass(frozen=True, slots=True)
class EssentialRangeMeasure:
    """Find the shortest interval that holds a proportion of the values of one signal.

    The interval is where the bulk of the data lies; the extreme tail is left out. It is information about the data
    and finds no rows. `RangesMeasure` is the check against limits. Missing and infinite values are left out; text
    that is a number is read as a number.

    Metrics, on the signal: `essential_min`, `essential_max`, `essential_width`, `n_values` (finite values used) and
    the applied `proportion`.
    """

    ASPECT: typing.ClassVar[str] = "distribution"
    METRICS: typing.ClassVar[tuple[str, ...]] = (
        "essential_min",
        "essential_max",
        "essential_width",
        "n_values",
        "proportion",
    )

    signal: str
    """Catalog name of the signal."""
    proportion: float = 0.995
    """Share of the values that the interval holds, above 0 and up to 1."""
    measure_id: str = "essential_range"
    """Identifier of this measure in a profile; the contract key when built from a contract."""

    def __post_init__(self) -> None:
        """Require a proportion above 0 and up to 1."""
        if not 0.0 < self.proportion <= 1.0:
            raise ValueError(f"proportion must be above 0 and at most 1, got {self.proportion:g}")

    @property
    def id(self) -> str:
        """Identifier of the measure."""
        return self.measure_id

    @property
    def aspect(self) -> str:
        """Aspect of data quality."""
        return self.ASPECT

    def measure(self, dataset: core.Dataset) -> core.Measured | core.NotEvaluable:
        """Measure the essential range.

        Parameters
        ----------
        dataset
            The data.

        Returns
        -------
        Measured | NotEvaluable
            Metrics; not evaluable if the signal is missing or ambiguous or has no finite value.
        """
        numbers = signals_module.read_numeric_signals(dataset, (self.signal,), self.id)
        if isinstance(numbers, core.NotEvaluable):
            return numbers
        values = numbers.iloc[:, 0].to_numpy()
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return core.NotEvaluable(
                self.id, core.NotEvaluableReason.INSUFFICIENT_DATA, f"'{self.signal}' has no finite value"
            )
        low, high = find_shortest_interval(values, self.proportion)
        metrics = (
            core.Metric("essential_min", low, signal=self.signal),
            core.Metric("essential_max", high, signal=self.signal),
            core.Metric("essential_width", high - low, signal=self.signal),
            core.Metric("n_values", float(len(values)), signal=self.signal),
            core.Metric("proportion", self.proportion, signal=self.signal),
        )
        return core.Measured(self.id, metrics)
