"""Generic measures of structure: what arrived."""

import dataclasses
import typing

from phoibe.dokimasia import core


@dataclasses.dataclass(frozen=True, slots=True)
class RequiredSignalsMeasure:
    """Report which of the required signals are in the dataset.

    Metrics: `n_missing_signals`, and `present` (1 or 0) for every required signal. The measure is always evaluable.
    """

    ASPECT: typing.ClassVar[str] = "structure"
    METRICS: typing.ClassVar[tuple[str, ...]] = ("n_missing_signals", "present")

    signals: tuple[str, ...]
    """Catalog names of the signals that must be present."""
    measure_id: str = "required_signals"
    """Identifier of this measure in a profile; the contract key when built from a contract."""

    def __post_init__(self) -> None:
        """Require at least one signal."""
        if not self.signals:
            raise ValueError("required_signals needs at least one signal")

    @property
    def id(self) -> str:
        """Identifier of the measure."""
        return self.measure_id

    @property
    def aspect(self) -> str:
        """Aspect of data quality."""
        return self.ASPECT

    def measure(self, dataset: core.Dataset) -> core.Measured | core.NotEvaluable:
        """Measure which required signals are present.

        Parameters
        ----------
        dataset
            The data; only its signal names are read.

        Returns
        -------
        Measured
            The metrics.
        """
        present = {name: name in dataset.signal_names for name in self.signals}
        metrics = [core.Metric("n_missing_signals", float(sum(not value for value in present.values())))]
        metrics.extend(core.Metric("present", float(value), signal=name) for name, value in present.items())
        return core.Measured(self.id, tuple(metrics))
