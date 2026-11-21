"""Signals and the catalog that names them (ADR-01)."""

import collections
import dataclasses
import enum


class Statistic(enum.StrEnum):
    """The statistic that a signal reports over one time step."""

    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    STD = "std"


@dataclasses.dataclass(frozen=True, slots=True)
class Signal:
    """A quantity, a statistic and a unit. Rules ask for signals, never for column names."""

    name: str
    """Catalog name; the column name after standardizing (ADR-11)."""
    quantity: str
    """What is measured, for example `wind_speed`."""
    statistic: Statistic
    """Statistic over one time step."""
    unit: str | None = None
    """Unit of the values, `None` for dimensionless or unknown."""
    measurement_point: str | None = None
    """Source or height when sources are combined, for example `nacelle`."""


@dataclasses.dataclass(frozen=True, slots=True)
class Catalog:
    """The signals of one domain pack, shared by all layers of a contract (ADR-01).

    Signal names are unique. Detection patterns belong to the domain pack, not to the model.
    """

    signals: tuple[Signal, ...]
    """All signals, in a stable order."""

    def __post_init__(self) -> None:
        """Reject duplicate signal names, naming all of them at once."""
        counts = collections.Counter(signal.name for signal in self.signals)
        duplicates = sorted(name for name, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate signal names in catalog: {', '.join(duplicates)}")

    @property
    def names(self) -> tuple[str, ...]:
        """The signal names in catalog order."""
        return tuple(signal.name for signal in self.signals)

    def find_signal(self, name: str) -> Signal | None:
        """Look a signal up by name.

        Parameters
        ----------
        name
            Catalog name of the signal.

        Returns
        -------
        Signal | None
            The signal, or `None` if the catalog has none of that name.

        """
        for signal in self.signals:
            if signal.name == name:
                return signal
        return None

    def get_signal(self, name: str) -> Signal:
        """Look a signal up by name, which must exist.

        Parameters
        ----------
        name
            Catalog name of the signal.

        Returns
        -------
        Signal
            The signal.

        Raises
        ------
        KeyError
            If the catalog has no signal of that name.

        """
        signal = self.find_signal(name)
        if signal is None:
            raise KeyError(name)
        return signal
