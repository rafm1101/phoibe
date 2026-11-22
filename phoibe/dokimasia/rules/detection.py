"""Find which column carries which signal, and say so where that is not clear (ADR-01, ADR-11).

Detection never picks one of several candidates. A signal with two candidate columns, a column that two signals
claim, and a column name that appears twice are reported as ambiguities and left out of the mapping, so that nothing
is renamed on a guess.
"""

import dataclasses
import re
import types
from collections.abc import Mapping, Sequence

import pandas as pd


class DetectionError(Exception):
    """Raised by `Detection.raise_if_ambiguous` when detection found ambiguities."""

    def __init__(self, ambiguities: "tuple[Ambiguity, ...]") -> None:
        """Create the error from the ambiguities."""
        self.ambiguities = ambiguities
        super().__init__("signal detection is ambiguous: " + "; ".join(item.describe() for item in ambiguities))


@dataclasses.dataclass(frozen=True, slots=True)
class SignalPattern:
    """The regular expressions that identify the column of one signal. Matching ignores case and uses `re.search`."""

    signal: str
    """Catalog name of the signal."""
    patterns: tuple[str, ...]
    """Regular expressions; a column matches if any of them matches."""

    def __post_init__(self) -> None:
        """Check that every expression compiles.

        Raises
        ------
        ValueError
            Naming the signal and the expression that does not compile.
        """
        for pattern in self.patterns:
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(f"signal {self.signal!r}: invalid pattern {pattern!r}: {error}") from error

    def matches(self, column: str) -> bool:
        """Tell whether a column name matches the signal.

        Parameters
        ----------
        column
            The column name.

        Returns
        -------
        bool
            Whether any pattern matches.
        """
        return any(re.search(pattern, column, re.IGNORECASE) for pattern in self.patterns)


@dataclasses.dataclass(frozen=True, slots=True)
class Ambiguity:
    """Something detection could not decide."""

    kind: str
    """`signal_has_several_columns`, `column_matches_several_signals` or `column_repeated`."""
    subject: str
    """The signal, or the column, that is affected."""
    candidates: tuple[str, ...]
    """The columns, or the signals, that compete; empty for a repeated column."""

    def describe(self) -> str:
        """Describe the ambiguity in one line for a person.

        Returns
        -------
        str
            The sentence.
        """
        listed = ", ".join(repr(item) for item in self.candidates)
        if self.kind == "signal_has_several_columns":
            return f"signal {self.subject!r} matches several columns: {listed}"
        if self.kind == "column_matches_several_signals":
            return f"column {self.subject!r} matches several signals: {listed}"
        return f"column name {self.subject!r} appears several times"


@dataclasses.dataclass(frozen=True, slots=True)
class Detection:
    """The result of detection: what is clear, what is missing and what is ambiguous."""

    mapping: Mapping[str, str]
    """Source column to signal name, only where both sides are unambiguous. A read-only copy. Columns that already
    carry the signal name are not listed."""
    detected: Mapping[str, str]
    """Signal name to the column that carries it, also where the column already has the signal name."""
    missing: tuple[str, ...]
    """Signals without any candidate column."""
    ambiguities: tuple[Ambiguity, ...]
    """What could not be decided."""
    unmatched: tuple[str, ...]
    """Columns that no signal claims."""

    def __post_init__(self) -> None:
        """Replace the mappings by read-only copies."""
        object.__setattr__(self, "mapping", types.MappingProxyType(dict(self.mapping)))
        object.__setattr__(self, "detected", types.MappingProxyType(dict(self.detected)))

    @property
    def is_clear(self) -> bool:
        """Whether there is no ambiguity. Missing signals do not make detection unclear."""
        return not self.ambiguities

    def raise_if_ambiguous(self) -> None:
        """Raise `DetectionError` if anything is ambiguous.

        Raises
        ------
        DetectionError
            Listing every ambiguity.
        """
        if self.ambiguities:
            raise DetectionError(self.ambiguities)

    def build_table(self) -> pd.DataFrame:
        """Build the result as a table, one row per fact.

        Returns
        -------
        pandas.DataFrame
            The columns `status` (`detected`, `missing`, `ambiguous` or `unmatched`), `signal`, `column` and
            `detail`.
        """
        rows: list[dict[str, str | None]] = [
            {"status": "detected", "signal": signal, "column": column, "detail": ""}
            for signal, column in self.detected.items()
        ]
        rows.extend({"status": "missing", "signal": signal, "column": None, "detail": ""} for signal in self.missing)
        rows.extend(
            {"status": "ambiguous", "signal": None, "column": None, "detail": item.describe()}
            for item in self.ambiguities
        )
        rows.extend(
            {"status": "unmatched", "signal": None, "column": column, "detail": ""} for column in self.unmatched
        )
        return pd.DataFrame(rows, columns=["status", "signal", "column", "detail"])


def detect_signals(columns: Sequence[str], patterns: Sequence[SignalPattern]) -> Detection:
    """Detect which column carries which signal.

    Parameters
    ----------
    columns
        The column names of the data. The time axis is not a column and does not belong here.
    patterns
        One entry per signal of the catalog.

    Returns
    -------
    Detection
        The mapping where it is clear, the signals without a column, the columns no signal claims, and every
        ambiguity. A signal is mapped only if exactly one column matches it and that column matches no other signal.

    Raises
    ------
    ValueError
        If two entries are for the same signal.
    """
    names = [entry.signal for entry in patterns]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate signal patterns: {', '.join(duplicates)}")
    unique = list(dict.fromkeys(columns))
    repeated = {column for column in unique if list(columns).count(column) > 1}
    by_signal = {entry.signal: [column for column in unique if entry.matches(column)] for entry in patterns}
    by_column: dict[str, list[str]] = {column: [] for column in unique}
    for signal, matched in by_signal.items():
        for column in matched:
            by_column[column].append(signal)

    ambiguities = [
        Ambiguity("signal_has_several_columns", signal, tuple(matched))
        for signal, matched in by_signal.items()
        if len(matched) > 1
    ]
    ambiguities.extend(
        Ambiguity("column_matches_several_signals", column, tuple(signals))
        for column, signals in by_column.items()
        if len(signals) > 1
    )
    ambiguities.extend(
        Ambiguity("column_repeated", column, ()) for column in unique if column in repeated and by_column[column]
    )

    detected: dict[str, str] = {}
    for signal, matched in by_signal.items():
        if len(matched) == 1 and len(by_column[matched[0]]) == 1 and matched[0] not in repeated:
            detected[signal] = matched[0]
    return Detection(
        mapping={column: signal for signal, column in detected.items() if column != signal},
        detected=detected,
        missing=tuple(signal for signal, matched in by_signal.items() if not matched),
        ambiguities=tuple(ambiguities),
        unmatched=tuple(column for column in unique if not by_column[column]),
    )
