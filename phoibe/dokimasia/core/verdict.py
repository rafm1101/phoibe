"""Verdicts of criteria (ADR-05)."""

import dataclasses
import enum
from collections.abc import Iterable

import pandas as pd


class Verdict(enum.StrEnum):
    """What a criterion found on a profile."""

    GOOD = "good"
    ACCEPTABLE = "acceptable"
    BAD = "bad"
    NOT_EVALUABLE = "not_evaluable"
    ERROR = "error"


class Severity(enum.StrEnum):
    """What a violated criterion does to the gate. The contract states it for every criterion; there is no default."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class CriterionVerdict:
    """The verdict of one check, with the one value it read. The dataset is never involved."""

    check_id: str
    """Identifier of the check in the contract layer."""
    measure_id: str
    """Identifier of the measurement the check reads."""
    severity: Severity
    """Severity the contract gives the check."""
    verdict: Verdict
    """The outcome."""
    value: float | None = None
    """The value that was read; `None` if the measurement could not be evaluated or failed."""
    reason: str = ""
    """Facts for a person: the bands applied, or why nothing could be evaluated."""
    clause: str | None = None
    """Reference to the source clause, as free text."""
    signal: str | None = None
    """Signal the check concerns, if any."""

    def __repr__(self) -> str:
        """Summarize the verdict in one line."""
        value = "no value" if self.value is None else f"value={self.value:g}"
        return f"CriterionVerdict({self.check_id}: {self.verdict.value} at {self.severity.value}, {value})"


def build_verdict_table(verdicts: Iterable[CriterionVerdict]) -> pd.DataFrame:
    """Build the verdicts as a table, in the order given.

    Parameters
    ----------
    verdicts
        The verdicts of a layer.

    Returns
    -------
    pandas.DataFrame
        One row per verdict with the columns `check_id`, `measure_id`, `signal`, `severity`, `verdict`, `value`,
        `reason` and `clause`.
    """
    rows = [
        {
            "check_id": verdict.check_id,
            "measure_id": verdict.measure_id,
            "signal": verdict.signal,
            "severity": verdict.severity.value,
            "verdict": verdict.verdict.value,
            "value": verdict.value,
            "reason": verdict.reason,
            "clause": verdict.clause,
        }
        for verdict in verdicts
    ]
    columns = ["check_id", "measure_id", "signal", "severity", "verdict", "value", "reason", "clause"]
    return pd.DataFrame(rows, columns=columns)
