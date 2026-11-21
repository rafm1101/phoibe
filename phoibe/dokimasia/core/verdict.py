"""Verdicts of criteria (ADR-05)."""

import dataclasses
import enum


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


@dataclasses.dataclass(frozen=True, slots=True)
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
