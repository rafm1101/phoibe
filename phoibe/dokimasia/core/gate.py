"""The gate: a policy over verdicts (ADR-05)."""

import collections
import dataclasses
import enum
import typing
from collections.abc import Mapping, Sequence

import pandas as pd

from phoibe.dokimasia.core import verdict as verdict_module

Verdict = verdict_module.Verdict
Severity = verdict_module.Severity


class Effect(enum.StrEnum):
    """What one criterion does to the gate, from least to most severe."""

    PASS = "pass"
    PASS_NOTED = "pass_noted"
    CONDITIONAL = "conditional"
    FAIL = "fail"


class GateOutcome(enum.StrEnum):
    """The result of a gate."""

    PASS = "pass"
    CONDITIONAL = "conditional"
    FAIL = "fail"


_EFFECTS: Mapping[tuple[Verdict, Severity], Effect] = {
    (Verdict.GOOD, Severity.CRITICAL): Effect.PASS,
    (Verdict.GOOD, Severity.WARNING): Effect.PASS,
    (Verdict.GOOD, Severity.INFO): Effect.PASS,
    (Verdict.ACCEPTABLE, Severity.CRITICAL): Effect.CONDITIONAL,
    (Verdict.ACCEPTABLE, Severity.WARNING): Effect.PASS_NOTED,
    (Verdict.ACCEPTABLE, Severity.INFO): Effect.PASS,
    (Verdict.BAD, Severity.CRITICAL): Effect.FAIL,
    (Verdict.BAD, Severity.WARNING): Effect.CONDITIONAL,
    (Verdict.BAD, Severity.INFO): Effect.PASS_NOTED,
    (Verdict.NOT_EVALUABLE, Severity.CRITICAL): Effect.FAIL,
    (Verdict.NOT_EVALUABLE, Severity.WARNING): Effect.CONDITIONAL,
    (Verdict.NOT_EVALUABLE, Severity.INFO): Effect.PASS_NOTED,
    (Verdict.ERROR, Severity.CRITICAL): Effect.FAIL,
    (Verdict.ERROR, Severity.WARNING): Effect.CONDITIONAL,
    (Verdict.ERROR, Severity.INFO): Effect.PASS_NOTED,
}
_ORDER = (Effect.PASS, Effect.PASS_NOTED, Effect.CONDITIONAL, Effect.FAIL)
_OUTCOME_OF = {
    Effect.PASS: GateOutcome.PASS,
    Effect.PASS_NOTED: GateOutcome.PASS,
    Effect.CONDITIONAL: GateOutcome.CONDITIONAL,
    Effect.FAIL: GateOutcome.FAIL,
}


def get_effect(verdict: Verdict, severity: Severity) -> Effect:
    """Look up the effect of a verdict at a severity in the first policy.

    Parameters
    ----------
    verdict
        The verdict of the criterion.
    severity
        The severity the contract gives the criterion.

    Returns
    -------
    Effect
        The effect. "Not evaluable" and "error" are never `PASS`.

    """
    return _EFFECTS[(verdict, severity)]


@dataclasses.dataclass(frozen=True, slots=True)
class GateReason:
    """Why a criterion influences the gate. Criteria with the plain effect `pass` are not listed."""

    check_id: str
    """Identifier of the check."""
    verdict: Verdict
    """Its verdict."""
    severity: Severity
    """Its severity."""
    effect: Effect
    """Its effect on the gate."""
    text: str
    """One line for a person."""


class GateFailedError(Exception):
    """Raised by `GateResult.raise_if_failed` when the gate failed."""

    def __init__(self, failures: tuple[GateReason, ...]) -> None:
        """Create the error from the reasons that failed the gate."""
        self.failures = failures
        super().__init__("gate failed: " + "; ".join(reason.text for reason in failures))


@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class GateResult:
    """The decision of a gate over the verdicts of one layer, with its reasons. It is returned, never hidden."""

    outcome: GateOutcome
    """Pass, conditional or fail."""
    reasons: tuple[GateReason, ...]
    """Criteria with an effect other than `pass`, in the order of the verdicts."""
    verdicts: tuple[verdict_module.CriterionVerdict, ...]
    """All verdicts the decision rests on."""
    policy: str
    """Name of the policy that decided."""

    def __repr__(self) -> str:
        """Summarize the decision in one line: outcome, policy, and the number of criteria per effect."""
        counts = self.count_by_verdict()
        verdicts = ", ".join(f"{count} {name}" for name, count in counts.items() if count)
        return (
            f"GateResult({self.outcome.value} by {self.policy}: {len(self.failures)} failures, "
            f"{len(self.conditions)} conditions, {len(self.notes)} notes; criteria: {verdicts})"
        )

    def to_frame(self) -> pd.DataFrame:
        """Build the verdicts the decision rests on as a table.

        Returns
        -------
        pandas.DataFrame
            See `build_verdict_table`.
        """
        return verdict_module.build_verdict_table(self.verdicts)

    @property
    def conditions(self) -> tuple[GateReason, ...]:
        """The reasons that make a pass conditional. They are recorded in the attestation."""
        return tuple(reason for reason in self.reasons if reason.effect == Effect.CONDITIONAL)

    @property
    def notes(self) -> tuple[GateReason, ...]:
        """The reasons that were noted but did not change the outcome."""
        return tuple(reason for reason in self.reasons if reason.effect == Effect.PASS_NOTED)

    @property
    def failures(self) -> tuple[GateReason, ...]:
        """The reasons that failed the gate."""
        return tuple(reason for reason in self.reasons if reason.effect == Effect.FAIL)

    @property
    def passed(self) -> bool:
        """Whether the gate let the data through, conditionally or not."""
        return self.outcome != GateOutcome.FAIL

    def count_by_verdict(self) -> dict[str, int]:
        """Count the verdicts. Every verdict appears, also with zero.

        Returns
        -------
        dict
            Verdict name to count.

        """
        counts = collections.Counter(verdict.verdict for verdict in self.verdicts)
        return {verdict.value: counts[verdict] for verdict in Verdict}

    def count_by_severity(self) -> dict[str, int]:
        """Count the criteria by severity. Every severity appears, also with zero.

        Returns
        -------
        dict
            Severity name to count.

        """
        counts = collections.Counter(verdict.severity for verdict in self.verdicts)
        return {severity.value: counts[severity] for severity in Severity}

    def compute_score(self) -> float:
        """Compute the informal score: the share of criteria that are good or acceptable. The gate never uses it.

        Returns
        -------
        float
            A number from 0 to 1.

        """
        met = sum(verdict.verdict in (Verdict.GOOD, Verdict.ACCEPTABLE) for verdict in self.verdicts)
        return met / len(self.verdicts)

    def raise_if_failed(self) -> None:
        """Raise `GateFailedError` if the gate failed, for pipelines that stop on a failed gate."""
        if self.outcome == GateOutcome.FAIL:
            raise GateFailedError(self.failures)


class GatePolicy(typing.Protocol):
    """A rule that turns the verdicts of a layer into a gate result. The policy is the extension point of ADR-05."""

    @property
    def name(self) -> str:
        """Name recorded in the result."""
        ...

    def apply(self, verdicts: Sequence[verdict_module.CriterionVerdict]) -> GateResult:
        """Decide on the verdicts of one layer."""
        ...


class WorstEffectPolicy:
    """The first policy: the gate takes the worst effect over all criteria (the table of ADR-05)."""

    name = "worst_effect"

    def apply(self, verdicts: Sequence[verdict_module.CriterionVerdict]) -> GateResult:
        """Decide on the verdicts of one layer.

        Parameters
        ----------
        verdicts
            The verdicts of all criteria of the layer.

        Returns
        -------
        GateResult
            The outcome is the worst effect: a failing criterion fails the gate, a conditional one makes it
            conditional, and a noted one only appears in the notes.

        Raises
        ------
        ValueError
            If there are no verdicts. A layer without criteria is a contract error, not a pass.

        """
        if not verdicts:
            raise ValueError("a layer without criteria cannot be judged; the contract must have at least one check")
        reasons = []
        worst = Effect.PASS
        for verdict in verdicts:
            effect = get_effect(verdict.verdict, verdict.severity)
            if _ORDER.index(effect) > _ORDER.index(worst):
                worst = effect
            if effect != Effect.PASS:
                text = f"{verdict.check_id}: {verdict.verdict.value} at severity {verdict.severity.value}"
                reasons.append(GateReason(verdict.check_id, verdict.verdict, verdict.severity, effect, text))
        return GateResult(
            outcome=_OUTCOME_OF[worst], reasons=tuple(reasons), verdicts=tuple(verdicts), policy=self.name
        )
