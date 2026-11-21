"""Judge a profile against a layer of a contract. The data and the dataset are never read (ADR-02, ADR-05)."""

import logging
import math
from collections.abc import Mapping

from phoibe.dokimasia import core
from phoibe.dokimasia.contract import limits, schema

logger = logging.getLogger(__name__)


def judge_layer(
    profile: core.Profile, layer: schema.LayerSpec, *, policy: core.GatePolicy | None = None
) -> core.GateResult:
    """Judge a profile against the checks of a layer and decide the gate.

    Every check reads one metric of one measurement from the profile. A measurement that was not evaluable or failed,
    a missing measurement or metric, and a limit that cannot be resolved give a verdict of their own and follow the
    check's severity. Nothing passes silently.

    Parameters
    ----------
    profile
        The profile of the dataset version, with the subject it was measured for.
    layer
        The layer of the contract whose checks apply.
    policy
        The gate policy. If `None`, the worst effect decides.

    Returns
    -------
    GateResult
        The verdicts and the gate decision.

    Raises
    ------
    ValueError
        If the layer has no checks.

    """
    verdicts = [_judge_check(profile, check) for check in layer.checks]
    return (policy or core.WorstEffectPolicy()).apply(verdicts)


def _find_outcome(profile: core.Profile, measure_id: str) -> core.Outcome | None:
    for outcome in profile.list_outcomes():
        if outcome.measure_id == measure_id:
            return outcome
    return None


def _read_metric(outcome: core.Measured, check: schema.CheckSpec) -> core.Metric | str:
    candidates = [metric for metric in outcome.metrics if metric.key == check.value]
    if check.signal is not None:
        candidates = [metric for metric in candidates if metric.signal == check.signal]
    if not candidates:
        known = ", ".join(sorted({metric.key for metric in outcome.metrics})) or "none"
        return (
            f"the result of '{check.measure}' has no value '{check.value}'"
            + (f" for signal '{check.signal}'" if check.signal else "")
            + f"; it has: {known}"
        )
    if len(candidates) > 1:
        signals = ", ".join(str(metric.signal) for metric in candidates)
        return (
            f"value '{check.value}' of '{check.measure}' exists for several signals ({signals}); name one in 'signal'"
        )
    return candidates[0]


def _contains(band: schema.Band, value: float, specification: Mapping[str, object]) -> tuple[bool, str]:
    parts = []
    inside = True
    if band.min is not None:
        lower = limits.resolve_limit(band.min, specification)
        inside = inside and value >= lower.value
        parts.append(f"min {lower.text}")
    if band.max is not None:
        upper = limits.resolve_limit(band.max, specification)
        inside = inside and value <= upper.value
        parts.append(f"max {upper.text}")
    return inside, ", ".join(parts)


def _judge_check(profile: core.Profile, check: schema.CheckSpec) -> core.CriterionVerdict:
    def verdict(kind: core.Verdict, reason: str, value: float | None = None) -> core.CriterionVerdict:
        return core.CriterionVerdict(
            check_id=check.id,
            measure_id=check.measure,
            severity=check.severity,
            verdict=kind,
            value=value,
            reason=reason,
            clause=check.clause,
            signal=check.signal,
        )

    outcome = _find_outcome(profile, check.measure)
    if outcome is None:
        return verdict(core.Verdict.ERROR, f"measure '{check.measure}' is not in the profile")
    if isinstance(outcome, core.NotEvaluable):
        return verdict(core.Verdict.NOT_EVALUABLE, f"{outcome.reason.value}: {outcome.detail}")
    if isinstance(outcome, core.MeasurementError):
        return verdict(core.Verdict.ERROR, f"{outcome.exception_type}: {outcome.message}")
    metric = _read_metric(outcome, check)
    if isinstance(metric, str):
        return verdict(core.Verdict.ERROR, metric)
    if math.isnan(metric.value):
        return verdict(core.Verdict.NOT_EVALUABLE, f"value '{check.value}' is not a number", None)

    specification = profile.subject.specification
    try:
        in_good, good_text = _contains(check.good, metric.value, specification)
        if in_good:
            return verdict(core.Verdict.GOOD, f"{metric.value:g} within good ({good_text})", metric.value)
        if check.acceptable is None:
            return verdict(core.Verdict.BAD, f"{metric.value:g} outside good ({good_text})", metric.value)
        in_acceptable, acceptable_text = _contains(check.acceptable, metric.value, specification)
    except (KeyError, TypeError) as error:
        logger.warning("check %s: cannot resolve a limit: %s", check.id, error)
        return verdict(core.Verdict.ERROR, f"cannot resolve a limit: {error!s}", metric.value)
    if in_acceptable:
        return verdict(
            core.Verdict.ACCEPTABLE,
            f"{metric.value:g} outside good ({good_text}), within acceptable ({acceptable_text})",
            metric.value,
        )
    return verdict(
        core.Verdict.BAD,
        f"{metric.value:g} outside good ({good_text}) and acceptable ({acceptable_text})",
        metric.value,
    )
