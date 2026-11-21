"""The generic rules of `rules` as registry entries."""

from collections.abc import Mapping

import pandas as pd

from phoibe.dokimasia import core, rules
from phoibe.dokimasia.compose import registry


def _number(params: Mapping[str, object], name: str) -> float | None:
    value = params.get(name)
    return None if value is None else float(value)  # type: ignore[arg-type]


def _build_time_gaps(measure_id: str, signal: str | None, params: Mapping[str, object]) -> core.Measure:
    step_s = _number(params, "expected_step_s")
    step = None if step_s is None else pd.Timedelta(seconds=step_s)
    return rules.GapMeasure(expected_step=step, measure_id=measure_id)


def _build_missing_runs(measure_id: str, signal: str | None, params: Mapping[str, object]) -> core.Measure:
    return rules.MissingRunsMeasure(signals=None if signal is None else (signal,), measure_id=measure_id)


def _build_ranges(measure_id: str, signal: str | None, params: Mapping[str, object]) -> core.Measure:
    assert signal is not None
    return rules.RangesMeasure(
        signal=signal, min=_number(params, "min"), max=_number(params, "max"), measure_id=measure_id
    )


def _build_required_signals(measure_id: str, signal: str | None, params: Mapping[str, object]) -> core.Measure:
    names = params["signals"]
    assert isinstance(names, list | tuple)
    return rules.RequiredSignalsMeasure(signals=tuple(str(name) for name in names), measure_id=measure_id)


def list_generic_rules() -> tuple[registry.Rule, ...]:
    """List the generic rules.

    Returns
    -------
    tuple
        The rules `time_gaps`, `missing_runs`, `ranges` and `required_signals`.
    """
    kind = registry.ParamKind
    use = registry.SignalUse
    return (
        registry.Rule(
            "time_gaps",
            _build_time_gaps,
            rules.GapMeasure.METRICS,
            {"expected_step_s": registry.ParamSpec(kind.NUMBER)},
        ),
        registry.Rule("missing_runs", _build_missing_runs, rules.MissingRunsMeasure.METRICS, signal=use.OPTIONAL),
        registry.Rule(
            "ranges",
            _build_ranges,
            rules.RangesMeasure.METRICS,
            {"min": registry.ParamSpec(kind.NUMBER_OR_SUBJECT), "max": registry.ParamSpec(kind.NUMBER_OR_SUBJECT)},
            signal=use.REQUIRED,
            at_least_one_of=("min", "max"),
        ),
        registry.Rule(
            "required_signals",
            _build_required_signals,
            rules.RequiredSignalsMeasure.METRICS,
            {"signals": registry.ParamSpec(kind.TEXT_LIST, required=True)},
        ),
    )
