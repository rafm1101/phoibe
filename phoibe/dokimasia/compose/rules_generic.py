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


def _text(params: Mapping[str, object], name: str) -> str:
    return str(params[name])


def _build_statistic_order(measure_id: str, signal: str | None, params: Mapping[str, object]) -> core.Measure:
    assert signal is not None
    return rules.StatisticOrderMeasure(
        lower=_text(params, "lower"),
        middle=signal,
        upper=_text(params, "upper"),
        tolerance=_number(params, "tolerance") or 0.0,
        measure_id=measure_id,
    )


def _build_frozen_runs(measure_id: str, signal: str | None, params: Mapping[str, object]) -> core.Measure:
    names = params["signals"]
    assert isinstance(names, list | tuple)
    min_steps = _number(params, "min_steps")
    assert min_steps is not None
    if min_steps != int(min_steps):
        raise ValueError(f"min_steps must be a whole number, got {min_steps:g}")
    return rules.FrozenRunsMeasure(
        signals=tuple(str(name) for name in names), min_steps=int(min_steps), measure_id=measure_id
    )


def list_generic_rules() -> tuple[registry.Rule, ...]:
    """List the generic rules.

    Returns
    -------
    tuple
        The rules `time_gaps`, `missing_runs`, `ranges`, `required_signals`, `statistic_order` and `frozen_runs`.
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
        registry.Rule(
            "statistic_order",
            _build_statistic_order,
            rules.StatisticOrderMeasure.METRICS,
            {
                "lower": registry.ParamSpec(kind.TEXT, required=True),
                "upper": registry.ParamSpec(kind.TEXT, required=True),
                "tolerance": registry.ParamSpec(kind.NUMBER),
            },
            signal=use.REQUIRED,
        ),
        registry.Rule(
            "frozen_runs",
            _build_frozen_runs,
            rules.FrozenRunsMeasure.METRICS,
            {
                "signals": registry.ParamSpec(kind.TEXT_LIST, required=True),
                "min_steps": registry.ParamSpec(kind.NUMBER, required=True),
            },
        ),
    )
