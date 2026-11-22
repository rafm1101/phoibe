"""The rules of the wind pack as registry entries."""

from collections.abc import Mapping

from phoibe.dokimasia import core, wind
from phoibe.dokimasia.compose import registry


def _number(params: Mapping[str, object], name: str, default: float) -> float:
    value = params.get(name)
    return default if value is None else float(value)  # type: ignore[arg-type]


def _whole(params: Mapping[str, object], name: str, default: int) -> int:
    value = _number(params, name, default)
    if value != int(value):
        raise ValueError(f"{name} must be a whole number, got {value:g}")
    return int(value)


def _build_power_levels(measure_id: str, signal: str | None, params: Mapping[str, object]) -> core.Measure:
    assert signal is not None
    return wind.PowerLevelsMeasure(
        power=signal,
        wind_speed=str(params["wind_speed"]),
        rated_power_kw=_number(params, "rated_power_kw", 0.0),
        min_wind_speed=_number(params, "min_wind_speed", 0.0),
        resolution_kw=_number(params, "resolution_kw", 10.0),
        method=wind.LevelMethod(str(params.get("method", wind.LevelMethod.COUNT))),
        min_count=_whole(params, "min_count", 5),
        prominence_threshold=_number(params, "prominence_threshold", 1e-7),
        min_rows=_whole(params, "min_rows", 20),
        measure_id=measure_id,
    )


def _build_night_limit(measure_id: str, signal: str | None, params: Mapping[str, object]) -> core.Measure:
    assert signal is not None
    return wind.NightLimitMeasure(
        power=signal,
        wind_speed=str(params["wind_speed"]),
        min_wind_speed=_number(params, "min_wind_speed", 0.0),
        night_start_hour=_whole(params, "night_start_hour", 0),
        night_end_hour=_whole(params, "night_end_hour", 0),
        resolution_kw=_number(params, "resolution_kw", 10.0),
        min_rows=_whole(params, "min_rows", 10),
        min_share=_number(params, "min_share", 0.5),
        measure_id=measure_id,
    )


def list_wind_rules() -> tuple[registry.Rule, ...]:
    """List the rules of the wind pack.

    Returns
    -------
    tuple
        The rules `power_levels` and `night_limit`. Both take the power signal as their signal and name the wind speed
        signal as a parameter.
    """
    kind = registry.ParamKind
    use = registry.SignalUse
    spec = registry.ParamSpec
    return (
        registry.Rule(
            "power_levels",
            _build_power_levels,
            wind.PowerLevelsMeasure.METRICS,
            {
                "wind_speed": spec(kind.TEXT, required=True),
                "rated_power_kw": spec(kind.NUMBER_OR_SUBJECT, required=True),
                "min_wind_speed": spec(kind.NUMBER_OR_SUBJECT, required=True),
                "resolution_kw": spec(kind.NUMBER),
                "method": spec(kind.TEXT),
                "min_count": spec(kind.NUMBER),
                "prominence_threshold": spec(kind.NUMBER),
                "min_rows": spec(kind.NUMBER),
            },
            signal=use.REQUIRED,
        ),
        registry.Rule(
            "night_limit",
            _build_night_limit,
            wind.NightLimitMeasure.METRICS,
            {
                "wind_speed": spec(kind.TEXT, required=True),
                "min_wind_speed": spec(kind.NUMBER_OR_SUBJECT, required=True),
                "night_start_hour": spec(kind.NUMBER, required=True),
                "night_end_hour": spec(kind.NUMBER, required=True),
                "resolution_kw": spec(kind.NUMBER),
                "min_rows": spec(kind.NUMBER),
                "min_share": spec(kind.NUMBER),
            },
            signal=use.REQUIRED,
        ),
    )
