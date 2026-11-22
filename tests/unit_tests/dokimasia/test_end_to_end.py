"""A contract written as YAML runs on generator data: load, check the rules, profile, judge, gate (Phase 2)."""

import numpy as np
import pandas as pd
import pytest

from phoibe.dokimasia import adapters, compose, contract, core
from phoibe.synthetic_data import turbine
from phoibe.synthetic_data._turbine_noise import RowDrop, UniformSegments

CONTRACT = adapters.parse_contract_yaml(
    """\
id: wind-scada-10min
version: 1.0.0
domain: wind
layers:
  raw:
    measures:
      required_signals:
        params: {signals: [power, wind_speed]}
      time_gaps:
        params: {expected_step_s: 600}
    checks:
      signals_present:
        measure: required_signals
        value: n_missing_signals
        severity: critical
        good: {max: 0}
        clause: "TR6 section 3.1"
      availability:
        measure: time_gaps
        value: share_missing_steps
        severity: critical
        good: {max: 0.02}
        acceptable: {max: 0.10}
  bronze:
    measures:
      wind_speed_limits:
        rule: ranges
        signal: wind_speed
        params: {min: 0, max: 60}
      power_limits:
        rule: ranges
        signal: power
        params:
          min: 0
          max: {subject: rated_power_kw, factor: 1.05}
    checks:
      no_wind_speed_outliers:
        measure: wind_speed_limits
        value: n_out_of_range
        severity: critical
        good: {max: 0}
        acceptable: {max: 10}
      power_within_rating:
        measure: power_limits
        value: share_out_of_range
        severity: warning
        good: {max: 0}
"""
)
SUBJECT = core.Subject("WEA 01", "wind_turbine", {"rated_power_kw": 5600})
REGISTRY = compose.create_default_registry()
FREQ = pd.Timedelta("10min")


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    return turbine.generate_wtg_scada(time=turbine.Time("2026-02-14T04:00:00", "10min", 576), seed=3)


def _validate(frame: pd.DataFrame, layer: str) -> compose.Validation:
    return compose.validate_layer(core.Dataset.from_frame(frame), SUBJECT, CONTRACT, layer, REGISTRY)


@pytest.mark.parametrize("layer", ["raw", "bronze"])
def test_clean_data_passes_both_layers(clean, layer):
    validation = _validate(clean, layer)

    assert validation.result.outcome == core.GateOutcome.PASS
    assert (validation.contract_id, validation.contract_version, validation.layer) == (
        "wind-scada-10min",
        "1.0.0",
        layer,
    )
    assert validation.profile.fingerprint == core.Dataset.from_frame(clean).fingerprint


def test_a_few_dropped_rows_make_the_raw_gate_conditional_and_many_make_it_fail(clean):
    few = clean.drop(clean.index[[10, 11, 12, 13, 14, 100, 101, 102, 200, 201, 202, 300, 301, 302]])
    many = clean.drop(clean.index[50:150])

    assert _validate(few, "raw").result.outcome == core.GateOutcome.CONDITIONAL
    assert _validate(many, "raw").result.outcome == core.GateOutcome.FAIL


def test_the_gap_check_agrees_with_the_generator_labels(clean):
    pipeline = turbine.MessUpPipeline(seed=4)
    pipeline.add(RowDrop(UniformSegments(n=4, min_len=2, max_len=6)))
    messy, labels = pipeline.apply_labelled(clean)
    n_dropped = sum(label.n_changed for label in labels)

    verdict = {v.check_id: v for v in _validate(messy, "raw").result.verdicts}["availability"]

    assert verdict.value == pytest.approx(n_dropped / len(clean), rel=0.2)


def test_a_missing_required_signal_fails_the_gate(clean):
    result = _validate(clean.drop(columns=["power"]), "raw").result

    assert result.outcome == core.GateOutcome.FAIL
    assert [reason.check_id for reason in result.failures] == ["signals_present"]


def test_outliers_in_a_few_values_make_the_bronze_gate_conditional_and_in_many_make_it_fail(clean):
    few, many = clean.copy(), clean.copy()
    few.iloc[[5, 6, 7], few.columns.get_loc("wind_speed")] = 99.0
    many.iloc[list(range(40)), many.columns.get_loc("wind_speed")] = 99.0

    assert _validate(few, "bronze").result.outcome == core.GateOutcome.CONDITIONAL
    assert _validate(many, "bronze").result.outcome == core.GateOutcome.FAIL


def test_power_above_the_limit_of_this_subject_is_a_warning_that_makes_the_pass_conditional(clean):
    messy = clean.copy()
    messy.iloc[3, messy.columns.get_loc("power")] = 6000.0
    other_subject = core.Subject("WEA 02", "wind_turbine", {"rated_power_kw": 7000})

    ours = _validate(messy, "bronze")
    theirs = compose.validate_layer(core.Dataset.from_frame(messy), other_subject, CONTRACT, "bronze", REGISTRY)

    assert ours.result.outcome == core.GateOutcome.CONDITIONAL
    assert theirs.result.outcome == core.GateOutcome.PASS
    limits = {m.key: m.value for o in ours.profile.list_outcomes() if o.measure_id == "power_limits" for m in o.metrics}
    assert limits["limit_max"] == pytest.approx(5880.0)


@pytest.mark.parametrize("breakage", ["no_column", "all_missing", "text"])
def test_data_that_cannot_be_evaluated_never_passes_a_critical_check(clean, breakage):
    messy = clean.copy()
    if breakage == "no_column":
        messy = messy.drop(columns=["wind_speed"])
    elif breakage == "all_missing":
        messy["wind_speed"] = np.nan
    else:
        messy["wind_speed"] = "n/a"

    result = _validate(messy, "bronze").result

    assert result.outcome == core.GateOutcome.FAIL
    verdicts = {v.check_id: v.verdict for v in result.verdicts}
    assert verdicts["no_wind_speed_outliers"] == core.Verdict.NOT_EVALUABLE


def test_a_changed_threshold_is_judged_again_on_the_same_profile(clean):
    validation = _validate(
        clean.drop(clean.index[[10, 11, 12, 13, 14, 100, 101, 102, 200, 201, 202, 300, 301, 302]]), "raw"
    )
    stricter = adapters.parse_contract_yaml(
        adapters.dump_contract_yaml(CONTRACT)
        .replace("max: 0.1\n", "max: 0.001\n")
        .replace("max: 0.02\n", "max: 0.0005\n")
    )

    rejudged = contract.judge_layer(validation.profile, stricter.get_layer("raw"))

    assert validation.result.outcome == core.GateOutcome.CONDITIONAL
    assert rejudged.outcome == core.GateOutcome.FAIL
