"""Exit criteria of Phase 2: no silent pass, and layer N+1 reports where its input came from."""

import datetime

import pytest

from phoibe.dokimasia import adapters, compose, core, standardize
from phoibe.synthetic_data import turbine

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
        signal: power_kw
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
        value: n_out_of_range
        severity: warning
        good: {max: 0}
"""
)
SUBJECT = core.Subject("WEA 01", "wind_turbine", {"rated_power_kw": 5600, "site_timezone": "Europe/Berlin"})
REGISTRY = compose.create_default_registry()
NOW = datetime.datetime(2026, 11, 21, 12, 0, tzinfo=datetime.UTC)
VERSION = "0.1.0"
TRANSFORM = standardize.Chain(
    (standardize.RenameToCatalog({"power": "power_kw"}), standardize.LocalizeTimeToUtc.from_subject(SUBJECT))
)


@pytest.fixture(scope="module")
def raw() -> core.Dataset:
    return core.Dataset.from_frame(
        turbine.generate_wtg_scada(time=turbine.Time("2026-02-14T04:00:00", "10min", 576), seed=3)
    )


def _validate(dataset: core.Dataset, layer: str) -> compose.Validation:
    return compose.validate_layer(dataset, SUBJECT, CONTRACT, layer, REGISTRY)


def test_a_dataset_moves_from_raw_to_bronze_and_the_bronze_profile_says_where_it_came_from(raw):
    raw_validation = _validate(raw, "raw")
    raw_attestation = compose.attest_validation(raw_validation, created_at=NOW, software_version=VERSION)

    bronze_input = core.advance_dataset(raw, raw_attestation, TRANSFORM, software_version=VERSION)
    bronze_validation = _validate(bronze_input, "bronze")

    provenance = bronze_validation.profile.provenance
    assert provenance.has_provenance
    assert provenance.fingerprint_matches is True
    assert provenance.parent_attestation_id == raw_attestation.id
    assert provenance.transform is not None
    assert (provenance.transform.name, provenance.transform.software_version) == ("chain", VERSION)
    assert "Europe/Berlin" in str(provenance.transform.parameters)
    assert bronze_validation.result.outcome == core.GateOutcome.PASS

    bronze_attestation = compose.attest_validation(bronze_validation, created_at=NOW, software_version=VERSION)
    assert bronze_attestation.parent_id == raw_attestation.id
    assert bronze_attestation.fingerprint == bronze_input.fingerprint
    assert bronze_attestation.fingerprint != raw_attestation.fingerprint


def test_the_raw_profile_says_that_its_input_has_no_provenance(raw):
    assert not _validate(raw, "raw").profile.provenance.has_provenance


def test_a_conditional_pass_is_attested_with_its_condition_and_advances(raw):
    index = list(raw.frame.index[[10, 11, 12, 13, 14, 100, 101, 102, 200, 201, 202, 300, 301, 302]])
    gappy = core.Dataset.from_frame(raw.frame.drop(index))

    validation = _validate(gappy, "raw")
    attestation = compose.attest_validation(validation, created_at=NOW, software_version=VERSION)

    assert attestation.outcome == core.GateOutcome.CONDITIONAL
    assert attestation.conditions == ("availability: acceptable at severity critical",)
    core.advance_dataset(gappy, attestation, TRANSFORM, software_version=VERSION)


def test_a_failed_gate_has_no_attestation_and_so_the_data_cannot_advance(raw):
    failed = _validate(core.Dataset.from_frame(raw.frame.drop(columns=["power"])), "raw")

    assert failed.result.outcome == core.GateOutcome.FAIL
    with pytest.raises(ValueError, match="no attestation"):
        compose.attest_validation(failed, created_at=NOW, software_version=VERSION)


def test_data_fixed_by_hand_in_bronze_keeps_the_lineage_and_the_mismatch_is_reported_and_attested(raw):
    raw_attestation = compose.attest_validation(_validate(raw, "raw"), created_at=NOW, software_version=VERSION)
    bronze_input = core.advance_dataset(raw, raw_attestation, TRANSFORM, software_version=VERSION)
    broken = bronze_input.frame.copy()
    broken.iloc[[5, 6, 7], broken.columns.get_loc("wind_speed")] = 99.0
    fixed = bronze_input.with_frame(broken).with_frame(bronze_input.frame)
    edited = bronze_input.with_frame(broken.clip(upper=60.0))

    unchanged_again = _validate(fixed, "bronze")
    edited_validation = _validate(edited, "bronze")

    assert unchanged_again.profile.provenance.fingerprint_matches is True
    assert edited_validation.profile.provenance.fingerprint_matches is False
    assert edited_validation.profile.provenance.parent_attestation_id == raw_attestation.id
    attestation = compose.attest_validation(edited_validation, created_at=NOW, software_version=VERSION)
    assert attestation.parent_id == raw_attestation.id
    assert attestation.fingerprint == edited.fingerprint
    with pytest.raises(core.ProvenanceMismatchError):
        core.profile_dataset(edited, SUBJECT, "bronze", [], strict_provenance=True)


def test_the_attestation_of_another_version_does_not_let_the_data_advance(raw):
    attestation = compose.attest_validation(_validate(raw, "raw"), created_at=NOW, software_version=VERSION)
    other = raw.with_frame(raw.frame.assign(power=0.0))

    with pytest.raises(ValueError, match="another version"):
        core.advance_dataset(other, attestation, TRANSFORM, software_version=VERSION)


def test_a_transform_that_cannot_be_applied_without_guessing_stops_the_move(raw):
    attestation = compose.attest_validation(_validate(raw, "raw"), created_at=NOW, software_version=VERSION)
    refusing = standardize.RenameToCatalog({"not_there": "power_kw"})

    with pytest.raises(core.TransformRefusedError, match="not_there"):
        core.advance_dataset(raw, attestation, refusing, software_version=VERSION)


def test_bronze_checks_on_unstandardized_data_do_not_pass_silently_and_a_warning_makes_the_pass_conditional(raw):
    result = _validate(raw, "bronze").result

    assert result.outcome == core.GateOutcome.CONDITIONAL
    verdicts = {verdict.check_id: verdict.verdict for verdict in result.verdicts}
    assert verdicts["power_within_rating"] == core.Verdict.NOT_EVALUABLE
