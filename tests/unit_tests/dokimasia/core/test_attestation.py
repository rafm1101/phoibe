import datetime

import pandas as pd
import pytest

from phoibe.dokimasia import core

SUBJECT = core.Subject(id="t1", device_type="wind_turbine")
FINGERPRINT = core.compute_fingerprint(
    pd.DataFrame({"a": [1.0]}, index=pd.DatetimeIndex(["2026-01-01"], name="datetime"))
)
NOW = datetime.datetime(2026, 11, 21, 12, 0, tzinfo=datetime.UTC)


def _result(*verdicts: tuple[core.Verdict, core.Severity]) -> core.GateResult:
    return core.WorstEffectPolicy().apply(
        [core.CriterionVerdict(f"c{i}", "m", severity, verdict) for i, (verdict, severity) in enumerate(verdicts)]
    )


def _build(result: core.GateResult, **changes) -> core.Attestation:
    arguments = {
        "subject": SUBJECT,
        "layer": "raw",
        "fingerprint": FINGERPRINT,
        "contract_id": "wind-scada",
        "contract_version": "1.0.0",
        "result": result,
        "created_at": NOW,
        "software_version": "0.1.0",
    } | changes
    return core.build_attestation(**arguments)


def test_build_attestation_given_a_pass_records_subject_layer_fingerprint_and_contract():
    attestation = _build(_result((core.Verdict.GOOD, core.Severity.CRITICAL)))

    assert attestation.outcome == core.GateOutcome.PASS
    assert attestation.conditions == ()
    assert (attestation.subject_id, attestation.layer, attestation.fingerprint) == ("t1", "raw", FINGERPRINT)
    assert (attestation.contract_id, attestation.contract_version) == ("wind-scada", "1.0.0")
    assert attestation.verdict_counts["good"] == 1
    assert attestation.parent_id is None


def test_build_attestation_given_a_conditional_pass_records_the_conditions():
    attestation = _build(_result((core.Verdict.ACCEPTABLE, core.Severity.CRITICAL)), parent_id="abc")

    assert attestation.outcome == core.GateOutcome.CONDITIONAL
    assert attestation.conditions == ("c0: acceptable at severity critical",)
    assert attestation.parent_id == "abc"


def test_build_attestation_given_a_failed_gate_raises():
    with pytest.raises(ValueError, match="no attestation"):
        _build(_result((core.Verdict.BAD, core.Severity.CRITICAL)))


@pytest.mark.parametrize(
    "created_at",
    [
        datetime.datetime(2026, 11, 21, 12, 0),
        datetime.datetime(2026, 11, 21, 12, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=1))),
    ],
)
def test_build_attestation_given_a_time_that_is_not_utc_raises(created_at):
    with pytest.raises(ValueError, match="UTC"):
        _build(_result((core.Verdict.GOOD, core.Severity.INFO)), created_at=created_at)


def test_build_attestation_gives_the_same_id_for_the_same_input_and_another_for_another_time():
    result = _result((core.Verdict.GOOD, core.Severity.INFO))

    assert _build(result).id == _build(result).id
    assert _build(result).id != _build(result, created_at=NOW + datetime.timedelta(seconds=1)).id


def test_transform_description_parameters_are_a_read_only_copy():
    parameters = {"mapping": "power->power_kw"}
    transform = core.TransformDescription("rename_to_catalog", parameters, "0.1.0")
    parameters["mapping"] = "changed"

    assert transform.parameters["mapping"] == "power->power_kw"
    with pytest.raises(TypeError):
        transform.parameters["x"] = 1  # type: ignore[index]
