import datetime
import json
import math

import numpy as np
import pandas as pd
import pytest

from phoibe.dokimasia import adapters, compose, core, standardize
from phoibe.synthetic_data import turbine

FORMATS = ["yaml", "json"]
NOW = datetime.datetime(2026, 11, 21, 12, 0, 0, 123456, tzinfo=datetime.UTC)
VERSION = "0.1.0"
SUBJECT = core.Subject("WEA 01", "wind_turbine", {"rated_power_kw": 5600, "site_timezone": "Europe/Berlin"})
CONTRACT = adapters.parse_contract_yaml(
    """\
id: wind-scada-10min
version: 1.0.0
domain: wind
layers:
  raw:
    measures:
      time_gaps:
        params: {expected_step_s: 600}
    checks:
      availability:
        measure: time_gaps
        value: share_missing_steps
        severity: critical
        good: {max: 0.001}
        acceptable: {max: 0.10}
        clause: "IEC 61400-12-1, 7.1"
  bronze:
    measures:
      power_limits:
        rule: ranges
        signal: power_kw
        params: {min: 0, max: {subject: rated_power_kw, factor: 1.05}}
    checks:
      power_within_rating:
        measure: power_limits
        value: n_out_of_range
        severity: warning
        good: {max: 0}
"""
)
REGISTRY = compose.create_default_registry()


@pytest.fixture(scope="module")
def run() -> dict[str, compose.Validation | core.Attestation]:
    """A real run from raw to bronze: a conditional raw pass with a gap, and a bronze layer with provenance."""
    frame = turbine.generate_wtg_scada(time=turbine.Time("2026-02-14T04:00:00", "10min", 288), seed=3)
    raw = core.Dataset.from_frame(frame.drop(frame.index[[10, 11, 12]]))
    raw_validation = compose.validate_layer(raw, SUBJECT, CONTRACT, "raw", REGISTRY)
    raw_attestation = compose.attest_validation(raw_validation, created_at=NOW, software_version=VERSION)
    transform = standardize.Chain(
        (standardize.RenameToCatalog({"power": "power_kw"}), standardize.LocalizeTimeToUtc.from_subject(SUBJECT))
    )
    bronze = core.advance_dataset(raw, raw_attestation, transform, software_version=VERSION)
    bronze_validation = compose.validate_layer(bronze, SUBJECT, CONTRACT, "bronze", REGISTRY)
    return {
        "raw": raw_validation,
        "raw_attestation": raw_attestation,
        "bronze": bronze_validation,
        "bronze_attestation": compose.attest_validation(bronze_validation, created_at=NOW, software_version=VERSION),
    }


def _round_trip(data: dict[str, object], fmt: str) -> object:
    return adapters.parse_artifact(adapters.format_artifact(data, fmt), fmt)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("layer", ["raw", "bronze"])
class TestRoundTripOfARealRun:
    def test_profile(self, run, fmt, layer):
        profile = run[layer].profile

        back = adapters.profile_from_data(_round_trip(adapters.profile_to_data(profile), fmt))

        assert back == profile
        assert [o.duration_s for o in back.list_outcomes()] == [o.duration_s for o in profile.list_outcomes()]
        assert back.provenance == profile.provenance
        assert adapters.profile_to_data(back) == adapters.profile_to_data(profile)

    def test_verdicts(self, run, fmt, layer):
        validation = run[layer]
        record = adapters.VerdictsRecord(
            subject_id=SUBJECT.id,
            layer=layer,
            fingerprint=validation.profile.fingerprint,
            contract_id=validation.contract_id,
            contract_version=validation.contract_version,
            result=validation.result,
        )

        back = adapters.verdicts_from_data(_round_trip(adapters.verdicts_to_data(record), fmt))

        assert back == record
        assert back.result.outcome == validation.result.outcome

    def test_attestation(self, run, fmt, layer):
        attestation = run[f"{layer}_attestation"]

        back = adapters.attestation_from_data(_round_trip(adapters.attestation_to_data(attestation), fmt))

        assert back == attestation
        assert back.created_at == NOW


def test_the_real_run_covers_a_conditional_pass_and_provenance(run):
    assert run["raw_attestation"].outcome == core.GateOutcome.CONDITIONAL
    assert run["bronze"].profile.provenance.has_provenance
    assert run["bronze"].profile.provenance.transform is not None


def test_every_artifact_states_schema_version_and_kind(run):
    assert adapters.profile_to_data(run["raw"].profile)["kind"] == "profile"
    assert adapters.attestation_to_data(run["raw_attestation"])["kind"] == "attestation"
    assert adapters.profile_to_data(run["raw"].profile)["schema_version"] == adapters.SCHEMA_VERSION == 1


def _awkward_profile() -> core.Profile:
    naive = pd.Timestamp("2026-01-01 00:00:00.000000001")
    aware = pd.Timestamp("2026-01-01 01:00:00", tz="Europe/Berlin")
    outcomes = (
        core.Measured(
            "m1",
            metrics=(
                core.Metric("nan", math.nan),
                core.Metric("inf", math.inf, unit="kW", signal="power_kw"),
                core.Metric("ninf", -math.inf),
                core.Metric("tiny", 1e-300),
                core.Metric("third", 1 / 3),
            ),
            findings=(
                core.Finding("gap", ("a", "b"), naive, naive, 3, magnitude=math.nan, magnitude_unit="s"),
                core.Finding("range", (), aware, aware, 1),
            ),
            duration_s=0.5,
        ),
        core.NotEvaluable("m2", core.NotEvaluableReason.INSUFFICIENT_DATA, "ünïcode: 1e3, null, yes, 2026-01-01", 0.1),
        core.MeasurementError("m3", "ValueError", "line one\nline two: 'quoted'", 0.2),
    )
    transform = core.TransformDescription(
        "t", {"pair": (1, 2.5), "np": np.float64(2.0), "n": None, "nested": {"x": [1, {"y": "z"}]}}, "1", rule="r"
    )
    subject = core.Subject(
        "T 1", "wind_turbine", {"rated_power_kw": np.int64(5600), "nan": math.nan, "t": "2026-01-01"}
    )
    return core.Profile(
        subject,
        "raw",
        core.Fingerprint("sha256-v1", "ab" * 32),
        {"temporal": outcomes[:1], "other": outcomes[1:]},
        core.ProvenanceReport(True, False, "p" * 64, transform),
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_a_profile_with_awkward_values_survives_the_round_trip(fmt):
    profile = _awkward_profile()

    data = adapters.profile_to_data(profile)
    back = adapters.profile_from_data(_round_trip(data, fmt))

    assert adapters.profile_to_data(back) == data
    metrics = {m.key: m.value for m in back.aspects["temporal"][0].metrics}
    assert math.isnan(metrics["nan"])
    assert metrics["inf"] == math.inf
    assert metrics["ninf"] == -math.inf
    assert metrics["third"] == 1 / 3
    assert metrics["tiny"] == 1e-300
    findings = back.list_findings()
    assert findings[0].start == pd.Timestamp("2026-01-01 00:00:00.000000001")
    assert findings[0].start.tzinfo is None
    assert findings[1].start == pd.Timestamp("2026-01-01 01:00:00", tz="Europe/Berlin")
    assert str(findings[1].start.tz) == "UTC"
    assert back.aspects["other"][0].detail == "ünïcode: 1e3, null, yes, 2026-01-01"
    assert back.aspects["other"][1].message == "line one\nline two: 'quoted'"
    assert back.subject.specification["rated_power_kw"] == 5600
    assert back.subject.specification["t"] == "2026-01-01"
    assert back.provenance.transform.parameters["pair"] == (1, 2.5)
    assert back.provenance.fingerprint_matches is False


def test_the_json_text_is_strict_json():
    text = adapters.format_artifact(adapters.profile_to_data(_awkward_profile()), "json")

    assert "NaN" not in text.replace('"nan"', "")
    json.loads(text, parse_constant=lambda name: pytest.fail(f"literal {name}"))


def test_profile_to_data_given_a_specification_that_is_no_plain_data_raises():
    profile = core.Profile(core.Subject("t", "d", {"bad": object()}), "raw", core.Fingerprint("a", "b"), {})

    with pytest.raises(adapters.ArtifactError, match=r"subject\.specification\.bad"):
        adapters.profile_to_data(profile)


def test_profile_to_data_given_a_reserved_key_raises():
    profile = core.Profile(core.Subject("t", "d", {"$float": 1}), "raw", core.Fingerprint("a", "b"), {})

    with pytest.raises(adapters.ArtifactError, match="reserved"):
        adapters.profile_to_data(profile)


@pytest.fixture
def profile_data(run) -> dict[str, object]:
    return adapters.profile_to_data(run["bronze"].profile)


@pytest.fixture
def attestation_data(run) -> dict[str, object]:
    return adapters.attestation_to_data(run["raw_attestation"])


class TestRefusals:
    def test_another_schema_version(self, profile_data):
        profile_data["schema_version"] = 2

        with pytest.raises(adapters.ArtifactError, match="unsupported version 2; this software reads 1"):
            adapters.profile_from_data(profile_data)

    def test_a_missing_schema_version(self, profile_data):
        del profile_data["schema_version"]

        with pytest.raises(adapters.ArtifactError, match="missing key 'schema_version'"):
            adapters.profile_from_data(profile_data)

    def test_another_kind(self, attestation_data):
        with pytest.raises(adapters.ArtifactError, match="expected 'profile', found 'attestation'"):
            adapters.profile_from_data(attestation_data)

    def test_an_unknown_key_is_refused_and_not_dropped(self, profile_data):
        profile_data["extra"] = 1

        with pytest.raises(adapters.ArtifactError, match="unknown keys: extra"):
            adapters.profile_from_data(profile_data)

    def test_an_unknown_key_deep_inside_is_refused_with_its_place(self, profile_data):
        profile_data["aspects"][0]["outcomes"][0]["surprise"] = 1

        with pytest.raises(adapters.ArtifactError, match=r"aspects\[0\]\.outcomes\[0\]: unknown keys: surprise"):
            adapters.profile_from_data(profile_data)

    def test_a_missing_key_names_its_place(self, profile_data):
        del profile_data["fingerprint"]["digest"]

        with pytest.raises(adapters.ArtifactError, match=r"fingerprint: missing key 'digest'"):
            adapters.profile_from_data(profile_data)

    def test_a_wrong_type_names_its_place(self, profile_data):
        profile_data["aspects"][0]["outcomes"][0]["metrics"][0]["value"] = "high"

        with pytest.raises(adapters.ArtifactError, match=r"aspects\[0\]\.outcomes\[0\]\.metrics\[0\]\.value"):
            adapters.profile_from_data(profile_data)

    def test_a_boolean_is_no_number(self, profile_data):
        profile_data["aspects"][0]["outcomes"][0]["duration_s"] = True

        with pytest.raises(adapters.ArtifactError, match="expected a number"):
            adapters.profile_from_data(profile_data)

    def test_an_unknown_status(self, profile_data):
        profile_data["aspects"][0]["outcomes"][0]["status"] = "fine"

        with pytest.raises(adapters.ArtifactError, match="unknown status 'fine'"):
            adapters.profile_from_data(profile_data)

    def test_an_unknown_verdict(self, run):
        validation = run["raw"]
        data = adapters.verdicts_to_data(
            adapters.VerdictsRecord("t", "raw", validation.profile.fingerprint, "c", "1", validation.result)
        )
        data["verdicts"][0]["verdict"] = "great"

        with pytest.raises(adapters.ArtifactError, match=r"verdicts\[0\]\.verdict"):
            adapters.verdicts_from_data(data)

    def test_a_time_that_is_not_iso(self, profile_data):
        data = adapters.profile_to_data(_awkward_profile())
        data["aspects"][0]["outcomes"][0]["findings"][0]["start"] = "yesterday"

        with pytest.raises(adapters.ArtifactError, match="not an ISO 8601 time"):
            adapters.profile_from_data(data)

    def test_a_time_that_is_no_text(self, attestation_data):
        attestation_data["created_at"] = 20261121

        with pytest.raises(adapters.ArtifactError, match="ISO 8601 time as text"):
            adapters.attestation_from_data(attestation_data)

    def test_an_attestation_time_that_is_not_utc(self, attestation_data):
        attestation_data["created_at"] = "2026-11-21T13:00:00+01:00"

        with pytest.raises(adapters.ArtifactError, match="zero UTC offset"):
            adapters.attestation_from_data(attestation_data)

    def test_a_naive_attestation_time(self, attestation_data):
        attestation_data["created_at"] = "2026-11-21T12:00:00"

        with pytest.raises(adapters.ArtifactError, match="zero UTC offset"):
            adapters.attestation_from_data(attestation_data)

    def test_a_malformed_float_tag(self, profile_data):
        profile_data["aspects"][0]["outcomes"][0]["duration_s"] = {"$float": "huge"}

        with pytest.raises(adapters.ArtifactError, match="nan, inf, -inf"):
            adapters.profile_from_data(profile_data)

    def test_the_root_must_be_a_mapping(self):
        with pytest.raises(adapters.ArtifactError, match="expected a mapping"):
            adapters.profile_from_data([1, 2])

    def test_a_repeated_aspect(self, profile_data):
        profile_data["aspects"].append(profile_data["aspects"][0])

        with pytest.raises(adapters.ArtifactError, match="appears twice"):
            adapters.profile_from_data(profile_data)


class TestParseArtifact:
    @pytest.mark.parametrize(
        ("text", "fmt", "match"),
        [
            ("a: 1\na: 2\n", "yaml", "duplicate key"),
            ('{"a": 1, "a": 2}', "json", "duplicate key"),
            ('{"a": NaN}', "json", "not allowed"),
            ('{"a": Infinity}', "json", "not allowed"),
            ("{a: [1, 2", "yaml", "invalid yaml"),
            ("{not json", "json", "invalid json"),
            ("{}", "toml", "unknown format"),
        ],
    )
    def test_refuses(self, text, fmt, match):
        with pytest.raises(adapters.ArtifactError, match=match):
            adapters.parse_artifact(text, fmt)

    def test_an_unknown_format_cannot_be_written(self):
        with pytest.raises(ValueError, match="unknown format"):
            adapters.format_artifact({}, "toml")

    def test_yaml_is_the_default_and_keeps_key_order(self, attestation_data):
        text = adapters.format_artifact(attestation_data)

        assert text.startswith("schema_version: 1\nkind: attestation\nid:")
        assert adapters.parse_artifact(text) == attestation_data
