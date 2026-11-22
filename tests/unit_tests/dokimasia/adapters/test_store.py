import dataclasses
import datetime
import pathlib

import pytest

from phoibe.dokimasia import adapters, compose, core, standardize
from phoibe.synthetic_data import turbine

T0 = datetime.datetime(2026, 11, 21, 12, 0, 0, tzinfo=datetime.UTC)
T1 = T0 + datetime.timedelta(minutes=5)
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
        good: {max: 0.001}
        acceptable: {max: 0.10}
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
TRANSFORM = standardize.Chain(
    (standardize.RenameToCatalog({"power": "power_kw"}), standardize.LocalizeTimeToUtc.from_subject(SUBJECT))
)


@pytest.fixture(scope="module")
def runs() -> dict[str, adapters.Run]:
    frame = turbine.generate_wtg_scada(time=turbine.Time("2026-02-14T04:00:00", "10min", 288), seed=3)
    raw = core.Dataset.from_frame(frame.drop(frame.index[[10, 11, 12]]))
    raw_validation = compose.validate_layer(raw, SUBJECT, CONTRACT, "raw", REGISTRY)
    raw_attestation = compose.attest_validation(raw_validation, created_at=T0, software_version=VERSION)
    bronze = core.advance_dataset(raw, raw_attestation, TRANSFORM, software_version=VERSION)
    bronze_validation = compose.validate_layer(bronze, SUBJECT, CONTRACT, "bronze", REGISTRY)
    bronze_attestation = compose.attest_validation(bronze_validation, created_at=T1, software_version=VERSION)
    failing = core.Dataset.from_frame(frame.drop(columns=["power"]))
    failed_validation = compose.validate_layer(failing, SUBJECT, CONTRACT, "raw", REGISTRY)
    return {
        "raw": compose.build_run(raw_validation, raw_attestation),
        "bronze": compose.build_run(bronze_validation, bronze_attestation),
        "failed": compose.build_run(failed_validation),
    }


@pytest.fixture
def store(tmp_path) -> adapters.FlatStore:
    return adapters.FlatStore(tmp_path / "runs")


@pytest.mark.parametrize("fmt", ["yaml", "json"])
def test_a_saved_run_loads_equal(runs, tmp_path, fmt):
    store = adapters.FlatStore(tmp_path, fmt=fmt)

    entry = store.save_run(runs["bronze"], created_at=T1)
    loaded = store.load_run(entry)

    assert loaded == runs["bronze"]
    assert loaded.attestation is not None
    assert loaded.profile.provenance.parent_attestation_id == runs["raw"].attestation.id


def test_a_run_is_three_files_named_by_layer_and_time(runs, store):
    store.save_run(runs["raw"], created_at=T0)

    names = sorted(path.name for path in store.directory.iterdir())

    assert names == [
        "raw-20261121T120000Z-attestation.yaml",
        "raw-20261121T120000Z-profile.yaml",
        "raw-20261121T120000Z-verdicts.yaml",
    ]


def test_a_failed_run_has_no_attestation_file_and_is_listed(runs, store):
    entry = store.save_run(runs["failed"], created_at=T0)

    assert entry.gate_outcome == core.GateOutcome.FAIL
    assert entry.attestation_id is None
    assert sorted(path.name for path in store.directory.iterdir()) == [
        "raw-20261121T120000Z-profile.yaml",
        "raw-20261121T120000Z-verdicts.yaml",
    ]
    assert store.load_run(entry) == runs["failed"]


def test_the_index_is_built_from_the_files_oldest_first(runs, store):
    store.save_run(runs["bronze"], created_at=T1)
    store.save_run(runs["raw"], created_at=T0)

    entries = store.list_runs()

    assert [(e.layer, e.created_at) for e in entries] == [("raw", T0), ("bronze", T1)]
    raw, bronze = entries
    assert raw.subject_id == "WEA 01"
    assert raw.gate_outcome == core.GateOutcome.CONDITIONAL
    assert bronze.parent_id == raw.attestation_id
    assert bronze.attestation_id == runs["bronze"].attestation.id


def test_a_new_store_object_sees_the_runs_of_an_old_one(runs, tmp_path):
    adapters.FlatStore(tmp_path).save_run(runs["raw"], created_at=T0)

    assert len(adapters.FlatStore(tmp_path).list_runs()) == 1


def test_listing_a_directory_that_does_not_exist_is_empty(tmp_path):
    assert adapters.FlatStore(tmp_path / "nowhere").list_runs() == ()


def test_files_that_are_not_ours_are_ignored(runs, store):
    store.save_run(runs["raw"], created_at=T0)
    (store.directory / "notes.txt").write_text("hello")
    (store.directory / "raw-today-profile.yaml").write_text("x")

    assert len(store.list_runs()) == 1


def test_a_run_is_never_overwritten(runs, store):
    store.save_run(runs["raw"], created_at=T0)

    with pytest.raises(adapters.StoreError, match="exists already"):
        store.save_run(runs["raw"], created_at=T0)
    with pytest.raises(adapters.StoreError, match="exists already"):
        adapters.FlatStore(store.directory, fmt="json").save_run(runs["raw"], created_at=T0)


def test_a_failed_save_leaves_no_partial_run(runs, store, monkeypatch):
    original = pathlib.Path.replace
    calls = []

    def flaky(self, target):
        calls.append(self.name)
        if len(calls) == 3:
            raise OSError("disk full")
        return original(self, target)

    monkeypatch.setattr(pathlib.Path, "replace", flaky)

    with pytest.raises(adapters.StoreError, match="disk full"):
        store.save_run(runs["raw"], created_at=T0)

    assert list(store.directory.iterdir()) == []


def test_save_run_given_a_time_that_is_not_utc_raises(runs, store):
    with pytest.raises(ValueError, match="UTC"):
        store.save_run(runs["raw"], created_at=datetime.datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="UTC"):
        store.save_run(runs["raw"], created_at=T0.astimezone(datetime.timezone(datetime.timedelta(hours=1))))


@pytest.mark.parametrize("layer", ["a/b", "../x", "", "a b", "_x", "a-b"])
def test_save_run_given_a_layer_that_is_no_file_name_part_raises(runs, store, layer):
    run = adapters.Run(
        core.Profile(SUBJECT, layer, runs["raw"].profile.fingerprint, {}),
        adapters.VerdictsRecord(
            "WEA 01", layer, runs["raw"].profile.fingerprint, "c", "1", runs["raw"].verdicts.result
        ),
    )

    with pytest.raises(ValueError, match="file name"):
        store.save_run(run, created_at=T0)


def test_a_missing_verdicts_file_is_reported_and_not_skipped(runs, store):
    store.save_run(runs["raw"], created_at=T0)
    (store.directory / "raw-20261121T120000Z-verdicts.yaml").unlink()

    with pytest.raises(adapters.StoreError, match=r"verdicts\.yaml"):
        store.list_runs()


def test_a_damaged_file_names_the_file_and_the_place(runs, store):
    entry = store.save_run(runs["raw"], created_at=T0)
    path = store.directory / "raw-20261121T120000Z-profile.yaml"
    path.write_text(path.read_text().replace("schema_version: 1", "schema_version: 9"))

    with pytest.raises(adapters.StoreError, match=r"raw-20261121T120000Z-profile\.yaml: schema_version"):
        store.load_run(entry)


def test_files_of_different_runs_are_refused_when_loaded(runs, store):
    entry = store.save_run(runs["raw"], created_at=T0)
    other = store.save_run(runs["failed"], created_at=T1)
    (store.directory / f"{other.stem}-profile.yaml").replace(store.directory / f"{entry.stem}-profile.yaml")

    with pytest.raises(adapters.StoreError, match="do not belong together"):
        store.load_run(entry)


def test_a_file_name_that_disagrees_with_its_content_is_refused(runs, store):
    store.save_run(runs["raw"], created_at=T0)
    for artifact in ("profile", "verdicts", "attestation"):
        (store.directory / f"raw-20261121T120000Z-{artifact}.yaml").rename(
            store.directory / f"bronze-20261121T120000Z-{artifact}.yaml"
        )

    with pytest.raises(adapters.StoreError, match="file name says layer 'bronze'"):
        store.list_runs()


class TestRun:
    def test_refuses_artifacts_of_different_dataset_versions(self, runs):
        with pytest.raises(ValueError, match="differ in subject, layer or fingerprint"):
            adapters.Run(runs["raw"].profile, runs["bronze"].verdicts)

    def test_refuses_an_attestation_of_another_run(self, runs):
        with pytest.raises(ValueError, match="differ in subject, layer or fingerprint"):
            adapters.Run(runs["raw"].profile, runs["raw"].verdicts, runs["bronze"].attestation)

    def test_refuses_an_attestation_for_a_failed_gate(self, runs):
        forged = dataclasses.replace(runs["raw"].attestation, fingerprint=runs["failed"].profile.fingerprint)

        with pytest.raises(ValueError, match="no attestation"):
            adapters.Run(runs["failed"].profile, runs["failed"].verdicts, forged)

    def test_refuses_an_attestation_that_disagrees_on_the_outcome(self, runs):
        forged = dataclasses.replace(runs["raw"].attestation, outcome=core.GateOutcome.PASS)

        with pytest.raises(ValueError, match="disagree on the gate outcome"):
            adapters.Run(runs["raw"].profile, runs["raw"].verdicts, forged)


def test_find_entries_selects_by_every_given_criterion(runs, store):
    store.save_run(runs["raw"], created_at=T0)
    store.save_run(runs["bronze"], created_at=T1)
    store.save_run(runs["failed"], created_at=T1)
    entries = store.list_runs()

    assert len(adapters.find_entries(entries)) == 3
    assert [e.layer for e in adapters.find_entries(entries, layer="bronze")] == ["bronze"]
    assert len(adapters.find_entries(entries, layer="raw", subject_id="WEA 01")) == 2
    assert adapters.find_entries(entries, subject_id="other") == ()
    found = adapters.find_entries(entries, attestation_id=runs["raw"].attestation.id)
    assert [e.layer for e in found] == ["raw"]
    assert len(adapters.find_entries(entries, fingerprint=runs["bronze"].profile.fingerprint)) == 1


def test_the_parent_of_a_run_is_found_through_the_index(runs, store):
    store.save_run(runs["raw"], created_at=T0)
    store.save_run(runs["bronze"], created_at=T1)
    entries = store.list_runs()
    bronze = adapters.find_entries(entries, layer="bronze")[0]

    parent = adapters.find_entries(entries, attestation_id=bronze.parent_id)[0]

    assert store.load_run(parent).attestation == runs["raw"].attestation


def test_build_run_table_has_a_row_per_run(runs, store):
    store.save_run(runs["raw"], created_at=T0)
    store.save_run(runs["failed"], created_at=T1)

    table = adapters.build_run_table(store.list_runs())

    assert table["layer"].tolist() == ["raw", "raw"]
    assert table["gate_outcome"].tolist() == ["conditional", "fail"]
    assert table["attested"].tolist() == [True, False]
    assert adapters.build_run_table([]).empty


def test_flat_store_given_an_unknown_format_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown format"):
        adapters.FlatStore(tmp_path, fmt="toml")  # type: ignore[arg-type]


def test_the_store_satisfies_the_protocol(store):
    protocol_user: adapters.Store = store

    assert protocol_user.list_runs() == ()
