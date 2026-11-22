import datetime

import pandas as pd
import pytest

from phoibe.dokimasia import core

NOW = datetime.datetime(2026, 11, 21, 12, 0, tzinfo=datetime.UTC)
SUBJECT = core.Subject("t1", "wind_turbine")


class _Add:
    name = "add_column"

    def apply(self, dataset):
        frame = dataset.frame.copy()
        frame["b"] = 1.0
        return core.Transformed(frame, {"column": "b"}, rule="constant")


class _Refuse:
    name = "refuser"

    def apply(self, dataset):
        return core.TransformRefused("would have to guess")


def _dataset() -> core.Dataset:
    index = pd.date_range("2026-01-01", periods=4, freq="10min", name="datetime")
    return core.Dataset.from_frame(pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=index))


def _attestation(dataset: core.Dataset) -> core.Attestation:
    verdict = core.CriterionVerdict("c", "m", core.Severity.INFO, core.Verdict.GOOD)
    result = core.WorstEffectPolicy().apply([verdict])
    return core.build_attestation(
        subject=SUBJECT,
        layer="raw",
        fingerprint=dataset.fingerprint,
        contract_id="c",
        contract_version="1",
        result=result,
        created_at=NOW,
        software_version="0.1.0",
    )


class TestAdvanceDataset:
    def test_without_a_transform_hands_the_data_on_with_provenance(self):
        dataset = _dataset()
        attestation = _attestation(dataset)

        advanced = core.advance_dataset(dataset, attestation, software_version="0.1.0")

        assert advanced.fingerprint == dataset.fingerprint
        assert advanced.provenance == core.ProvenanceRecord(attestation.id, None, dataset.fingerprint)

    def test_with_a_transform_records_description_rule_and_output_fingerprint(self):
        dataset = _dataset()
        attestation = _attestation(dataset)

        advanced = core.advance_dataset(dataset, attestation, _Add(), software_version="0.1.0")

        assert advanced.signal_names == ("a", "b")
        assert advanced.fingerprint != dataset.fingerprint
        record = advanced.provenance
        assert record is not None
        assert record.parent_attestation_id == attestation.id
        assert record.output_fingerprint == advanced.fingerprint
        assert record.transform == core.TransformDescription("add_column", {"column": "b"}, "0.1.0", "constant")

    def test_does_not_change_the_dataset_that_was_given(self):
        dataset = _dataset()

        core.advance_dataset(dataset, _attestation(dataset), _Add(), software_version="0.1.0")

        assert dataset.signal_names == ("a",)

    def test_refuses_an_attestation_of_another_version_of_the_data(self):
        dataset = _dataset()
        edited = dataset.with_frame(dataset.frame.assign(a=0.0))

        with pytest.raises(ValueError, match="another version"):
            core.advance_dataset(edited, _attestation(dataset), software_version="0.1.0")

    def test_a_transform_that_refuses_stops_the_move_and_says_why(self):
        dataset = _dataset()

        with pytest.raises(core.TransformRefusedError, match=r"refuser.*would have to guess") as raised:
            core.advance_dataset(dataset, _attestation(dataset), _Refuse(), software_version="0.1.0")

        assert (raised.value.transform, raised.value.reason) == ("refuser", "would have to guess")


class TestReportProvenance:
    def test_a_dataset_without_provenance_says_so(self):
        report = core.report_provenance(_dataset())

        assert (report.has_provenance, report.fingerprint_matches, report.parent_attestation_id) == (False, None, None)

    def test_an_advanced_dataset_reports_parent_transform_and_a_matching_fingerprint(self):
        dataset = _dataset()
        attestation = _attestation(dataset)

        report = core.report_provenance(core.advance_dataset(dataset, attestation, _Add(), software_version="0.1.0"))

        assert report.has_provenance
        assert report.fingerprint_matches is True
        assert report.parent_attestation_id == attestation.id
        assert report.transform is not None
        assert report.transform.name == "add_column"

    def test_data_edited_after_it_advanced_keeps_its_lineage_and_reports_a_mismatch(self):
        dataset = _dataset()
        attestation = _attestation(dataset)
        advanced = core.advance_dataset(dataset, attestation, software_version="0.1.0")

        edited = advanced.with_frame(advanced.frame.assign(a=99.0))
        report = core.report_provenance(edited)

        assert (report.has_provenance, report.fingerprint_matches) == (True, False)
        assert report.parent_attestation_id == attestation.id
        assert edited.fingerprint != advanced.fingerprint


class TestProfileDatasetProvenance:
    def test_the_profile_reports_where_its_input_came_from(self):
        dataset = _dataset()
        attestation = _attestation(dataset)
        advanced = core.advance_dataset(dataset, attestation, software_version="0.1.0")

        profile = core.profile_dataset(advanced, SUBJECT, "bronze", [])

        assert profile.provenance.parent_attestation_id == attestation.id
        assert profile.provenance.fingerprint_matches is True

    def test_a_profile_of_data_without_provenance_says_so(self):
        assert not core.profile_dataset(_dataset(), SUBJECT, "raw", []).provenance.has_provenance

    def test_by_default_a_mismatch_is_information_and_in_strict_mode_it_is_refused(self):
        dataset = _dataset()
        advanced = core.advance_dataset(dataset, _attestation(dataset), software_version="0.1.0")
        edited = advanced.with_frame(advanced.frame.assign(a=99.0))

        profile = core.profile_dataset(edited, SUBJECT, "bronze", [])

        assert profile.provenance.fingerprint_matches is False
        with pytest.raises(core.ProvenanceMismatchError):
            core.profile_dataset(edited, SUBJECT, "bronze", [], strict_provenance=True)

    def test_strict_mode_accepts_data_without_provenance_and_data_that_still_matches(self):
        dataset = _dataset()
        advanced = core.advance_dataset(dataset, _attestation(dataset), software_version="0.1.0")

        core.profile_dataset(dataset, SUBJECT, "raw", [], strict_provenance=True)
        core.profile_dataset(advanced, SUBJECT, "bronze", [], strict_provenance=True)
