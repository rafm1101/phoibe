import datetime
import types

import pytest

from phoibe.geography.complexity.rix.writer import RIXWriter, WriterProfile, _is_datetime


def _make_result(
    summary=None, trix_table=None, meta=None, locations_site=None, locations_reference=None, steep_segments=None
):
    return types.SimpleNamespace(
        summary=summary,
        trix_table=trix_table,
        meta=meta if meta is not None else {},
        locations_site=locations_site,
        locations_reference=locations_reference,
        steep_segments=steep_segments,
    )


class TestWriterProfile:
    def test_accepts_enum_member(self):
        assert WriterProfile(WriterProfile.SUMMARY) is WriterProfile.SUMMARY

    def test_accepts_plain_string(self):
        assert WriterProfile("full") is WriterProfile.FULL

    def test_rejects_unknown_value(self):
        with pytest.raises(ValueError):
            WriterProfile("nonexistent")


class TestIsDatetime:
    def test_true_for_datetime_instance(self):
        assert _is_datetime(datetime.datetime.now())

    def test_false_for_date_instance(self):
        assert not _is_datetime(datetime.date.today())

    def test_false_for_string(self):
        assert not _is_datetime("2026-01-01")

    def test_false_for_none(self):
        assert not _is_datetime(None)


class TestRIXWriterConstruction:
    def test_stores_result_and_defaults_to_summary_profile(self):
        result = _make_result()
        writer = RIXWriter(result)
        assert writer._profile is WriterProfile.SUMMARY

    def test_accepts_full_profile(self):
        result = _make_result()
        writer = RIXWriter(result, profile=WriterProfile.FULL)
        assert writer._profile is WriterProfile.FULL

    def test_no_longer_accepts_separate_locations_parameters(self):
        import inspect

        signature = inspect.signature(RIXWriter.__init__)
        assert "locations_site" not in signature.parameters
        assert "locations_reference" not in signature.parameters
