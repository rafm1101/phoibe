import numpy as np
import pandas as pd
import pytest

from phoibe.dokimasia import core, standardize


def _dataset(index: pd.DatetimeIndex, **columns) -> core.Dataset:
    return core.Dataset.from_frame(pd.DataFrame(columns, index=index))


GRID = pd.date_range("2026-01-01", periods=4, freq="10min", name="datetime")


def _transformed(outcome) -> core.Transformed:
    assert isinstance(outcome, core.Transformed), outcome
    return outcome


class TestRenameToCatalog:
    def test_renames_the_named_columns_and_leaves_the_rest(self):
        dataset = _dataset(GRID, power=[1.0] * 4, wind_speed=[2.0] * 4)

        outcome = _transformed(standardize.RenameToCatalog({"power": "power_kw"}).apply(dataset))

        assert list(outcome.frame.columns) == ["power_kw", "wind_speed"]
        assert outcome.parameters == {"mapping": {"power": "power_kw"}}
        assert list(dataset.frame.columns) == ["power", "wind_speed"]

    def test_refuses_and_names_every_problem(self):
        dataset = _dataset(GRID, a=[1.0] * 4, b=[1.0] * 4, c=[1.0] * 4)

        outcome = standardize.RenameToCatalog({"a": "x", "b": "x", "z": "y", "c": "a"}).apply(dataset)

        assert isinstance(outcome, core.TransformRefused)
        assert "column 'z' is not in the data" in outcome.reason
        assert "'x' is the target of several renames" in outcome.reason

    def test_refuses_a_target_that_is_the_name_of_another_column(self):
        outcome = standardize.RenameToCatalog({"a": "b"}).apply(_dataset(GRID, a=[1.0] * 4, b=[1.0] * 4))

        assert isinstance(outcome, core.TransformRefused)
        assert "'b' is already the name of another column" in outcome.reason

    def test_refuses_a_column_that_appears_several_times(self):
        frame = pd.DataFrame(np.ones((4, 2)), index=GRID, columns=["a", "a"])

        outcome = standardize.RenameToCatalog({"a": "x"}).apply(core.Dataset.from_frame(frame))

        assert isinstance(outcome, core.TransformRefused)
        assert "appears several times" in outcome.reason


class TestLocalizeTimeToUtc:
    def test_a_naive_axis_is_read_as_local_time_and_converted(self):
        dataset = _dataset(pd.date_range("2026-01-15 12:00", periods=3, freq="10min", name="datetime"), a=[1.0] * 3)

        outcome = _transformed(standardize.LocalizeTimeToUtc("Europe/Berlin").apply(dataset))

        assert str(outcome.frame.index.tz) == "UTC"
        assert outcome.frame.index[0] == pd.Timestamp("2026-01-15 11:00", tz="UTC")
        assert outcome.frame.index.name == "datetime"
        assert outcome.parameters == {"timezone": "Europe/Berlin"}
        assert outcome.rule is not None

    def test_the_repeated_hour_is_resolved_by_order_when_the_data_is_sorted_and_complete(self):
        utc = pd.date_range("2026-10-24 21:00", periods=36, freq="10min", tz="UTC")
        local = pd.DatetimeIndex(utc.tz_convert("Europe/Berlin").tz_localize(None), name="datetime")
        assert local.duplicated().any()

        outcome = _transformed(standardize.LocalizeTimeToUtc("Europe/Berlin").apply(_dataset(local, a=np.arange(36.0))))

        assert outcome.frame.index.equals(utc.rename("datetime"))
        assert "ambiguous=infer" in str(outcome.rule)

    def test_refuses_where_the_repeated_hour_cannot_be_resolved(self):
        utc = pd.date_range("2026-10-24 21:00", periods=36, freq="10min", tz="UTC")
        local = pd.DatetimeIndex(utc.tz_convert("Europe/Berlin").tz_localize(None))[::-1]

        outcome = standardize.LocalizeTimeToUtc("Europe/Berlin").apply(_dataset(local, a=np.arange(36.0)))

        assert isinstance(outcome, core.TransformRefused)
        assert "without guessing" in outcome.reason

    def test_refuses_a_local_time_that_does_not_exist(self):
        index = pd.date_range("2026-03-29 01:30", periods=4, freq="10min")

        outcome = standardize.LocalizeTimeToUtc("Europe/Berlin").apply(_dataset(index, a=[1.0] * 4))

        assert isinstance(outcome, core.TransformRefused)

    def test_an_axis_with_a_time_zone_is_converted(self):
        index = pd.date_range("2026-01-15 12:00", periods=3, freq="10min", tz="Europe/Berlin")

        outcome = _transformed(standardize.LocalizeTimeToUtc("Europe/Berlin").apply(_dataset(index, a=[1.0] * 3)))

        assert outcome.frame.index[0] == pd.Timestamp("2026-01-15 11:00", tz="UTC")
        assert "already had time zone" in str(outcome.rule)

    def test_rejects_an_unknown_time_zone(self):
        with pytest.raises(ValueError, match="Nowhere/Land"):
            standardize.LocalizeTimeToUtc("Nowhere/Land")

    def test_from_subject_reads_the_site_time_zone(self):
        subject = core.Subject("t1", "wind_turbine", {"site_timezone": "Europe/Berlin"})

        assert standardize.LocalizeTimeToUtc.from_subject(subject).timezone == "Europe/Berlin"
        with pytest.raises(KeyError):
            standardize.LocalizeTimeToUtc.from_subject(core.Subject("t2", "wind_turbine"))
        with pytest.raises(ValueError, match="not a time zone name"):
            standardize.LocalizeTimeToUtc.from_subject(core.Subject("t3", "wind_turbine", {"site_timezone": 5}))


class TestChain:
    def test_applies_the_steps_in_order_and_records_each(self):
        dataset = _dataset(pd.date_range("2026-01-15 12:00", periods=3, freq="10min", name="datetime"), power=[1.0] * 3)
        chain = standardize.Chain(
            (standardize.RenameToCatalog({"power": "power_kw"}), standardize.LocalizeTimeToUtc("Europe/Berlin"))
        )

        outcome = _transformed(chain.apply(dataset))

        assert list(outcome.frame.columns) == ["power_kw"]
        assert str(outcome.frame.index.tz) == "UTC"
        assert [step["name"] for step in outcome.parameters["steps"]] == ["rename_to_catalog", "localize_time_to_utc"]
        assert "localize_time_to_utc:" in str(outcome.rule)

    def test_the_first_refusal_stops_the_chain_and_names_the_step(self):
        dataset = _dataset(GRID, power=[1.0] * 4)
        chain = standardize.Chain((standardize.RenameToCatalog({"missing": "x"}), standardize.LocalizeTimeToUtc("UTC")))

        outcome = chain.apply(dataset)

        assert isinstance(outcome, core.TransformRefused)
        assert outcome.reason.startswith("rename_to_catalog:")

    def test_needs_a_step(self):
        with pytest.raises(ValueError):
            standardize.Chain(())
