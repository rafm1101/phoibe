import pandas as pd
import pytest

from phoibe.dokimasia import core


def _finding(n_steps: int) -> core.Finding:
    start = pd.Timestamp("2026-01-01")
    return core.Finding(kind="gap", signals=(), start=start, end=start, n_steps=n_steps)


class _Measure:
    def __init__(self, id, aspect, result):
        self.id = id
        self.aspect = aspect
        self.result = result

    def measure(self, dataset):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def dataset() -> core.Dataset:
    index = pd.date_range("2026-01-01", periods=3, freq="10min")
    return core.Dataset.from_frame(pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=index))


SUBJECT = core.Subject(id="t1", device_type="wind_turbine")


class TestProfileDataset:
    def test_groups_outcomes_by_aspect_in_run_order(self, dataset):
        measures = [
            _Measure("m1", "temporal", core.Measured("m1", findings=(_finding(2),))),
            _Measure("m2", "completeness", core.Measured("m2")),
            _Measure("m3", "temporal", core.Measured("m3", findings=(_finding(5),))),
        ]

        profile = core.profile_dataset(dataset, SUBJECT, "raw", measures)

        assert list(profile.aspects) == ["temporal", "completeness"]
        assert [outcome.measure_id for outcome in profile.aspects["temporal"]] == ["m1", "m3"]
        assert profile.layer == "raw"
        assert profile.subject is SUBJECT
        assert profile.fingerprint == dataset.fingerprint

    def test_one_failing_measure_does_not_stop_the_others(self, dataset):
        measures = [
            _Measure("m1", "temporal", RuntimeError("boom")),
            _Measure("m2", "temporal", core.Measured("m2")),
        ]

        profile = core.profile_dataset(dataset, SUBJECT, "raw", measures)

        first, second = profile.aspects["temporal"]
        assert isinstance(first, core.MeasurementError)
        assert isinstance(second, core.Measured)

    def test_rejects_duplicate_measure_ids(self, dataset):
        measures = [_Measure("m1", "a", core.Measured("m1")), _Measure("m1", "b", core.Measured("m1"))]

        with pytest.raises(ValueError, match="m1"):
            core.profile_dataset(dataset, SUBJECT, "raw", measures)

    def test_a_profile_without_measures_is_empty(self, dataset):
        profile = core.profile_dataset(dataset, SUBJECT, "raw", [])

        assert profile.list_outcomes() == ()
        assert profile.build_findings_table().empty


class TestProfile:
    @pytest.fixture
    def profile(self, dataset) -> core.Profile:
        measures = [
            _Measure(
                "m1",
                "temporal",
                core.Measured("m1", metrics=(core.Metric("n", 2.0),), findings=(_finding(2), _finding(7))),
            ),
            _Measure("m2", "completeness", core.NotEvaluable("m2", core.NotEvaluableReason.MISSING_SIGNAL, "no a")),
            _Measure("m3", "completeness", ValueError("bad")),
        ]
        return core.profile_dataset(dataset, SUBJECT, "raw", measures)

    def test_aspects_are_read_only(self, profile):
        with pytest.raises(TypeError):
            profile.aspects["x"] = ()  # type: ignore[index]

    def test_lists_findings_of_measured_outcomes_only(self, profile):
        assert [finding.n_steps for finding in profile.list_findings()] == [2, 7]
        assert profile.list_findings(kind="missing") == ()

    def test_findings_table_is_sorted_largest_first(self, profile):
        table = profile.build_findings_table()

        assert table["n_steps"].tolist() == [7, 2]
        assert set(table["measure_id"]) == {"m1"}

    def test_summary_table_holds_the_metrics(self, profile):
        table = profile.build_summary_table()

        assert table[["measure_id", "key", "value"]].to_dict("records") == [
            {"measure_id": "m1", "key": "n", "value": 2.0}
        ]

    def test_status_table_says_what_became_of_each_measure(self, profile):
        table = profile.build_status_table()

        assert table["status"].tolist() == ["measured", "not_evaluable", "error"]
        assert "missing_signal" in table["detail"].iloc[1]
        assert "ValueError: bad" in table["detail"].iloc[2]

    def test_status_table_holds_the_duration_of_each_measure(self, profile):
        table = profile.build_status_table()

        assert (table["duration_s"] >= 0).all()
        assert table["duration_s"].sum() == pytest.approx(profile.compute_duration_s())

    def test_counts_by_status_include_zeros(self, profile):
        assert profile.count_by_status() == {"measured": 1, "not_evaluable": 1, "error": 1}
        empty = core.Profile(SUBJECT, "raw", profile.fingerprint, {})
        assert empty.count_by_status() == {"measured": 0, "not_evaluable": 0, "error": 0}

    def test_to_frame_has_a_row_per_metric_and_a_row_per_measure_without_metrics(self, profile):
        frame = profile.to_frame()

        assert frame["measure_id"].tolist() == ["m1", "m2", "m3"]
        assert frame["status"].tolist() == ["measured", "not_evaluable", "error"]
        assert frame.loc[0, "key"] == "n"
        assert frame.loc[0, "value"] == 2.0
        assert frame["key"].iloc[1:].isna().all()
        assert list(frame.columns) == [
            "aspect",
            "measure_id",
            "status",
            "key",
            "signal",
            "value",
            "unit",
            "duration_s",
        ]

    def test_to_frame_of_an_empty_profile_is_empty_with_its_columns(self, dataset):
        profile = core.profile_dataset(dataset, SUBJECT, "raw", [])

        assert profile.to_frame().empty
        assert "duration_s" in profile.to_frame().columns

    def test_repr_is_one_line_with_the_facts(self, profile):
        text = repr(profile)

        assert "\n" not in text
        assert text.startswith("Profile(subject='t1', layer='raw', fingerprint=")
        assert "3 measures: 1 measured, 1 not_evaluable, 1 error" in text
        assert "2 findings" in text

    def test_repr_of_an_empty_profile(self, dataset):
        assert "0 measures: none, 0 findings" in repr(core.profile_dataset(dataset, SUBJECT, "raw", []))
