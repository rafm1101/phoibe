import numpy as np
import pandas as pd
import pytest

from phoibe.dokimasia import core, rules

GRID = pd.date_range("2026-01-01", periods=10, freq="10min", name="datetime")


def _dataset(**columns) -> core.Dataset:
    return core.Dataset.from_frame(pd.DataFrame(columns, index=GRID))


def _metrics(outcome: core.Measured) -> dict[str, float]:
    return {metric.key: metric.value for metric in outcome.metrics}


class TestRangesMeasure:
    def test_counts_and_locates_values_outside_the_limits(self):
        values = np.array([1.0, 5.0, 5.0, 70.0, 80.0, 5.0, -1.0, 5.0, 5.0, 5.0])

        outcome = rules.RangesMeasure("a", min=0, max=60).measure(_dataset(a=values))

        assert isinstance(outcome, core.Measured)
        assert [(f.kind, f.signals, f.start, f.end, f.n_steps) for f in outcome.findings] == [
            ("out_of_range", ("a",), GRID[3], GRID[4], 2),
            ("out_of_range", ("a",), GRID[6], GRID[6], 1),
        ]
        metrics = _metrics(outcome)
        assert (metrics["n_out_of_range"], metrics["share_out_of_range"], metrics["n_checked"]) == (3.0, 0.3, 10.0)
        assert (metrics["limit_min"], metrics["limit_max"]) == (0.0, 60.0)

    def test_bounds_are_inside_the_range_and_one_limit_is_enough(self):
        outcome = rules.RangesMeasure("a", max=5).measure(_dataset(a=[5.0] * 9 + [5.1]))

        assert isinstance(outcome, core.Measured)
        assert _metrics(outcome)["n_out_of_range"] == 1.0
        assert "limit_min" not in _metrics(outcome)

    def test_text_is_read_as_numbers_and_what_cannot_be_read_is_counted_not_checked(self):
        outcome = rules.RangesMeasure("a", min=0, max=10).measure(
            _dataset(a=["1", "99", "1,5", None, "x", [1], 2.0, 3, 4, 5])
        )

        assert isinstance(outcome, core.Measured)
        metrics = _metrics(outcome)
        assert metrics["n_unreadable"] == 3.0
        assert metrics["n_checked"] == 6.0
        assert metrics["n_out_of_range"] == 1.0

    def test_infinity_is_outside_a_finite_limit(self):
        outcome = rules.RangesMeasure("a", max=10).measure(_dataset(a=[np.inf] + [1.0] * 9))

        assert isinstance(outcome, core.Measured)
        assert _metrics(outcome)["n_out_of_range"] == 1.0

    def test_given_a_missing_signal_is_not_evaluable(self):
        outcome = rules.RangesMeasure("b", min=0).measure(_dataset(a=[1.0] * 10))

        assert isinstance(outcome, core.NotEvaluable)
        assert outcome.reason == core.NotEvaluableReason.MISSING_SIGNAL

    def test_given_duplicate_columns_is_not_evaluable(self):
        frame = pd.DataFrame(np.ones((10, 2)), index=GRID, columns=["a", "a"])

        outcome = rules.RangesMeasure("a", min=0).measure(core.Dataset.from_frame(frame))

        assert isinstance(outcome, core.NotEvaluable)
        assert outcome.reason == core.NotEvaluableReason.MISSING_STRUCTURE

    def test_given_nothing_to_check_is_not_evaluable(self):
        outcome = rules.RangesMeasure("a", min=0).measure(_dataset(a=[np.nan] * 10))

        assert isinstance(outcome, core.NotEvaluable)
        assert outcome.reason == core.NotEvaluableReason.INSUFFICIENT_DATA

    @pytest.mark.parametrize(("lower", "upper"), [(None, None), (5.0, 1.0)])
    def test_rejects_limits_that_make_no_sense(self, lower, upper):
        with pytest.raises(ValueError):
            rules.RangesMeasure("a", min=lower, max=upper)

    def test_findings_are_in_time_order_and_leave_out_rows_without_a_time(self):
        index = pd.DatetimeIndex([GRID[2], pd.NaT, GRID[0], GRID[1]])
        dataset = core.Dataset.from_frame(pd.DataFrame({"a": [99.0, 99.0, 1.0, 99.0]}, index=index))

        outcome = rules.RangesMeasure("a", max=10).measure(dataset)

        assert isinstance(outcome, core.Measured)
        assert [(f.start, f.end) for f in outcome.findings] == [(GRID[1], GRID[2])]
        assert _metrics(outcome)["n_out_of_range"] == 3.0


class TestRequiredSignalsMeasure:
    def test_reports_which_signals_are_present(self):
        outcome = rules.RequiredSignalsMeasure(("a", "b", "c")).measure(_dataset(a=[1.0] * 10, c=[1.0] * 10))

        assert isinstance(outcome, core.Measured)
        assert {(m.signal, m.value) for m in outcome.metrics if m.key == "present"} == {
            ("a", 1.0),
            ("b", 0.0),
            ("c", 1.0),
        }
        assert _metrics(outcome)["n_missing_signals"] == 1.0

    def test_is_evaluable_on_a_dataset_without_rows(self):
        dataset = core.Dataset.from_frame(pd.DataFrame({"a": []}, index=GRID[:0]))

        assert isinstance(rules.RequiredSignalsMeasure(("a",)).measure(dataset), core.Measured)

    def test_rejects_an_empty_list(self):
        with pytest.raises(ValueError):
            rules.RequiredSignalsMeasure(())


class TestMissingRunsMeasureSignals:
    def test_measures_only_the_named_signals(self):
        dataset = _dataset(a=[np.nan] + [1.0] * 9, b=[np.nan] * 10)

        outcome = rules.MissingRunsMeasure(signals=("a",)).measure(dataset)

        assert isinstance(outcome, core.Measured)
        assert {f.signals for f in outcome.findings} == {("a",)}

    def test_given_a_signal_that_is_absent_is_not_evaluable(self):
        outcome = rules.MissingRunsMeasure(signals=("z",)).measure(_dataset(a=[1.0] * 10))

        assert isinstance(outcome, core.NotEvaluable)
        assert outcome.reason == core.NotEvaluableReason.MISSING_SIGNAL


@pytest.mark.parametrize(
    "measure",
    [
        rules.GapMeasure(),
        rules.MissingRunsMeasure(),
        rules.RangesMeasure("a", min=0, max=1),
        rules.RequiredSignalsMeasure(("a",)),
    ],
    ids=lambda measure: measure.id,
)
def test_the_metrics_a_measure_reports_are_the_ones_it_declares(measure):
    index = pd.date_range("2026-01-01", periods=20, freq="10min").delete([5, 6])
    dataset = core.Dataset.from_frame(pd.DataFrame({"a": np.where(np.arange(18) % 5 == 0, np.nan, 0.5)}, index=index))

    outcome = measure.measure(dataset)

    assert isinstance(outcome, core.Measured)
    assert {metric.key for metric in outcome.metrics} == set(measure.METRICS)
