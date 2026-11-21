import numpy as np
import pandas as pd

from phoibe.dokimasia import core, rules

GRID = pd.date_range("2026-01-01 00:00", periods=10, freq="10min", name="datetime")


def _dataset(frame: pd.DataFrame) -> core.Dataset:
    return core.Dataset.from_frame(frame)


class TestMissingRunsMeasure:
    def test_locates_each_run_per_signal(self):
        frame = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)}, index=GRID)
        frame.iloc[2:5, 0] = np.nan
        frame.iloc[8, 0] = np.nan
        frame.iloc[0:2, 1] = np.nan

        outcome = rules.MissingRunsMeasure().measure(_dataset(frame))

        assert isinstance(outcome, core.Measured)
        assert [(f.signals, f.start, f.end, f.n_steps) for f in outcome.findings] == [
            (("a",), GRID[2], GRID[4], 3),
            (("a",), GRID[8], GRID[8], 1),
            (("b",), GRID[0], GRID[1], 2),
        ]
        metrics = {(m.signal, m.key): m.value for m in outcome.metrics}
        assert metrics[("a", "n_missing")] == 4.0
        assert metrics[("a", "share_missing")] == 0.4
        assert metrics[("a", "n_runs")] == 2.0
        assert metrics[("a", "longest_run_steps")] == 3.0
        assert metrics[("b", "n_missing")] == 2.0

    def test_text_infinity_and_none_in_object_columns(self):
        frame = pd.DataFrame({"a": ["1,5", None, "x", np.inf, np.nan] + [1.0] * 5}, index=GRID)

        outcome = rules.MissingRunsMeasure().measure(_dataset(frame))

        assert isinstance(outcome, core.Measured)
        assert [(f.start, f.n_steps) for f in outcome.findings] == [(GRID[1], 1), (GRID[4], 1)]

    def test_leaves_out_rows_without_a_valid_timestamp_and_reads_in_time_order(self):
        index = pd.DatetimeIndex([GRID[2], pd.NaT, GRID[0], GRID[1]])
        frame = pd.DataFrame({"a": [np.nan, np.nan, 1.0, np.nan]}, index=index)

        outcome = rules.MissingRunsMeasure().measure(_dataset(frame))

        assert isinstance(outcome, core.Measured)
        assert [(f.start, f.end, f.n_steps) for f in outcome.findings] == [(GRID[1], GRID[2], 2)]

    def test_handles_duplicate_column_names(self):
        frame = pd.DataFrame(np.ones((10, 2)), index=GRID, columns=["a", "a"])
        frame.iloc[3, 1] = np.nan

        outcome = rules.MissingRunsMeasure().measure(_dataset(frame))

        assert isinstance(outcome, core.Measured)
        assert [(f.signals, f.start) for f in outcome.findings] == [(("a",), GRID[3])]

    def test_given_no_signal_is_not_evaluable(self):
        outcome = rules.MissingRunsMeasure().measure(_dataset(pd.DataFrame(index=GRID)))

        assert isinstance(outcome, core.NotEvaluable)
        assert outcome.reason == core.NotEvaluableReason.MISSING_SIGNAL

    def test_given_no_row_is_not_evaluable(self):
        outcome = rules.MissingRunsMeasure().measure(_dataset(pd.DataFrame({"a": []}, index=GRID[:0])))

        assert isinstance(outcome, core.NotEvaluable)
        assert outcome.reason == core.NotEvaluableReason.INSUFFICIENT_DATA
