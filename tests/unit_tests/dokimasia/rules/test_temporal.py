import numpy as np
import pandas as pd
import pytest

from phoibe.dokimasia import core, rules


def _dataset(index: pd.DatetimeIndex) -> core.Dataset:
    return core.Dataset.from_frame(pd.DataFrame({"a": np.arange(len(index), dtype=float)}, index=index))


def _metrics(outcome: core.Measured) -> dict[str, float]:
    return {metric.key: metric.value for metric in outcome.metrics}


GRID = pd.date_range("2026-01-01 00:00", periods=20, freq="10min", name="datetime")


class TestGapMeasure:
    def test_locates_each_gap_by_the_missing_timestamps(self):
        index = GRID.delete([5, 6, 7, 12])

        outcome = rules.GapMeasure().measure(_dataset(index))

        assert isinstance(outcome, core.Measured)
        assert [(f.kind, f.start, f.end, f.n_steps, f.magnitude) for f in outcome.findings] == [
            ("gap", GRID[5], GRID[7], 3, 1800.0),
            ("gap", GRID[12], GRID[12], 1, 600.0),
        ]
        assert _metrics(outcome) | {} == pytest.approx(
            {
                "step_s": 600.0,
                "n_gaps": 2.0,
                "n_missing_steps": 4.0,
                "longest_gap_steps": 3.0,
                "share_missing_steps": 4 / 20,
                "n_missing_timestamps": 0.0,
                "n_duplicate_timestamps": 0.0,
                "n_decreasing_steps": 0.0,
            }
        )

    def test_complete_data_has_no_finding(self):
        outcome = rules.GapMeasure().measure(_dataset(GRID))

        assert isinstance(outcome, core.Measured)
        assert outcome.findings == ()
        assert _metrics(outcome)["longest_gap_steps"] == 0.0

    def test_uses_the_expected_step_when_given(self):
        index = GRID[::2]

        inferred = rules.GapMeasure().measure(_dataset(index))
        expected = rules.GapMeasure(expected_step=pd.Timedelta("10min")).measure(_dataset(index))

        assert isinstance(inferred, core.Measured)
        assert isinstance(expected, core.Measured)
        assert inferred.findings == ()
        assert len(expected.findings) == len(index) - 1

    def test_rejects_a_step_that_is_not_positive(self):
        with pytest.raises(ValueError, match="positive"):
            rules.GapMeasure(expected_step=pd.Timedelta(0))

    @pytest.mark.parametrize("n_rows", [0, 1])
    def test_given_fewer_than_two_timestamps_is_not_evaluable(self, n_rows):
        outcome = rules.GapMeasure().measure(_dataset(GRID[:n_rows]))

        assert isinstance(outcome, core.NotEvaluable)
        assert outcome.reason == core.NotEvaluableReason.INSUFFICIENT_DATA

    def test_given_only_duplicates_is_not_evaluable(self):
        outcome = rules.GapMeasure().measure(_dataset(GRID[[3, 3, 3]]))

        assert isinstance(outcome, core.NotEvaluable)

    def test_counts_missing_duplicate_and_decreasing_timestamps_without_repairing_them(self):
        index = pd.DatetimeIndex([GRID[0], GRID[2], GRID[1], GRID[1], pd.NaT, GRID[3], GRID[4]])

        outcome = rules.GapMeasure().measure(_dataset(index))

        assert isinstance(outcome, core.Measured)
        metrics = _metrics(outcome)
        assert metrics["n_missing_timestamps"] == 1.0
        assert metrics["n_duplicate_timestamps"] == 1.0
        assert metrics["n_decreasing_steps"] == 1.0
        assert outcome.findings == ()

    def test_measures_absolute_time_on_a_time_axis_with_a_time_zone(self):
        index = GRID.tz_localize("Europe/Berlin").delete([4, 5])

        outcome = rules.GapMeasure().measure(_dataset(index))

        assert isinstance(outcome, core.Measured)
        assert [(f.start, f.n_steps) for f in outcome.findings] == [(GRID.tz_localize("Europe/Berlin")[4], 2)]

    def test_a_gap_at_the_edges_cannot_be_seen(self):
        outcome = rules.GapMeasure().measure(_dataset(GRID[3:-3]))

        assert isinstance(outcome, core.Measured)
        assert outcome.findings == ()
