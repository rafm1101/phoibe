import numpy as np
import pandas as pd
import pytest

from phoibe.dokimasia import core, rules
from phoibe.dokimasia.rules import distribution
from phoibe.synthetic_data import turbine

GRID = pd.date_range("2026-01-01 00:00", periods=20, freq="10min", name="datetime")


def _dataset(index: pd.DatetimeIndex) -> core.Dataset:
    return core.Dataset.from_frame(pd.DataFrame({"a": np.arange(len(index), dtype=float)}, index=index))


def _metrics(outcome: core.Measured) -> dict[str, float]:
    return {metric.key: metric.value for metric in outcome.metrics}


class TestTimeSpanMeasure:
    def test_reports_the_period_of_a_naive_regular_axis(self):
        outcome = rules.TimeSpanMeasure().measure(_dataset(GRID))

        assert isinstance(outcome, core.Measured)
        assert [
            (f.kind, f.signals, f.start, f.end, f.n_steps, f.magnitude, f.magnitude_unit) for f in outcome.findings
        ] == [("time_span", (), GRID[0], GRID[-1], 20, 19 * 600.0, "s")]
        epoch = pd.Timestamp("2026-01-01")
        assert _metrics(outcome) == {
            "n_timestamps": 20.0,
            "span_s": 19 * 600.0,
            "start_epoch_s": (epoch - pd.Timestamp("1970-01-01")).total_seconds(),
            "end_epoch_s": (epoch - pd.Timestamp("1970-01-01")).total_seconds() + 19 * 600.0,
            "is_timezone_aware": 0.0,
            "n_utc_offsets": 1.0,
            "n_short_steps": 0.0,
        }

    def test_an_axis_in_utc_has_one_offset_and_epoch_seconds_of_the_instants(self):
        outcome = rules.TimeSpanMeasure().measure(_dataset(GRID.tz_localize("UTC")))

        metrics = _metrics(outcome)
        assert (metrics["is_timezone_aware"], metrics["n_utc_offsets"]) == (1.0, 1.0)
        assert metrics["start_epoch_s"] == (pd.Timestamp("2026-01-01") - pd.Timestamp("1970-01-01")).total_seconds()

    def test_a_zone_with_a_fixed_offset_gives_the_epoch_of_the_instant_not_the_wall_time(self):
        index = GRID.tz_localize("Etc/GMT-1")  # UTC+1 all year

        outcome = rules.TimeSpanMeasure().measure(_dataset(index))

        metrics = _metrics(outcome)
        assert metrics["n_utc_offsets"] == 1.0
        assert (
            metrics["start_epoch_s"] == (pd.Timestamp("2025-12-31 23:00") - pd.Timestamp("1970-01-01")).total_seconds()
        )
        assert outcome.findings[0].start == index[0]

    def test_daylight_saving_time_gives_two_offsets(self):
        index = pd.date_range("2026-03-28", periods=4, freq="D", tz="Europe/Berlin")

        outcome = rules.TimeSpanMeasure().measure(_dataset(index))

        assert _metrics(outcome)["n_utc_offsets"] == 2.0

    def test_locates_runs_of_oversampled_steps(self):
        index = GRID.delete([4, 5]).append(
            pd.DatetimeIndex([GRID[3] + pd.Timedelta("1min"), GRID[3] + pd.Timedelta("2min")])
        )

        outcome = rules.TimeSpanMeasure(expected_step=pd.Timedelta("10min")).measure(_dataset(index.sort_values()))

        short = [f for f in outcome.findings if f.kind == "short_step"]
        assert [(f.start, f.end, f.n_steps) for f in short] == [(GRID[3], GRID[3] + pd.Timedelta("2min"), 3)]
        assert _metrics(outcome)["n_short_steps"] == 2.0

    def test_a_step_of_half_the_expected_length_or_more_is_not_short(self):
        index = GRID[:5].append(pd.DatetimeIndex([GRID[4] + pd.Timedelta("5min")])).sort_values()

        outcome = rules.TimeSpanMeasure(expected_step=pd.Timedelta("10min")).measure(_dataset(index))

        assert _metrics(outcome)["n_short_steps"] == 0.0

    def test_missing_duplicate_and_unsorted_timestamps_do_not_change_the_period(self):
        index = pd.DatetimeIndex([GRID[3], pd.NaT, GRID[1], GRID[1], GRID[2]])

        outcome = rules.TimeSpanMeasure().measure(_dataset(index))

        assert (outcome.findings[0].start, outcome.findings[0].end) == (GRID[1], GRID[3])
        assert _metrics(outcome)["n_timestamps"] == 3.0

    def test_one_timestamp_is_a_period_of_zero_with_one_step(self):
        outcome = rules.TimeSpanMeasure().measure(_dataset(GRID[:1]))

        assert [(f.n_steps, f.magnitude) for f in outcome.findings] == [(1, 0.0)]
        assert _metrics(outcome)["span_s"] == 0.0

    def test_no_valid_timestamp_is_not_evaluable(self):
        outcome = rules.TimeSpanMeasure().measure(_dataset(pd.DatetimeIndex([pd.NaT, pd.NaT])))

        assert isinstance(outcome, core.NotEvaluable)
        assert outcome.reason is core.NotEvaluableReason.INSUFFICIENT_DATA

    def test_rejects_a_step_that_is_not_positive(self):
        with pytest.raises(ValueError, match="expected_step"):
            rules.TimeSpanMeasure(expected_step=pd.Timedelta(0))


class TestFindShortestInterval:
    def test_finds_the_densest_stretch_of_sorted_values(self):
        values = np.array([0.0, 1, 2, 3, 4, 100, 5, 6])

        assert distribution.find_shortest_interval(values, 0.75) == (0.0, 6.0)

    def test_the_whole_range_for_a_proportion_of_one_but_the_last_value_is_a_must(self):
        values = np.array([3.0, 1.0, 2.0])

        assert distribution.find_shortest_interval(values, 1.0) == (1.0, 3.0)

    def test_one_value_is_an_interval_of_width_zero(self):
        assert distribution.find_shortest_interval(np.array([4.0]), 0.995) == (4.0, 4.0)

    def test_the_lowest_interval_wins_a_tie(self):
        assert distribution.find_shortest_interval(np.array([0.0, 1, 2, 10, 11, 12]), 0.34) == (0.0, 10.0)


class TestEssentialRangeMeasure:
    def test_leaves_out_the_tail_and_reports_the_bulk(self):
        index = pd.date_range("2026-01-01", periods=1000, freq="10min")
        frame = pd.DataFrame({"a": np.r_[np.linspace(10, 20, 999), -500.0]}, index=index)

        outcome = rules.EssentialRangeMeasure("a").measure(core.Dataset.from_frame(frame))

        metrics = _metrics(outcome)
        assert metrics["essential_min"] == 10.0
        assert 19.9 < metrics["essential_max"] < 20.0
        assert metrics["essential_width"] == metrics["essential_max"] - 10.0
        assert metrics["n_values"] == 1000.0
        assert metrics["proportion"] == 0.995
        assert outcome.findings == ()

    def test_skips_missing_and_infinite_values_and_reads_text_numbers(self):
        index = pd.date_range("2026-01-01", periods=6, freq="10min")
        frame = pd.DataFrame({"a": [np.nan, np.inf, "5", 1.0, 3.0, "x"]}, index=index)

        outcome = rules.EssentialRangeMeasure("a", proportion=1.0).measure(core.Dataset.from_frame(frame))

        metrics = _metrics(outcome)
        assert (metrics["essential_min"], metrics["essential_max"], metrics["n_values"]) == (1.0, 5.0, 3.0)

    def test_a_signal_without_a_finite_value_is_not_evaluable(self):
        index = pd.date_range("2026-01-01", periods=3, freq="10min")
        frame = pd.DataFrame({"a": [np.nan, np.inf, np.nan]}, index=index)

        outcome = rules.EssentialRangeMeasure("a").measure(core.Dataset.from_frame(frame))

        assert isinstance(outcome, core.NotEvaluable)
        assert outcome.reason is core.NotEvaluableReason.INSUFFICIENT_DATA

    def test_a_missing_signal_is_not_evaluable(self):
        outcome = rules.EssentialRangeMeasure("b").measure(_dataset(GRID))

        assert outcome == core.NotEvaluable(
            "essential_range", core.NotEvaluableReason.MISSING_SIGNAL, "signals not in the dataset: b"
        )

    @pytest.mark.parametrize("proportion", [0.0, -0.1, 1.1, float("nan")])
    def test_rejects_a_proportion_outside_zero_to_one(self, proportion):
        with pytest.raises(ValueError, match="proportion"):
            rules.EssentialRangeMeasure("a", proportion=proportion)

    def test_matches_the_essential_range_of_layered(self):
        from phoibe.layered.application.context import ValidationContext
        from phoibe.layered.rules.rules_values import EssentialRange

        time = turbine.Time(start="2026-02-14T00:00:00", freq="10min", periods=1008)
        frame = turbine.generate_wtg_scada(time=time, seed=3)
        legacy = EssentialRange(["wind_speed", "power_kw"]).execute(
            frame.reset_index(drop=True),
            ValidationContext(
                layer_name="t", detected_variables={"wind_speed": "wind_speed", "power_kw": "power"}, turbine_id="t"
            ),
        )

        for name, key in (("wind_speed", "wind_speed"), ("power", "power_kw")):
            metrics = _metrics(rules.EssentialRangeMeasure(name).measure(core.Dataset.from_frame(frame)))
            assert [metrics["essential_min"], metrics["essential_max"]] == legacy.details["checked"][key]
