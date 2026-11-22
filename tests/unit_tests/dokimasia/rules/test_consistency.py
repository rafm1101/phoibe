import numpy as np
import pandas as pd
import pytest

from phoibe.dokimasia import core, rules, wind

GRID = pd.date_range("2026-01-01", periods=10, freq="10min", name="datetime")


def _dataset(**columns) -> core.Dataset:
    return core.Dataset.from_frame(pd.DataFrame(columns, index=GRID))


def _metrics(outcome: core.Measured) -> dict[str, float]:
    return {metric.key: metric.value for metric in outcome.metrics}


class TestStatisticOrderMeasure:
    def _measure(self, **kwargs) -> rules.StatisticOrderMeasure:
        return rules.StatisticOrderMeasure("lo", "mid", "hi", **kwargs)

    def test_a_consistent_triple_has_no_finding(self):
        lo, mid, hi = np.arange(10.0), np.arange(10.0) + 1, np.arange(10.0) + 2

        outcome = self._measure().measure(_dataset(lo=lo, mid=mid, hi=hi))

        assert isinstance(outcome, core.Measured)
        assert outcome.findings == ()
        assert _metrics(outcome) == {
            "n_violations": 0.0,
            "share_violations": 0.0,
            "n_checked": 10.0,
            "n_runs": 0.0,
            "tolerance": 0.0,
        }

    def test_equal_values_are_in_order(self):
        constant = np.full(10, 3.0)

        outcome = self._measure().measure(_dataset(lo=constant, mid=constant, hi=constant))

        assert outcome.findings == ()

    def test_finds_a_mean_above_the_maximum_and_below_the_minimum_as_runs(self):
        lo = np.zeros(10)
        mid = np.array([1.0, 1, 9, 9, 1, 1, -3, 1, 1, 1])
        hi = np.full(10, 5.0)

        outcome = self._measure().measure(_dataset(lo=lo, mid=mid, hi=hi))

        assert [(f.kind, f.signals, f.start, f.end, f.n_steps) for f in outcome.findings] == [
            ("statistic_order", ("lo", "mid", "hi"), GRID[2], GRID[3], 2),
            ("statistic_order", ("lo", "mid", "hi"), GRID[6], GRID[6], 1),
        ]
        assert _metrics(outcome)["n_violations"] == 3.0
        assert _metrics(outcome)["n_runs"] == 2.0

    def test_a_minimum_above_the_maximum_is_found_wherever_the_mean_lies(self):
        for mean in (1.0, 3.0, 6.0):
            outcome = self._measure().measure(
                _dataset(lo=np.full(10, 4.0), mid=np.full(10, mean), hi=np.r_[np.full(9, 5.0), 2.0])
            )

            assert GRID[9] in {f.end for f in outcome.findings}, mean

    def test_does_not_check_rows_with_a_missing_value(self):
        mid = np.array([np.nan, 9, 1, 1, 1, 1, 1, 1, 1, 1])

        outcome = self._measure().measure(_dataset(lo=np.zeros(10), mid=mid, hi=np.full(10, 5.0)))

        assert [(f.start, f.n_steps) for f in outcome.findings] == [(GRID[1], 1)]
        assert _metrics(outcome)["n_checked"] == 9.0
        assert _metrics(outcome)["share_violations"] == pytest.approx(1 / 9)

    def test_the_tolerance_allows_rounding(self):
        mid = np.full(10, 5.0 + 1e-9)
        data = _dataset(lo=np.zeros(10), mid=mid, hi=np.full(10, 5.0))

        assert len(self._measure().measure(data).findings) == 1
        assert self._measure(tolerance=1e-6).measure(data).findings == ()

    def test_reads_text_numbers_and_never_checks_text_that_is_no_number(self):
        mid = ["1", "x", 1, 1, 1, 1, 1, 1, 1, "9"]

        outcome = self._measure().measure(_dataset(lo=np.zeros(10), mid=mid, hi=np.full(10, 5.0)))

        assert [(f.start, f.n_steps) for f in outcome.findings] == [(GRID[9], 1)]
        assert _metrics(outcome)["n_checked"] == 9.0

    def test_orders_by_time_before_locating_runs(self):
        index = GRID[::-1]
        frame = pd.DataFrame({"lo": np.zeros(10), "mid": [9.0] * 2 + [1.0] * 8, "hi": np.full(10, 5.0)}, index=index)

        outcome = self._measure().measure(core.Dataset.from_frame(frame))

        assert [(f.start, f.end) for f in outcome.findings] == [(GRID[8], GRID[9])]

    def test_names_every_missing_signal(self):
        outcome = self._measure().measure(_dataset(mid=np.zeros(10)))

        assert outcome == core.NotEvaluable(
            "statistic_order", core.NotEvaluableReason.MISSING_SIGNAL, "signals not in the dataset: lo, hi"
        )

    def test_duplicate_columns_are_not_evaluable(self):
        frame = pd.DataFrame(np.zeros((10, 4)), index=GRID, columns=["lo", "mid", "mid", "hi"])

        outcome = self._measure().measure(core.Dataset.from_frame(frame))

        assert isinstance(outcome, core.NotEvaluable)
        assert outcome.reason is core.NotEvaluableReason.MISSING_STRUCTURE

    def test_no_row_with_all_three_values_is_not_evaluable(self):
        outcome = self._measure().measure(_dataset(lo=np.zeros(10), mid=np.full(10, np.nan), hi=np.ones(10)))

        assert isinstance(outcome, core.NotEvaluable)
        assert outcome.reason is core.NotEvaluableReason.INSUFFICIENT_DATA

    @pytest.mark.parametrize(
        ("signals", "tolerance"),
        [(("a", "a", "b"), 0.0), (("a", "b", "b"), 0.0), (("a", "b", "c"), -1.0), (("a", "b", "c"), float("nan"))],
    )
    def test_rejects_an_unusable_definition(self, signals, tolerance):
        with pytest.raises(ValueError, match=r"three different signals|tolerance"):
            rules.StatisticOrderMeasure(*signals, tolerance=tolerance)


class TestListStatisticTriples:
    def test_lists_a_triple_for_every_quantity_of_the_wind_catalog(self):
        triples = rules.list_statistic_triples(wind.create_catalog())

        assert triples == (
            ("power_kw_min", "power_kw", "power_kw_max"),
            ("wind_speed_min", "wind_speed", "wind_speed_max"),
            ("rotor_speed_min", "rotor_speed", "rotor_speed_max"),
            ("pitch_angle_min", "pitch_angle", "pitch_angle_max"),
        )

    def test_leaves_out_a_quantity_without_all_three_statistics(self):
        catalog = core.Catalog(
            (
                core.Signal("t", "temperature", core.Statistic.MEAN),
                core.Signal("t_max", "temperature", core.Statistic.MAX),
                core.Signal("p", "pressure", core.Statistic.MEAN),
                core.Signal("p_min", "pressure", core.Statistic.MIN),
                core.Signal("p_max", "pressure", core.Statistic.MAX),
            )
        )

        assert rules.list_statistic_triples(catalog) == (("p_min", "p", "p_max"),)

    def test_keeps_measurement_points_apart(self):
        catalog = core.Catalog(
            tuple(
                core.Signal(f"{point}{suffix}", "temperature", statistic, measurement_point=point)
                for point in ("a", "b")
                for suffix, statistic in (
                    ("", core.Statistic.MEAN),
                    ("_lo", core.Statistic.MIN),
                    ("_hi", core.Statistic.MAX),
                )
            )
        )

        assert rules.list_statistic_triples(catalog) == (("a_lo", "a", "a_hi"), ("b_lo", "b", "b_hi"))


class TestFrozenRunsMeasure:
    def _measure(self, **kwargs) -> rules.FrozenRunsMeasure:
        return rules.FrozenRunsMeasure(("a", "b"), **{"min_steps": 2, **kwargs})

    def test_varying_signals_have_no_finding(self):
        outcome = self._measure().measure(_dataset(a=np.arange(10.0), b=np.arange(10.0) * 2))

        assert isinstance(outcome, core.Measured)
        assert outcome.findings == ()
        assert _metrics(outcome) == {
            "n_frozen_steps": 0.0,
            "share_frozen_steps": 0.0,
            "n_runs": 0.0,
            "longest_run_steps": 0.0,
            "min_steps": 2.0,
        }

    def test_a_state_held_by_all_signals_is_one_finding_including_the_row_that_starts_it(self):
        a = np.array([0.0, 1, 2, 2, 2, 5, 6, 7, 8, 9])
        b = np.array([0.0, 1, 7, 7, 7, 3, 3, 4, 5, 6])

        outcome = self._measure().measure(_dataset(a=a, b=b))

        assert [(f.kind, f.signals, f.start, f.end, f.n_steps) for f in outcome.findings] == [
            ("frozen", ("a", "b"), GRID[2], GRID[4], 3)
        ]
        assert _metrics(outcome)["n_frozen_steps"] == 3.0
        assert _metrics(outcome)["share_frozen_steps"] == 0.3
        assert _metrics(outcome)["longest_run_steps"] == 3.0

    def test_one_signal_staying_constant_is_not_a_finding(self):
        outcome = self._measure().measure(_dataset(a=np.zeros(10), b=np.arange(10.0)))

        assert outcome.findings == ()

    def test_missing_values_never_repeat(self):
        nan = np.full(10, np.nan)

        outcome = self._measure().measure(_dataset(a=nan, b=nan))

        assert outcome.findings == ()

    def test_a_run_shorter_than_min_steps_is_no_finding(self):
        a = np.array([0.0, 1, 1, 2, 3, 3, 3, 4, 5, 6])

        short = self._measure(min_steps=3).measure(_dataset(a=a, b=a))

        assert [(f.start, f.n_steps) for f in short.findings] == [(GRID[4], 3)]
        assert _metrics(short)["min_steps"] == 3.0
        assert _metrics(short)["n_runs"] == 1.0

    def test_three_signals_must_all_repeat(self):
        rule = rules.FrozenRunsMeasure(("a", "b", "c"), min_steps=2)
        a = np.array([0.0, 1, 1, 1, 4, 5, 6, 7, 8, 9])

        outcome = rule.measure(_dataset(a=a, b=a, c=np.arange(10.0)))

        assert outcome.findings == ()

    def test_orders_by_time_and_leaves_out_rows_without_a_timestamp(self):
        index = GRID[::-1].insert(0, pd.NaT)
        values = np.r_[0.0, [5.0] * 3, np.arange(7.0)]
        frame = pd.DataFrame({"a": values, "b": values}, index=index)

        outcome = self._measure().measure(core.Dataset.from_frame(frame))

        assert [(f.start, f.end, f.n_steps) for f in outcome.findings] == [(GRID[7], GRID[9], 3)]
        assert _metrics(outcome)["share_frozen_steps"] == 0.3

    def test_names_every_missing_signal(self):
        outcome = self._measure().measure(_dataset(a=np.zeros(10)))

        assert outcome == core.NotEvaluable(
            "frozen_runs", core.NotEvaluableReason.MISSING_SIGNAL, "signals not in the dataset: b"
        )

    def test_a_single_row_is_not_evaluable(self):
        frame = pd.DataFrame({"a": [1.0], "b": [1.0]}, index=GRID[:1])

        outcome = self._measure().measure(core.Dataset.from_frame(frame))

        assert isinstance(outcome, core.NotEvaluable)
        assert outcome.reason is core.NotEvaluableReason.INSUFFICIENT_DATA

    @pytest.mark.parametrize(
        ("signals", "min_steps"), [(("a",), 2), (("a", "a"), 2), ((), 2), (("a", "b"), 1), (("a", "b"), 0)]
    )
    def test_rejects_an_unusable_definition(self, signals, min_steps):
        with pytest.raises(ValueError, match=r"at least two|min_steps"):
            rules.FrozenRunsMeasure(signals, min_steps=min_steps)
