"""ADR-12: every measure is total. A hostile corpus runs through every measure.

Each case must give a valid outcome without an exception and, where nothing can be evaluated, `NotEvaluable`
and never a measured result.
"""

import numpy as np
import pandas as pd
import pytest

from phoibe.dokimasia import core, rules

MEASURES = [rules.GapMeasure(), rules.MissingRunsMeasure()]
GRID = pd.date_range("2026-01-01", periods=12, freq="10min", name="datetime")


def _frame(index: pd.DatetimeIndex, **columns: object) -> pd.DataFrame:
    return pd.DataFrame(columns, index=index)


def _corpus() -> dict[str, pd.DataFrame]:
    numbers = np.arange(12.0)
    return {
        "empty_frame": pd.DataFrame({"a": []}, index=pd.DatetimeIndex([], name="datetime")),
        "no_columns": pd.DataFrame(index=GRID),
        "one_row": _frame(GRID[:1], a=[1.0]),
        "all_missing_values": _frame(GRID, a=np.nan, b=np.nan),
        "all_missing_timestamps": _frame(pd.DatetimeIndex([pd.NaT] * 4), a=[1.0, 2.0, 3.0, 4.0]),
        "text_in_numeric_column": _frame(GRID, a=["1,5", "2.0", "x"] * 4),
        "comma_decimals_and_sentinels": _frame(GRID, a=["1,5"] * 6 + [-9999] * 6),
        "infinities": _frame(GRID, a=np.where(numbers % 2 == 0, np.inf, -np.inf)),
        "unsorted_timestamps": _frame(GRID[::-1], a=numbers),
        "duplicate_timestamps": _frame(GRID[[0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5]], a=numbers),
        "constant_timestamps": _frame(GRID[[0] * 12], a=numbers),
        "duplicate_column_names": pd.DataFrame(np.ones((12, 3)), index=GRID, columns=["a", "a", "b"]),
        "mixed_object_cells": _frame(GRID, a=[1.0, "x", None, [1, 2], {"k": 1}, pd.NaT] * 2),
        "naive_and_far_apart": _frame(pd.DatetimeIndex(["1970-01-01", "2262-01-01"]), a=[1.0, 2.0]),
        "time_zone_aware": _frame(GRID.tz_localize("Europe/Berlin"), a=numbers),
        "large_frame": pd.DataFrame(
            np.random.default_rng(0).normal(size=(200_000, 10)),
            index=pd.date_range("2020-01-01", periods=200_000, freq="1min"),
        ),
    }


CORPUS = _corpus()
MUST_NOT_BE_EVALUATED = {
    ("empty_frame", "time_gaps"),
    ("empty_frame", "missing_runs"),
    ("no_columns", "missing_runs"),
    ("one_row", "time_gaps"),
    ("all_missing_timestamps", "time_gaps"),
    ("all_missing_timestamps", "missing_runs"),
    ("constant_timestamps", "time_gaps"),
}


@pytest.mark.parametrize("measure", MEASURES, ids=lambda measure: measure.id)
@pytest.mark.parametrize("case", CORPUS)
def test_measure_is_total_on_hostile_input(measure, case):
    dataset = core.Dataset.from_frame(CORPUS[case])

    outcome = core.run_measure(measure, dataset)

    assert not isinstance(outcome, core.MeasurementError), outcome
    if (case, measure.id) in MUST_NOT_BE_EVALUATED:
        assert isinstance(outcome, core.NotEvaluable)
    else:
        assert isinstance(outcome, core.Measured)
        assert all(np.isfinite(metric.value) for metric in outcome.metrics)
