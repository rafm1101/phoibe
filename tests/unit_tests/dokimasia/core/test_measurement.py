import logging

import pandas as pd
import pytest

from phoibe.dokimasia import core


class _Measure:
    id = "probe"
    aspect = "testing"

    def __init__(self, result):
        self.result = result

    def measure(self, dataset):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.fixture
def dataset() -> core.Dataset:
    index = pd.date_range("2026-01-01", periods=3, freq="10min")
    return core.Dataset.from_frame(pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=index))


def test_run_measure_returns_the_outcome_of_the_measure(dataset):
    measured = core.Measured(measure_id="probe", metrics=(core.Metric("n", 1.0),))
    not_evaluable = core.NotEvaluable("probe", core.NotEvaluableReason.MISSING_SIGNAL, "no a")

    assert core.run_measure(_Measure(measured), dataset) is measured
    assert core.run_measure(_Measure(not_evaluable), dataset) is not_evaluable


def test_run_measure_given_an_exception_returns_an_error_outcome(dataset, caplog):
    with caplog.at_level(logging.WARNING, logger="phoibe.dokimasia.core.measurement"):
        outcome = core.run_measure(_Measure(ZeroDivisionError("boom")), dataset)

    assert outcome == core.MeasurementError("probe", "ZeroDivisionError", "boom")
    assert "ZeroDivisionError" in caplog.text


def test_run_measure_given_a_result_that_is_no_outcome_returns_an_error_outcome(dataset):
    outcome = core.run_measure(_Measure({"n": 1}), dataset)

    assert isinstance(outcome, core.MeasurementError)
    assert outcome.exception_type == "dict"
