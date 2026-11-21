"""Exit criterion of Phase 1: located findings match the ground truth that the generator labels."""

import pandas as pd
import pytest

from phoibe.dokimasia import core, rules
from phoibe.synthetic_data import turbine
from phoibe.synthetic_data._turbine_noise import DeleteEntries, RowDrop, UniformSegments

FREQ = pd.Timedelta("10min")
TIME = turbine.Time(start="2026-02-14T04:00:00", freq="10min", periods=576)


def _runs_of(timestamps: set[pd.Timestamp], grid: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return the maximal intervals of consecutive grid steps that are all in `timestamps`."""
    runs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = previous = None
    for timestamp in grid:
        if timestamp in timestamps:
            start = timestamp if start is None else start
            previous = timestamp
        elif start is not None:
            runs.append((start, previous))
            start = None
    if start is not None:
        runs.append((start, previous))
    return runs


def _timestamps_of(labels: list, *, kind: str, column: str | None = None) -> set[pd.Timestamp]:
    covered: set[pd.Timestamp] = set()
    for label in labels:
        if label.kind != kind or label.n_changed == 0:
            continue
        if column is not None and column not in label.columns:
            continue
        covered.update(pd.date_range(label.start, label.end, freq=FREQ))
    return covered


@pytest.fixture(scope="module")
def clean() -> pd.DataFrame:
    return turbine.generate_wtg_scada(time=TIME, seed=3)


@pytest.mark.parametrize("seed", range(8))
def test_gap_measure_finds_the_rows_that_row_drop_removed(clean, seed):
    pipeline = turbine.MessUpPipeline(seed=seed)
    pipeline.add(RowDrop(UniformSegments(n=6, min_len=1, max_len=15)))
    messy, labels = pipeline.apply_labelled(clean)

    outcome = rules.GapMeasure(expected_step=FREQ).measure(core.Dataset.from_frame(messy))

    dropped = _timestamps_of(labels, kind="RowDrop")
    expected = [run for run in _runs_of(dropped, clean.index) if run[0] != clean.index[0] and run[1] != clean.index[-1]]
    assert isinstance(outcome, core.Measured)
    assert [(finding.start, finding.end) for finding in outcome.findings] == expected
    assert [finding.n_steps for finding in outcome.findings] == [
        int((end - start) / FREQ) + 1 for start, end in expected
    ]
    assert expected, "the seed must produce at least one detectable gap"


@pytest.mark.parametrize("seed", range(8))
def test_gap_measure_infers_the_step_that_row_drop_leaves_alone(clean, seed):
    pipeline = turbine.MessUpPipeline(seed=seed)
    pipeline.add(RowDrop(UniformSegments(n=6, min_len=1, max_len=15)))
    messy, _ = pipeline.apply_labelled(clean)

    outcome = rules.GapMeasure().measure(core.Dataset.from_frame(messy))

    assert isinstance(outcome, core.Measured)
    assert {metric.key: metric.value for metric in outcome.metrics}["step_s"] == FREQ.total_seconds()


@pytest.mark.parametrize("seed", range(8))
def test_missing_runs_measure_finds_the_values_that_delete_entries_removed(clean, seed):
    pipeline = turbine.MessUpPipeline(seed=seed)
    pipeline.add(DeleteEntries(UniformSegments(n=6, min_len=1, max_len=15), columns=["power", "wind_speed"]))
    messy, labels = pipeline.apply_labelled(clean)

    outcome = rules.MissingRunsMeasure().measure(core.Dataset.from_frame(messy))

    assert isinstance(outcome, core.Measured)
    for column in ["power", "wind_speed"]:
        expected = _runs_of(_timestamps_of(labels, kind="DeleteEntries", column=column), clean.index)
        found = [(f.start, f.end) for f in outcome.findings if f.signals == (column,)]
        assert found == expected
    assert not [f for f in outcome.findings if f.signals not in {("power",), ("wind_speed",)}]


def test_the_default_pipeline_gaps_are_found_among_other_faults(clean):
    messy, labels = turbine.create_default_messup_pipeline(turbine.MessUpPipeline(seed=5), 7, 5).apply_labelled(clean)

    outcome = rules.GapMeasure(expected_step=FREQ).measure(core.Dataset.from_frame(messy))

    dropped = _timestamps_of(labels, kind="RowDrop")
    expected = [run for run in _runs_of(dropped, clean.index) if run[0] != clean.index[0] and run[1] != clean.index[-1]]
    assert isinstance(outcome, core.Measured)
    assert [(finding.start, finding.end) for finding in outcome.findings] == expected
    assert expected
