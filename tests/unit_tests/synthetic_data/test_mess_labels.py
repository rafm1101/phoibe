"""Ground-truth labels of the messup pipeline.

The labels are the oracle of the rule tests: a rule is only as testable as the labels are right. These tests
check that the labels describe exactly what the steps did, and that the labelled path changes nothing.
"""

import numpy as np
import pandas as pd
import pytest

from phoibe.synthetic_data import turbine
from phoibe.synthetic_data._turbine_noise import (
    CurtailmentNight,
    CurtailmentToZero,
    DeleteEntries,
    Freeze,
    RowDrop,
    Spike,
    ZeroOut,
)

TIME = turbine.Time(start="2026-02-14T04:00:00", freq="10min", periods=288)
PIPELINES = {
    "default": (turbine.create_default_messup_pipeline, None),
    "extended": (turbine.create_extended_messup_pipeline, "1min"),
}


class FixedSegments:
    """Segments chosen by the test instead of the random generator."""

    def __init__(self, *segments: tuple[int, int]):
        self.segments = list(segments)

    def generate(self, n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
        return self.segments


@pytest.fixture(scope="module")
def clean_frames() -> dict[str, pd.DataFrame]:
    return {
        name: turbine.generate_wtg_scada(time=TIME, latent_freq=latent_freq, seed=11)
        for name, (_, latent_freq) in PIPELINES.items()
    }


@pytest.fixture
def frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=8, freq="10min", name="datetime")
    return pd.DataFrame({"a": np.arange(1.0, 9.0), "b": np.arange(11.0, 19.0), "c": np.arange(21.0, 29.0)}, index=index)


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


class TestGenerator:
    def test_seed_makes_the_clean_data_reproducible(self):
        first = turbine.generate_wtg_scada(time=TIME, seed=5)
        second = turbine.generate_wtg_scada(time=TIME, seed=5)
        other = turbine.generate_wtg_scada(time=TIME, seed=6)

        pd.testing.assert_frame_equal(first, second)
        assert not first.equals(other)


class TestLabelledPathChangesNothing:
    @pytest.mark.parametrize("name", PIPELINES)
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    @pytest.mark.parametrize(("incidence", "level"), [(1, 1), (3, 2), (7, 5)])
    def test_data_equals_the_unlabelled_data(self, clean_frames, name, seed, incidence, level):
        create, _ = PIPELINES[name]
        plain = create(turbine.MessUpPipeline(seed=seed), incidence, level).apply(clean_frames[name])
        labelled, _ = create(turbine.MessUpPipeline(seed=seed), incidence, level).apply_labelled(clean_frames[name])

        pd.testing.assert_frame_equal(plain, labelled)

    def test_input_is_not_modified(self, clean_frames):
        before = clean_frames["default"].copy()
        turbine.create_default_messup_pipeline(turbine.MessUpPipeline(seed=0), 3, 2).apply_labelled(
            clean_frames["default"]
        )

        pd.testing.assert_frame_equal(clean_frames["default"], before)

    def test_step_without_labels_is_applied_and_contributes_none(self, frame):
        class AddOne:
            def apply(self, df, rng):
                return df + 1

        pipeline = turbine.MessUpPipeline(seed=0)
        pipeline.add(AddOne())
        pipeline.add(Freeze(segments=FixedSegments((1, 4)), column="a"))

        result, labels = pipeline.apply_labelled(frame)

        assert result["b"].iloc[0] == frame["b"].iloc[0] + 1
        assert [(label.step, label.kind) for label in labels] == [(1, "Freeze")]


class TestLabelsDescribeTheSteps:
    def test_freeze(self, frame):
        result, (label,) = Freeze(segments=FixedSegments((2, 6)), column="a").apply_labelled(frame, _rng())

        assert (label.kind, label.columns, label.n_rows) == ("Freeze", ("a",), 4)
        assert (label.start, label.end) == (frame.index[2], frame.index[5])
        assert label.n_changed == 3  # the first value is the frozen one
        assert label.detail["value"] == frame["a"].iloc[2]
        assert (result["a"].iloc[2:6] == frame["a"].iloc[2]).all()

    def test_freeze_of_a_constant_signal_changes_nothing(self, frame):
        frame["a"] = 5.0

        _, (label,) = Freeze(segments=FixedSegments((2, 6)), column="a").apply_labelled(frame, _rng())

        assert label.n_changed == 0

    def test_spike(self, frame):
        result, (label,) = Spike(segments=FixedSegments((1, 4)), column="b", magnitude=100).apply_labelled(
            frame, _rng()
        )

        added = (result["b"] - frame["b"]).iloc[1:4]
        assert (label.n_rows, label.n_changed) == (3, 3)
        assert label.detail["amplitude"] == pytest.approx(added.abs().max())
        assert (result.drop(index=frame.index[1:4]) == frame.drop(index=frame.index[1:4])).all().all()

    def test_delete_entries_counts_only_newly_missing_values(self, frame):
        frame.loc[frame.index[2], "b"] = np.nan

        result, (label,) = DeleteEntries(segments=FixedSegments((1, 4)), columns=["b"]).apply_labelled(frame, _rng())

        assert result["b"].iloc[1:4].isna().all()
        assert (label.columns, label.n_rows, label.n_changed) == (("b",), 3, 2)

    def test_curtailment_to_zero_labels_the_zeroed_columns(self, frame):
        result, (label,) = CurtailmentToZero(segments=FixedSegments((0, 3)), columns_to_keep=["a"]).apply_labelled(
            frame, _rng()
        )

        assert label.columns == ("b", "c")
        assert label.n_changed == 6
        assert (result[["b", "c"]].iloc[:3] == 0).all().all()
        assert (result["a"] == frame["a"]).all()

    def test_zero_out_keeps_the_requested_column(self, frame):
        result, (label,) = ZeroOut(segments=FixedSegments((5, 8)), keep_column="c").apply_labelled(frame, _rng())

        assert label.columns == ("a", "b")
        assert label.n_changed == 6
        assert (result["c"] == frame["c"]).all()

    def test_row_drop_labels_rows_that_no_longer_exist(self, frame):
        result, labels = RowDrop(segments=FixedSegments((1, 3), (5, 6))).apply_labelled(frame, _rng())

        assert len(result) == 5
        assert [(label.start, label.end, label.n_rows, label.n_changed) for label in labels] == [
            (frame.index[1], frame.index[2], 2, 2),
            (frame.index[5], frame.index[5], 1, 1),
        ]
        assert not result.index.isin([frame.index[1], frame.index[2], frame.index[5]]).any()

    def test_curtailment_night_labels_each_night_run(self):
        index = pd.date_range("2026-01-01", periods=48, freq="h", name="datetime")
        df = pd.DataFrame({"power": 5000.0}, index=index)

        result, labels = CurtailmentNight(limit=3200, column="power").apply_labelled(df, _rng())

        assert [(label.start.hour, label.end.hour, label.n_rows, label.n_changed) for label in labels] == [
            (0, 5, 6, 6),
            (22, 5, 8, 8),
            (22, 23, 2, 2),
        ]
        assert labels[0].detail["limit"] == 3200
        assert (result["power"][index.hour.isin([22, 23, 0, 1, 2, 3, 4, 5])] == 3200).all()
        assert (result["power"][~index.hour.isin([22, 23, 0, 1, 2, 3, 4, 5])] == 5000).all()


class TestPipelineLabels:
    @pytest.mark.parametrize("name", PIPELINES)
    def test_labels_carry_the_step_position_and_kind(self, clean_frames, name):
        create, _ = PIPELINES[name]
        pipeline = create(turbine.MessUpPipeline(seed=0), 3, 2)

        _, labels = pipeline.apply_labelled(clean_frames[name])

        steps = [type(step).__name__ for step in pipeline.steps]
        assert labels
        assert all(steps[label.step] == label.kind for label in labels)
        assert [label.step for label in labels] == sorted(label.step for label in labels)

    @pytest.mark.parametrize("name", PIPELINES)
    @pytest.mark.parametrize("seed", range(6))
    @pytest.mark.parametrize(("incidence", "level"), [(1, 1), (3, 2), (7, 5)])
    def test_every_change_is_labelled(self, clean_frames, name, seed, incidence, level):
        """No value differs from the clean data without a label that says so. That makes the labels an oracle."""
        create, _ = PIPELINES[name]
        clean = clean_frames[name]
        messy, labels = create(turbine.MessUpPipeline(seed=seed), incidence, level).apply_labelled(clean)

        kept = clean.index.intersection(messy.index)
        for column in clean.columns:
            before, after = clean.loc[kept, column], messy.loc[kept, column]
            changed = (before != after) & ~(before.isna() & after.isna())
            labelled = turbine.messlabel_mask(labels, kept, column=column)

            assert not (changed & ~labelled).any(), f"unlabelled change in {column}"

    @pytest.mark.parametrize("name", PIPELINES)
    @pytest.mark.parametrize("seed", range(6))
    def test_dropped_rows_are_exactly_the_row_drop_labels(self, clean_frames, name, seed):
        create, _ = PIPELINES[name]
        clean = clean_frames[name]
        messy, labels = create(turbine.MessUpPipeline(seed=seed), 3, 2).apply_labelled(clean)

        labelled = turbine.messlabel_mask(labels, clean.index, kind="RowDrop", effective_only=False)

        assert (labelled.to_numpy() == ~clean.index.isin(messy.index)).all()

    def test_effective_labels_exist(self, clean_frames):
        """Guard: the checks above are not vacuous."""
        _, labels = turbine.create_extended_messup_pipeline(turbine.MessUpPipeline(seed=0), 3, 2).apply_labelled(
            clean_frames["extended"]
        )

        kinds = {label.kind for label in labels if label.n_changed > 0}
        assert {"Freeze", "RowDrop", "Spike", "DeleteEntries", "CurtailmentToZero"} <= kinds


class TestMessLabelMask:
    def test_filters_by_kind_column_and_effect(self, frame):
        pipeline = turbine.MessUpPipeline(seed=0)
        pipeline.add(Freeze(segments=FixedSegments((1, 3)), column="a"))
        pipeline.add(DeleteEntries(segments=FixedSegments((5, 7)), columns=["b"]))
        frame["c"] = 1.0
        pipeline.add(Freeze(segments=FixedSegments((0, 8)), column="c"))
        messy, labels = pipeline.apply_labelled(frame)
        index = messy.index

        assert turbine.messlabel_mask(labels, index, kind="Freeze", column="a").sum() == 2
        assert turbine.messlabel_mask(labels, index, kind="DeleteEntries").sum() == 2
        assert turbine.messlabel_mask(labels, index, column="b").sum() == 2
        assert turbine.messlabel_mask(labels, index, column="c").sum() == 0
        assert turbine.messlabel_mask(labels, index, column="c", effective_only=False).sum() == 8

    def test_labels_tabulate(self, frame):
        pipeline = turbine.MessUpPipeline(seed=0)
        pipeline.add(Freeze(segments=FixedSegments((1, 3)), column="a"))
        _, labels = pipeline.apply_labelled(frame)

        table = turbine.messlabel_to_frame(labels)

        assert list(table.columns) == ["step", "kind", "columns", "start", "end", "n_rows", "n_changed", "detail"]
        assert len(table) == 1
        assert len(turbine.messlabel_to_frame([])) == 0
