import numpy as np
import pandas as pd
import pytest

from phoibe.dokimasia import core


@pytest.fixture
def frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=6, freq="10min", name="datetime")
    return pd.DataFrame({"a": np.arange(6.0), "b": np.arange(6.0) * 2}, index=index)


class TestDatasetFromFrame:
    def test_rejects_an_index_that_is_not_a_datetime_index(self, frame):
        with pytest.raises(TypeError, match="DatetimeIndex"):
            core.Dataset.from_frame(frame.reset_index())

    def test_copies_the_frame(self, frame):
        dataset = core.Dataset.from_frame(frame)
        frame.iloc[0, 0] = 99.0

        assert dataset.frame.iloc[0, 0] == 0.0

    def test_exposes_time_axis_rows_and_signal_names(self, frame):
        dataset = core.Dataset.from_frame(frame)

        assert dataset.time_axis.equals(frame.index)
        assert dataset.n_rows == 6
        assert dataset.signal_names == ("a", "b")

    def test_accepts_a_time_axis_with_missing_duplicate_and_unsorted_values(self):
        index = pd.DatetimeIndex(["2026-01-01 00:10", pd.NaT, "2026-01-01 00:00", "2026-01-01 00:00"])

        dataset = core.Dataset.from_frame(pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=index))

        assert dataset.n_rows == 4


class TestComputeFingerprint:
    def test_equal_frames_give_equal_digests(self, frame):
        assert core.compute_fingerprint(frame) == core.compute_fingerprint(frame.copy())

    def test_a_changed_value_changes_the_digest(self, frame):
        changed = frame.copy()
        changed.iloc[2, 1] = -1.0

        assert core.compute_fingerprint(changed) != core.compute_fingerprint(frame)

    def test_a_renamed_column_changes_the_digest(self, frame):
        assert core.compute_fingerprint(frame.rename(columns={"a": "c"})) != core.compute_fingerprint(frame)

    def test_a_shifted_index_changes_the_digest(self, frame):
        shifted = frame.copy()
        shifted.index = shifted.index + pd.Timedelta("10min")

        assert core.compute_fingerprint(shifted) != core.compute_fingerprint(frame)

    def test_names_the_algorithm(self, frame):
        assert core.compute_fingerprint(frame).algorithm == core.FINGERPRINT_ALGORITHM

    def test_handles_cells_that_are_not_hashable(self, frame):
        frame = frame.astype(object)
        frame.iloc[0, 0] = [1, 2, 3]

        assert len(core.compute_fingerprint(frame).digest) == 64

    def test_handles_duplicate_column_names_and_an_empty_frame(self, frame):
        duplicated = pd.concat([frame, frame], axis=1)
        empty = frame.iloc[0:0]

        assert core.compute_fingerprint(duplicated) != core.compute_fingerprint(frame)
        assert len(core.compute_fingerprint(empty).digest) == 64


def test_transform_description_parameters_are_frozen_deeply():
    description = core.TransformDescription("t", {"steps": [{"name": "a", "parameters": {"k": [1]}}]}, "1")

    assert description.parameters["steps"][0]["parameters"]["k"] == (1,)
    with pytest.raises(TypeError):
        description.parameters["steps"][0]["name"] = "b"  # type: ignore[index]
