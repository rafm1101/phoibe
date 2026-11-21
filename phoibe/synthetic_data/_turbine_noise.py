import dataclasses
import typing

import numpy as np
import pandas as pd


@dataclasses.dataclass(frozen=True)
class MessLabel:
    """Ground truth of one injected fault: where a messup step altered the data.

    Labels refer to timestamps, not positions, because later steps may drop rows. A label is emitted for each
    segment a step touched, whether or not the segment altered any value; `n_changed` tells which did.
    """

    kind: str
    """Class name of the step that injected the fault, e.g. `Freeze`."""
    columns: tuple[str, ...]
    """Columns the step addressed in this segment. A dropped row addresses all columns."""
    start: pd.Timestamp
    """First affected timestamp, inclusive."""
    end: pd.Timestamp
    """Last affected timestamp, inclusive."""
    n_rows: int
    """Number of rows in the segment."""
    n_changed: int
    """Number of values actually altered. For `RowDrop`, the number of rows removed."""
    step: int = 0
    """Position of the step in the pipeline. Set by `MessUpPipeline.apply_labelled`."""
    detail: dict[str, typing.Any] = dataclasses.field(default_factory=dict)
    """Step-specific facts such as the frozen value, the spike amplitude or the clip limit."""


def _count_differences(before: typing.Any, after: typing.Any) -> int:
    """Count positions at which values differ. Two missing values are equal."""
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    both_missing = np.isnan(before) & np.isnan(after)
    return int(np.sum((before != after) & ~both_missing))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return the half-open position intervals of consecutive `True` values."""
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2], strict=True)]


def _label(
    kind: str, df: pd.DataFrame, columns: typing.Iterable[typing.Any], start: int, end: int, n_changed: int, **detail
) -> MessLabel:
    """Create a label for the positions `start` to `end` (exclusive) of `df`."""
    return MessLabel(
        kind=kind,
        columns=tuple(str(column) for column in columns),
        start=df.index[start],
        end=df.index[end - 1],
        n_rows=end - start,
        n_changed=n_changed,
        detail=detail,
    )


class _LabelledStep:
    """Base of the messup steps. `apply` returns the data only; `apply_labelled` also returns the ground truth."""

    def apply_labelled(self, df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, list[MessLabel]]:
        raise NotImplementedError

    def apply(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        return self.apply_labelled(df, rng)[0]


class UniformSegments:
    """Generate `n` segments of uniformly distributed lengths."""

    def __init__(self, n: int, min_len: int, max_len: int, allow_overlap: bool = False, max_attempts: int = 1000):
        self.n = n
        self.min_len = min_len
        self.max_len = max_len
        self.allow_overlap = allow_overlap
        self.max_attempts = max_attempts

    def generate(self, n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
        segments_start_end: list[tuple[typing.Any, typing.Any]] = []
        occupation_mask = np.zeros(n, dtype=bool)

        attempts = 0
        while len(segments_start_end) < self.n and attempts < self.max_attempts:
            attempts += 1

            length = rng.integers(self.min_len, self.max_len + 1)
            if length >= n:
                continue

            start = rng.integers(0, n - length)
            end = start + length

            if self.allow_overlap or not occupation_mask[start:end].any():
                segments_start_end.append((start, end))
                occupation_mask[start:end] = True

        return segments_start_end


class GeometricSegments:
    """Generate `n` segments of geometrically distributed lengths of expected length `1/p`."""

    def __init__(
        self, n: int, p: float, max_len: int | None = None, allow_overlap: bool = False, max_attempts: int = 1000
    ):
        self.n = n
        self.p = p
        self.max_len = max_len
        self.allow_overlap = allow_overlap
        self.max_attempts = max_attempts

    def generate(self, n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
        segments_start_end: list[tuple[typing.Any, typing.Any]] = []
        occupation_mask = np.zeros(n, dtype=bool)

        attempts = 0
        while len(segments_start_end) < self.n and attempts < self.max_attempts:
            attempts += 1

            length = rng.geometric(self.p)
            if self.max_len is not None:
                length = min(length, self.max_len)
            if length >= n:
                continue

            start = rng.integers(0, n - length)
            end = start + length

            if self.allow_overlap or not occupation_mask[start:end].any():
                segments_start_end.append((int(start), int(end)))
                occupation_mask[start:end] = True

        return segments_start_end


class BernoulliStartSegments:
    """Generate a random number of segments of geometrically distributed lengths of expected lengths `1/p_len`."""

    def __init__(self, p_start: float, p_len: float, max_len: int | None = None):
        self.p_start = p_start
        self.p_len = p_len
        self.max_len = max_len

    def generate(self, n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
        segments_start_end: list[tuple[typing.Any, typing.Any]] = []

        i = 0
        while i < n:
            if rng.random() < self.p_start:
                length = rng.geometric(self.p_len)

                if self.max_len is not None:
                    length = min(length, self.max_len)

                end = min(i + length, n)
                segments_start_end.append((i, end))
                i = end
            else:
                i += 1

        return segments_start_end


class CurtailmentNight(_LabelledStep):
    """Clip values of `column` at nighttimes to `limit`. Interval is left-closed and right-open."""

    def __init__(self, limit: float, column: typing.Any, start_hour: int = 22, end_hour: int = 6):
        self.limit = limit
        self.column = column
        self.start_hour = 22
        self.end_hour = 6

    def apply_labelled(self, df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, list[MessLabel]]:
        df = df.copy()
        index = pd.DatetimeIndex(df.index)
        night_times = (index.hour >= self.start_hour) | (index.hour < self.end_hour)
        before = df.loc[:, self.column].to_numpy(copy=True)
        after = np.where(night_times, np.minimum(before, self.limit), before)
        df.loc[:, self.column] = after

        labels = [
            _label(
                "CurtailmentNight",
                df,
                [self.column],
                start,
                end,
                _count_differences(before[start:end], after[start:end]),
                limit=self.limit,
            )
            for start, end in _runs(night_times)
        ]
        return df, labels


class CurtailmentToZero(_LabelledStep):
    """Set all columns but `columns_to_keep` to zero in the given segments."""

    def __init__(self, segments, columns_to_keep: list[typing.Any]):
        self.segments = segments
        self.columns_to_keep = columns_to_keep

    def apply_labelled(self, df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, list[MessLabel]]:
        df = df.copy()
        columns_to_zero = df.columns.difference(self.columns_to_keep)
        labels = []

        for start, end in self.segments.generate(len(df), rng):
            before = df.loc[df.index[start:end], columns_to_zero].to_numpy()
            df.loc[df.index[start:end], columns_to_zero] = 0
            labels.append(
                _label(
                    "CurtailmentToZero",
                    df,
                    columns_to_zero,
                    start,
                    end,
                    _count_differences(before, df.loc[df.index[start:end], columns_to_zero].to_numpy()),
                )
            )

        return df, labels


class DeleteEntries(_LabelledStep):
    """Set values to NaN."""

    def __init__(self, segments, columns=None):
        self.segments = segments
        self.columns = columns

    def apply_labelled(self, df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, list[MessLabel]]:
        df = df.copy()
        n = len(df)

        columns = self.columns if self.columns is not None else df.columns
        labels = []

        for start, end in self.segments.generate(n, rng):
            before = df.loc[df.index[start:end], columns].to_numpy()
            df.loc[df.index[start:end], columns] = np.nan
            labels.append(
                _label(
                    "DeleteEntries", df, columns, start, end, _count_differences(before, np.full(before.shape, np.nan))
                )
            )

        return df, labels


class RowDrop(_LabelledStep):
    """Drop entire rows."""

    def __init__(self, segments):
        self.segments = segments

    def apply_labelled(self, df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, list[MessLabel]]:
        df = df.copy()
        n = len(df)
        drop_indices: list[typing.Any] = []
        labels = []

        for start, end in self.segments.generate(n, rng):
            drop_indices.extend(df.index[start:end])
            labels.append(_label("RowDrop", df, df.columns, start, end, end - start))

        df = df.drop(drop_indices)
        return df, labels


class ZeroOut(_LabelledStep):
    """Set column values to zero. Optional: Prevent one."""

    def __init__(self, segments, keep_column=None):
        self.segments = segments
        self.keep_column = keep_column

    def apply_labelled(self, df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, list[MessLabel]]:
        df = df.copy()
        n = len(df)

        if self.keep_column:
            columns = df.columns.difference([self.keep_column])
        else:
            columns = df.columns
        labels = []

        for start, end in self.segments.generate(n, rng):
            before = df.loc[df.index[start:end], columns].to_numpy()
            df.loc[df.index[start:end], columns] = 0
            labels.append(
                _label(
                    "ZeroOut",
                    df,
                    columns,
                    start,
                    end,
                    _count_differences(before, df.loc[df.index[start:end], columns].to_numpy()),
                )
            )

        return df, labels


class Freeze(_LabelledStep):
    """Freeze values."""

    def __init__(self, segments, column):
        self.segments = segments
        self.column = column

    def apply_labelled(self, df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, list[MessLabel]]:
        df = df.copy()
        n = len(df)
        labels = []

        for start, end in self.segments.generate(n, rng):
            value = df.loc[df.index[start], self.column]
            before = df.loc[df.index[start:end], self.column].to_numpy()
            df.loc[df.index[start:end], self.column] = value
            labels.append(
                _label(
                    "Freeze",
                    df,
                    [self.column],
                    start,
                    end,
                    _count_differences(before, df.loc[df.index[start:end], self.column].to_numpy()),
                    value=float(value),
                )
            )

        return df, labels


class Spike(_LabelledStep):
    """Add spikes to values."""

    def __init__(self, segments, column, magnitude):
        self.segments = segments
        self.column = column
        self.magnitude = magnitude

    def apply_labelled(self, df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, list[MessLabel]]:
        df = df.copy()
        n = len(df)
        labels = []

        for start, end in self.segments.generate(n, rng):
            noise = rng.normal(scale=self.magnitude, size=end - start)
            before = df.loc[df.index[start:end], self.column].to_numpy()
            df.loc[df.index[start:end], self.column] += noise
            labels.append(
                _label(
                    "Spike",
                    df,
                    [self.column],
                    start,
                    end,
                    _count_differences(before, df.loc[df.index[start:end], self.column].to_numpy()),
                    amplitude=float(np.max(np.abs(noise))),
                )
            )

        return df, labels
