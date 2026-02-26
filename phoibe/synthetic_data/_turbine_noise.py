import typing

import numpy as np
import pandas as pd


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


class CurtailmentNight:
    """Clip values of `column` at nighttimes to `limit`. Interval is left-closed and right-open."""

    def __init__(self, limit: float, column: typing.Any, start_hour: int = 22, end_hour: int = 6):
        self.limit = limit
        self.column = column
        self.start_hour = 22
        self.end_hour = 6

    def apply(self, df: pd.DataFrame, rng: np.random.Generator):
        df = df.copy()
        index = pd.DatetimeIndex(df.index)
        night_times = (index.hour >= self.start_hour) | (index.hour < self.end_hour)
        df.loc[:, self.column] = np.where(
            night_times, np.minimum(df.loc[:, self.column].values, self.limit), df.loc[:, self.column].values
        )
        return df


class CurtailmentToZero:
    """Clip values of `column` at nighttimes to `limit`. Interval is left-closed and right-open."""

    def __init__(self, segments, columns_to_keep: list[typing.Any]):
        self.segments = segments
        self.columns_to_keep = columns_to_keep

    def apply(self, df: pd.DataFrame, rng: np.random.Generator):
        df = df.copy()
        columns_to_zero = df.columns.difference(self.columns_to_keep)

        for start, end in self.segments.generate(len(df), rng):
            df.loc[df.index[start:end], columns_to_zero] = 0

        return df


class DeleteEntries:
    """Set values to NaN."""

    def __init__(self, segments, columns=None):
        self.segments = segments
        self.columns = columns

    def apply(self, df: pd.DataFrame, rng: np.random.Generator):
        df = df.copy()
        n = len(df)

        columns = self.columns if self.columns is not None else df.columns

        for start, end in self.segments.generate(n, rng):
            df.loc[df.index[start:end], columns] = np.nan

        return df


class RowDrop:
    """Drop entire rows."""

    def __init__(self, segments):
        self.segments = segments

    def apply(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        df = df.copy()
        n = len(df)
        drop_indices: list[typing.Any] = []

        for start, end in self.segments.generate(n, rng):
            drop_indices.extend(df.index[start:end])

        df = df.drop(drop_indices)
        return df


class ZeroOut:
    """Set column values to zero. Optional: Prevent one."""

    def __init__(self, segments, keep_column=None):
        self.segments = segments
        self.keep_column = keep_column

    def apply(self, df: pd.DataFrame, rng: np.random.Generator):
        df = df.copy()
        n = len(df)

        if self.keep_column:
            columns = df.columns.difference([self.keep_column])
        else:
            columns = df.columns

        for start, end in self.segments.generate(n, rng):
            df.loc[df.index[start:end], columns] = 0

        return df


class Freeze:
    """Freeze values."""

    def __init__(self, segments, column):
        self.segments = segments
        self.column = column

    def apply(self, df: pd.DataFrame, rng: np.random.Generator):
        df = df.copy()
        n = len(df)

        for start, end in self.segments.generate(n, rng):
            value = df.loc[df.index[start], self.column]
            df.loc[df.index[start:end], self.column] = value

        return df


class Spike:
    """Add spikes to values."""

    def __init__(self, segments, column, magnitude):
        self.segments = segments
        self.column = column
        self.magnitude = magnitude

    def apply(self, df: pd.DataFrame, rng: np.random.Generator):
        df = df.copy()
        n = len(df)

        for start, end in self.segments.generate(n, rng):
            noise = rng.normal(scale=self.magnitude, size=end - start)
            df.loc[df.index[start:end], self.column] += noise

        return df
