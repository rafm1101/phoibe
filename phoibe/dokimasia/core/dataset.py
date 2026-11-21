"""The dataset: an immutable time series with a fingerprint (ADR-01, ADR-06)."""

import dataclasses
import hashlib

import pandas as pd

FINGERPRINT_ALGORITHM = "sha256-pandas-row-hash"


@dataclasses.dataclass(frozen=True, slots=True)
class Fingerprint:
    """A consistency check of a dataset. It is not tamper protection."""

    algorithm: str
    """How the digest was computed."""
    digest: str
    """Hexadecimal digest of the index, the values and the column names."""


def _hash_values(values: pd.Series | pd.Index) -> bytes:
    """Hash all values. Cells that are not numbers are hashed by their text."""
    if values.dtype == object:
        values = values.astype(str)
    return pd.util.hash_pandas_object(values, index=False).to_numpy().tobytes()


def compute_fingerprint(frame: pd.DataFrame) -> Fingerprint:
    """Compute the fingerprint of a frame.

    The hash covers the index, every column in order, the dtypes and the column names. Duplicate column names are
    allowed. Cells that are not numbers, timestamps or strings are hashed by their text.

    Parameters
    ----------
    frame
        The frame to fingerprint.

    Returns
    -------
    Fingerprint
        The fingerprint; equal frames give equal digests.

    """
    digest = hashlib.sha256()
    digest.update(_hash_values(frame.index))
    for position in range(frame.shape[1]):
        column = frame.iloc[:, position]
        digest.update(f"{frame.columns[position]!s}\x1f{column.dtype}\x1e".encode())
        digest.update(_hash_values(column))
    return Fingerprint(algorithm=FINGERPRINT_ALGORITHM, digest=digest.hexdigest())


@dataclasses.dataclass(frozen=True, slots=True)
class Dataset:
    """An immutable time series plus its fingerprint.

    The time axis is the index of the frame, whether it was an index or a column in the source; no rule searches for
    it. Unsorted or duplicate timestamps and missing values are allowed, because measuring them is the point.
    Build a dataset with `Dataset.from_frame`. The frame must not be modified.
    """

    frame: pd.DataFrame
    """The data. Index: time axis. Columns: signals, by catalog name after standardizing."""
    fingerprint: Fingerprint
    """Fingerprint of `frame`."""

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "Dataset":
        """Create a dataset from a frame with a datetime index.

        Parameters
        ----------
        frame
            Data indexed by time. The frame is copied, so later changes to it do not reach the dataset.

        Returns
        -------
        Dataset
            The dataset with its fingerprint.

        Raises
        ------
        TypeError
            If the index is not a `pandas.DatetimeIndex`. An adapter has to turn the time column into the index.

        """
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(f"the time axis must be a DatetimeIndex, got {type(frame.index).__name__}")
        copied = frame.copy(deep=True)
        return cls(frame=copied, fingerprint=compute_fingerprint(copied))

    @property
    def time_axis(self) -> pd.DatetimeIndex:
        """The time axis. It may be naive or aware, unsorted, with duplicates and missing values."""
        index = self.frame.index
        assert isinstance(index, pd.DatetimeIndex)
        return index

    @property
    def n_rows(self) -> int:
        """Number of rows."""
        return len(self.frame)

    @property
    def signal_names(self) -> tuple[str, ...]:
        """Column names as text, in order. Duplicates are kept."""
        return tuple(str(name) for name in self.frame.columns)
