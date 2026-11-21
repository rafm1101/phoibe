"""The dataset: an immutable time series with a fingerprint (ADR-01, ADR-06)."""

import dataclasses
import hashlib
import types
from collections.abc import Mapping

import pandas as pd

FINGERPRINT_ALGORITHM = "sha256-pandas-row-hash"


@dataclasses.dataclass(frozen=True, slots=True)
class Fingerprint:
    """A consistency check of a dataset. It is not tamper protection."""

    algorithm: str
    """How the digest was computed."""
    digest: str
    """Hexadecimal digest of the index, the values and the column names."""


@dataclasses.dataclass(frozen=True, slots=True)
class TransformDescription:
    """What a transform is: its name, parameters and the software that ran it (ADR-11)."""

    name: str
    """Name of the transform, for example `rename_to_catalog`."""
    parameters: Mapping[str, object]
    """Parameters it ran with. A read-only copy."""
    software_version: str
    """Version of the software that applied it."""
    rule: str | None = None
    """A rule applied where the data left a choice, such as how a repeated hour was resolved."""

    def __post_init__(self) -> None:
        """Replace the parameters by a read-only copy."""
        object.__setattr__(self, "parameters", types.MappingProxyType(dict(self.parameters)))


@dataclasses.dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Where a dataset came from when it moved to the next layer."""

    parent_attestation_id: str
    """Attestation of the layer the data left."""
    transform: TransformDescription | None
    """The transform that was applied on the way, if any."""
    output_fingerprint: Fingerprint
    """Fingerprint of the data that the next layer receives."""


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
    provenance: "ProvenanceRecord | None" = None
    """Where the data came from, if it reached this layer through `advance_dataset`."""

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, *, provenance: "ProvenanceRecord | None" = None) -> "Dataset":
        """Create a dataset from a frame with a datetime index.

        Parameters
        ----------
        frame
            Data indexed by time. The frame is copied, so later changes to it do not reach the dataset.
        provenance
            Where the data came from, if it is known.

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
        return cls(frame=copied, fingerprint=compute_fingerprint(copied), provenance=provenance)

    def with_frame(self, frame: pd.DataFrame) -> "Dataset":
        """Create the next version of this dataset after the data was edited by hand.

        The provenance is kept, so that the lineage shows where the data came from. Its output fingerprint no longer
        matches, which a profile reports.

        Parameters
        ----------
        frame
            The edited data, indexed by time.

        Returns
        -------
        Dataset
            A new dataset with its own fingerprint and the provenance of this one.
        """
        return Dataset.from_frame(frame, provenance=self.provenance)

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
