"""Plain-data codecs and a strict reader for the artifact files.

Plain data is what YAML and JSON both hold: mappings with text keys, lists, text, numbers, booleans and `None`.
Non-finite floats are written as `{"$float": "nan" | "inf" | "-inf"}`, so that JSON stays valid and a number never
turns into text by accident. Timestamps are ISO 8601 text; aware ones are written in UTC.
"""

import datetime
import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

FLOAT_TAG = "$float"
_NONFINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


class ArtifactError(Exception):
    """Raised when an artifact file cannot be read, with the path inside the file where it went wrong."""

    def __init__(self, path: str, problem: str) -> None:
        """Create the error.

        Parameters
        ----------
        path
            Where in the file, for example `aspects[1].outcomes[0].metrics[2].value`.
        problem
            What is wrong.
        """
        self.path = path
        self.problem = problem
        super().__init__(f"{path or '<root>'}: {problem}")


def encode_float(value: float) -> float | dict[str, str]:
    """Encode a float; a non-finite one becomes its tag."""
    number = float(value)
    if math.isfinite(number):
        return number
    return {FLOAT_TAG: "nan" if math.isnan(number) else ("inf" if number > 0 else "-inf")}


def encode_time(value: pd.Timestamp | datetime.datetime) -> str:
    """Encode a time as ISO 8601 text; an aware time is converted to UTC, a naive one is kept naive."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def encode_plain(value: object, path: str = "") -> object:
    """Encode free-form data (a subject specification, transform parameters) as plain data.

    Parameters
    ----------
    value
        `None`, booleans, integers, floats, text, sequences and mappings with text keys, nested. Numpy scalars are
        converted. Sequences are written as lists.
    path
        Where the value sits, for the error message.

    Returns
    -------
    object
        Plain data.

    Raises
    ------
    ArtifactError
        For anything else, so that nothing is written as a guess.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return encode_float(value)
    if isinstance(value, Mapping):
        encoded: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArtifactError(path, f"mapping key {key!r} is not text")
            if key.startswith("$"):
                raise ArtifactError(path, f"mapping key {key!r} is reserved")
            encoded[key] = encode_plain(item, f"{path}.{key}" if path else key)
        return encoded
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [encode_plain(item, f"{path}[{i}]") for i, item in enumerate(value)]
    raise ArtifactError(path, f"a value of type {type(value).__name__} cannot be written as plain data")


def decode_plain(value: object, path: str = "") -> object:
    """Decode free-form plain data; lists become tuples, and `$float` tags become floats.

    Parameters
    ----------
    value
        Plain data as read from a file.
    path
        Where the value sits, for the error message.

    Returns
    -------
    object
        The value, with immutable sequences.

    Raises
    ------
    ArtifactError
        For a reserved key other than a well-formed `$float` tag.
    """
    if isinstance(value, Mapping):
        if FLOAT_TAG in value:
            return _decode_tagged_float(value, path)
        decoded: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or key.startswith("$"):
                raise ArtifactError(path, f"mapping key {key!r} is not allowed")
            decoded[key] = decode_plain(item, f"{path}.{key}" if path else key)
        return decoded
    if isinstance(value, list):
        return tuple(decode_plain(item, f"{path}[{i}]") for i, item in enumerate(value))
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise ArtifactError(path, f"a value of type {type(value).__name__} is not plain data")


def _decode_tagged_float(value: Mapping[object, object], path: str) -> float:
    if len(value) != 1 or value[FLOAT_TAG] not in _NONFINITE:
        raise ArtifactError(path, f"{FLOAT_TAG} takes exactly one of nan, inf, -inf")
    return _NONFINITE[str(value[FLOAT_TAG])]


class Reader:
    """A strict view on one mapping of a file: typed access, a path for every error, and no unknown keys."""

    def __init__(self, data: object, path: str = "") -> None:
        """Wrap a mapping.

        Parameters
        ----------
        data
            What the file holds at this place; it must be a mapping with text keys.
        path
            Where this place is in the file.

        Raises
        ------
        ArtifactError
            If `data` is not a mapping with text keys.
        """
        if not isinstance(data, Mapping) or not all(isinstance(key, str) for key in data):
            raise ArtifactError(path, "expected a mapping with text keys")
        self._data: Mapping[str, object] = data
        self.path = path
        self._used: set[str] = set()

    def child_path(self, key: str) -> str:
        """Give the path of a key of this mapping."""
        return f"{self.path}.{key}" if self.path else key

    def _get(self, key: str) -> object:
        if key not in self._data:
            raise ArtifactError(self.path, f"missing key {key!r}")
        self._used.add(key)
        return self._data[key]

    def list_keys(self) -> list[str]:
        """List the keys of the mapping, in file order."""
        return list(self._data)

    def text(self, key: str) -> str:
        """Read a text value."""
        value = self._get(key)
        if not isinstance(value, str):
            raise ArtifactError(self.child_path(key), "expected text")
        return value

    def text_or_none(self, key: str) -> str | None:
        """Read a text value that may be `null`."""
        value = self._get(key)
        if value is not None and not isinstance(value, str):
            raise ArtifactError(self.child_path(key), "expected text or null")
        return value

    def boolean(self, key: str) -> bool:
        """Read a boolean."""
        value = self._get(key)
        if not isinstance(value, bool):
            raise ArtifactError(self.child_path(key), "expected true or false")
        return value

    def boolean_or_none(self, key: str) -> bool | None:
        """Read a boolean that may be `null`."""
        value = self._get(key)
        if value is not None and not isinstance(value, bool):
            raise ArtifactError(self.child_path(key), "expected true, false or null")
        return value

    def integer(self, key: str) -> int:
        """Read an integer."""
        value = self._get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ArtifactError(self.child_path(key), "expected an integer")
        return value

    def number(self, key: str) -> float:
        """Read a number, also a `$float` tag."""
        return self._number(self._get(key), self.child_path(key))

    def number_or_none(self, key: str) -> float | None:
        """Read a number that may be `null`."""
        value = self._get(key)
        return None if value is None else self._number(value, self.child_path(key))

    @staticmethod
    def _number(value: object, path: str) -> float:
        if isinstance(value, Mapping):
            return _decode_tagged_float(value, path)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ArtifactError(path, "expected a number")
        return float(value)

    def time(self, key: str) -> pd.Timestamp:
        """Read an ISO 8601 time."""
        return self._time(self._get(key), self.child_path(key))

    def time_utc(self, key: str) -> datetime.datetime:
        """Read an ISO 8601 time that must carry a zero UTC offset."""
        timestamp = self._time(self._get(key), self.child_path(key))
        if timestamp.tzinfo is None or timestamp.utcoffset() != datetime.timedelta(0):
            raise ArtifactError(self.child_path(key), "expected a time with a zero UTC offset")
        return timestamp.to_pydatetime()

    @staticmethod
    def _time(value: object, path: str) -> pd.Timestamp:
        if not isinstance(value, str):
            raise ArtifactError(path, "expected an ISO 8601 time as text")
        try:
            return pd.Timestamp(value)
        except ValueError as error:
            raise ArtifactError(path, f"not an ISO 8601 time: {value!r}") from error

    def plain_mapping(self, key: str) -> dict[str, object]:
        """Read free-form data that must be a mapping."""
        value = decode_plain(self._get(key), self.child_path(key))
        if not isinstance(value, dict):
            raise ArtifactError(self.child_path(key), "expected a mapping")
        return value

    def texts(self, key: str) -> tuple[str, ...]:
        """Read a list of text."""
        items = self.items(key)
        for i, item in enumerate(items):
            if not isinstance(item, str):
                raise ArtifactError(f"{self.child_path(key)}[{i}]", "expected text")
        return tuple(str(item) for item in items)

    def items(self, key: str) -> list[object]:
        """Read a list."""
        value = self._get(key)
        if not isinstance(value, list):
            raise ArtifactError(self.child_path(key), "expected a list")
        return value

    def readers(self, key: str) -> list["Reader"]:
        """Read a list of mappings, each with its own reader."""
        return [Reader(item, f"{self.child_path(key)}[{i}]") for i, item in enumerate(self.items(key))]

    def reader(self, key: str) -> "Reader":
        """Read a nested mapping with its own reader."""
        return Reader(self._get(key), self.child_path(key))

    def reader_or_none(self, key: str) -> "Reader | None":
        """Read a nested mapping that may be `null`."""
        value = self._get(key)
        return None if value is None else Reader(value, self.child_path(key))

    def finish(self) -> None:
        """Refuse keys that were never read, so that a misspelled or unknown key is not silently dropped.

        Raises
        ------
        ArtifactError
            Listing the unknown keys.
        """
        unknown = sorted(set(self._data) - self._used)
        if unknown:
            raise ArtifactError(self.path, f"unknown keys: {', '.join(unknown)}")
