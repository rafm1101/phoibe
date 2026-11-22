"""Where the artifacts of a run are kept (ADR-10): a `Store` protocol and the flat default store.

A run is the profile, the verdicts and, if the gate let the data through, the attestation of one layer. The flat store
keeps them as three files next to each other, named `<layer>-<UTC timestamp>-<artifact>.<yaml|json>`. The subject is
a field inside the files, so a store directory is meant for one subject or one campaign. There is no separate index
file: the index is built from the files, so it cannot go stale.
"""

import dataclasses
import datetime
import pathlib
import re
import typing

import pandas as pd

from phoibe.dokimasia import core
from phoibe.dokimasia.adapters import _text, artifacts

_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
_LAYER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.]*$")
_FILE_PATTERN = re.compile(
    r"^(?P<layer>[A-Za-z0-9][A-Za-z0-9_.]*)-(?P<stamp>\d{8}T\d{6}Z)-(?P<artifact>profile|verdicts|attestation)"
    r"\.(?P<fmt>yaml|json)$"
)
_ARTIFACTS = ("profile", "verdicts", "attestation")


class StoreError(Exception):
    """Raised when a run cannot be saved or loaded; the message names the file."""


@dataclasses.dataclass(frozen=True, slots=True)
class Run:
    """The artifacts of one layer of one dataset version."""

    profile: core.Profile
    """What was measured."""
    verdicts: artifacts.VerdictsRecord
    """What the contract made of it."""
    attestation: core.Attestation | None = None
    """The record of a passed gate; `None` if the gate failed or the run was not attested."""

    def __post_init__(self) -> None:
        """Check that the artifacts belong together.

        Raises
        ------
        ValueError
            If subject, layer or fingerprint differ between the artifacts, or a failed gate has an attestation.
        """
        profile, verdicts, attestation = self.profile, self.verdicts, self.attestation
        keys = {
            "profile": (profile.subject.id, profile.layer, profile.fingerprint),
            "verdicts": (verdicts.subject_id, verdicts.layer, verdicts.fingerprint),
        }
        if attestation is not None:
            keys["attestation"] = (attestation.subject_id, attestation.layer, attestation.fingerprint)
        if len(set(keys.values())) != 1:
            raise ValueError(f"the artifacts differ in subject, layer or fingerprint: {keys}")
        if attestation is not None:
            if verdicts.result.outcome == core.GateOutcome.FAIL:
                raise ValueError("a failed gate has no attestation")
            if attestation.outcome != verdicts.result.outcome:
                raise ValueError("the attestation and the verdicts disagree on the gate outcome")


@dataclasses.dataclass(frozen=True, slots=True)
class RunEntry:
    """One line of the run index: what a run is, without loading its profile."""

    subject_id: str
    """The subject."""
    layer: str
    """The layer."""
    created_at: datetime.datetime
    """The time in the file names, UTC, to the second."""
    fingerprint: core.Fingerprint
    """Fingerprint of the dataset version."""
    contract_id: str
    """Contract that was applied."""
    contract_version: str
    """Version of that contract."""
    gate_outcome: core.GateOutcome
    """What the gate decided."""
    attestation_id: str | None
    """The attestation, if there is one."""
    parent_id: str | None
    """The attestation of the layer the data came from."""
    stem: str
    """File name without artifact and suffix, `<layer>-<UTC timestamp>`."""
    fmt: _text.Format
    """Format of the files."""


class Store(typing.Protocol):
    """Where runs are kept. The flat store is the default; a database could be another."""

    def save_run(self, run: Run, *, created_at: datetime.datetime) -> RunEntry:
        """Save a run and return its index entry; a run that exists is never overwritten."""
        ...

    def list_runs(self) -> tuple[RunEntry, ...]:
        """List the runs, oldest first."""
        ...

    def load_run(self, entry: RunEntry) -> Run:
        """Load the run of an entry."""
        ...


def _format_stamp(created_at: datetime.datetime) -> str:
    if created_at.tzinfo is None or created_at.utcoffset() != datetime.timedelta(0):
        raise ValueError("created_at must be a timezone-aware UTC timestamp")
    return created_at.strftime(_STAMP_FORMAT)


class FlatStore:
    """A directory with three files per run. One writer at a time is assumed."""

    def __init__(self, directory: pathlib.Path, *, fmt: _text.Format = "yaml") -> None:
        """Create the store; the directory is made on the first save.

        Parameters
        ----------
        directory
            Where the files live.
        fmt
            Format of the files that are written, `yaml` or `json`; both are read.
        """
        if fmt not in _text.FORMATS:
            raise ValueError(f"unknown format {fmt!r}; use one of {', '.join(_text.FORMATS)}")
        self.directory = directory
        self.fmt: _text.Format = fmt

    def _path(self, stem: str, artifact: str, fmt: _text.Format) -> pathlib.Path:
        return self.directory / f"{stem}-{artifact}.{fmt}"

    def save_run(self, run: Run, *, created_at: datetime.datetime) -> RunEntry:
        """Save a run as files.

        The files are written under temporary names and renamed at the end, attestation last, so that an attested
        run is never seen half written.

        Parameters
        ----------
        run
            The run.
        created_at
            The time that goes into the file names, timezone-aware UTC. It is passed in; the store reads no clock.

        Returns
        -------
        RunEntry
            The entry of the saved run.

        Raises
        ------
        ValueError
            If the layer name is not usable in a file name, or the time is not UTC.
        StoreError
            If a run of this layer and time exists already, or a file cannot be written.
        """
        layer = run.profile.layer
        if not _LAYER_PATTERN.match(layer):
            raise ValueError(f"layer {layer!r} cannot be used in a file name; use letters, digits, '_' and '.'")
        stem = f"{layer}-{_format_stamp(created_at)}"
        payload = {
            "profile": artifacts.profile_to_data(run.profile),
            "verdicts": artifacts.verdicts_to_data(run.verdicts),
        }
        if run.attestation is not None:
            payload["attestation"] = artifacts.attestation_to_data(run.attestation)
        targets = {name: self._path(stem, name, self.fmt) for name in payload}
        for name in _ARTIFACTS:
            for fmt in _text.FORMATS:
                existing = self._path(stem, name, fmt)
                if existing.exists():
                    raise StoreError(f"{existing.name} exists already; a run is never overwritten")
        self.directory.mkdir(parents=True, exist_ok=True)
        temporaries = {name: path.with_name(f".{path.name}.tmp") for name, path in targets.items()}
        moved: list[pathlib.Path] = []
        try:
            for name, temporary in temporaries.items():
                temporary.write_text(artifacts.format_artifact(payload[name], self.fmt), encoding="utf-8")
            for name in payload:
                temporaries[name].replace(targets[name])
                moved.append(targets[name])
        except OSError as error:
            for path in moved:
                path.unlink(missing_ok=True)
            raise StoreError(f"cannot write the run {stem}: {error}") from error
        finally:
            for temporary in temporaries.values():
                temporary.unlink(missing_ok=True)
        return self._build_entry(stem, self.fmt)

    def list_runs(self) -> tuple[RunEntry, ...]:
        """List the runs in the directory, oldest first.

        Files whose names do not follow the pattern are not ours and are ignored. A run that has a profile but no
        verdicts is damaged, and listing says so.

        Returns
        -------
        tuple
            The entries, ordered by time, then layer.

        Raises
        ------
        StoreError
            If a run is incomplete or a file cannot be read.
        """
        if not self.directory.is_dir():
            return ()
        found: dict[str, _text.Format] = {}
        for path in self.directory.iterdir():
            match = _FILE_PATTERN.match(path.name)
            if match and match["artifact"] == "profile":
                found[f"{match['layer']}-{match['stamp']}"] = typing.cast("_text.Format", match["fmt"])
        entries = [self._build_entry(stem, fmt) for stem, fmt in found.items()]
        return tuple(sorted(entries, key=lambda entry: (entry.created_at, entry.layer)))

    def load_run(self, entry: RunEntry) -> Run:
        """Load the run of an entry, checking that its artifacts belong together.

        Parameters
        ----------
        entry
            An entry from `list_runs` or `save_run`.

        Returns
        -------
        Run
            The run.

        Raises
        ------
        StoreError
            If a file is missing, cannot be read, is not a valid artifact or the artifacts do not belong together.
        """
        profile = self._read(entry.stem, "profile", entry.fmt, artifacts.profile_from_data)
        verdicts = self._read(entry.stem, "verdicts", entry.fmt, artifacts.verdicts_from_data)
        attestation = None
        if self._path(entry.stem, "attestation", entry.fmt).exists():
            attestation = self._read(entry.stem, "attestation", entry.fmt, artifacts.attestation_from_data)
        try:
            return Run(profile, verdicts, attestation)
        except ValueError as error:
            raise StoreError(f"the files of {entry.stem} do not belong together: {error}") from error

    def _read[T](self, stem: str, artifact: str, fmt: _text.Format, build: typing.Callable[[object], T]) -> T:
        path = self._path(stem, artifact, fmt)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise StoreError(f"cannot read {path.name}: {error}") from error
        try:
            return build(artifacts.parse_artifact(text, fmt))
        except artifacts.ArtifactError as error:
            raise StoreError(f"{path.name}: {error}") from error

    def _build_entry(self, stem: str, fmt: _text.Format) -> RunEntry:
        verdicts = self._read(stem, "verdicts", fmt, artifacts.verdicts_from_data)
        attestation = None
        if self._path(stem, "attestation", fmt).exists():
            attestation = self._read(stem, "attestation", fmt, artifacts.attestation_from_data)
        layer, _, stamp = stem.rpartition("-")
        created_at = datetime.datetime.strptime(stamp, _STAMP_FORMAT).replace(tzinfo=datetime.UTC)
        if verdicts.layer != layer:
            raise StoreError(f"{stem}: the file name says layer {layer!r}, the verdicts say {verdicts.layer!r}")
        return RunEntry(
            subject_id=verdicts.subject_id,
            layer=layer,
            created_at=created_at,
            fingerprint=verdicts.fingerprint,
            contract_id=verdicts.contract_id,
            contract_version=verdicts.contract_version,
            gate_outcome=verdicts.result.outcome,
            attestation_id=None if attestation is None else attestation.id,
            parent_id=None if attestation is None else attestation.parent_id,
            stem=stem,
            fmt=fmt,
        )


def find_entries(
    entries: typing.Iterable[RunEntry],
    *,
    layer: str | None = None,
    subject_id: str | None = None,
    attestation_id: str | None = None,
    fingerprint: core.Fingerprint | None = None,
) -> tuple[RunEntry, ...]:
    """Select entries of the run index; every given criterion must match.

    Parameters
    ----------
    entries
        The index, for example from `list_runs`.
    layer
        Only this layer.
    subject_id
        Only this subject.
    attestation_id
        Only the run with this attestation.
    fingerprint
        Only runs of this dataset version.

    Returns
    -------
    tuple
        The matching entries in the order given.
    """
    return tuple(
        entry
        for entry in entries
        if (layer is None or entry.layer == layer)
        and (subject_id is None or entry.subject_id == subject_id)
        and (attestation_id is None or entry.attestation_id == attestation_id)
        and (fingerprint is None or entry.fingerprint == fingerprint)
    )


def build_run_table(entries: typing.Iterable[RunEntry]) -> pd.DataFrame:
    """Build the run index as a table, in the order given.

    Parameters
    ----------
    entries
        The index.

    Returns
    -------
    pandas.DataFrame
        One row per run with the columns `subject_id`, `layer`, `created_at`, `gate_outcome`, `attested`,
        `attestation_id`, `parent_id`, `fingerprint`, `contract_id` and `contract_version`.
    """
    rows = [
        {
            "subject_id": entry.subject_id,
            "layer": entry.layer,
            "created_at": entry.created_at,
            "gate_outcome": entry.gate_outcome.value,
            "attested": entry.attestation_id is not None,
            "attestation_id": entry.attestation_id,
            "parent_id": entry.parent_id,
            "fingerprint": entry.fingerprint.digest,
            "contract_id": entry.contract_id,
            "contract_version": entry.contract_version,
        }
        for entry in entries
    ]
    columns = [
        "subject_id",
        "layer",
        "created_at",
        "gate_outcome",
        "attested",
        "attestation_id",
        "parent_id",
        "fingerprint",
        "contract_id",
        "contract_version",
    ]
    return pd.DataFrame(rows, columns=columns)
