"""Attestation and provenance (ADR-06)."""

import dataclasses
import datetime
import hashlib
import types
from collections.abc import Mapping

from phoibe.dokimasia.core import dataset as dataset_module
from phoibe.dokimasia.core import gate as gate_module
from phoibe.dokimasia.core import subject as subject_module


@dataclasses.dataclass(frozen=True, slots=True)
class Attestation:
    """The record that a dataset version passed the gate of a layer, conditionally or not."""

    id: str
    """Identifier derived from the fields below and the time."""
    subject_id: str
    """The subject."""
    layer: str
    """The layer."""
    fingerprint: dataset_module.Fingerprint
    """Fingerprint of the dataset version that was judged."""
    contract_id: str
    """Contract that was applied."""
    contract_version: str
    """Version of that contract."""
    outcome: gate_module.GateOutcome
    """`pass` or `conditional`; a failed gate has no attestation."""
    conditions: tuple[str, ...]
    """Texts of the conditions of a conditional pass; empty otherwise."""
    verdict_counts: Mapping[str, int]
    """Number of criteria per verdict. A read-only copy."""
    created_at: datetime.datetime
    """Time of the attestation, in UTC."""
    software_version: str
    """Version of the software that judged."""
    parent_id: str | None = None
    """Attestation of the previous layer, if this layer received its data through `advance`."""

    def __post_init__(self) -> None:
        """Replace the verdict counts by a read-only copy."""
        object.__setattr__(self, "verdict_counts", types.MappingProxyType(dict(self.verdict_counts)))


def build_attestation(
    *,
    subject: subject_module.Subject,
    layer: str,
    fingerprint: dataset_module.Fingerprint,
    contract_id: str,
    contract_version: str,
    result: gate_module.GateResult,
    created_at: datetime.datetime,
    software_version: str,
    parent_id: str | None = None,
) -> Attestation:
    """Build the attestation of a passed gate.

    The time is passed in, so that the core reads no clock and a run can be reproduced.

    Parameters
    ----------
    subject
        The subject that was judged.
    layer
        The layer.
    fingerprint
        Fingerprint of the judged dataset version.
    contract_id
        Identifier of the contract.
    contract_version
        Version of the contract.
    result
        The gate result; it must not be a failure.
    created_at
        Time of the attestation, timezone-aware, with a zero UTC offset.
    software_version
        Version of the software that judged.
    parent_id
        Attestation of the previous layer, if any.

    Returns
    -------
    Attestation
        The attestation, with the conditions of a conditional pass.

    Raises
    ------
    ValueError
        If the gate failed, or if `created_at` is not UTC.

    """
    if result.outcome == gate_module.GateOutcome.FAIL:
        raise ValueError("a failed gate has no attestation")
    if created_at.tzinfo is None or created_at.utcoffset() != datetime.timedelta(0):
        raise ValueError("created_at must be a timezone-aware UTC timestamp")
    identity = "\x1f".join(
        [subject.id, layer, fingerprint.digest, contract_id, contract_version, created_at.isoformat()]
    )
    return Attestation(
        id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        subject_id=subject.id,
        layer=layer,
        fingerprint=fingerprint,
        contract_id=contract_id,
        contract_version=contract_version,
        outcome=result.outcome,
        conditions=tuple(reason.text for reason in result.conditions),
        verdict_counts=result.count_by_verdict(),
        created_at=created_at,
        software_version=software_version,
        parent_id=parent_id,
    )
