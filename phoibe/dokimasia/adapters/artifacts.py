"""Map profiles, verdicts and attestations to plain data and back (ADR-10).

Every artifact carries `schema_version` and `kind`, so a file says what it is. One version number covers the whole
catalogue of artifact schemas; a file of another version is refused. The three artifacts of a run are keyed by the
subject, the layer and the fingerprint of the dataset version.
"""

import dataclasses

from phoibe.dokimasia import core
from phoibe.dokimasia.adapters import _plain, _text

SCHEMA_VERSION = 1
"""Version of the catalogue of artifact schemas."""

PROFILE_KIND = "profile"
VERDICTS_KIND = "verdicts"
ATTESTATION_KIND = "attestation"

ArtifactError = _plain.ArtifactError


@dataclasses.dataclass(frozen=True, slots=True)
class VerdictsRecord:
    """The verdicts of one layer with the keys that tie them to a run."""

    subject_id: str
    """The subject."""
    layer: str
    """The layer."""
    fingerprint: core.Fingerprint
    """Fingerprint of the judged dataset version."""
    contract_id: str
    """Contract that was applied."""
    contract_version: str
    """Version of that contract."""
    result: core.GateResult
    """The gate decision with its verdicts."""


def _start(kind: str) -> dict[str, object]:
    return {"schema_version": SCHEMA_VERSION, "kind": kind}


def _open(data: object, kind: str) -> _plain.Reader:
    reader = _plain.Reader(data)
    version = reader.integer("schema_version")
    if version != SCHEMA_VERSION:
        raise ArtifactError("schema_version", f"unsupported version {version}; this software reads {SCHEMA_VERSION}")
    found = reader.text("kind")
    if found != kind:
        raise ArtifactError("kind", f"expected {kind!r}, found {found!r}")
    return reader


def _fingerprint_to_data(fingerprint: core.Fingerprint) -> dict[str, object]:
    return {"algorithm": fingerprint.algorithm, "digest": fingerprint.digest}


def _fingerprint_from_data(reader: _plain.Reader) -> core.Fingerprint:
    fingerprint = core.Fingerprint(algorithm=reader.text("algorithm"), digest=reader.text("digest"))
    reader.finish()
    return fingerprint


def _transform_to_data(transform: core.TransformDescription | None) -> dict[str, object] | None:
    if transform is None:
        return None
    return {
        "name": transform.name,
        "parameters": _plain.encode_plain(transform.parameters, "transform.parameters"),
        "software_version": transform.software_version,
        "rule": transform.rule,
    }


def _transform_from_data(reader: _plain.Reader | None) -> core.TransformDescription | None:
    if reader is None:
        return None
    transform = core.TransformDescription(
        name=reader.text("name"),
        parameters=reader.plain_mapping("parameters"),
        software_version=reader.text("software_version"),
        rule=reader.text_or_none("rule"),
    )
    reader.finish()
    return transform


def _finding_to_data(finding: core.Finding) -> dict[str, object]:
    return {
        "kind": finding.kind,
        "signals": list(finding.signals),
        "start": _plain.encode_time(finding.start),
        "end": _plain.encode_time(finding.end),
        "n_steps": int(finding.n_steps),
        "magnitude": None if finding.magnitude is None else _plain.encode_float(finding.magnitude),
        "magnitude_unit": finding.magnitude_unit,
    }


def _finding_from_data(reader: _plain.Reader) -> core.Finding:
    finding = core.Finding(
        kind=reader.text("kind"),
        signals=reader.texts("signals"),
        start=reader.time("start"),
        end=reader.time("end"),
        n_steps=reader.integer("n_steps"),
        magnitude=reader.number_or_none("magnitude"),
        magnitude_unit=reader.text_or_none("magnitude_unit"),
    )
    reader.finish()
    return finding


def _outcome_to_data(outcome: core.Outcome) -> dict[str, object]:
    if isinstance(outcome, core.Measured):
        return {
            "status": "measured",
            "measure_id": outcome.measure_id,
            "duration_s": _plain.encode_float(outcome.duration_s),
            "metrics": [
                {"key": m.key, "signal": m.signal, "value": _plain.encode_float(m.value), "unit": m.unit}
                for m in outcome.metrics
            ],
            "findings": [_finding_to_data(finding) for finding in outcome.findings],
        }
    if isinstance(outcome, core.NotEvaluable):
        return {
            "status": "not_evaluable",
            "measure_id": outcome.measure_id,
            "duration_s": _plain.encode_float(outcome.duration_s),
            "reason": outcome.reason.value,
            "detail": outcome.detail,
        }
    return {
        "status": "error",
        "measure_id": outcome.measure_id,
        "duration_s": _plain.encode_float(outcome.duration_s),
        "exception_type": outcome.exception_type,
        "message": outcome.message,
    }


def _metric_from_data(reader: _plain.Reader) -> core.Metric:
    metric = core.Metric(
        key=reader.text("key"),
        value=reader.number("value"),
        unit=reader.text_or_none("unit"),
        signal=reader.text_or_none("signal"),
    )
    reader.finish()
    return metric


def _outcome_from_data(reader: _plain.Reader) -> core.Outcome:
    status = reader.text("status")
    measure_id = reader.text("measure_id")
    duration_s = reader.number("duration_s")
    outcome: core.Outcome
    if status == "measured":
        outcome = core.Measured(
            measure_id=measure_id,
            metrics=tuple(_metric_from_data(item) for item in reader.readers("metrics")),
            findings=tuple(_finding_from_data(item) for item in reader.readers("findings")),
            duration_s=duration_s,
        )
    elif status == "not_evaluable":
        reason = reader.text("reason")
        try:
            parsed = core.NotEvaluableReason(reason)
        except ValueError as error:
            raise ArtifactError(reader.child_path("reason"), f"unknown reason {reason!r}") from error
        outcome = core.NotEvaluable(measure_id, parsed, reader.text("detail"), duration_s)
    elif status == "error":
        outcome = core.MeasurementError(measure_id, reader.text("exception_type"), reader.text("message"), duration_s)
    else:
        raise ArtifactError(reader.child_path("status"), f"unknown status {status!r}")
    reader.finish()
    return outcome


def profile_to_data(profile: core.Profile) -> dict[str, object]:
    """Map a profile to plain data.

    Parameters
    ----------
    profile
        The profile.

    Returns
    -------
    dict
        Plain data in a fixed key order, with `schema_version` and `kind`. Aspects and outcomes keep their run order.

    Raises
    ------
    ArtifactError
        If the subject specification or the transform parameters hold something that is not plain data.
    """
    provenance = profile.provenance
    data = _start(PROFILE_KIND)
    data |= {
        "subject": {
            "id": profile.subject.id,
            "device_type": profile.subject.device_type,
            "specification": _plain.encode_plain(profile.subject.specification, "subject.specification"),
        },
        "layer": profile.layer,
        "fingerprint": _fingerprint_to_data(profile.fingerprint),
        "provenance": {
            "has_provenance": provenance.has_provenance,
            "fingerprint_matches": provenance.fingerprint_matches,
            "parent_attestation_id": provenance.parent_attestation_id,
            "transform": _transform_to_data(provenance.transform),
        },
        "aspects": [
            {"name": name, "outcomes": [_outcome_to_data(outcome) for outcome in outcomes]}
            for name, outcomes in profile.aspects.items()
        ],
    }
    return data


def profile_from_data(data: object) -> core.Profile:
    """Map plain data back to a profile, the inverse of `profile_to_data`.

    Parameters
    ----------
    data
        Plain data as read from a file.

    Returns
    -------
    Profile
        The profile.

    Raises
    ------
    ArtifactError
        If the version is not supported, the kind is not `profile`, or anything is missing, unknown or of the wrong
        type; the error names the place.
    """
    reader = _open(data, PROFILE_KIND)
    subject_reader = reader.reader("subject")
    subject = core.Subject(
        id=subject_reader.text("id"),
        device_type=subject_reader.text("device_type"),
        specification=subject_reader.plain_mapping("specification"),
    )
    subject_reader.finish()
    layer = reader.text("layer")
    fingerprint = _fingerprint_from_data(reader.reader("fingerprint"))
    provenance_reader = reader.reader("provenance")
    provenance = core.ProvenanceReport(
        has_provenance=provenance_reader.boolean("has_provenance"),
        fingerprint_matches=provenance_reader.boolean_or_none("fingerprint_matches"),
        parent_attestation_id=provenance_reader.text_or_none("parent_attestation_id"),
        transform=_transform_from_data(provenance_reader.reader_or_none("transform")),
    )
    provenance_reader.finish()
    aspects: dict[str, tuple[core.Outcome, ...]] = {}
    for aspect_reader in reader.readers("aspects"):
        name = aspect_reader.text("name")
        if name in aspects:
            raise ArtifactError(aspect_reader.child_path("name"), f"aspect {name!r} appears twice")
        aspects[name] = tuple(_outcome_from_data(item) for item in aspect_reader.readers("outcomes"))
        aspect_reader.finish()
    reader.finish()
    return core.Profile(subject=subject, layer=layer, fingerprint=fingerprint, aspects=aspects, provenance=provenance)


def _verdict_to_data(verdict: core.CriterionVerdict) -> dict[str, object]:
    return {
        "check_id": verdict.check_id,
        "measure_id": verdict.measure_id,
        "signal": verdict.signal,
        "severity": verdict.severity.value,
        "verdict": verdict.verdict.value,
        "value": None if verdict.value is None else _plain.encode_float(verdict.value),
        "reason": verdict.reason,
        "clause": verdict.clause,
    }


def _enum[E: core.Verdict | core.Severity | core.Effect | core.GateOutcome](
    enum_type: type[E], reader: _plain.Reader, key: str
) -> E:
    text = reader.text(key)
    try:
        return enum_type(text)
    except ValueError as error:
        raise ArtifactError(reader.child_path(key), f"unknown {key} {text!r}") from error


def _verdict_from_data(reader: _plain.Reader) -> core.CriterionVerdict:
    verdict = core.CriterionVerdict(
        check_id=reader.text("check_id"),
        measure_id=reader.text("measure_id"),
        signal=reader.text_or_none("signal"),
        severity=_enum(core.Severity, reader, "severity"),
        verdict=_enum(core.Verdict, reader, "verdict"),
        value=reader.number_or_none("value"),
        reason=reader.text("reason"),
        clause=reader.text_or_none("clause"),
    )
    reader.finish()
    return verdict


def _reason_from_data(reader: _plain.Reader) -> core.GateReason:
    reason = core.GateReason(
        check_id=reader.text("check_id"),
        verdict=_enum(core.Verdict, reader, "verdict"),
        severity=_enum(core.Severity, reader, "severity"),
        effect=_enum(core.Effect, reader, "effect"),
        text=reader.text("text"),
    )
    reader.finish()
    return reason


def verdicts_to_data(record: VerdictsRecord) -> dict[str, object]:
    """Map the verdicts of a layer to plain data.

    Parameters
    ----------
    record
        The verdicts with their keys.

    Returns
    -------
    dict
        Plain data in a fixed key order, with `schema_version` and `kind`.
    """
    result = record.result
    data = _start(VERDICTS_KIND)
    data |= {
        "subject_id": record.subject_id,
        "layer": record.layer,
        "fingerprint": _fingerprint_to_data(record.fingerprint),
        "contract_id": record.contract_id,
        "contract_version": record.contract_version,
        "gate": {
            "outcome": result.outcome.value,
            "policy": result.policy,
            "reasons": [
                {
                    "check_id": reason.check_id,
                    "verdict": reason.verdict.value,
                    "severity": reason.severity.value,
                    "effect": reason.effect.value,
                    "text": reason.text,
                }
                for reason in result.reasons
            ],
        },
        "verdicts": [_verdict_to_data(verdict) for verdict in result.verdicts],
    }
    return data


def verdicts_from_data(data: object) -> VerdictsRecord:
    """Map plain data back to the verdicts of a layer, the inverse of `verdicts_to_data`.

    Parameters
    ----------
    data
        Plain data as read from a file.

    Returns
    -------
    VerdictsRecord
        The verdicts with their keys.

    Raises
    ------
    ArtifactError
        As `profile_from_data`, for the kind `verdicts`.
    """
    reader = _open(data, VERDICTS_KIND)
    subject_id = reader.text("subject_id")
    layer = reader.text("layer")
    fingerprint = _fingerprint_from_data(reader.reader("fingerprint"))
    contract_id = reader.text("contract_id")
    contract_version = reader.text("contract_version")
    gate = reader.reader("gate")
    outcome = _enum(core.GateOutcome, gate, "outcome")
    policy = gate.text("policy")
    reasons = tuple(_reason_from_data(item) for item in gate.readers("reasons"))
    gate.finish()
    verdicts = tuple(_verdict_from_data(item) for item in reader.readers("verdicts"))
    reader.finish()
    result = core.GateResult(outcome=outcome, reasons=reasons, verdicts=verdicts, policy=policy)
    return VerdictsRecord(subject_id, layer, fingerprint, contract_id, contract_version, result)


def attestation_to_data(attestation: core.Attestation) -> dict[str, object]:
    """Map an attestation to plain data.

    Parameters
    ----------
    attestation
        The attestation.

    Returns
    -------
    dict
        Plain data in a fixed key order, with `schema_version` and `kind`. The time is written in UTC.
    """
    data = _start(ATTESTATION_KIND)
    data |= {
        "id": attestation.id,
        "subject_id": attestation.subject_id,
        "layer": attestation.layer,
        "fingerprint": _fingerprint_to_data(attestation.fingerprint),
        "contract_id": attestation.contract_id,
        "contract_version": attestation.contract_version,
        "outcome": attestation.outcome.value,
        "conditions": list(attestation.conditions),
        "verdict_counts": dict(attestation.verdict_counts),
        "created_at": _plain.encode_time(attestation.created_at),
        "software_version": attestation.software_version,
        "parent_id": attestation.parent_id,
    }
    return data


def attestation_from_data(data: object) -> core.Attestation:
    """Map plain data back to an attestation, the inverse of `attestation_to_data`.

    Parameters
    ----------
    data
        Plain data as read from a file.

    Returns
    -------
    Attestation
        The attestation.

    Raises
    ------
    ArtifactError
        As `profile_from_data`, for the kind `attestation`; also if the time is not in UTC.
    """
    reader = _open(data, ATTESTATION_KIND)
    counts_reader = reader.reader("verdict_counts")
    counts = {name: counts_reader.integer(name) for name in counts_reader.list_keys()}
    attestation = core.Attestation(
        id=reader.text("id"),
        subject_id=reader.text("subject_id"),
        layer=reader.text("layer"),
        fingerprint=_fingerprint_from_data(reader.reader("fingerprint")),
        contract_id=reader.text("contract_id"),
        contract_version=reader.text("contract_version"),
        outcome=_enum(core.GateOutcome, reader, "outcome"),
        conditions=reader.texts("conditions"),
        verdict_counts=counts,
        created_at=reader.time_utc("created_at"),
        software_version=reader.text("software_version"),
        parent_id=reader.text_or_none("parent_id"),
    )
    counts_reader.finish()
    reader.finish()
    return attestation


def format_artifact(data: dict[str, object], fmt: _text.Format = "yaml") -> str:
    """Format the plain data of an artifact as text.

    Parameters
    ----------
    data
        The result of `profile_to_data`, `verdicts_to_data` or `attestation_to_data`.
    fmt
        `yaml` (default) or `json`.

    Returns
    -------
    str
        The text.
    """
    return _text.format_text(data, fmt)


def parse_artifact(text: str, fmt: _text.Format = "yaml") -> object:
    """Parse the text of an artifact into plain data, ready for one of the `*_from_data` functions.

    Parameters
    ----------
    text
        The text.
    fmt
        `yaml` (default) or `json`.

    Returns
    -------
    object
        The plain data.

    Raises
    ------
    ArtifactError
        If the text is not valid, repeats a key or holds a non-finite number literal.
    """
    try:
        return _text.parse_text(text, fmt)
    except ValueError as error:
        raise ArtifactError("", str(error)) from error
