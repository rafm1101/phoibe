"""The profile: what was measured, per aspect, with no verdicts (ADR-02, ADR-03)."""

import dataclasses
import types
from collections.abc import Iterable, Mapping

import pandas as pd

from phoibe.dokimasia.core import dataset as dataset_module
from phoibe.dokimasia.core import finding as finding_module
from phoibe.dokimasia.core import measurement as measurement_module
from phoibe.dokimasia.core import subject as subject_module


@dataclasses.dataclass(frozen=True, slots=True)
class ProvenanceReport:
    """What a profile says about where its input came from (ADR-06). By default this is information."""

    has_provenance: bool
    """Whether the input reached this layer through `advance_dataset`."""
    fingerprint_matches: bool | None
    """Whether the data still has the fingerprint that was recorded when it advanced; `None` without provenance."""
    parent_attestation_id: str | None
    """Attestation of the layer the data came from."""
    transform: dataset_module.TransformDescription | None
    """The transform applied on the way, if any."""


_NO_PROVENANCE = ProvenanceReport(
    has_provenance=False, fingerprint_matches=None, parent_attestation_id=None, transform=None
)


class ProvenanceMismatchError(Exception):
    """Raised by `profile_dataset` in strict mode when the data changed after it advanced."""


def report_provenance(dataset: dataset_module.Dataset) -> ProvenanceReport:
    """Report where a dataset came from and whether it is still what advanced.

    Parameters
    ----------
    dataset
        The dataset.

    Returns
    -------
    ProvenanceReport
        The report. A dataset edited by hand after it advanced reports a fingerprint that does not match.
    """
    record = dataset.provenance
    if record is None:
        return _NO_PROVENANCE
    matches = dataset_module.compute_fingerprint(dataset.frame) == record.output_fingerprint
    return ProvenanceReport(
        has_provenance=True,
        fingerprint_matches=matches,
        parent_attestation_id=record.parent_attestation_id,
        transform=record.transform,
    )


@dataclasses.dataclass(frozen=True, slots=True, repr=False)
class Profile:
    """Summaries and findings per aspect for one subject, layer and dataset version. It holds no verdicts.

    A profile can exist without a contract; the measures then run with default parameters.
    """

    subject: subject_module.Subject
    """The subject that was measured."""
    layer: str
    """Identifier of the layer; its content is defined by the contract."""
    fingerprint: dataset_module.Fingerprint
    """Fingerprint of the dataset version that was measured."""
    aspects: Mapping[str, tuple[measurement_module.Outcome, ...]]
    """Outcomes per aspect, in the order the measures ran. A read-only copy."""
    provenance: ProvenanceReport = _NO_PROVENANCE
    """Where the input came from; the default is an input without provenance."""

    def __post_init__(self) -> None:
        """Replace the aspects by a read-only copy."""
        object.__setattr__(self, "aspects", types.MappingProxyType(dict(self.aspects)))

    def __repr__(self) -> str:
        """Summarize the profile in one line: subject, layer, fingerprint, and what became of the measures."""
        counts = self.count_by_status()
        statuses = ", ".join(f"{count} {status}" for status, count in counts.items() if count)
        return (
            f"Profile(subject={self.subject.id!r}, layer={self.layer!r}, fingerprint={self.fingerprint.digest[:8]}, "
            f"{len(self.aspects)} aspects, {sum(counts.values())} measures: {statuses or 'none'}, "
            f"{len(self.list_findings())} findings)"
        )

    def count_by_status(self) -> dict[str, int]:
        """Count the outcomes by what became of them. Every status appears, also with zero.

        Returns
        -------
        dict
            `measured`, `not_evaluable` and `error` to their counts.
        """
        counts = {"measured": 0, "not_evaluable": 0, "error": 0}
        for outcome in self.list_outcomes():
            counts[_get_status(outcome)] += 1
        return counts

    def compute_duration_s(self) -> float:
        """Add up the durations of all measurements.

        Returns
        -------
        float
            Seconds spent in the measures.
        """
        return sum(outcome.duration_s for outcome in self.list_outcomes())

    def list_outcomes(self) -> tuple[measurement_module.Outcome, ...]:
        """List all outcomes, aspect by aspect.

        Returns
        -------
        tuple
            Every outcome in run order.

        """
        return tuple(outcome for outcomes in self.aspects.values() for outcome in outcomes)

    def list_findings(self, *, kind: str | None = None) -> tuple[finding_module.Finding, ...]:
        """List the findings of all measured outcomes.

        Parameters
        ----------
        kind
            If given, only findings of this kind.

        Returns
        -------
        tuple
            The findings in run order.

        """
        return tuple(
            finding
            for outcome in self.list_outcomes()
            if isinstance(outcome, measurement_module.Measured)
            for finding in outcome.findings
            if kind is None or finding.kind == kind
        )

    def build_findings_table(self) -> pd.DataFrame:
        """Build the findings as a table, largest first.

        Returns
        -------
        pandas.DataFrame
            One row per finding with the columns `aspect`, `measure_id`, `kind`, `signals`, `start`, `end`,
            `n_steps`, `magnitude` and `magnitude_unit`, sorted by `n_steps`, largest first.

        """
        rows = [
            {
                "aspect": aspect,
                "measure_id": outcome.measure_id,
                "kind": finding.kind,
                "signals": ", ".join(finding.signals),
                "start": finding.start,
                "end": finding.end,
                "n_steps": finding.n_steps,
                "magnitude": finding.magnitude,
                "magnitude_unit": finding.magnitude_unit,
            }
            for aspect, outcomes in self.aspects.items()
            for outcome in outcomes
            if isinstance(outcome, measurement_module.Measured)
            for finding in outcome.findings
        ]
        columns = [
            "aspect",
            "measure_id",
            "kind",
            "signals",
            "start",
            "end",
            "n_steps",
            "magnitude",
            "magnitude_unit",
        ]
        table = pd.DataFrame(rows, columns=columns)
        return table.sort_values("n_steps", ascending=False, kind="stable", ignore_index=True)

    def build_summary_table(self) -> pd.DataFrame:
        """Build the metrics as a table.

        Returns
        -------
        pandas.DataFrame
            One row per metric with the columns `aspect`, `measure_id`, `key`, `signal`, `value` and `unit`.

        """
        rows = [
            {
                "aspect": aspect,
                "measure_id": outcome.measure_id,
                "key": metric.key,
                "signal": metric.signal,
                "value": metric.value,
                "unit": metric.unit,
            }
            for aspect, outcomes in self.aspects.items()
            for outcome in outcomes
            if isinstance(outcome, measurement_module.Measured)
            for metric in outcome.metrics
        ]
        return pd.DataFrame(rows, columns=["aspect", "measure_id", "key", "signal", "value", "unit"])

    def build_status_table(self) -> pd.DataFrame:
        """Build one row per measure, with what became of it.

        Returns
        -------
        pandas.DataFrame
            The columns `aspect`, `measure_id`, `status` (`measured`, `not_evaluable` or `error`),
            `detail` and `duration_s`.
        """
        rows = [
            {
                "aspect": aspect,
                "measure_id": outcome.measure_id,
                "status": _get_status(outcome),
                "detail": _get_detail(outcome),
                "duration_s": outcome.duration_s,
            }
            for aspect, outcomes in self.aspects.items()
            for outcome in outcomes
        ]
        return pd.DataFrame(rows, columns=["aspect", "measure_id", "status", "detail", "duration_s"])

    def to_frame(self) -> pd.DataFrame:
        """Build the whole profile as one table: every metric, and every measure that produced none.

        Returns
        -------
        pandas.DataFrame
            One row per metric and one row per measure that was not measured. The columns are `aspect`,
            `measure_id`, `status`, `key`, `signal`, `value`, `unit` and `duration_s`; the metric columns are
            empty for a measure that was not measured.
        """
        rows: list[dict[str, object]] = []
        for aspect, outcomes in self.aspects.items():
            for outcome in outcomes:
                base = {"aspect": aspect, "measure_id": outcome.measure_id, "status": _get_status(outcome)}
                if isinstance(outcome, measurement_module.Measured) and outcome.metrics:
                    rows.extend(
                        {
                            **base,
                            "key": metric.key,
                            "signal": metric.signal,
                            "value": metric.value,
                            "unit": metric.unit,
                            "duration_s": outcome.duration_s,
                        }
                        for metric in outcome.metrics
                    )
                else:
                    rows.append(
                        {
                            **base,
                            "key": None,
                            "signal": None,
                            "value": None,
                            "unit": None,
                            "duration_s": outcome.duration_s,
                        }
                    )
        columns = ["aspect", "measure_id", "status", "key", "signal", "value", "unit", "duration_s"]
        return pd.DataFrame(rows, columns=columns)


def _get_status(outcome: measurement_module.Outcome) -> str:
    """Name what became of an outcome."""
    if isinstance(outcome, measurement_module.Measured):
        return "measured"
    if isinstance(outcome, measurement_module.NotEvaluable):
        return "not_evaluable"
    return "error"


def _get_detail(outcome: measurement_module.Outcome) -> str:
    """Give the facts a person needs about an outcome that was not measured."""
    if isinstance(outcome, measurement_module.NotEvaluable):
        return f"{outcome.reason}: {outcome.detail}"
    if isinstance(outcome, measurement_module.MeasurementError):
        return f"{outcome.exception_type}: {outcome.message}"
    return ""


def profile_dataset(
    dataset: dataset_module.Dataset,
    subject: subject_module.Subject,
    layer: str,
    measures: Iterable[measurement_module.Measure],
    *,
    strict_provenance: bool = False,
) -> Profile:
    """Run the measures on a dataset and collect the outcomes per aspect. No verdicts are made.

    Parameters
    ----------
    dataset
        The data to measure.
    subject
        The subject the data belongs to.
    layer
        Identifier of the layer.
    measures
        The measures to run, already configured. One failing measure does not stop the others.
    strict_provenance
        If true, data that carries provenance but no longer has the recorded fingerprint is refused.

    Returns
    -------
    Profile
        The profile of this dataset version.

    Raises
    ------
    ValueError
        If two measures have the same id.
    ProvenanceMismatchError
        In strict mode, if the data changed after it advanced.
    """
    provenance = report_provenance(dataset)
    if strict_provenance and provenance.fingerprint_matches is False:
        raise ProvenanceMismatchError("the data changed after it advanced; its fingerprint no longer matches")
    measure_list = list(measures)
    ids = [measure.id for measure in measure_list]
    duplicates = sorted({measure_id for measure_id in ids if ids.count(measure_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate measure ids: {', '.join(duplicates)}")

    aspects: dict[str, list[measurement_module.Outcome]] = {}
    for measure in measure_list:
        outcome = measurement_module.run_measure(measure, dataset)
        aspects.setdefault(measure.aspect, []).append(outcome)
    return Profile(
        subject=subject,
        layer=layer,
        fingerprint=dataset.fingerprint,
        aspects={aspect: tuple(outcomes) for aspect, outcomes in aspects.items()},
        provenance=provenance,
    )
