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


@dataclasses.dataclass(frozen=True, slots=True)
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
            The columns `aspect`, `measure_id`, `status` (`measured`, `not_evaluable` or `error`) and `detail`.

        """
        rows = []
        for aspect, outcomes in self.aspects.items():
            for outcome in outcomes:
                if isinstance(outcome, measurement_module.Measured):
                    status, detail = "measured", ""
                elif isinstance(outcome, measurement_module.NotEvaluable):
                    status, detail = "not_evaluable", f"{outcome.reason}: {outcome.detail}"
                else:
                    status, detail = "error", f"{outcome.exception_type}: {outcome.message}"
                rows.append({"aspect": aspect, "measure_id": outcome.measure_id, "status": status, "detail": detail})
        return pd.DataFrame(rows, columns=["aspect", "measure_id", "status", "detail"])


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
