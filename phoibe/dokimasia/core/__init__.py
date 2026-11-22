"""The core model: subject, dataset, signal, finding, measurement outcome and profile (ADR-01 to ADR-03, ADR-12).

The core imports nothing of ours and does no file I/O. Public names are those without a leading underscore.
"""

from phoibe.dokimasia.core.attestation import Attestation, build_attestation
from phoibe.dokimasia.core.dataset import (
    FINGERPRINT_ALGORITHM,
    Dataset,
    Fingerprint,
    ProvenanceRecord,
    TransformDescription,
    compute_fingerprint,
)
from phoibe.dokimasia.core.finding import Finding, build_finding_mask
from phoibe.dokimasia.core.gate import (
    Effect,
    GateFailedError,
    GateOutcome,
    GatePolicy,
    GateReason,
    GateResult,
    WorstEffectPolicy,
    get_effect,
)
from phoibe.dokimasia.core.measurement import (
    Measure,
    Measured,
    MeasurementError,
    Metric,
    NotEvaluable,
    NotEvaluableReason,
    Outcome,
    run_measure,
)
from phoibe.dokimasia.core.profile import (
    Profile,
    ProvenanceMismatchError,
    ProvenanceReport,
    profile_dataset,
    report_provenance,
)
from phoibe.dokimasia.core.signal import Catalog, Signal, Statistic
from phoibe.dokimasia.core.subject import Subject
from phoibe.dokimasia.core.transform import (
    Transform,
    Transformed,
    TransformRefused,
    TransformRefusedError,
    advance_dataset,
)
from phoibe.dokimasia.core.verdict import CriterionVerdict, Severity, Verdict, build_verdict_table

__all__ = [
    "FINGERPRINT_ALGORITHM",
    "Attestation",
    "Catalog",
    "CriterionVerdict",
    "Dataset",
    "Effect",
    "Finding",
    "Fingerprint",
    "GateFailedError",
    "GateOutcome",
    "GatePolicy",
    "GateReason",
    "GateResult",
    "Measure",
    "Measured",
    "MeasurementError",
    "Metric",
    "NotEvaluable",
    "NotEvaluableReason",
    "Outcome",
    "Profile",
    "ProvenanceMismatchError",
    "ProvenanceRecord",
    "ProvenanceReport",
    "Severity",
    "Signal",
    "Statistic",
    "Subject",
    "Transform",
    "TransformDescription",
    "TransformRefused",
    "TransformRefusedError",
    "Transformed",
    "Verdict",
    "WorstEffectPolicy",
    "advance_dataset",
    "build_attestation",
    "build_finding_mask",
    "build_verdict_table",
    "compute_fingerprint",
    "get_effect",
    "profile_dataset",
    "report_provenance",
    "run_measure",
]
