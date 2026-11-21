"""The core model: subject, dataset, signal, finding, measurement outcome and profile (ADR-01 to ADR-03, ADR-12).

The core imports nothing of ours and does no file I/O. Public names are those without a leading underscore.
"""

from phoibe.dokimasia.core.dataset import FINGERPRINT_ALGORITHM, Dataset, Fingerprint, compute_fingerprint
from phoibe.dokimasia.core.finding import Finding, build_finding_mask
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
from phoibe.dokimasia.core.profile import Profile, profile_dataset
from phoibe.dokimasia.core.signal import Catalog, Signal, Statistic
from phoibe.dokimasia.core.subject import Subject

__all__ = [
    "FINGERPRINT_ALGORITHM",
    "Catalog",
    "Dataset",
    "Finding",
    "Fingerprint",
    "Measure",
    "Measured",
    "MeasurementError",
    "Metric",
    "NotEvaluable",
    "NotEvaluableReason",
    "Outcome",
    "Profile",
    "Signal",
    "Statistic",
    "Subject",
    "build_finding_mask",
    "compute_fingerprint",
    "profile_dataset",
    "run_measure",
]
