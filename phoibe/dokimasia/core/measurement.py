"""Measurements and their outcomes. Every measurement is total (ADR-02, ADR-12)."""

import dataclasses
import enum
import logging
import typing

from phoibe.dokimasia.core import dataset as dataset_module
from phoibe.dokimasia.core import finding as finding_module

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class Metric:
    """A typed summary value, addressed by key."""

    key: str
    """Name of the metric, for example `n_gaps`; unique per measure and signal."""
    value: float
    """The value."""
    unit: str | None = None
    """Unit of the value, `None` for counts and shares."""
    signal: str | None = None
    """Signal the metric belongs to, `None` for the dataset as a whole."""


class NotEvaluableReason(enum.StrEnum):
    """Why a measurement could not be evaluated."""

    MISSING_SIGNAL = "missing_signal"
    MISSING_STRUCTURE = "missing_structure"
    INSUFFICIENT_DATA = "insufficient_data"
    LIMIT_EXCEEDED = "limit_exceeded"


@dataclasses.dataclass(frozen=True, slots=True)
class Measured:
    """A measurement that was evaluated."""

    measure_id: str
    """Identifier of the measure that produced this."""
    metrics: tuple[Metric, ...] = ()
    """Summaries."""
    findings: tuple[finding_module.Finding, ...] = ()
    """Located findings."""


@dataclasses.dataclass(frozen=True, slots=True)
class NotEvaluable:
    """A measurement that could not be evaluated for an expected reason. This is never a pass."""

    measure_id: str
    """Identifier of the measure that produced this."""
    reason: NotEvaluableReason
    """Why."""
    detail: str = ""
    """Facts for a person, for example what was missing."""


@dataclasses.dataclass(frozen=True, slots=True)
class MeasurementError:
    """A measurement that failed unexpectedly. This is never a pass."""

    measure_id: str
    """Identifier of the measure that produced this."""
    exception_type: str
    """Class name of the exception, or of the invalid result."""
    message: str
    """Message of the exception."""


type Outcome = Measured | NotEvaluable | MeasurementError


class Measure(typing.Protocol):
    """A measurement on a dataset. It returns an outcome and never judges.

    Expected problems, such as a missing signal or too few rows, give `NotEvaluable`. Only a programming error
    raises, and `run_measure` contains it.
    """

    @property
    def id(self) -> str:
        """Identifier of the measure, unique within a profile."""
        ...

    @property
    def aspect(self) -> str:
        """Aspect of data quality the measure belongs to, for example `temporal`."""
        ...

    def measure(self, dataset: dataset_module.Dataset) -> Measured | NotEvaluable:
        """Measure the dataset."""
        ...


def run_measure(measure: Measure, dataset: dataset_module.Dataset) -> Outcome:
    """Run one measure and contain every failure as a typed outcome.

    This is the one place that catches `Exception` (ADR-12): one failing measurement does not stop the others.

    Parameters
    ----------
    measure
        The measure to run.
    dataset
        The data to measure.

    Returns
    -------
    Measured | NotEvaluable | MeasurementError
        The measure's own outcome, or an error outcome carrying the measure id and the exception type and message.
        A result that is none of the outcome types is an error too.

    """
    try:
        result = measure.measure(dataset)
    # The single containment point of ADR-12. Ruff accepts it because the failure is logged with its traceback.
    except Exception as error:
        logger.warning("measure %s failed with %s: %s", measure.id, type(error).__name__, error, exc_info=True)
        return MeasurementError(measure_id=measure.id, exception_type=type(error).__name__, message=str(error))
    if not isinstance(result, Measured | NotEvaluable):
        logger.warning("measure %s returned %s, not an outcome", measure.id, type(result).__name__)
        return MeasurementError(
            measure_id=measure.id, exception_type=type(result).__name__, message="the measure returned no outcome"
        )
    return result
