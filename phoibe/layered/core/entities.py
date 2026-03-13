from __future__ import annotations

import dataclasses
import datetime
import enum
import logging
import typing

logger = logging.getLogger(__name__)


class Status(enum.Enum):
    """Status of a check."""

    PASSED = "passed"
    """Check passed."""
    FAILED = "failed"
    """Check failed for some reason."""
    WARNING = "warning"
    """Check issued a warning."""
    NOT_CHECKED = "not_checked"
    """Check was skipped."""
    ERROR = "error"
    """An error occurred during the check."""


class Severity(enum.Enum):
    """Severity of a check."""

    CRITICAL = "critical"
    """Critical: Resolve the failure reason."""
    WARNING = "warning"
    """Warning: Passed but has some issues."""
    INFO = "info"
    """Info."""


class ValidationMode(str, enum.Enum):
    """Validation execution mode."""

    PROFILING = "profiling"
    """Descriptive analysis: all rules run, no gates."""
    CONTRACT = "contract"
    """Prescriptive validation against contract: CRITICAL failures trigger gates."""


@dataclasses.dataclass(frozen=True)
class FileMetadata:
    """Value object of file metadata."""

    filename: str
    """File name."""
    size_bytes: int
    """File size [B]."""
    format: str
    """File format."""
    modified_at: datetime.datetime
    """Last modified."""

    @property
    def size_mb(self) -> float:
        """File size [MB]."""
        return round(self.size_bytes / 1024**2, 2)


@dataclasses.dataclass(frozen=True)
class RuleExecutionResult:
    """Return value object of each single check."""

    rule_name: str
    """Unique name of the check."""
    status: Status
    """Final status of the check."""
    severity: Severity
    """Criticality of the check."""
    required: typing.Any
    """Required/expected value."""
    actual: typing.Any
    """Acutal value."""
    points_max: int
    """Maximal reward for passing the check."""
    points_achieved: int
    """Actual reward."""
    message: str = ""
    """Passed message for logging."""
    details: dict = dataclasses.field(default_factory=dict)
    """Additional details to be passed."""

    def __post_init__(self):
        if self.points_achieved > self.points_max:
            raise ValueError(f"Points achieved ({self.points_achieved}) > max ({self.points_max}).")
        if self.points_achieved < 0:
            raise ValueError("Points cannot be negative.")


@dataclasses.dataclass
class LayerReport:
    """Report aggregating layer checks."""

    layer_name: str
    """Name of the layer."""
    turbine_id: str
    """Turbine identifier."""
    timestamp: datetime.datetime
    """Time of the layer validation start."""
    file_metadata: FileMetadata
    """File metadata."""
    detected_variables: dict[str, str | None]
    """Variables detected in the current data.."""
    rule_execution_results: list[RuleExecutionResult]
    """Results of the checks."""

    @property
    def score_max(self) -> int:
        return sum(result.points_max for result in self.rule_execution_results)

    @property
    def score_achieved(self) -> int:
        return sum(result.points_achieved for result in self.rule_execution_results)

    @property
    def percentage(self) -> float:
        return (self.score_achieved / self.score_max * 100) if self.score_max > 0 else 0

    @property
    def critical_failures(self) -> int:
        return sum(
            1
            for result in self.rule_execution_results
            if result.status == Status.FAILED and result.severity == Severity.CRITICAL
        )

    @property
    def warnings(self) -> int:
        return sum(1 for result in self.rule_execution_results if result.status == Status.WARNING)

    @property
    def overall_status(self) -> Status:
        if any(result.status == Status.ERROR for result in self.rule_execution_results):
            return Status.ERROR
        if self.critical_failures > 0:
            return Status.FAILED
        if self.warnings > 0:
            return Status.WARNING
        return Status.PASSED


@dataclasses.dataclass
class LayerGateKeeper:
    """Layer gate keeper. Decision whether data passes layer gate.

    Created from LayerReport when in CONTRACT mode.
    """

    passed: bool
    """True if data can proceed to next layer."""
    failed_critical_rules: list[str]
    """List of CRITICAL failures that blocked gate."""
    report: LayerReport
    """Full validation report."""
    statistics: dict[str, int]
    """Summary statistics of the gate."""

    @classmethod
    def from_report(cls, report: LayerReport) -> "LayerGateKeeper":
        """Create a layer gate keeper from validation report.

        Gate passes if no CRITICAL failures or CRITICAL errors exist.

        Parameters
        ----------
        report
            Validation report from a contract validation.

        Returns
        -------
        LayerGateKeeper
            GateDecision with pass/fail status and blocking failures.
        """

        if not report.rule_execution_results:
            logger.warning(
                f"Gate decision for empty report (layer={report.layer_name}, "
                f"turbine={report.turbine_id}): PASSED by default."
            )
            return cls(passed=True, failed_critical_rules=[], report=report, statistics=cls._empty_statistics())

        failed_critical_rules = [
            result.rule_name
            for result in report.rule_execution_results
            if result.severity == Severity.CRITICAL and result.status in (Status.FAILED, Status.ERROR)
        ]
        passed = len(failed_critical_rules) == 0
        statistics = cls._compute_statistics(report)

        if passed:
            logger.info(
                f"Gate PASSED: layer={report.layer_name}, turbine={report.turbine_id}, "
                f"score={report.score_achieved}/{report.score_max} ({report.percentage:.1f}%), "
                f"critical_failures=0."
            )
        else:
            logger.error(
                f"Gate FAILED: layer={report.layer_name}, turbine={report.turbine_id}, "
                f"failed_critical_rules={len(failed_critical_rules)}, "
                f"rules={', '.join(failed_critical_rules)}, "
                f"score={report.score_achieved}/{report.score_max} ({report.percentage:.1f}%)."
            )

        return cls(passed=passed, failed_critical_rules=failed_critical_rules, report=report, statistics=statistics)

    @staticmethod
    def _compute_statistics(report: LayerReport) -> dict[str, int]:
        """Calculate gate statistics from report.

        Parameters
        ----------
        LayerReport
            Report to evaluate.

        Returns
        -------
        stats
            Dictionary with counts for each status and severity combination.
        """
        stats = {
            "total_checks": len(report.rule_execution_results),
            "passed": 0,
            "failed": 0,
            "warning": 0,
            "error": 0,
            "not_checked": 0,
            "critical_failed": 0,
            "critical_error": 0,
            "warning_failed": 0,
            "info_failed": 0,
        }

        for result in report.rule_execution_results:
            if result.status == Status.PASSED:
                stats["passed"] += 1
            elif result.status == Status.FAILED:
                stats["failed"] += 1
            elif result.status == Status.WARNING:
                stats["warning"] += 1
            elif result.status == Status.ERROR:
                stats["error"] += 1
            elif result.status == Status.NOT_CHECKED:
                stats["not_checked"] += 1

            if result.severity == Severity.CRITICAL and result.status == Status.FAILED:
                stats["critical_failed"] += 1
            elif result.severity == Severity.CRITICAL and result.status == Status.ERROR:
                stats["critical_error"] += 1
            elif result.severity == Severity.WARNING and result.status == Status.FAILED:
                stats["warning_failed"] += 1
            elif result.severity == Severity.INFO and result.status == Status.FAILED:
                stats["info_failed"] += 1

        return stats

    @staticmethod
    def _empty_statistics() -> dict[str, int]:
        """Return empty statistics dict."""
        return {
            "total_checks": 0,
            "passed": 0,
            "failed": 0,
            "warning": 0,
            "error": 0,
            "not_checked": 0,
            "critical_failed": 0,
            "critical_error": 0,
            "warning_failed": 0,
            "info_failed": 0,
        }


class LayerGateFailureError(Exception):
    """Raised when data fails a contract validation gate.

    Attributes
    ----------
    decision
        LayerGateKeeper that caused failure.
    message
        Human-readable error message.
    """

    def __init__(self, decision: LayerGateKeeper):
        self.decision = decision
        count = len(decision.failed_critical_rules)
        plural = "failure" if count == 1 else "failures"

        failures_str = ", ".join(decision.failed_critical_rules)

        message = (
            f"Contract validation gate FAILED for layer '{decision.report.layer_name}'. "
            f"{count} CRITICAL {plural}: {failures_str}. "
            f"Score: {decision.report.score_achieved}/{decision.report.score_max}. "
            f"({decision.report.percentage:.1f}%). "
            f"Turbine: {decision.report.turbine_id}."
        )

        super().__init__(message)

        logger.error(f"GateFailureError raised: {count} blocking failures, " f"statistics={decision.statistics}")


__all__ = ["LayerGateKeeper", "LayerGateFailureError"]
