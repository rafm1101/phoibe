import dataclasses
import datetime
import enum
import typing


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
