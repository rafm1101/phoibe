import abc
import logging
import typing

import pandas as pd

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.core.entities import RuleExecutionResult
from phoibe.layered.core.entities import Severity
from phoibe.layered.core.entities import Status


class RuleExecutionResultBuilder:
    """Builder for creating RuleExecutionResult objects.

    Notes
    -----
    1. Provides convenience methods for the common result patterns: passed, failed, not_checked, error.
    """

    def __init__(self, rule_name: str, points: int, severity: Severity = Severity.CRITICAL):
        self.rule_name = rule_name
        self.points = points
        self.severity = severity

    def passed(
        self, required: typing.Any, actual: typing.Any, message: str = "", details: dict | None = None
    ) -> RuleExecutionResult:
        """Create a PASSED result.

        Parameters
        ----------
        required
            Expected value.
        actual
            Actual value.
        message
            Optional success message.
        details
            Optional additional details.

        Returns
        -------
        RuleExecutionResult
            Status PASSED.
        """
        return RuleExecutionResult(
            rule_name=self.rule_name,
            status=Status.PASSED,
            severity=self.severity,
            required=required,
            actual=actual,
            points_max=self.points,
            points_achieved=self.points,
            message=message,
            details=details or {},
        )

    def failed(
        self, required: typing.Any, actual: typing.Any, message: str = "", details: dict | None = None
    ) -> RuleExecutionResult:
        """Create a FAILED result.

        Parameters
        ----------
        required
            Expected value.
        actual
            Actual value.
        message
            Optional success message.
        details
            Optional additional details.

        Returns
        -------
        RuleExecutionResult
            Status FAILED.
        """
        return RuleExecutionResult(
            rule_name=self.rule_name,
            status=Status.FAILED,
            severity=self.severity,
            required=required,
            actual=actual,
            points_max=self.points,
            points_achieved=0,
            message=message,
            details=details or {},
        )

    def not_checked(self, message: str) -> RuleExecutionResult:
        """Create a NOT CHECKED result.

        Parameters
        ----------
        required
            Expected value.
        actual
            Actual value.
        message
            Optional success message.
        details
            Optional additional details.

        Returns
        -------
        RuleExecutionResult
            Status NOT CHECKED.
        """
        return RuleExecutionResult(
            rule_name=self.rule_name,
            status=Status.NOT_CHECKED,
            severity=self.severity,
            required="N/A",
            actual=None,
            points_max=self.points,
            points_achieved=0,
            message=message,
        )

    def error(self, message: str, exception: Exception | None = None) -> RuleExecutionResult:
        """Create a ERROR result.

        Parameters
        ----------
        required
            Expected value.
        actual
            Actual value.
        message
            Optional success message.
        details
            Optional additional details.

        Returns
        -------
        RuleExecutionResult
            Status ERROR.
        """
        full_message = message
        if exception:
            full_message = f"{message}: {str(exception)}"

        return RuleExecutionResult(
            rule_name=self.rule_name,
            status=Status.ERROR,
            severity=self.severity,
            required="N/A",
            actual=None,
            points_max=self.points,
            points_achieved=0,
            message=full_message,
        )


class ValidationRule(abc.ABC):
    """Interface for each validation rule.

    Properties
    ----------
    name
        Name of the rule.

    Methods
    -------
    execute
        Execute rule on a given table.
    """

    def __init__(self, points: int, severity: Severity = Severity.CRITICAL, logger: logging.Logger | None = None):
        """Initialize the validation rule.

        Parameters
        ----------
        points
            Maximum points for passing this rule
        severity
            Severity level (CRITICAL, WARNING, INFO).
        logger
            Logger instance.
        """
        self.points = points
        self.severity = severity
        self._logger = logger
        self._result_builder: RuleExecutionResultBuilder | None = None

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique identifier of this rule."""
        pass

    @property
    def result_builder(self) -> RuleExecutionResultBuilder:
        if self._result_builder is None:
            self._result_builder = RuleExecutionResultBuilder(self.name, self.points, self.severity)
        return self._result_builder

    @abc.abstractmethod
    def execute(self, df: pd.DataFrame, context: ValidationContext) -> RuleExecutionResult:
        """Execute a validation rule.

        Parameters
        ----------
        df
            Dataframe holding the actual data.
        context
            Context/configuration of the rule.

        Returns
        -------
        RuleExecutionResult
            Result of the rule execution.
        """
        pass


__all__ = ["ValidationRule", "RuleExecutionResultBuilder"]
