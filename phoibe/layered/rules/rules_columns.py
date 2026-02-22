import logging

import pandas as pd

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.application.factory import RuleRegistry
from phoibe.layered.core.entities import RuleExecutionResult
from phoibe.layered.core.entities import Severity
from phoibe.layered.rules.rule import ValidationRule


@RuleRegistry.register("required_variable")
class RequiredVariableRule(ValidationRule):
    """Validate the presence of required variables in the data.

    Primer validation rule. Downstream rules may not be executed due to missing variables.
    """

    def __init__(
        self,
        variable_name: str,
        points: int = 10,
        severity: Severity = Severity.CRITICAL,
        logger: logging.Logger | None = None,
    ):
        super().__init__(points, severity, logger)
        self.variable_name = variable_name

    @property
    def name(self) -> str:
        return f"variable_{self.variable_name}"

    def execute(self, df: pd.DataFrame, context: ValidationContext) -> RuleExecutionResult:
        """Verify that a variable is present.

        Parameters
        ----------
        df
            Dataframe to execute the validation rule on.
        context
            Validation context with detected variables.

        Returns
        -------
        RuleExecutionResult
            PASSED if signal detected, FAILED if not
        """
        column_name = context.get_column_key(self.variable_name)

        if column_name is None:
            return self.result_builder.failed(
                required=f"Variable '{self.variable_name}' detected",
                actual="Not detected",
                message=f"Required signal '{self.variable_name}' not found in data",
            )

        return self.result_builder.passed(
            required=f"Variable '{self.variable_name}' detected",
            actual=f"Detected as '{column_name}'",
            message=f"Variable '{self.variable_name}' found",
            details={"column_name": column_name},
        )


__all__ = ["RequiredVariableRule"]
