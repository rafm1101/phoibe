import abc

import pandas as pd

from ..application.context import ValidationContext
from ..core.entities import RuleExecutionResult


class ValidationRule(abc.ABC):
    """Interface for each check.

    Properties
    ----------
    name
        Name of the check.

    Methods
    -------
    execute
        Execute check on a given table.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def execute(self, df: pd.DataFrame, context: ValidationContext) -> RuleExecutionResult:
        # def execute(self, df: pd.DataFrame, context: dict[str, typing.Any]) -> RuleExecutionResult:
        """Execute a check.

        Parameters
        ----------
        df
            Dataframe holding the actual data.
        context
            Context/configuration of the check.

        Returns
        -------
        CheckResult
            Result of the check.
        """
        pass
