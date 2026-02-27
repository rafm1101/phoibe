import logging
import typing

# import numpy as np
import pandas as pd

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.application.factory import RuleRegistry

# from phoibe.layered.core.entities import RuleExecutionResult
from phoibe.layered.core.entities import Severity
from phoibe.layered.rules.rule import ValidationRule

# TODO: Document properly.


@RuleRegistry.register("ranges")
class RangeRules(ValidationRule):
    """Validate ranges of variables."""

    def __init__(
        self,
        variable_ranges: dict[str, typing.Any],
        points: int = 10,
        severity: Severity = Severity.INFO,
        logger: logging.Logger | None = None,
    ):
        super().__init__(points, severity, logger)
        self.variable_ranges = variable_ranges

    @property
    def name(self):
        return "ranges"

    def execute(self, df: pd.DataFrame, context: ValidationContext):
        """Verify the temporal properties.

        Parameters
        ----------
        df
            Dataframe with a pandas datetime index.
        context
            Validation context.

        Returns
        -------
        RuleExecutionResult
            PASSED.
        """
        variables_skipped: list[str] = []
        variables_checked = {}
        for variable_name, viable_range in self.variable_ranges.items():
            key = context.get_column_key(variable_name)
            if key is None:
                # return self.result_builder.not_checked("Datetime variable not detected.")
                variables_skipped.append(variable_name)
                continue

            out_of_range = (df.loc[:, key] < viable_range[0]) | (df.loc[:, key] > viable_range[1])
            #
            # essential_min = df.loc[df.loc[:, key], key]
            variables_checked[key] = int(out_of_range.sum())

        details = {"checked": variables_checked, "skipped": variables_skipped}
        if len(variables_skipped) == 0:
            return self.result_builder.passed(required="", actual="", message="", details=details)
        else:
            return self.result_builder.warning(
                required="", actual="", points=int(self.points / 2), message="", details=details
            )
