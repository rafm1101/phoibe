import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.application.registry import RuleRegistry
from phoibe.layered.core.entities import RuleExecutionResult
from phoibe.layered.core.entities import Severity
from phoibe.layered.rules.rule import ValidationRule

# TODO: Document properly.


@RuleRegistry.register("ranges")
class RangeRule(ValidationRule):
    """Validate ranges of variables.

    `RangeRule` validates for each requested variable the provided range, and counts the outliers.

    Parameters
    ----------
    variable_ranges
        Dictionary providing for each variable to be checked a tuple of the respective range bounds.
    """

    def __init__(
        self,
        variable_ranges: dict[str, tuple[float, float]],
        points: int = 10,
        severity: Severity = Severity.INFO,
        logger: logging.Logger | None = None,
    ):
        super().__init__(points, severity, logger)
        self.variable_ranges = variable_ranges

    @property
    def name(self):
        return "ranges"

    def execute(self, df: pd.DataFrame, context: ValidationContext) -> RuleExecutionResult:
        """Verify the range properties of variables.

        Parameters
        ----------
        df
            Dataframe with a pandas datetime index.
        context
            Validation context.

        Returns
        -------
        RuleExecutionResult
            PASSED|WARNING.
        """
        variables_skipped: list[str] = []
        variables_checked = {}
        for variable_name, viable_range in self.variable_ranges.items():
            key = context.get_column_key(variable_name)
            if key is None:
                variables_skipped.append(variable_name)
                continue

            out_of_range = (df.loc[:, key] < viable_range[0]) | (df.loc[:, key] > viable_range[1])
            variables_checked[variable_name] = int(out_of_range.sum())

        details = {"n_out_of_range": variables_checked, "skipped": variables_skipped}
        if len(variables_skipped) == 0:
            message = f"Checked ranges of {len(variables_checked)} variables."
            return self.result_builder.passed(required="", actual="", message=message, details=details)
        else:
            points = int(self.points * len(variables_checked) / len(self.variable_ranges))
            message = f"Checked ranges of {len(variables_checked)}/{len(self.variable_ranges)} variables."
            return self.result_builder.warning(required="", actual="", points=points, message="", details=details)


@RuleRegistry.register("essential_ranges")
class EssentialRange(ValidationRule):
    """Determine essential ranges of variables.

    The heuristic to determine essential ranges or essential minimum and maximum, respectively, consists in
    choosing the shortest inverval containing at least a percentage of `proportion` data points.

    Parameters
    ----------
    variable_names
        A list of all variable names to be checked.
    proportion
        The amount of points that make up the essential part.
    """

    def __init__(
        self,
        variable_names: list[str],
        proportion: float = 0.995,
        points: int = 10,
        severity: Severity = Severity.INFO,
        logger: logging.Logger | None = None,
    ):
        super().__init__(points, severity, logger)
        self.variable_names = variable_names
        self.proportion = proportion

    @property
    def name(self):
        return "essential_range"

    def execute(self, df: pd.DataFrame, context: ValidationContext) -> RuleExecutionResult:
        """Determine the range properties of variables.

        Parameters
        ----------
        df
            Dataframe with a pandas datetime index.
        context
            Validation context.

        Returns
        -------
        RuleExecutionResult
            PASSED|NOT_CHECKED.
        """
        variables_skipped: list[str] = []
        variables_checked = {}
        for variable_name in self.variable_names:
            key = context.get_column_key(variable_name)
            if key is None:
                variables_skipped.append(variable_name)
                continue

            x = np.asarray(df.loc[:, key], dtype=float)
            variables_checked[variable_name] = self._compute_high_density_interval(x=x, proportion=self.proportion)

        details = {"checked": variables_checked, "skipped": variables_skipped}
        if len(variables_skipped) == 0:
            return self.result_builder.passed(
                required="", actual="", message="All essential ranges determined.", details=details
            )
        else:
            points = int(self.points * len(variables_checked) / len(self.variable_names))
            message = f"Checked essential ranges of {len(variables_checked)}/{len(self.variable_names)} variables."
            return self.result_builder.warning(required="", actual="", points=points, message=message, details=details)

    def _compute_high_density_interval(self, x: NDArray[np.floating], proportion: float = 0.995) -> list[float]:
        """Determine the shortest interval that contains the given amount of data."""
        x = x[np.isfinite(x)]
        x_sorted = np.sort(x)

        if len(x_sorted) == 0:
            return [np.nan, np.nan]

        total_mass = len(x_sorted)
        k = int(min(np.ceil(proportion * total_mass), total_mass - 1))

        widths = x_sorted[k:] - x_sorted[: total_mass - k]
        i = np.argmin(widths)

        return [float(x_sorted[i]), float(x_sorted[i + k])]
