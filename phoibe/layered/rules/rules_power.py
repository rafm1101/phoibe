import logging

import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.application.registry import RuleRegistry
from phoibe.layered.core.entities import RuleExecutionResult
from phoibe.layered.core.entities import Severity
from phoibe.layered.rules.rule import ValidationRule


@RuleRegistry.register("curtailments_power")
class CurtailmentRule(ValidationRule):
    """Determine curtailments.

    The heuristic to determine curtailments based on determining clusters of observed powers at full load.
    Values are rounded to multiples of 10.

    Parameters
    ----------
    wind_speed_threshold
        Threshold of wind speeds [m/s] above which the wtg is expected to run on full load.
    prominence_threshold
        Threshold above which clusters of powers are accepted as actual curtailments.
    n_samples
        Number of points of the regular grid where the KDE es evaluated.

    Notes
    -----
    1. Algorithm:
       1. Determine a Gaussian KDE on the set of observed powers at high wind speed.
       2. Determine its values on a regular grid and peak candidates of these values.
       3. Filter for peaks whose prominence is above the given threshold. These are the curtailments.
    2. Output:
       1. Number of candidates identified as peaks.
       2. Number of curtailments.
       3. Power values of the curtailments.
       4. Values of the respective densities.
       5. Value of the density of the next following peak (the first on not considered as curtailment).
    3. Awards 0 is keys are not found or data is insufficient.
    4. Observations:
       - For a single power level, sufficiently many values (more than 100) pass
         while few raise a `numpy.linalg.LinAlgError`.
    """

    def __init__(
        self,
        wind_speed_threshold: float = 14.0,
        prominence_threshold: float = 1e-7,
        points: int = 10,
        severity: Severity = Severity.INFO,
        logger: logging.Logger | None = None,
    ):
        super().__init__(points, severity, logger)
        self.wind_speed_threshold = wind_speed_threshold
        self.prominence_threshold = prominence_threshold

    @property
    def name(self):
        return "curtailments_power"

    def execute(self, df: pd.DataFrame, context: ValidationContext) -> RuleExecutionResult:
        """Verify the range properties of variables.

        Parameters
        ----------
        df
            Dataframe w/ columns for `power_kw` and `wind_speed`.
        context
            Validation context.

        Returns
        -------
        RuleExecutionResult
            PASSED|NOT_CHECKED.
        """
        power_key = context.get_column_key("power_kw")
        windspeed_key = context.get_column_key("wind_speed")
        if power_key is None:
            return self.result_builder.not_checked("Power variable not detected.")
        if windspeed_key is None:
            return self.result_builder.not_checked("Wind speed variable not detected.")

        mask_curtailments = df.loc[:, windspeed_key] > self.wind_speed_threshold
        if mask_curtailments.sum() <= 1:
            return self.result_builder.not_checked(
                f"Missing sufficient data at threshold {self.wind_speed_threshold}. Please lower threshold."
            )

        extent_min = np.floor(df.loc[mask_curtailments, power_key].min() / 10) * 10
        extent_max = np.ceil(df.loc[mask_curtailments, power_key].max() / 10) * 10

        power_kde = scipy.stats.gaussian_kde(df.loc[mask_curtailments, power_key])
        sample_locations = np.arange(extent_min - 50, extent_max + 60, step=10)
        sample_values = power_kde(sample_locations)

        peak_indices, peak_properties = scipy.signal.find_peaks(sample_values, prominence=(None, None))
        n_candidates = len(peak_indices)

        peaks_prominent = peak_properties["prominences"] > self.prominence_threshold
        peaks_remaining = ~peaks_prominent

        peak_powers = sample_locations[peak_indices[peaks_prominent]]
        peak_densities = sample_values[peak_indices[peaks_prominent]]
        first_non_peak_density = (
            sample_values[peak_indices[peaks_remaining]].max() if peaks_remaining.sum() > 0 else 0.0
        )

        details = {
            "n_curtailments": int(np.sum(peaks_prominent)),
            "n_candidates_detected": n_candidates,
            "power": [round(float(power), 1) for power in peak_powers],
            "height": [round(float(density), 6) for density in peak_densities],
            "ignored_below": round(float(first_non_peak_density), 6),
        }
        message = f"Found {details['n_curtailments']} full load levels."
        return self.result_builder.passed(required="", actual="", message=message, details=details)
