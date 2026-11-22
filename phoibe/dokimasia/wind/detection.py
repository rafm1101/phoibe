"""Detect the signals of a wind turbine in column names."""

from collections.abc import Sequence

from phoibe.dokimasia import rules
from phoibe.dokimasia.wind import patterns


def detect_wind_signals(columns: Sequence[str]) -> rules.Detection:
    """Detect the wind turbine signals among column names.

    Parameters
    ----------
    columns
        The column names of the data.

    Returns
    -------
    Detection
        The result. `detection.mapping` is what `RenameToCatalog` takes; nothing ambiguous is in it, and
        `detection.ambiguities` says what a person has to decide.
    """
    return rules.detect_signals(columns, patterns.PATTERNS)
