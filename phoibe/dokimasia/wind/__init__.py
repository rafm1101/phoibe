"""The wind turbine domain pack: catalog, column patterns and curtailment measures (ADR-01, ADR-07).

Imports `core` and `rules`.
"""

from phoibe.dokimasia.wind.catalog import create_catalog
from phoibe.dokimasia.wind.curtailment import LevelMethod, NightLimitMeasure, PowerLevelsMeasure
from phoibe.dokimasia.wind.detection import detect_wind_signals
from phoibe.dokimasia.wind.patterns import PATTERNS

__all__ = [
    "PATTERNS",
    "LevelMethod",
    "NightLimitMeasure",
    "PowerLevelsMeasure",
    "create_catalog",
    "detect_wind_signals",
]
