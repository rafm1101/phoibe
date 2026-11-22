"""The wind turbine domain pack: catalog and column patterns (ADR-01, ADR-07). Imports `core` and `rules`."""

from phoibe.dokimasia.wind.catalog import create_catalog
from phoibe.dokimasia.wind.detection import detect_wind_signals
from phoibe.dokimasia.wind.patterns import PATTERNS

__all__ = ["PATTERNS", "create_catalog", "detect_wind_signals"]
