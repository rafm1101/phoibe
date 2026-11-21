"""Generic measures that do not assume a domain (ADR-07). Imports `core` only."""

from phoibe.dokimasia.rules.completeness import MissingRunsMeasure
from phoibe.dokimasia.rules.temporal import GapMeasure

__all__ = ["GapMeasure", "MissingRunsMeasure"]
