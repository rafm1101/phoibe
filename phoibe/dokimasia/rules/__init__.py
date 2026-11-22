"""Generic measures that do not assume a domain (ADR-07). Imports `core` only."""

from phoibe.dokimasia.rules.completeness import MissingRunsMeasure
from phoibe.dokimasia.rules.detection import Ambiguity, Detection, DetectionError, SignalPattern, detect_signals
from phoibe.dokimasia.rules.structure import RequiredSignalsMeasure
from phoibe.dokimasia.rules.temporal import GapMeasure
from phoibe.dokimasia.rules.validity import RangesMeasure

__all__ = [
    "Ambiguity",
    "Detection",
    "DetectionError",
    "GapMeasure",
    "MissingRunsMeasure",
    "RangesMeasure",
    "RequiredSignalsMeasure",
    "SignalPattern",
    "detect_signals",
]
