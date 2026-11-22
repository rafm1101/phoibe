"""Generic measures that do not assume a domain (ADR-07). Imports `core` only."""

from phoibe.dokimasia.rules._runs import find_runs
from phoibe.dokimasia.rules.completeness import MissingRunsMeasure
from phoibe.dokimasia.rules.consistency import FrozenRunsMeasure, StatisticOrderMeasure, list_statistic_triples
from phoibe.dokimasia.rules.detection import Ambiguity, Detection, DetectionError, SignalPattern, detect_signals
from phoibe.dokimasia.rules.signals import read_numeric_signals
from phoibe.dokimasia.rules.structure import RequiredSignalsMeasure
from phoibe.dokimasia.rules.temporal import GapMeasure
from phoibe.dokimasia.rules.validity import RangesMeasure

__all__ = [
    "Ambiguity",
    "Detection",
    "DetectionError",
    "FrozenRunsMeasure",
    "GapMeasure",
    "MissingRunsMeasure",
    "RangesMeasure",
    "RequiredSignalsMeasure",
    "SignalPattern",
    "StatisticOrderMeasure",
    "detect_signals",
    "find_runs",
    "list_statistic_triples",
    "read_numeric_signals",
]
