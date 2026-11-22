"""Recall and precision of findings against known faulty rows (ADR-09).

Both are counted over rows, because a finding covers an interval and a fault is a set of rows. A row counts as
found when some finding covers it; a row counts as faulty when the ground truth says so.

- Recall is the share of faulty rows that findings cover: what the rule finds of what is there.
- Precision is the share of covered rows that are faulty: how much of what the rule reports is real.

Faults are rare in real data, so precision decides whether a rule can be used. A rule must also stay quiet on clean
data; that is checked with `Score.n_flagged` of a run whose truth is empty.
"""

import dataclasses
import math
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import pandas as pd

from phoibe.dokimasia.core import finding as finding_module


@dataclasses.dataclass(frozen=True, slots=True)
class Score:
    """Counts of rows, and the recall and precision they give."""

    n_truth: int
    """Rows that are faulty."""
    n_flagged: int
    """Rows covered by findings."""
    n_hit: int
    """Rows that are both."""

    @property
    def recall(self) -> float:
        """Share of faulty rows that are flagged; `nan` if there are no faulty rows."""
        return self.n_hit / self.n_truth if self.n_truth else math.nan

    @property
    def precision(self) -> float:
        """Share of flagged rows that are faulty; `nan` if nothing is flagged."""
        return self.n_hit / self.n_flagged if self.n_flagged else math.nan

    @property
    def n_false_alarms(self) -> int:
        """Rows that are flagged but not faulty."""
        return self.n_flagged - self.n_hit

    def __add__(self, other: "Score") -> "Score":
        """Pool two scores by adding their counts, for example over seeds."""
        return Score(self.n_truth + other.n_truth, self.n_flagged + other.n_flagged, self.n_hit + other.n_hit)


def compute_score(flagged: npt.ArrayLike, truth: npt.ArrayLike) -> Score:
    """Count the rows of two boolean masks.

    Parameters
    ----------
    flagged
        True where a finding covers the row.
    truth
        True where the row is faulty.

    Returns
    -------
    Score
        The counts.

    Raises
    ------
    ValueError
        If the masks differ in length.
    """
    flagged_array = np.asarray(flagged, dtype=bool)
    truth_array = np.asarray(truth, dtype=bool)
    if flagged_array.shape != truth_array.shape:
        raise ValueError(f"the masks differ in shape: {flagged_array.shape} and {truth_array.shape}")
    return Score(
        n_truth=int(truth_array.sum()),
        n_flagged=int(flagged_array.sum()),
        n_hit=int((flagged_array & truth_array).sum()),
    )


def score_findings(
    findings: Iterable[finding_module.Finding],
    truth: npt.ArrayLike,
    index: pd.DatetimeIndex,
    *,
    kind: str | None = None,
    signal: str | None = None,
) -> Score:
    """Score findings against the faulty rows of a time grid.

    Parameters
    ----------
    findings
        The findings of a measure.
    truth
        True where the row of `index` is faulty.
    index
        The time grid the truth is defined on. For rows that were dropped, this is the grid before the drop.
    kind
        If given, only findings of this kind count.
    signal
        If given, only findings that concern this signal count.

    Returns
    -------
    Score
        The counts, with recall and precision.
    """
    flagged = finding_module.build_finding_mask(findings, index, kind=kind, signal=signal)
    return compute_score(flagged, truth)
