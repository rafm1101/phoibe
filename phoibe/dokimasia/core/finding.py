"""Located findings (ADR-03)."""

import dataclasses
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclasses.dataclass(frozen=True, slots=True)
class Finding:
    """One run-length interval where a measurement found something. Never one finding per row.

    "Gap 10:20 to 11:10, 6 steps" is one finding.
    """

    kind: str
    """What was found, for example `gap` or `missing`; defined by the measure that reports it."""
    signals: tuple[str, ...]
    """Catalog names of the signals concerned. Empty for findings about the time axis."""
    start: pd.Timestamp
    """First affected timestamp, inclusive."""
    end: pd.Timestamp
    """Last affected timestamp, inclusive."""
    n_steps: int
    """Number of time steps the finding covers."""
    magnitude: float | None = None
    """Size of the finding where steps do not tell it, in `magnitude_unit`."""
    magnitude_unit: str | None = None
    """Unit of `magnitude`."""


def build_finding_mask(
    findings: Iterable[Finding],
    index: pd.DatetimeIndex,
    *,
    kind: str | None = None,
    signal: str | None = None,
) -> npt.NDArray[np.bool_]:
    """Build the row mask of findings, for plotting and fixing.

    Parameters
    ----------
    findings
        Findings to draw from.
    index
        Time axis of the rows the mask is for.
    kind
        If given, only findings of this kind count.
    signal
        If given, only findings that concern this signal count.

    Returns
    -------
    numpy.ndarray
        Boolean array with one entry per index entry; `True` where a selected finding covers the timestamp.

    """
    mask = np.zeros(len(index), dtype=bool)
    for finding in findings:
        if kind is not None and finding.kind != kind:
            continue
        if signal is not None and signal not in finding.signals:
            continue
        mask |= (index >= finding.start) & (index <= finding.end)
    return mask
