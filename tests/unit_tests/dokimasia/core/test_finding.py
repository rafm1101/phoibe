import numpy as np
import pandas as pd

from phoibe.dokimasia import core


def _finding(kind: str, start: str, end: str, signals: tuple[str, ...] = ()) -> core.Finding:
    return core.Finding(kind=kind, signals=signals, start=pd.Timestamp(start), end=pd.Timestamp(end), n_steps=1)


INDEX = pd.date_range("2026-01-01 00:00", periods=6, freq="10min")


def test_build_finding_mask_marks_the_rows_of_all_findings():
    findings = [
        _finding("gap", "2026-01-01 00:10", "2026-01-01 00:20"),
        _finding("gap", "2026-01-01 00:50", "2026-01-01 00:50"),
    ]

    mask = core.build_finding_mask(findings, INDEX)

    assert mask.tolist() == [False, True, True, False, False, True]


def test_build_finding_mask_given_kind_selects_that_kind():
    findings = [
        _finding("gap", "2026-01-01 00:10", "2026-01-01 00:10"),
        _finding("missing", "2026-01-01 00:30", "2026-01-01 00:30"),
    ]

    assert np.flatnonzero(core.build_finding_mask(findings, INDEX, kind="missing")).tolist() == [3]


def test_build_finding_mask_given_signal_selects_findings_of_that_signal():
    findings = [
        _finding("missing", "2026-01-01 00:10", "2026-01-01 00:10", ("a",)),
        _finding("missing", "2026-01-01 00:30", "2026-01-01 00:30", ("b",)),
    ]

    assert np.flatnonzero(core.build_finding_mask(findings, INDEX, signal="b")).tolist() == [3]


def test_build_finding_mask_given_no_findings_is_all_false():
    assert not core.build_finding_mask([], INDEX).any()
