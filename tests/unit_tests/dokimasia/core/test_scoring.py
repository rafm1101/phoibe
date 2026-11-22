import math

import numpy as np
import pandas as pd
import pytest

from phoibe.dokimasia import core

INDEX = pd.date_range("2026-01-01", periods=10, freq="10min")


def _finding(start: int, end: int, *, kind: str = "gap", signals: tuple[str, ...] = ()) -> core.Finding:
    return core.Finding(kind, signals, INDEX[start], INDEX[end], end - start + 1)


class TestScore:
    def test_recall_is_the_share_of_faulty_rows_found(self):
        score = core.Score(n_truth=10, n_flagged=4, n_hit=3)

        assert score.recall == pytest.approx(0.3)

    def test_precision_is_the_share_of_flagged_rows_that_are_faulty(self):
        score = core.Score(n_truth=10, n_flagged=4, n_hit=3)

        assert score.precision == pytest.approx(0.75)
        assert score.n_false_alarms == 1

    def test_recall_without_faulty_rows_is_nan(self):
        assert math.isnan(core.Score(0, 3, 0).recall)

    def test_precision_without_flagged_rows_is_nan(self):
        assert math.isnan(core.Score(3, 0, 0).precision)

    def test_scores_pool_by_adding_counts(self):
        pooled = core.Score(10, 4, 3) + core.Score(2, 2, 2)

        assert pooled == core.Score(12, 6, 5)


def test_compute_score_counts_the_rows_of_both_masks():
    flagged = [True, True, False, False, True]
    truth = [True, False, True, False, True]

    assert core.compute_score(flagged, truth) == core.Score(n_truth=3, n_flagged=3, n_hit=2)


def test_compute_score_given_masks_of_different_length_raises():
    with pytest.raises(ValueError, match="differ in shape"):
        core.compute_score([True], [True, False])


def test_compute_score_given_no_rows_counts_nothing():
    assert core.compute_score(np.array([], dtype=bool), np.array([], dtype=bool)) == core.Score(0, 0, 0)


class TestScoreFindings:
    truth = np.array([False, True, True, False, False, False, True, True, True, False])

    def test_a_finding_that_covers_a_fault_exactly_is_perfect(self):
        score = core.score_findings([_finding(1, 2), _finding(6, 8)], self.truth, INDEX)

        assert (score.recall, score.precision) == (1.0, 1.0)

    def test_a_finding_that_is_too_wide_costs_precision(self):
        score = core.score_findings([_finding(0, 3), _finding(6, 8)], self.truth, INDEX)

        assert score.recall == 1.0
        assert score.precision == pytest.approx(5 / 7)

    def test_a_missed_fault_costs_recall(self):
        score = core.score_findings([_finding(1, 2)], self.truth, INDEX)

        assert score.recall == pytest.approx(2 / 5)
        assert score.precision == 1.0

    def test_no_findings_on_clean_data_flag_nothing(self):
        score = core.score_findings([], np.zeros(10, dtype=bool), INDEX)

        assert score == core.Score(0, 0, 0)

    def test_findings_of_another_kind_or_signal_are_left_out_when_asked(self):
        findings = [_finding(1, 2, kind="gap"), _finding(6, 8, kind="missing", signals=("a",))]

        by_kind = core.score_findings(findings, self.truth, INDEX, kind="gap")
        by_signal = core.score_findings(findings, self.truth, INDEX, signal="a")

        assert by_kind.n_flagged == 2
        assert by_signal.n_flagged == 3
