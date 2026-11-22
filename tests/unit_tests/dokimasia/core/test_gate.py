import pytest

from phoibe.dokimasia import core

V, S, E = core.Verdict, core.Severity, core.Effect

EFFECT_TABLE = {
    (V.GOOD, S.CRITICAL): E.PASS,
    (V.GOOD, S.WARNING): E.PASS,
    (V.GOOD, S.INFO): E.PASS,
    (V.ACCEPTABLE, S.CRITICAL): E.CONDITIONAL,
    (V.ACCEPTABLE, S.WARNING): E.PASS_NOTED,
    (V.ACCEPTABLE, S.INFO): E.PASS,
    (V.BAD, S.CRITICAL): E.FAIL,
    (V.BAD, S.WARNING): E.CONDITIONAL,
    (V.BAD, S.INFO): E.PASS_NOTED,
    (V.NOT_EVALUABLE, S.CRITICAL): E.FAIL,
    (V.NOT_EVALUABLE, S.WARNING): E.CONDITIONAL,
    (V.NOT_EVALUABLE, S.INFO): E.PASS_NOTED,
    (V.ERROR, S.CRITICAL): E.FAIL,
    (V.ERROR, S.WARNING): E.CONDITIONAL,
    (V.ERROR, S.INFO): E.PASS_NOTED,
}


def _verdict(check_id: str, verdict: core.Verdict, severity: core.Severity) -> core.CriterionVerdict:
    return core.CriterionVerdict(check_id=check_id, measure_id="m", severity=severity, verdict=verdict)


@pytest.mark.parametrize(("verdict", "severity"), list(EFFECT_TABLE))
def test_get_effect(verdict, severity):
    assert core.get_effect(verdict, severity) == EFFECT_TABLE[(verdict, severity)]


def test_get_effect_covers_every_combination():
    assert set(EFFECT_TABLE) == {(v, s) for v in core.Verdict for s in core.Severity}


@pytest.mark.parametrize("verdict", [V.NOT_EVALUABLE, V.ERROR])
def test_get_effect_given_a_verdict_without_value_is_never_pass(verdict):
    assert all(core.get_effect(verdict, severity) != E.PASS for severity in core.Severity)


class TestWorstEffectPolicy:
    policy = core.WorstEffectPolicy()

    def test_all_good_passes_without_reasons(self):
        result = self.policy.apply([_verdict("a", V.GOOD, S.CRITICAL), _verdict("b", V.GOOD, S.INFO)])

        assert result.outcome == core.GateOutcome.PASS
        assert result.reasons == ()
        assert result.passed
        assert result.policy == "worst_effect"

    def test_takes_the_worst_effect(self):
        result = self.policy.apply(
            [
                _verdict("a", V.GOOD, S.CRITICAL),
                _verdict("b", V.ACCEPTABLE, S.CRITICAL),
                _verdict("c", V.BAD, S.CRITICAL),
            ]
        )

        assert result.outcome == core.GateOutcome.FAIL
        assert not result.passed
        assert [reason.check_id for reason in result.failures] == ["c"]
        assert [reason.check_id for reason in result.conditions] == ["b"]

    def test_an_acceptable_critical_criterion_gives_a_conditional_pass_with_the_condition_recorded(self):
        result = self.policy.apply([_verdict("a", V.ACCEPTABLE, S.CRITICAL), _verdict("b", V.BAD, S.INFO)])

        assert result.outcome == core.GateOutcome.CONDITIONAL
        assert [reason.check_id for reason in result.conditions] == ["a"]
        assert [reason.check_id for reason in result.notes] == ["b"]
        assert result.passed

    def test_noted_criteria_do_not_change_the_outcome(self):
        result = self.policy.apply([_verdict("a", V.ACCEPTABLE, S.WARNING), _verdict("b", V.BAD, S.INFO)])

        assert result.outcome == core.GateOutcome.PASS
        assert len(result.notes) == 2

    @pytest.mark.parametrize("verdict", [V.NOT_EVALUABLE, V.ERROR])
    def test_a_critical_criterion_that_could_not_be_evaluated_fails_the_gate(self, verdict):
        assert self.policy.apply([_verdict("a", verdict, S.CRITICAL)]).outcome == core.GateOutcome.FAIL

    def test_a_warning_that_could_not_be_evaluated_makes_the_pass_conditional(self):
        assert self.policy.apply([_verdict("a", V.NOT_EVALUABLE, S.WARNING)]).outcome == core.GateOutcome.CONDITIONAL

    def test_a_layer_without_criteria_is_an_error_and_not_a_pass(self):
        with pytest.raises(ValueError, match="without criteria"):
            self.policy.apply([])


class TestGateResult:
    result = core.WorstEffectPolicy().apply(
        [
            _verdict("a", V.GOOD, S.CRITICAL),
            _verdict("b", V.ACCEPTABLE, S.CRITICAL),
            _verdict("c", V.BAD, S.WARNING),
            _verdict("d", V.ERROR, S.INFO),
        ]
    )

    def test_counts_by_verdict_and_severity_include_zeros(self):
        assert self.result.count_by_verdict() == {"good": 1, "acceptable": 1, "bad": 1, "not_evaluable": 0, "error": 1}
        assert self.result.count_by_severity() == {"critical": 2, "warning": 1, "info": 1}

    def test_score_is_the_share_of_good_and_acceptable_criteria(self):
        assert self.result.compute_score() == 0.5

    def test_raise_if_failed_does_nothing_for_a_conditional_pass(self):
        self.result.raise_if_failed()

    def test_raise_if_failed_raises_with_the_failures_for_a_failed_gate(self):
        failed = core.WorstEffectPolicy().apply([_verdict("a", V.BAD, S.CRITICAL), _verdict("b", V.GOOD, S.INFO)])

        with pytest.raises(core.GateFailedError, match="a: bad at severity critical") as raised:
            failed.raise_if_failed()

        assert [reason.check_id for reason in raised.value.failures] == ["a"]

    def test_to_frame_has_a_row_per_verdict_in_order(self):
        frame = self.result.to_frame()

        assert frame["check_id"].tolist() == ["a", "b", "c", "d"]
        assert frame["verdict"].tolist() == ["good", "acceptable", "bad", "error"]
        assert frame["severity"].tolist() == ["critical", "critical", "warning", "info"]

    def test_repr_is_one_line_with_outcome_and_counts(self):
        text = repr(self.result)

        assert "\n" not in text
        assert text.startswith("GateResult(conditional by worst_effect: 0 failures, 2 conditions, 1 notes")
        assert "1 good, 1 acceptable, 1 bad, 1 error" in text


def test_build_verdict_table_given_no_verdicts_is_empty_with_its_columns():
    table = core.build_verdict_table([])

    assert table.empty
    assert list(table.columns)[:2] == ["check_id", "measure_id"]


def test_criterion_verdict_repr_names_check_verdict_severity_and_value():
    verdict = core.CriterionVerdict("gaps", "m", core.Severity.CRITICAL, core.Verdict.BAD, value=0.25)
    unread = core.CriterionVerdict("gaps", "m", core.Severity.INFO, core.Verdict.ERROR)

    assert repr(verdict) == "CriterionVerdict(gaps: bad at critical, value=0.25)"
    assert repr(unread) == "CriterionVerdict(gaps: error at info, no value)"
