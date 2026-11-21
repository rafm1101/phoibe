import dataclasses

import pandas as pd
import pytest

from phoibe.dokimasia import contract, core

SUBJECT = core.Subject("t1", "wind_turbine", {"rated_power_kw": 5600})
V, S = core.Verdict, core.Severity


@dataclasses.dataclass
class _Fixed:
    """A measure that returns a prepared outcome and counts its runs."""

    id: str
    outcome: object
    aspect: str = "test"
    runs: int = 0

    def measure(self, dataset):
        self.runs += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _profile(*measures: _Fixed) -> core.Profile:
    index = pd.date_range("2026-01-01", periods=2, freq="10min")
    dataset = core.Dataset.from_frame(pd.DataFrame({"a": [1.0, 2.0]}, index=index))
    return core.profile_dataset(dataset, SUBJECT, "bronze", measures)


def _measured(measure_id: str, **values: float) -> _Fixed:
    return _Fixed(measure_id, core.Measured(measure_id, metrics=tuple(core.Metric(k, v) for k, v in values.items())))


def _layer(good, acceptable=None, severity="critical", value="n", **extra) -> contract.LayerSpec:
    check = {"measure": "m", "value": value, "severity": severity, "good": good} | extra
    if acceptable is not None:
        check["acceptable"] = acceptable
    data = {
        "id": "c",
        "version": "1",
        "domain": "wind",
        "layers": {"bronze": {"measures": {"m": {}}, "checks": {"k": check}}},
    }
    return contract.build_contract(data).get_layer("bronze")


def _only(result: core.GateResult) -> core.CriterionVerdict:
    (verdict,) = result.verdicts
    return verdict


class TestJudgeLayerBands:
    @pytest.mark.parametrize(
        ("value", "verdict", "outcome"),
        [
            (0, V.GOOD, core.GateOutcome.PASS),
            (10, V.ACCEPTABLE, core.GateOutcome.CONDITIONAL),
            (11, V.BAD, core.GateOutcome.FAIL),
        ],
    )
    def test_two_bands_give_good_acceptable_or_bad(self, value, verdict, outcome):
        result = contract.judge_layer(_profile(_measured("m", n=value)), _layer({"max": 0}, {"max": 10}))

        assert _only(result).verdict == verdict
        assert result.outcome == outcome

    def test_bounds_are_included(self):
        layer = _layer({"min": 1, "max": 2})

        assert _only(contract.judge_layer(_profile(_measured("m", n=1)), layer)).verdict == V.GOOD
        assert _only(contract.judge_layer(_profile(_measured("m", n=2)), layer)).verdict == V.GOOD
        assert _only(contract.judge_layer(_profile(_measured("m", n=2.5)), layer)).verdict == V.BAD

    def test_one_band_gives_good_or_bad_only(self):
        layer = _layer({"max": 0})

        assert _only(contract.judge_layer(_profile(_measured("m", n=1)), layer)).verdict == V.BAD

    def test_the_verdict_carries_value_reason_clause_and_severity(self):
        layer = _layer({"max": 0}, {"max": 10}, severity="warning", clause="TR6")

        verdict = _only(contract.judge_layer(_profile(_measured("m", n=3)), layer))

        assert (verdict.value, verdict.clause, verdict.severity) == (3.0, "TR6", S.WARNING)
        assert verdict.reason == "3 outside good (max 0), within acceptable (max 10)"

    def test_a_limit_that_depends_on_the_subject_is_resolved_and_recorded(self):
        layer = _layer(
            {"min": {"subject": "rated_power_kw", "factor": 0.99}, "max": {"subject": "rated_power_kw", "factor": 1.01}}
        )

        verdict = _only(contract.judge_layer(_profile(_measured("m", n=5600)), layer))

        assert verdict.verdict == V.GOOD
        assert "5544 = 0.99 × rated_power_kw 5600" in verdict.reason

    def test_a_limit_that_cannot_be_resolved_gives_an_error_verdict_not_an_exception(self):
        layer = _layer({"max": {"subject": "hub_height"}})

        verdict = _only(contract.judge_layer(_profile(_measured("m", n=1)), layer))

        assert verdict.verdict == V.ERROR
        assert "hub_height" in verdict.reason


class TestJudgeLayerWithoutValue:
    def test_a_measure_that_is_not_in_the_profile_is_an_error(self):
        verdict = _only(contract.judge_layer(_profile(_measured("other", n=0)), _layer({"max": 0})))

        assert verdict.verdict == V.ERROR
        assert "not in the profile" in verdict.reason

    def test_a_measure_that_was_not_evaluable_gives_not_evaluable(self):
        measure = _Fixed("m", core.NotEvaluable("m", core.NotEvaluableReason.MISSING_SIGNAL, "no wind_speed"))

        result = contract.judge_layer(_profile(measure), _layer({"max": 0}))

        assert _only(result).verdict == V.NOT_EVALUABLE
        assert result.outcome == core.GateOutcome.FAIL

    def test_a_measure_that_failed_gives_an_error_that_follows_the_severity_of_the_check(self):
        measure = _Fixed("m", RuntimeError("boom"))

        critical = contract.judge_layer(_profile(measure), _layer({"max": 0}, severity="critical"))
        warning = contract.judge_layer(_profile(measure), _layer({"max": 0}, severity="warning"))

        assert _only(critical).verdict == V.ERROR
        assert critical.outcome == core.GateOutcome.FAIL
        assert warning.outcome == core.GateOutcome.CONDITIONAL

    def test_a_value_the_measure_does_not_report_is_an_error_that_names_what_it_reports(self):
        verdict = _only(contract.judge_layer(_profile(_measured("m", n=0)), _layer({"max": 0}, value="missing")))

        assert verdict.verdict == V.ERROR
        assert "has no value 'missing'" in verdict.reason
        assert "n" in verdict.reason

    def test_a_value_that_is_not_a_number_is_not_evaluable(self):
        verdict = _only(contract.judge_layer(_profile(_measured("m", n=float("nan"))), _layer({"max": 0})))

        assert verdict.verdict == V.NOT_EVALUABLE

    def test_a_value_reported_for_several_signals_needs_a_signal(self):
        measure = _Fixed(
            "m", core.Measured("m", metrics=(core.Metric("n", 0.0, signal="a"), core.Metric("n", 5.0, signal="b")))
        )

        ambiguous = _only(contract.judge_layer(_profile(measure), _layer({"max": 0})))
        chosen = _only(contract.judge_layer(_profile(measure), _layer({"max": 0}, signal="b")))

        assert ambiguous.verdict == V.ERROR
        assert "several signals" in ambiguous.reason
        assert (chosen.verdict, chosen.value, chosen.signal) == (V.BAD, 5.0, "b")


class TestJudgeLayerGate:
    def test_the_gate_takes_the_worst_effect_over_all_checks(self):
        data = {
            "id": "c",
            "version": "1",
            "domain": "wind",
            "layers": {
                "bronze": {
                    "measures": {"m": {}},
                    "checks": {
                        "a": {"measure": "m", "value": "n", "severity": "critical", "good": {"max": 5}},
                        "b": {"measure": "m", "value": "n", "severity": "warning", "good": {"max": 0}},
                    },
                }
            },
        }
        layer = contract.build_contract(data).get_layer("bronze")

        result = contract.judge_layer(_profile(_measured("m", n=3)), layer)

        assert result.outcome == core.GateOutcome.CONDITIONAL
        assert [reason.check_id for reason in result.conditions] == ["b"]

    def test_a_layer_without_checks_raises(self):
        with pytest.raises(ValueError, match="without criteria"):
            contract.judge_layer(_profile(), contract.LayerSpec("bronze", (), ()))

    def test_another_policy_can_decide(self):
        class _Lenient:
            name = "lenient"

            def apply(self, verdicts):
                return core.GateResult(core.GateOutcome.PASS, (), tuple(verdicts), self.name)

        result = contract.judge_layer(_profile(_measured("m", n=99)), _layer({"max": 0}), policy=_Lenient())

        assert (result.outcome, result.policy) == (core.GateOutcome.PASS, "lenient")

    def test_a_changed_threshold_is_judged_again_without_measuring_again(self):
        measure = _measured("m", n=3)
        profile = _profile(measure)

        strict = contract.judge_layer(profile, _layer({"max": 0}))
        lenient = contract.judge_layer(profile, _layer({"max": 5}))

        assert (strict.outcome, lenient.outcome) == (core.GateOutcome.FAIL, core.GateOutcome.PASS)
        assert measure.runs == 1
