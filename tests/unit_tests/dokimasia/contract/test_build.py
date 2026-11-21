import pytest

from phoibe.dokimasia import contract, core

BRONZE = {
    "id": "wind-scada-10min",
    "version": "1.0.0",
    "domain": "wind",
    "layers": {
        "bronze": {
            "measures": {
                "ranges": {"signal": "wind_speed", "params": {"min": 0, "max": 60}},
                "curtailment_levels": {"rule": "curtailment", "signal": "power_kw"},
            },
            "checks": {
                "no_wind_speed_outliers": {
                    "measure": "ranges",
                    "value": "n_out_of_range",
                    "severity": "critical",
                    "good": {"max": 0},
                    "acceptable": {"max": 10},
                    "clause": "TR6 section 3",
                },
                "full_load_matches_rating": {
                    "measure": "curtailment_levels",
                    "value": "highest",
                    "severity": "warning",
                    "good": {
                        "min": {"subject": "rated_power_kw", "factor": 0.99},
                        "max": {"subject": "rated_power_kw", "factor": 1.01},
                    },
                },
            },
        }
    },
}


def _problems(data) -> tuple[str, ...]:
    with pytest.raises(contract.ContractError) as raised:
        contract.build_contract(data)
    return raised.value.problems


def _with_check(**changes) -> dict:
    check = dict(BRONZE["layers"]["bronze"]["checks"]["no_wind_speed_outliers"]) | changes
    layer = {"measures": BRONZE["layers"]["bronze"]["measures"], "checks": {"c": check}}
    return BRONZE | {"layers": {"bronze": layer}}


class TestBuildContract:
    def test_builds_the_layers_measures_and_checks_in_contract_order(self):
        built = contract.build_contract(BRONZE)

        assert (built.id, built.version, built.domain) == ("wind-scada-10min", "1.0.0", "wind")
        layer = built.get_layer("bronze")
        assert [measure.id for measure in layer.measures] == ["ranges", "curtailment_levels"]
        assert [check.id for check in layer.checks] == ["no_wind_speed_outliers", "full_load_matches_rating"]

    def test_the_rule_defaults_to_the_key(self):
        layer = contract.build_contract(BRONZE).get_layer("bronze")

        assert [measure.rule for measure in layer.measures] == ["ranges", "curtailment"]

    def test_reads_bands_clauses_severity_and_subject_references(self):
        layer = contract.build_contract(BRONZE).get_layer("bronze")
        outliers, rating = layer.checks

        assert outliers.severity == core.Severity.CRITICAL
        assert outliers.good == contract.Band(max=0.0)
        assert outliers.acceptable == contract.Band(max=10.0)
        assert outliers.clause == "TR6 section 3"
        assert rating.acceptable is None
        assert rating.good.min == contract.SubjectRef("rated_power_kw", 0.99)

    def test_a_subject_reference_in_a_parameter_becomes_a_subject_ref(self):
        data = BRONZE | {
            "layers": {
                "bronze": {
                    "measures": {"ranges": {"params": {"max": {"subject": "rated_power_kw", "factor": 1.05}}}},
                    "checks": {"c": {"measure": "ranges", "value": "n", "severity": "info", "good": {"max": 0}}},
                }
            }
        }

        measure = contract.build_contract(data).get_layer("bronze").measures[0]

        assert measure.params["max"] == contract.SubjectRef("rated_power_kw", 1.05)

    def test_a_measure_without_content_is_allowed(self):
        data = BRONZE | {
            "layers": {
                "bronze": {
                    "measures": {"time_gaps": None},
                    "checks": {
                        "c": {"measure": "time_gaps", "value": "n_gaps", "severity": "info", "good": {"max": 0}}
                    },
                }
            }
        }

        assert contract.build_contract(data).get_layer("bronze").measures[0].rule == "time_gaps"

    def test_get_layer_raises_for_an_unknown_layer(self):
        with pytest.raises(KeyError):
            contract.build_contract(BRONZE).get_layer("gold")


class TestBuildContractProblems:
    def test_collects_every_problem_with_its_path(self):
        data = {
            "id": "",
            "version": 1.0,
            "layers": {
                "bronze": {"measures": {"m": {"signal": 3}}, "checks": {"c": {"measure": "x", "severity": "urgent"}}}
            },
        }

        problems = _problems(data)

        joined = "\n".join(problems)
        for expected in [
            "id: must be non-empty text",
            "version: must be non-empty text, got 1.0",
            "contract: missing key 'domain'",
            "layers.bronze.measures.m.signal: must be non-empty text",
            "layers.bronze.checks.c: missing key 'value'",
            "layers.bronze.checks.c: missing key 'good'",
            "layers.bronze.checks.c.severity: must be one of critical, warning, info, got 'urgent'",
            "layers.bronze.checks.c.measure: unknown measure 'x'",
        ]:
            assert expected in joined, expected

    def test_a_check_without_severity_is_an_error(self):
        data = _with_check()
        del data["layers"]["bronze"]["checks"]["c"]["severity"]

        assert any("missing key 'severity'" in problem for problem in _problems(data))

    def test_unknown_keys_are_reported_not_ignored(self):
        problems = _problems(_with_check(sevrity="critical"))

        assert any("unknown key 'sevrity'" in problem for problem in problems)

    def test_the_root_must_be_a_mapping(self):
        assert "must be a mapping" in _problems(["not", "a", "mapping"])[0]

    def test_a_contract_without_layers_is_an_error(self):
        assert any("at least one layer" in problem for problem in _problems(BRONZE | {"layers": {}}))

    def test_a_layer_without_checks_is_an_error_and_not_a_pass(self):
        layer = {"measures": BRONZE["layers"]["bronze"]["measures"], "checks": {}}

        assert any("without criteria" in problem for problem in _problems(BRONZE | {"layers": {"bronze": layer}}))

    def test_a_layer_without_measures_is_an_error(self):
        layer = {"measures": {}, "checks": BRONZE["layers"]["bronze"]["checks"]}

        assert any("at least one measure" in problem for problem in _problems(BRONZE | {"layers": {"bronze": layer}}))

    @pytest.mark.parametrize(
        ("good", "expected"),
        [
            ({}, "needs at least one of 'min' and 'max'"),
            ({"min": 5, "max": 1}, "min 5 is above max 1"),
            ({"max": "ten"}, "must be a number or"),
            ({"max": True}, "must be a number or"),
            ({"top": 1}, "unknown key 'top'"),
            ({"max": {"subject": "x", "scale": 2}}, "unknown key 'scale'"),
            ({"max": {"subject": "x", "factor": "two"}}, "factor: must be a number"),
        ],
    )
    def test_a_bad_band_is_an_error(self, good, expected):
        assert any(expected in problem for problem in _problems(_with_check(good=good)))

    @pytest.mark.parametrize(
        ("good", "acceptable", "expected"),
        [
            ({"max": 10}, {"max": 5}, "outside the acceptable band at 'max'"),
            ({"min": 1}, {"min": 2}, "outside the acceptable band at 'min'"),
            ({"min": 1, "max": 10}, {"max": 20}, None),
            ({"max": 10}, {"min": 0, "max": 20}, "narrower"),
            ({"min": 1}, {"min": 0}, None),
        ],
    )
    def test_the_acceptable_band_must_contain_the_good_band(self, good, acceptable, expected):
        data = _with_check(good=good, acceptable=acceptable)

        if expected is None:
            contract.build_contract(data)
        else:
            assert any(expected in problem for problem in _problems(data))

    def test_an_acceptable_bound_that_the_good_band_lacks_is_an_error(self):
        problems = _problems(_with_check(good={"max": 1}, acceptable={"min": 0, "max": 5}))

        assert any("narrower" in problem for problem in problems)
