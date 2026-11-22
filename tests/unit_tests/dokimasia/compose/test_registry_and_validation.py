import pytest

from phoibe.dokimasia import compose, contract, core, rules, wind


def _layer(measures: dict, checks: dict | None = None) -> contract.LayerSpec:
    first = next(iter(measures))
    checks = checks or {"c": {"measure": first, "value": "n_gaps", "severity": "info", "good": {"max": 0}}}
    data = {"id": "c", "version": "1", "domain": "wind", "layers": {"raw": {"measures": measures, "checks": checks}}}
    return contract.build_contract(data).get_layer("raw")


def _problems(layer: contract.LayerSpec) -> str:
    with pytest.raises(contract.ContractError) as raised:
        compose.validate_layer_rules(layer, compose.create_default_registry())
    return "\n".join(raised.value.problems)


class TestRegistry:
    def test_rejects_a_rule_name_that_is_registered_twice(self):
        rule = compose.list_generic_rules()[0]

        with pytest.raises(ValueError, match=rule.name):
            compose.Registry([rule, rule])

    def test_each_call_creates_a_registry_of_its_own(self):
        first, second = compose.create_default_registry(), compose.create_default_registry()

        assert first is not second
        assert first.names == (
            "frozen_runs",
            "missing_runs",
            "night_limit",
            "power_levels",
            "ranges",
            "required_signals",
            "statistic_order",
            "time_gaps",
        )
        assert first.find_rule("nope") is None

    def test_the_metrics_of_every_rule_are_the_ones_its_measure_declares(self):
        declared = {
            "time_gaps": rules.GapMeasure.METRICS,
            "missing_runs": rules.MissingRunsMeasure.METRICS,
            "ranges": rules.RangesMeasure.METRICS,
            "required_signals": rules.RequiredSignalsMeasure.METRICS,
            "statistic_order": rules.StatisticOrderMeasure.METRICS,
            "frozen_runs": rules.FrozenRunsMeasure.METRICS,
            "power_levels": wind.PowerLevelsMeasure.METRICS,
            "night_limit": wind.NightLimitMeasure.METRICS,
        }
        registry = compose.create_default_registry()

        assert {name: registry.find_rule(name).metrics for name in registry.names} == declared


class TestValidateLayerRules:
    def test_a_valid_layer_passes(self):
        layer = _layer({"time_gaps": {"params": {"expected_step_s": 600}}})

        compose.validate_layer_rules(layer, compose.create_default_registry())

    def test_names_an_unknown_rule_and_lists_the_known_ones(self):
        problems = _problems(
            _layer(
                {"gaps": {"rule": "gap_finder"}},
                {"c": {"measure": "gaps", "value": "x", "severity": "info", "good": {"max": 0}}},
            )
        )

        assert "unknown rule 'gap_finder'" in problems
        assert "time_gaps" in problems

    def test_collects_parameter_problems_of_all_measures(self):
        layer = _layer(
            {
                "time_gaps": {"params": {"expected_step_s": "ten", "speed": 1}},
                "ranges": {"params": {}},
                "required_signals": {"params": {"signals": []}},
                "missing_runs": {"signal": "a", "params": {"x": 1}},
            },
            {"c": {"measure": "time_gaps", "value": "n_gaps", "severity": "info", "good": {"max": 0}}},
        )

        problems = _problems(layer)

        for expected in [
            "time_gaps.params.expected_step_s: must be a number",
            "time_gaps.params.speed: unknown parameter of rule 'time_gaps'; known: expected_step_s",
            "ranges: rule 'ranges' needs a 'signal'",
            "ranges.params: rule 'ranges' needs at least one of min, max",
            "required_signals.params.signals: must be a non-empty list of text",
            "missing_runs.params.x: unknown parameter",
        ]:
            assert expected in problems, expected

    def test_a_missing_required_parameter_is_reported(self):
        problems = _problems(
            _layer(
                {"required_signals": {}},
                {
                    "c": {
                        "measure": "required_signals",
                        "value": "n_missing_signals",
                        "severity": "info",
                        "good": {"max": 0},
                    }
                },
            )
        )

        assert "missing parameter 'signals'" in problems

    def test_a_rule_without_a_signal_refuses_one(self):
        assert "takes no signal" in _problems(_layer({"time_gaps": {"signal": "a"}}))

    def test_a_check_may_only_read_a_metric_that_the_rule_reports(self):
        layer = _layer(
            {"time_gaps": {}},
            {"c": {"measure": "time_gaps", "value": "n_gapz", "severity": "info", "good": {"max": 0}}},
        )

        problems = _problems(layer)

        assert "rule 'time_gaps' reports no 'n_gapz'" in problems
        assert "n_gaps" in problems

    def test_a_subject_reference_is_allowed_where_the_rule_takes_a_limit(self):
        layer = _layer(
            {"ranges": {"signal": "a", "params": {"max": {"subject": "rated_power_kw", "factor": 1.05}}}},
            {"c": {"measure": "ranges", "value": "n_out_of_range", "severity": "info", "good": {"max": 0}}},
        )

        compose.validate_layer_rules(layer, compose.create_default_registry())

    def test_a_subject_reference_is_refused_where_the_rule_takes_a_plain_number(self):
        layer = _layer({"time_gaps": {"params": {"expected_step_s": {"subject": "step"}}}})

        assert "expected_step_s: must be a number" in _problems(layer)


class TestBuildMeasures:
    def test_builds_measures_with_the_contract_keys_as_ids_and_resolves_subject_references(self):
        layer = _layer(
            {
                "power_limits": {
                    "rule": "ranges",
                    "signal": "power",
                    "params": {"min": 0, "max": {"subject": "rated_power_kw", "factor": 1.05}},
                },
                "gaps": {"rule": "time_gaps", "params": {"expected_step_s": 600}},
            },
            {"c": {"measure": "gaps", "value": "n_gaps", "severity": "info", "good": {"max": 0}}},
        )
        subject = core.Subject("t1", "wind_turbine", {"rated_power_kw": 5600})

        ranges, gaps = compose.build_measures(layer, compose.create_default_registry(), subject)

        assert (ranges.id, gaps.id) == ("power_limits", "gaps")
        assert isinstance(ranges, rules.RangesMeasure)
        assert ranges.max == pytest.approx(5880.0)
        assert isinstance(gaps, rules.GapMeasure)
        assert gaps.expected_step.total_seconds() == 600

    def test_names_every_subject_attribute_that_cannot_be_resolved(self):
        layer = _layer(
            {
                "a": {"rule": "ranges", "signal": "x", "params": {"max": {"subject": "rated_power_kw"}}},
                "b": {"rule": "ranges", "signal": "y", "params": {"max": {"subject": "hub_height"}}},
            },
            {"c": {"measure": "a", "value": "n_out_of_range", "severity": "info", "good": {"max": 0}}},
        )

        with pytest.raises(contract.ContractError) as raised:
            compose.build_measures(layer, compose.create_default_registry(), core.Subject("t1", "wind_turbine"))

        assert len(raised.value.problems) == 2
