import pytest

from phoibe.layered.application.registry import RuleRegistry

MINIMAL_PARAMS = {
    "required_variable": {"variable_name": "wind_speed"},
    "ranges": {"variable_ranges": {"wind_speed": [0, 40]}},
    "essential_ranges": {"variable_names": ["wind_speed"]},
    "curtailments_power": {},
    "temporal_attributes": {},
    "data_gaps": {},
    "availability": {},
}

# Registry keys whose rule reports a different name. Remove a mark when its rule is fixed.
MISMATCHED = {
    "required_variable": "defect 9: required_variable is reported as variable_<x>",
    "essential_ranges": "defect 9: essential_ranges is reported as essential_range",
}


def _case(key: str):
    marks = []
    if key in MISMATCHED:
        marks = [pytest.mark.characterization, pytest.mark.xfail(strict=True, reason=MISMATCHED[key])]
    return pytest.param(key, id=key, marks=marks)


def test_every_rule_has_minimal_params(fresh_registry):
    """Guard: a new rule needs an entry in `MINIMAL_PARAMS`, so that the identity test covers it."""
    assert sorted(RuleRegistry.list_rules()) == sorted(MINIMAL_PARAMS)


@pytest.mark.parametrize("key", [_case(key) for key in sorted(MINIMAL_PARAMS)])
def test_rule_reports_the_name_it_is_registered_under(fresh_registry, key):
    rule = RuleRegistry.get(key)(**MINIMAL_PARAMS[key])

    assert rule.name == key
