import types

import pytest

from phoibe.geography.complexity.rix.pipelines import product, requirements
from phoibe.geography.complexity.rix.pipelines.requirements import (
    REQUIRED_CHECKS,
    REQUIRED_FIXED_PARAMETERS,
    REQUIRED_INPUTS,
    REQUIRED_TABLES,
    check_specification_complete,
    specification_problems,
)


def dummy_spec(**changes):
    """Exactly what the pipeline requires, as far as `specification_problems` looks at it."""
    spec = {
        "checks": {name: object() for name in REQUIRED_CHECKS},
        "parameters": [types.SimpleNamespace(group=g, name=n, locked=True) for g, n in REQUIRED_FIXED_PARAMETERS],
        "tables": {name: object() for name in REQUIRED_TABLES},
        "inputs": {name: object() for name in REQUIRED_INPUTS},
    }
    return types.SimpleNamespace(**{**spec, **changes})


def without(mapping, name):
    return {key: value for key, value in mapping.items() if key != name}


def test_specification_problems_returns_nothing_given_what_the_pipeline_requires():
    assert specification_problems(dummy_spec()) == []


@pytest.mark.parametrize(
    "section, required", [("checks", REQUIRED_CHECKS), ("tables", REQUIRED_TABLES), ("inputs", REQUIRED_INPUTS)]
)
def test_product_specification_defines_exactly_what_the_pipeline_uses(product_spec, section, required):
    assert set(getattr(product_spec, section)) == set(required)  # not more: a check never raised misleads the contract


def test_product_specification_is_complete_for_the_pipeline(product_spec):
    check_specification_complete(product_spec)


@pytest.mark.parametrize("name", REQUIRED_CHECKS)
def test_specification_problems_returns_a_missing_check(name):
    spec = dummy_spec(checks=without(dummy_spec().checks, name))
    assert specification_problems(spec) == [f"check {name!r} is raised by the pipeline but not defined in 'checks'."]


def test_specification_problems_returns_a_missing_fixed_parameter():
    spec = dummy_spec(parameters=[p for p in dummy_spec().parameters if p.name != "decimals"])
    assert specification_problems(spec) == [
        "parameter 'trix.decimals' is used by the pipeline but not defined in 'parameters'."
    ]


def test_specification_problems_returns_a_fixed_parameter_that_is_not_locked():
    parameters = [
        types.SimpleNamespace(group=p.group, name=p.name, locked=p.name != "nan_policy")
        for p in dummy_spec().parameters
    ]
    assert specification_problems(dummy_spec(parameters=parameters)) == [
        "parameter 'sampler.nan_policy' must be locked, it is used as a fixed value."
    ]


def test_specification_problems_returns_a_parameter_of_another_group_as_missing():
    parameters = [types.SimpleNamespace(group="other", name=p.name, locked=True) for p in dummy_spec().parameters]
    assert len(specification_problems(dummy_spec(parameters=parameters))) == len(REQUIRED_FIXED_PARAMETERS)


@pytest.mark.parametrize("name", REQUIRED_TABLES)
def test_specification_problems_returns_a_missing_table(name):
    spec = dummy_spec(tables=without(dummy_spec().tables, name))
    assert specification_problems(spec) == [f"table {name!r} is built by the pipeline but not defined in 'tables'."]


@pytest.mark.parametrize("name", REQUIRED_INPUTS)
def test_specification_problems_returns_a_missing_input(name):
    spec = dummy_spec(inputs=without(dummy_spec().inputs, name))
    assert specification_problems(spec) == [f"input {name!r} is used by the pipeline but not defined in 'inputs'."]


def test_specification_problems_returns_everything_missing_at_once_in_the_order_of_the_sections():
    spec = dummy_spec(
        checks=without(dummy_spec().checks, REQUIRED_CHECKS[0]),
        tables=without(dummy_spec().tables, REQUIRED_TABLES[0]),
        inputs=without(dummy_spec().inputs, REQUIRED_INPUTS[0]),
    )
    problems = specification_problems(spec)
    assert [problem.split(" ")[0] for problem in problems] == ["check", "table", "input"]


def test_check_specification_complete_raises_listing_everything_missing():
    spec = dummy_spec(checks={}, inputs={})
    with pytest.raises(product.SpecificationError) as error:
        check_specification_complete(spec)
    assert list(error.value.problems) == specification_problems(spec)
    assert len(error.value.problems) == len(REQUIRED_CHECKS) + len(REQUIRED_INPUTS)


def test_check_specification_complete_returns_nothing_given_a_complete_specification():
    check_specification_complete(dummy_spec())
    assert requirements.REQUIRED_FIXED_PARAMETERS == (("trix", "decimals"), ("sampler", "nan_policy"))
