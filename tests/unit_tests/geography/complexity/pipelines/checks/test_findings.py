import logging

import pytest

from phoibe.geography.complexity.rix.computations import checks  # .checks import (

#     CENTRE_ELEVATION,
#     N_ANGLE,
#     NAN_COUNT,
#     RIX_NAN,
#     UNIQUE_IDS,
#     Finding,
# )
from phoibe.geography.complexity.rix.pipelines import errors
from phoibe.geography.complexity.rix.pipelines.checks.findings import (
    LOCKED_PARAMETER_CHANGED,
    LocatedFinding,
    check_parameter,
    locate,
    location_findings,
    locked_parameter_findings,
    log_findings,
    parameter_findings,
    raise_on_abort,
)


def locked_values(spec):
    return {parameter.name: parameter.value for parameter in spec.parameters if parameter.locked}


def test_locate_returns_the_findings_with_the_severity_of_their_check_and_the_place(make_spec):
    spec = make_spec(severities={checks.NAN_COUNT: "abort", checks.UNIQUE_IDS: "warn"})
    found = locate(
        spec,
        [checks.Finding(checks.NAN_COUNT, "a", 10.0), checks.Finding(checks.UNIQUE_IDS, "b")],
        collection="locations_site",
        location_id="s1",
    )
    assert [(f.finding.check, f.severity, f.collection, f.location_id) for f in found] == [
        (checks.NAN_COUNT, "abort", "locations_site", "s1"),
        (checks.UNIQUE_IDS, "warn", "locations_site", "s1"),
    ]


def test_locate_returns_nothing_given_no_findings(product_spec):
    assert locate(product_spec, []) == ()


def test_locate_raises_given_a_check_the_specification_does_not_define(product_spec):
    with pytest.raises(KeyError):
        locate(product_spec, [checks.Finding("no_such_check", "x")])


def test_check_parameter_returns_the_parameter_of_the_check_or_the_default(make_spec):
    spec = make_spec(check_parameters={checks.NAN_COUNT: {"max_fraction": 0.3}})
    assert check_parameter(spec, checks.NAN_COUNT, "max_fraction", 0.0) == 0.3
    assert check_parameter(spec, checks.NAN_COUNT, "unknown", "default") == "default"


def test_location_findings_returns_the_findings_of_every_location_with_its_id(make_spec, make_radial):
    spec = make_spec(
        check_parameters={checks.NAN_COUNT: {"max_fraction": 0.0}, checks.CENTRE_ELEVATION: {"atol": 1e-6}}
    )
    results = {"a": make_radial([0, 1], [0, 1]), "b": make_radial(([0, 1], 2, 2), [0, 1])}
    found = location_findings(spec, "locations_site", results)
    assert [(f.finding.check, f.collection, f.location_id, f.finding.theta) for f in found] == [
        (checks.NAN_COUNT, "locations_site", "b", 0.0)
    ]


@pytest.mark.parametrize(
    "parameters, expected_count",
    [({"max_fraction": 0.0}, 1), ({"max_fraction": 0.5}, 0), ({}, 1)],  # a missing threshold means: no NaN tolerated
    ids=["none tolerated", "tolerated", "not defined"],
)
def test_location_findings_returns_the_nan_findings_above_the_threshold_of_the_specification(
    make_spec, make_radial, parameters, expected_count
):
    spec = make_spec(check_parameters={checks.NAN_COUNT: parameters})
    found = location_findings(spec, "locations_site", {"a": make_radial(([0, 1], 1, 10))})  # 10 % NaN
    assert [f.finding.check for f in found] == [checks.NAN_COUNT] * expected_count


@pytest.mark.parametrize(
    "parameters, expected_count",
    [({"atol": 1.0}, 0), ({"atol": 0.1}, 1), ({}, 1)],  # a missing tolerance means: 1e-6
    ids=["tolerated", "exceeded", "not defined"],
)
def test_location_findings_returns_the_centre_findings_above_the_tolerance_of_the_specification(
    make_spec, make_radial, parameters, expected_count
):
    spec = make_spec(check_parameters={checks.CENTRE_ELEVATION: parameters})
    found = location_findings(spec, "locations_site", {"a": make_radial([10, 1], [10, 1], [10.5, 1])})  # 0.5 m
    assert [f.finding.check for f in found] == [checks.CENTRE_ELEVATION] * expected_count


def test_location_findings_returns_one_rix_finding_per_location(product_spec, make_radial):
    results = {"a": make_radial([5], [5], [5, 6]), "b": make_radial([0, 1])}  # two rays without a segment in "a"
    found = location_findings(product_spec, "locations_reference", results)
    assert [(f.finding.check, f.collection, f.location_id) for f in found] == [
        (checks.RIX_NAN, "locations_reference", "a")
    ]


def test_locked_parameter_findings_returns_nothing_given_the_locked_values(product_spec):
    assert locked_parameter_findings(product_spec, locked_values(product_spec)) == ()


def test_locked_parameter_findings_returns_a_finding_per_changed_parameter(product_spec):
    values = locked_values(product_spec)
    locked = dict(values)
    values.update(n_angles=36, R_km=5.0)
    found = locked_parameter_findings(product_spec, values)
    groups = {parameter.name: parameter.group for parameter in product_spec.parameters}
    assert sorted(f.finding.detail.split(" ")[0] for f in found) == sorted(
        [f"{groups['n_angles']}.n_angles", f"{groups['R_km']}.R_km"]
    )
    detail = next(f.finding.detail for f in found if ".n_angles" in f.finding.detail)
    assert "36" in detail and repr(locked["n_angles"]) in detail


@pytest.mark.parametrize("values", [{}, {"dr_km": 0.01}], ids=["absent", "not locked"])
def test_locked_parameter_findings_returns_nothing_given_parameters_that_are_absent_or_not_locked(product_spec, values):
    assert locked_parameter_findings(product_spec, values) == ()


def test_locked_parameter_findings_returns_nothing_given_an_equal_number_of_another_type(product_spec):
    values = {
        name: float(value) for name, value in locked_values(product_spec).items() if isinstance(value, (int, float))
    }
    assert locked_parameter_findings(product_spec, values) == ()


def test_locked_parameter_findings_returns_a_finding_given_another_string(product_spec):
    name = next(p.name for p in product_spec.parameters if p.locked and isinstance(p.value, str))
    found = locked_parameter_findings(product_spec, {name: "something else"})
    assert [f.finding.check for f in found] == [LOCKED_PARAMETER_CHANGED]


def test_locked_parameter_findings_returns_the_severity_of_the_check(make_spec):
    spec = make_spec(severities={LOCKED_PARAMETER_CHANGED: "abort"})
    found = locked_parameter_findings(spec, {"n_angles": 36})
    assert [f.severity for f in found] == ["abort"]


@pytest.mark.parametrize(
    "values, check_parameters, expected",
    [
        ({"n_angles": 72}, {checks.N_ANGLE: {"range": [1, 10]}}, [checks.N_ANGLE]),  # the range of the specification
        ({"n_angles": 5}, {checks.N_ANGLE: {"range": [1, 10]}}, [LOCKED_PARAMETER_CHANGED]),
        (
            {"n_angles": 361},
            {checks.N_ANGLE: {}},
            [checks.N_ANGLE, LOCKED_PARAMETER_CHANGED],
        ),  # the default range is [1, 360]
        ({"R_km": 5.0}, {checks.N_ANGLE: {"range": [1, 10]}}, [LOCKED_PARAMETER_CHANGED]),  # no angles, no range check
    ],
    ids=["outside the range", "inside the range", "default range", "no angles"],
)
def test_parameter_findings_returns_the_locked_and_the_angle_findings(make_spec, values, check_parameters, expected):
    spec = make_spec(check_parameters=check_parameters)
    base = {name: value for name, value in locked_values(spec).items() if name not in values and name != "n_angles"}
    found = parameter_findings(spec, {**base, **values})
    assert sorted(f.finding.check for f in found) == sorted(expected)


def test_raise_on_abort_returns_nothing_given_no_abort():
    raise_on_abort([])
    raise_on_abort([LocatedFinding(checks.Finding(checks.NAN_COUNT, "x"), "warn")])


def test_raise_on_abort_raises_given_any_abort():
    findings = [
        LocatedFinding(checks.Finding(checks.NAN_COUNT, "x"), "warn"),
        LocatedFinding(checks.Finding(checks.UNIQUE_IDS, "y"), "abort"),
    ]
    with pytest.raises(errors.CheckAborted):
        raise_on_abort(findings)


def test_log_findings_returns_one_message_per_location_and_check(caplog):
    logger = logging.getLogger("tests.findings")
    items = [
        LocatedFinding(checks.Finding(checks.NAN_COUNT, f"d{k}", 10.0 * k), "warn", "locations_site", "a")
        for k in range(3)
    ]
    items += [LocatedFinding(checks.Finding(checks.NAN_COUNT, "other"), "warn", "locations_site", "b")]
    items += [LocatedFinding(checks.Finding(checks.UNIQUE_IDS, "dup"), "abort", "locations_site")]
    with caplog.at_level(logging.DEBUG, logger="tests.findings"):
        log_findings(items, logger)
    assert [(record.levelno, record.getMessage()) for record in caplog.records] == [
        (logging.WARNING, "Check 'nan_count' at locations_site, id='a': 3 finding(s). First: d0"),
        (logging.WARNING, "Check 'nan_count' at locations_site, id='b': 1 finding(s). First: other"),
        (logging.ERROR, "Check 'unique_ids' at locations_site: 1 finding(s). First: dup"),
    ]


@pytest.mark.parametrize(
    "item, expected",
    [
        (
            LocatedFinding(checks.Finding(checks.NAN_COUNT, "detail", 10.0), "warn", "locations_site", "a"),
            "[warn] nan_count at locations_site, id='a', theta=10.0: detail",
        ),
        (LocatedFinding(checks.Finding(checks.UNIQUE_IDS, "detail"), "abort"), "[abort] unique_ids: detail"),
        (
            LocatedFinding(checks.Finding(checks.NAN_COUNT, "detail"), "warn", "locations_site", 0),
            "[warn] nan_count at locations_site, id=0: detail",
        ),
    ],
    ids=["full place", "no place", "falsy identifier"],
)
def test_located_finding_str_returns_severity_check_place_and_detail(item, expected):
    assert str(item) == expected
