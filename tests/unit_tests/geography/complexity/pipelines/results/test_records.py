import datetime

import pytest

from phoibe.geography.complexity.rix.computations import checks  # .checks import NAN_COUNT, UNIQUE_IDS, Finding
from phoibe.geography.complexity.rix.computations.fieldsampler import FieldInfo
from phoibe.geography.complexity.rix.computations.results import RadialRuggedness
from phoibe.geography.complexity.rix.pipelines.checks.findings import LocatedFinding
from phoibe.geography.complexity.rix.pipelines.checks.inputs import LOCATIONS_REFERENCE, LOCATIONS_SITE
from phoibe.geography.complexity.rix.pipelines.product.parameters import RunParameters, SampleParameters
from phoibe.geography.complexity.rix.pipelines.results.records import (
    CollectionRecord,
    build_run_record,
    fixed_parameter,
    parameter_values,
)

UTC = datetime.UTC
CREATED = datetime.datetime(2026, 10, 24, 12, 0, tzinfo=UTC)


def build(results, **changes):
    arguments = {
        "created_at": CREATED,
        "project_name": "p",
        "parameters": {},
        "field_info": FieldInfo(None, None, None),
        "dem_provenance": None,
        "results": results,
        "transferability_counts": None,
        "findings": [],
    }
    return build_run_record(**{**arguments, **changes})


def located(check, detail="d"):
    return LocatedFinding(checks.Finding(check, detail), "warn")


@pytest.fixture
def sites(make_radial):
    return {"s": make_radial([0, 1], [0, 1])}


def test_from_results_returns_the_sampling_summary_of_the_collection(make_ray):
    def radial(*nan_counts, message="", crs=None):
        return RadialRuggedness(
            rays=tuple(
                make_ray([0, 1], theta=10.0 * k, nan_count=n, message=message, crs=crs)
                for k, n in enumerate(nan_counts)
            )
        )

    results = {
        "a": radial(0, 2, message="m1"),
        "b": radial(0, 0, message="m2"),
        "c": radial(1, message="m1"),
        "d": radial(0, message=""),
    }
    record = CollectionRecord.from_results(results)
    assert (record.n_locations, record.n_locations_with_nans, record.nan_count) == (4, 2, 3)
    assert record.messages == ("m1", "m2")  # distinct, in order of appearance, without empty ones


def test_from_results_returns_the_distinct_ray_crs_in_order_of_appearance(make_ray):
    results = {
        "a": RadialRuggedness(rays=(make_ray([0, 1], crs="EPSG:32633"),)),
        "b": RadialRuggedness(rays=(make_ray([0, 1]),)),
        "c": RadialRuggedness(rays=(make_ray([0, 1], crs="EPSG:32633"),)),
    }
    assert CollectionRecord.from_results(results).ray_crs == ("EPSG:32633", None)


def test_from_results_returns_zeros_given_no_locations():
    assert CollectionRecord.from_results({}) == CollectionRecord(0, 0, 0, (), ())


def test_build_run_record_raises_given_a_naive_time(sites):
    with pytest.raises(ValueError, match="timezone-aware"):
        build({LOCATIONS_SITE: sites}, created_at=datetime.datetime(2026, 10, 24))


def test_build_run_record_raises_given_no_site_collection(sites):
    with pytest.raises(ValueError, match="locations_site"):
        build({LOCATIONS_REFERENCE: sites})


def test_build_run_record_returns_the_collections_by_name_and_copies_of_the_mappings(sites):
    parameters = {"n_angles": 72}
    record = build(
        {LOCATIONS_SITE: sites, LOCATIONS_REFERENCE: sites}, parameters=parameters, dem_provenance={"name": "dem"}
    )
    parameters["n_angles"] = 1
    assert {name: c.n_locations for name, c in record.collections.items()} == {
        LOCATIONS_SITE: 1,
        LOCATIONS_REFERENCE: 1,
    }
    assert (record.parameters, record.dem_provenance, build({LOCATIONS_SITE: sites}).dem_provenance) == (
        {"n_angles": 72},
        {"name": "dem"},
        {},
    )


@pytest.mark.parametrize(
    "collections, counts, present_inputs, computed",
    [
        ([LOCATIONS_SITE], None, {"dem", LOCATIONS_SITE}, ("rix_site",)),
        (
            [LOCATIONS_SITE, LOCATIONS_REFERENCE],
            None,
            {"dem", LOCATIONS_SITE, LOCATIONS_REFERENCE},
            ("rix_site", "rix_reference"),
        ),
        (
            [LOCATIONS_SITE, LOCATIONS_REFERENCE],
            {},
            {"dem", LOCATIONS_SITE, LOCATIONS_REFERENCE},
            ("rix_site", "rix_reference", "trix"),
        ),
    ],
    ids=["sites", "references", "trix"],
)
def test_run_record_returns_the_present_inputs_and_what_has_been_computed(
    sites, collections, counts, present_inputs, computed
):
    record = build(dict.fromkeys(collections, sites), transferability_counts=counts)
    # record = build({name: sites for name in collections}, transferability_counts=counts)
    assert (record.present_inputs, record.computed) == (frozenset(present_inputs), computed)


def test_build_run_record_returns_the_findings_capped_per_check_and_the_counts_of_all(sites):
    findings = [located(checks.NAN_COUNT, f"n{k}") for k in range(5)] + [
        located(checks.UNIQUE_IDS, "u0"),
        located(checks.UNIQUE_IDS, "u1"),
    ]
    record = build({LOCATIONS_SITE: sites}, findings=findings, max_findings_per_check=3)
    assert [item.finding.detail for item in record.findings] == ["n0", "n1", "n2", "u0", "u1"]
    assert (dict(record.finding_counts), record.n_findings_omitted) == ({checks.NAN_COUNT: 5, checks.UNIQUE_IDS: 2}, 2)


def test_build_run_record_returns_integer_counts_of_transferability(sites):
    import numpy as np

    record = build({LOCATIONS_SITE: sites}, transferability_counts={np.int64(2): np.int64(5)})
    assert record.transferability_counts == {2: 5} and type(next(iter(record.transferability_counts.values()))) is int


def test_fixed_parameter_returns_the_value_of_a_locked_parameter(product_spec):
    locked = next(p for p in product_spec.parameters if p.locked)
    assert fixed_parameter(product_spec, locked.group, locked.name) == locked.value


def test_fixed_parameter_raises_given_a_parameter_that_is_not_locked(product_spec):
    free = next(p for p in product_spec.parameters if not p.locked)
    with pytest.raises(ValueError, match="not fixed by the specification"):
        fixed_parameter(product_spec, free.group, free.name)


def test_parameter_values_returns_the_locked_values_overridden_by_the_supplied_ones(product_spec):
    run = RunParameters(n_angles=36, R_km=3.5, dr_km=0.05, slope_critical=0.033, crs="EPSG:2056")
    values = parameter_values(product_spec, run, SampleParameters(interpolation_method="nearest"))
    locked = {p.name: p.value for p in product_spec.parameters if p.locked}
    assert values == {
        **locked,
        "n_angles": 36,
        "R_km": 3.5,
        "dr_km": 0.05,
        "slope_critical": 0.033,
        "crs": "EPSG:2056",
        "interpolation_method": "nearest",
    }


def test_parameter_values_raises_given_a_parameter_that_two_sets_define(product_spec):
    run = RunParameters(n_angles=72, R_km=3.5, dr_km=0.05, slope_critical=0.033, crs=None)
    with pytest.raises(ValueError, match="defined twice"):
        parameter_values(product_spec, run, run)
