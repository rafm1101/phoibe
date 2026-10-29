import geopandas as gpd
import pytest

from phoibe.geography.complexity.rix.computations import checks
from phoibe.geography.complexity.rix.pipelines import errors
from phoibe.geography.complexity.rix.pipelines.checks.inputs import (
    LOCATIONS_REFERENCE,
    LOCATIONS_SITE,
    check_required_inputs,
    input_findings,
    normalise_locations,
)
from phoibe.geography.complexity.rix.pipelines.product.schema import PRODUCT_SPECIFICATION

REQUIRED = [name for name, item in PRODUCT_SPECIFICATION.inputs.items() if item.required]


def make_locations(ids=("a", "b"), crs="EPSG:2056"):
    frame = gpd.GeoDataFrame(
        {"site_id": list(ids), "height": range(len(ids))},
        geometry=gpd.points_from_xy(range(len(ids)), range(len(ids))),
        crs=crs,
    )
    return normalise_locations(frame)


def found(**kwargs):
    parameters = {"locations_site": make_locations(), "locations_reference": None, "crs_config": None}
    findings = input_findings(PRODUCT_SPECIFICATION, **{**parameters, **kwargs})
    return [(item.finding.check, item.collection) for item in findings]


def test_check_required_inputs_returns_nothing_given_the_required_and_any_optional_inputs(product_spec):
    check_required_inputs(product_spec, REQUIRED)
    check_required_inputs(product_spec, list(product_spec.inputs))


@pytest.mark.parametrize("missing", REQUIRED)
def test_check_required_inputs_raises_given_a_missing_required_input(product_spec, missing):
    with pytest.raises(errors.InputRequirementError, match=f"missing: {missing}"):
        check_required_inputs(product_spec, set(product_spec.inputs) - {missing})


def test_check_required_inputs_raises_given_an_input_the_specification_does_not_define(product_spec):
    with pytest.raises(errors.InputRequirementError, match="not defined in the specification: extra"):
        check_required_inputs(product_spec, {*product_spec.inputs, "extra"})


def test_normalise_locations_returns_the_identifier_column_as_index():
    frame = gpd.GeoDataFrame({"site_id": ["a", "b"], "height": [1, 2]}, geometry=gpd.points_from_xy([0, 1], [0, 1]))
    result = normalise_locations(frame)
    assert (result.index.name, list(result.index), list(result.columns)) == (
        "site_id",
        ["a", "b"],
        ["height", "geometry"],
    )


def test_normalise_locations_returns_a_copy_with_the_existing_index_given_no_identifier_column():
    frame = gpd.GeoDataFrame(geometry=gpd.points_from_xy([0, 1], [0, 1]), index=["x", "y"])
    result = normalise_locations(frame)
    assert list(result.index) == ["x", "y"] and result is not frame


def test_normalise_locations_returns_a_copy_and_takes_another_identifier_column():
    frame = gpd.GeoDataFrame({"name": ["a", "b"]}, geometry=gpd.points_from_xy([0, 1], [0, 1]))
    result = normalise_locations(frame, id_column="name")
    assert list(result.index) == ["a", "b"] and "name" in frame.columns  # the input is untouched


def test_input_findings_returns_nothing_given_clean_inputs():
    assert found(locations_reference=make_locations(("r1", "r2")), crs_config="EPSG:2056") == []


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"locations_site": make_locations(("a", "a"))}, [(checks.UNIQUE_IDS, LOCATIONS_SITE)]),
        ({"crs_config": "EPSG:32633"}, [(checks.CRS_VS_CONFIG, LOCATIONS_SITE)]),
        ({"locations_reference": make_locations(("r", "r"))}, [(checks.UNIQUE_IDS, LOCATIONS_REFERENCE)]),
        (
            {"locations_reference": make_locations(("r1", "r2"), crs="EPSG:32633")},
            [(checks.CRS_SITE_VS_REFERENCE, None)],
        ),
        ({"locations_reference": None, "crs_config": "EPSG:2056"}, []),  # the reference rules need a reference
    ],
    ids=["duplicate sites", "configured crs", "duplicate references", "crs of sites and references", "no references"],
)
def test_input_findings_returns_the_finding_of_the_collection_it_concerns(kwargs, expected):
    assert found(**kwargs) == expected


def test_input_findings_returns_the_findings_in_the_order_of_the_rules():
    result = found(
        locations_site=make_locations(("a", "a")),
        locations_reference=make_locations(("r", "r"), crs="EPSG:32633"),
        crs_config="EPSG:32633",
    )
    assert result == [
        (checks.UNIQUE_IDS, LOCATIONS_SITE),
        (checks.CRS_VS_CONFIG, LOCATIONS_SITE),
        (checks.UNIQUE_IDS, LOCATIONS_REFERENCE),
        (checks.CRS_SITE_VS_REFERENCE, None),
    ]
