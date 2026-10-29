import datetime
import pathlib

import numpy as np
import pytest
import yaml

from phoibe.geography.complexity.rix import computations
from phoibe.geography.complexity.rix.computations import checks
from phoibe.geography.complexity.rix.pipelines.checks.findings import LocatedFinding
from phoibe.geography.complexity.rix.pipelines.checks.inputs import LOCATIONS_REFERENCE, LOCATIONS_SITE
from phoibe.geography.complexity.rix.pipelines.ports import WrittenArtifacts
from phoibe.geography.complexity.rix.pipelines.product import RunParameters, SampleParameters
from phoibe.geography.complexity.rix.pipelines.publishing import build_manifest, render_contract
from phoibe.geography.complexity.rix.pipelines.results.records import build_run_record, parameter_values

UTC = datetime.UTC
CREATED = datetime.datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
PARAMETERS = {"n": 3, "r": 2.5, "crs": None, "method": "linear"}  # the parameters of the minimal specification
WRITTEN = WrittenArtifacts(
    profile="small",
    files={"table_a": "a.csv"},
    table_columns={"table_a": ("id", "v")},
    table_records={"table_a": 2},
    layer_columns={},
    layer_geometry_types={},
    layer_records={},
)


@pytest.fixture
def make_record(make_radial):
    def make(**changes):
        sites = {"s1": make_radial(([0, 1], 2, 2), [0, 1]), "s2": make_radial([0, 1])}  # 2 NaN in 1 of 2 locations
        arguments = {
            "created_at": CREATED,
            "project_name": "proj",
            "parameters": PARAMETERS,
            "field_info": computations.FieldInfo(None, None, None),
            "dem_provenance": None,
            "results": {LOCATIONS_SITE: sites},
            "transferability_counts": None,
            "findings": [],
        }

        return build_run_record(**{**arguments, **changes})

    return make


def test_build_manifest_returns_the_contract_first_and_the_sections_in_order(spec, make_record):
    manifest = build_manifest(spec, make_record(), WRITTEN)
    assert list(manifest) == ["contract", "meta", "parameters", "spatial_context", "run", "artifacts"]
    assert manifest["contract"] == render_contract(spec)


@pytest.mark.parametrize(
    "created_at",
    [CREATED, datetime.datetime(2026, 10, 24, 14, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=2)))],
    ids=["utc", "another zone"],
)
def test_build_manifest_returns_the_meta_data_with_the_time_in_utc(spec, make_record, created_at):
    meta = build_manifest(spec, make_record(created_at=created_at), WRITTEN)["meta"]
    assert meta == {
        "name": "Demo",
        "version": "1.2.3",
        "description": "A demo product.",
        "created_at": "2026-10-24T12:00:00Z",
    }


def test_build_manifest_returns_the_parameters_grouped_as_in_the_specification(spec, make_record):
    assert build_manifest(spec, make_record(), WRITTEN)["parameters"] == {
        "ray": {"n": 3, "r": 2.5, "crs": None},
        "mode": {"method": "linear"},
    }


@pytest.mark.parametrize(
    "parameters, expected",
    [
        ({"n": 3, "r": 2.5, "crs": None}, "missing in the run: ['method']; not in the specification: none"),
        ({**PARAMETERS, "extra": 1}, "missing in the run: none; not in the specification: ['extra']"),
    ],
    ids=["missing", "undefined"],
)
def test_build_manifest_raises_given_parameters_that_differ_from_the_specification(
    spec, make_record, parameters, expected
):
    with pytest.raises(ValueError, match="differ") as error:
        build_manifest(spec, make_record(parameters=parameters), WRITTEN)
    assert expected in str(error.value)


def test_build_manifest_returns_the_provenance_of_the_dem_and_what_is_read_from_the_data(spec, make_record):
    field_info = computations.FieldInfo("EPSG:32633", (0.0, 1.0, 2.0, 3.0), (30.0, -30.0))
    provenance = {"name": "dem", "source": {"url": "x"}, "crs": "from the user"}
    source_dem = build_manifest(spec, make_record(field_info=field_info, dem_provenance=provenance), WRITTEN)[
        "spatial_context"
    ]["source_dem"]
    assert source_dem == {
        "name": "dem",
        "source": {"url": "x"},
        "crs": "EPSG:32633",  # the data wins over the statement of the user
        "extent": {"west": 0.0, "south": 1.0, "east": 2.0, "north": 3.0},
        "resolution": {"dx": 30.0, "dy": -30.0},
    }
    assert provenance["crs"] == "from the user"


def test_build_manifest_returns_null_given_a_dem_without_georeferencing(spec, make_record):
    source_dem = build_manifest(spec, make_record(), WRITTEN)["spatial_context"]["source_dem"]
    assert source_dem == {"crs": None, "extent": None, "resolution": None}


def test_build_manifest_returns_the_distinct_crs_and_messages_of_all_collections(spec, make_ray):
    def collection(crs, *messages):
        rays = (make_ray([0, 1], theta=10.0 * k, crs=crs, message=message) for k, message in enumerate(messages))
        return {"a": computations.RadialRuggedness(rays=tuple(rays))}

    record = build_run_record(
        created_at=CREATED,
        project_name="",
        parameters=PARAMETERS,
        field_info=computations.FieldInfo(None, None, None),
        dem_provenance=None,
        results={LOCATIONS_SITE: collection(None, "m1"), LOCATIONS_REFERENCE: collection("EPSG:32633", "m1", "m2")},
        transferability_counts=None,
        findings=[],
    )
    context = build_manifest(spec, record, WRITTEN)["spatial_context"]
    assert (context["source_ray"], context["alignment"]) == ({"crs": [None, "EPSG:32633"]}, {"messages": ["m1", "m2"]})


def test_build_manifest_returns_the_counts_of_the_run_per_collection(spec, make_record, make_radial):
    references = {"r1": make_radial([0, 1]), "r2": make_radial([0, 1]), "r3": make_radial([0, 1])}
    record = make_record(
        results={
            LOCATIONS_SITE: {"s1": make_radial(([0, 1], 2, 2)), "s2": make_radial([0, 1])},
            LOCATIONS_REFERENCE: references,
        },
        transferability_counts={2: 4, 0: 2},
    )
    run = build_manifest(spec, record, WRITTEN)["run"]
    assert (run["project_name"], run["n_sites"], run["n_references"], run["computed"]) == (
        "proj",
        2,
        3,
        ["rix_site", "rix_reference", "trix"],
    )
    assert run["diagnostics"] == {
        "nan_count": {LOCATIONS_SITE: 2, LOCATIONS_REFERENCE: 0},
        "n_locations_with_nans": {LOCATIONS_SITE: 1, LOCATIONS_REFERENCE: 0},
        "transferability_counts": {2: 4, 0: 2},
        "finding_counts": {},
        "findings_omitted": 0,
    }


def test_build_manifest_returns_zero_references_and_no_transferability_given_only_sites(spec, make_record):
    run = build_manifest(spec, make_record(), WRITTEN)["run"]
    assert (run["n_references"], run["computed"], run["diagnostics"]["transferability_counts"]) == (0, ["rix_site"], {})
    assert list(run["diagnostics"]["nan_count"]) == [LOCATIONS_SITE]


def test_build_manifest_returns_the_findings_with_severity_and_place(spec, make_record):
    findings = [
        LocatedFinding(checks.Finding(checks.NAN_COUNT, "some NaN", 10.0), "warn", LOCATIONS_SITE, np.int64(7)),
        LocatedFinding(checks.Finding(checks.CENTRE_ELEVATION, "elevation"), "abort"),
    ]
    run = build_manifest(spec, make_record(findings=findings), WRITTEN)["run"]
    assert run["findings"] == [
        {
            "check": checks.NAN_COUNT,
            "severity": "warn",
            "collection": LOCATIONS_SITE,
            "location_id": 7,
            "theta": 10.0,
            "detail": "some NaN",
        },
        {
            "check": checks.CENTRE_ELEVATION,
            "severity": "abort",
            "collection": None,
            "location_id": None,
            "theta": None,
            "detail": "elevation",
        },
    ]
    assert type(run["findings"][0]["location_id"]) is int
    assert run["diagnostics"]["finding_counts"] == {checks.NAN_COUNT: 1, checks.CENTRE_ELEVATION: 1}


def test_build_manifest_returns_the_findings_omitted_by_the_cap(spec, make_record):
    findings = [LocatedFinding(checks.Finding(checks.NAN_COUNT, f"d{k}"), "warn") for k in range(5)]
    record = make_record(findings=findings, max_findings_per_check=2)
    run = build_manifest(spec, record, WRITTEN)["run"]
    assert (len(run["findings"]), run["diagnostics"]["findings_omitted"], run["diagnostics"]["finding_counts"]) == (
        2,
        3,
        {checks.NAN_COUNT: 5},
    )


def test_build_manifest_returns_the_files_with_the_manifest_added_and_copies_of_the_counts(spec, make_record):
    manifest = build_manifest(spec, make_record(), WRITTEN)
    assert manifest["artifacts"] == {
        "profile": "small",
        "files": {"table_a": "a.csv", "manifest": "manifest.yaml"},
        "table_records": {"table_a": 2},
        "layer_records": {},
    }
    assert "manifest" not in WRITTEN.files  # the written artifacts are untouched


def test_build_manifest_returns_python_values_for_what_a_yaml_dump_cannot_write(spec, make_record):
    provenance = {
        "count": np.int64(3),
        "ratio": np.float64(0.5),
        "path": pathlib.Path("/x"),
        "pair": (1, 2),
        "nested": {np.int64(1): "k"},
    }
    source_dem = build_manifest(spec, make_record(dem_provenance=provenance), WRITTEN)["spatial_context"]["source_dem"]
    assert source_dem["count"] == 3 and type(source_dem["count"]) is int
    assert type(source_dem["ratio"]) is float
    assert (source_dem["path"], source_dem["pair"], source_dem["nested"]) == ("/x", [1, 2], {1: "k"})


def test_build_manifest_returns_a_copy_of_the_provenance_of_the_record(spec, make_record):
    provenance = {"source": {"url": "x"}}
    record = make_record(dem_provenance=provenance)
    build_manifest(spec, record, WRITTEN)["spatial_context"]["source_dem"]["source"]["url"] = "changed"
    assert record.dem_provenance["source"]["url"] == "x"


def test_build_manifest_returns_values_that_survive_a_yaml_round_trip_given_the_product_specification(
    product_spec, make_radial
):
    parameters = parameter_values(
        product_spec,
        RunParameters(n_angles=72, R_km=3.5, dr_km=0.05, slope_critical=0.033, crs="EPSG:2056"),
        SampleParameters("linear"),
    )
    findings = [LocatedFinding(checks.Finding(checks.NAN_COUNT, "some NaN", 10.0), "warn", LOCATIONS_SITE, "s1")]
    record = build_run_record(
        created_at=CREATED,
        project_name="p",
        parameters=parameters,
        field_info=computations.FieldInfo("EPSG:2056", (0, 0, 1, 1), (1.0, -1.0)),
        dem_provenance={"name": "dem"},
        results={LOCATIONS_SITE: {"s1": make_radial([0, 1])}},
        transferability_counts={0: 1},
        findings=findings,
    )
    manifest = build_manifest(product_spec, record, WRITTEN)
    assert yaml.safe_load(yaml.safe_dump(manifest)) == manifest
