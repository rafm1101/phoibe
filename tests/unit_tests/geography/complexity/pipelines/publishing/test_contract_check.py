import dataclasses

import pytest

from phoibe.geography.complexity.rix.pipelines import errors
from phoibe.geography.complexity.rix.pipelines.ports import WrittenArtifacts
from phoibe.geography.complexity.rix.pipelines.publishing.contract_check import (
    check_written_artifacts,
    tables_missing_for,
    written_artifact_problems,
)

BOTH = {"dem", "refs"}


def written(**changes):
    """What a correct writer reports for profile "big" given the DEM and the references."""
    default = WrittenArtifacts(
        profile="big",
        files={"table_a": "a.csv", "table_b": "b.csv", "container": "c.gpkg"},
        table_columns={"table_a": ("id", "v"), "table_b": ("code",)},
        table_records={"table_a": 2, "table_b": 3},
        layer_columns={"geo": ("id",), "attrs": ("code",)},  # the geometry column is not listed
        layer_geometry_types={"geo": "Point", "attrs": None},
        layer_records={"geo": 2, "attrs": 3},
    )
    return dataclasses.replace(default, **changes)


def without(mapping, *keys):
    return {key: value for key, value in mapping.items() if key not in keys}


@pytest.mark.parametrize(
    "available, present_inputs, expected",
    [
        ({"t_a", "t_b", "t_geo"}, BOTH, []),
        ({"t_a", "t_geo"}, BOTH, ["t_b"]),  # needed by an artifact and a layer, reported once
        (set(), BOTH, ["t_a", "t_b", "t_geo"]),  # sorted
        ({"t_a", "t_geo"}, {"dem"}, []),  # t_b is only needed with the references
    ],
    ids=["complete", "once", "sorted", "optional"],
)
def test_tables_missing_for_returns_the_tables_a_profile_needs_that_are_not_available(
    spec, available, present_inputs, expected
):
    assert tables_missing_for(spec, available, "big", present_inputs) == expected


def test_written_artifact_problems_returns_nothing_given_what_the_specification_expects(spec):
    assert written_artifact_problems(spec, written(), BOTH) == []


def test_written_artifact_problems_returns_nothing_given_the_artifacts_without_the_references(spec):
    expected = written(
        files={"table_a": "a.csv", "container": "c.gpkg"},
        table_columns={"table_a": ("id", "v")},
        table_records={"table_a": 2},
        layer_columns={"geo": ("id",)},
        layer_geometry_types={"geo": "Point"},
        layer_records={"geo": 2},
    )
    assert written_artifact_problems(spec, expected, {"dem"}) == []


@pytest.mark.parametrize(
    "changes, inputs, expected",
    [
        (lambda w: {"files": without(w.files, "table_b")}, BOTH, "artifact 'table_b' was not written."),
        (
            lambda w: {"files": {**w.files, "extra": "x.csv"}},
            BOTH,
            "artifact 'extra' was written but is not expected for profile 'big'.",
        ),
        (
            lambda w: {"files": {**w.files, "manifest": "manifest.yaml"}},
            BOTH,
            "artifact 'manifest' was written but is not expected for profile 'big'.",
        ),
        (
            lambda w: {"files": {**w.files, "table_a": "x.csv"}},
            BOTH,
            "artifact 'table_a': file 'x.csv', expected 'a.csv'.",
        ),
        (
            lambda w: {"table_columns": {**w.table_columns, "table_a": ("v", "id")}},
            BOTH,
            "artifact 'table_a': columns ['v', 'id'], expected ['id', 'v'].",
        ),
        (
            lambda w: {"table_columns": without(w.table_columns, "table_a")},
            BOTH,
            "artifact 'table_a': columns were not reported.",
        ),
        (
            lambda w: {"table_records": without(w.table_records, "table_a")},
            BOTH,
            "artifact 'table_a': number of rows was not reported.",
        ),
        (lambda w: {"layer_columns": without(w.layer_columns, "geo")}, BOTH, "layer 'geo' was not written."),
        (
            lambda w: {"layer_columns": {**w.layer_columns, "geo": ("id", "geometry")}},
            BOTH,
            "layer 'geo': columns ['id', 'geometry'], expected ['id'].",
        ),
        (
            lambda w: {"layer_geometry_types": {**w.layer_geometry_types, "geo": "Polygon"}},
            BOTH,
            "layer 'geo': geometry type 'Polygon', expected 'Point'.",
        ),
        (
            lambda w: {"layer_geometry_types": {**w.layer_geometry_types, "attrs": "Point"}},
            BOTH,
            "layer 'attrs': geometry type 'Point', expected None.",
        ),
        (
            lambda w: {"layer_records": without(w.layer_records, "geo")},
            BOTH,
            "layer 'geo': number of records was not reported.",
        ),
        (
            lambda w: {"layer_columns": {**w.layer_columns, "ghost": ("id",)}},
            BOTH,
            "layer 'ghost' was written but is not expected.",
        ),
        (
            lambda w: {},
            {"dem"},
            "artifact 'table_b' was written but is not expected for profile 'big'.",
        ),  # no references
        (lambda w: {}, {"dem"}, "layer 'attrs' was written but is not expected."),
    ],
    ids=[
        "not written",
        "unexpected",
        "manifest",
        "filename",
        "column order",
        "columns not reported",
        "rows not reported",
        "layer not written",
        "layer columns",
        "geometry type",
        "attribute layer with geometry",
        "records not reported",
        "unexpected layer",
        "artifact without references",
        "layer without references",
    ],
)
def test_written_artifact_problems_returns_the_deviation_from_the_specification(spec, changes, inputs, expected):
    base = written()
    problems = written_artifact_problems(spec, dataclasses.replace(base, **changes(base)), inputs)
    assert expected in problems


def test_written_artifact_problems_returns_all_deviations_but_no_follow_up_of_a_missing_artifact(spec):
    base = written()
    broken = dataclasses.replace(
        base,
        files=without(base.files, "table_a"),
        table_columns=without(base.table_columns, "table_a"),  # would be a second problem if reported
        layer_geometry_types={**base.layer_geometry_types, "geo": "Polygon"},
    )
    assert written_artifact_problems(spec, broken, BOTH) == [
        "artifact 'table_a' was not written.",
        "layer 'geo': geometry type 'Polygon', expected 'Point'.",
    ]


def test_check_written_artifacts_returns_nothing_given_what_the_specification_expects(spec):
    check_written_artifacts(spec, written(), BOTH)


def test_check_written_artifacts_raises_listing_all_deviations(spec):
    base = written()
    broken = dataclasses.replace(base, files={**without(base.files, "table_b"), "table_a": "x.csv"})
    with pytest.raises(errors.ContractViolation) as error:
        check_written_artifacts(spec, broken, BOTH)
    assert error.value.problems == (
        "artifact 'table_b' was not written.",
        "artifact 'table_a': file 'x.csv', expected 'a.csv'.",
    )
