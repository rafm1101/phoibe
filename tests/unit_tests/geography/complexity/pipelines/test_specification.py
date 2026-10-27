import dataclasses

import pytest

from phoibe.geography.complexity.rix.pipelines.product.specification import ProductSpecification, SpecificationError


def test_from_dict_returns_the_parameters_with_their_inferred_dtype(spec):
    n, r, crs, method = (
        spec.parameter("ray", "n"),
        spec.parameter("ray", "r"),
        spec.parameter("ray", "crs"),
        spec.parameter("mode", "method"),
    )
    assert (n.value, n.dtype, n.locked, n.source, n.unit) == (3, "int", True, "REF", None)
    assert (r.value, r.dtype, r.locked, r.unit, r.valid_range) == (2.5, "float", False, "km", (0, 10))
    assert (crs.value, crs.dtype) == (None, "str")
    assert (method.value, method.dtype) == ("linear", "str")
    assert [(p.group, p.name) for p in spec.parameters] == [
        ("ray", "n"),
        ("ray", "r"),
        ("ray", "crs"),
        ("mode", "method"),
    ]


def test_from_dict_returns_the_inputs_and_checks(spec):
    dem, refs = spec.inputs["dem"], spec.inputs["refs"]
    assert (dem.required, dem.type, dem.notes, dem.conditions) == (True, "array", ("a note",), ("a rule",))
    assert (refs.required, refs.type, refs.notes) == (False, None, ())
    assert (spec.check("c1").severity, dict(spec.check("c1").parameters)) == ("warn", {"max": 0.1})
    assert (spec.check("c2").severity, dict(spec.check("c2").parameters)) == ("abort", {})


def test_from_dict_returns_the_tables_with_columns_in_their_order(spec):
    assert spec.table("t_a").column_names == ("id", "v")
    column = spec.table("t_a").column("v")
    assert (column.unit, column.dtype, column.valid_range, column.values) == ("m", "float", (0, 1), None)
    assert dict(spec.table("t_b").column("code").values) == {0: "No", 1: "Yes"}


def test_from_dict_returns_the_artifacts_layers_and_profiles(spec):
    assert spec.filename("table_a") == "a.csv"
    assert (spec.artifact("table_b").table, spec.artifact("table_b").requires_input) == ("t_b", "refs")
    assert (spec.artifact("container").table, spec.artifact("container").requires_input) == (None, None)
    layer = spec.layer("geo")
    assert (layer.artifact, layer.table, layer.name, layer.geometry_type, layer.requires_input) == (
        "container",
        "t_geo",
        "geo",
        "Point",
        None,
    )
    assert spec.layer("attrs").geometry_type is None
    assert spec.profiles["small"] == ("table_a", "manifest")


def test_from_dict_returns_immutable_sections(spec):
    for section in (spec.tables, spec.artifacts, spec.layers, spec.profiles, spec.references):
        with pytest.raises(TypeError):
            section["new"] = None
    assert isinstance(spec.parameters, tuple) and isinstance(spec.profiles["big"], tuple)


def test_paths_of_the_manifest_return_dotted_keys(spec):
    assert list(spec.manifest.paths()) == ["meta", "meta.name", "run", "run.n"]
    assert list(spec.manifest.paths(leaves_only=True)) == ["meta.name", "run.n"]


@pytest.mark.parametrize(
    "changes, deleted, expected",
    [
        ({}, ["name"], "definition: missing 'name'"),
        ({"name": ""}, [], "definition.name: expected non-empty text"),
        ({"version": "v1"}, [], "definition.version: expected dotted numbers"),
        ({"surprise": {}}, [], "unknown key 'surprise'"),
        ({}, ["manifest"], "missing 'manifest'"),
        ({"parameters.ray.n.value": "three"}, [], "'three' does not match dtype 'int'"),
        ({"parameters.ray.n.dtype": "number"}, [], "must be one of"),
        ({}, ["parameters.ray.crs.dtype"], "'dtype' is required when 'value' is None"),
        ({"parameters.ray.r.range": [5, 1]}, [], "lower bound must be smaller"),
        ({"parameters.ray.r.range": [0]}, [], "expected [lower, upper]"),
        ({"parameters.ray.r.value": 11}, [], "outside range"),
        ({"parameters.ray.n.value": None}, [], "a locked parameter needs a value"),
        ({}, ["parameters.ray.n.source"], "a locked parameter needs a 'source'"),
        ({"parameters.ray.n.source": "NOPE"}, [], "unknown reference 'NOPE'"),
        ({"parameters.ray.r.unit": ""}, [], "parameters.ray.r.unit: expected non-empty text"),
        ({"parameters.ray.r.extra": 1}, [], "unknown key 'extra'"),
        ({"parameters.mode.n": {"description": "Twice.", "value": 1}}, [], "name 'n'"),
        ({}, ["inputs.dem.required"], "inputs.dem: missing 'required'"),
        ({"inputs.dem.notes": "text"}, [], "expected a list of text"),
        ({"checks.c1.severity": "fatal"}, [], "must be one of ('abort', 'warn')"),
        ({}, ["checks.c1.severity"], "got None"),
        ({"checks.c1.parameters": [1]}, [], "expected a mapping"),
        ({"tables.t_a.columns": {}}, [], "needs at least one column"),
        ({"tables.t_a.columns.id.name": "other"}, [], "differs from its key"),
        ({"tables.t_a.columns.v.dtype": "decimal"}, [], "must be one of"),
        ({"tables.t_a.columns.v.range": [1, 0]}, [], "lower bound must be smaller"),
        ({"artifacts.files.table_a.table": "nope"}, [], "unknown table 'nope'"),
        ({"artifacts.files.table_b.requires_input": "nope"}, [], "unknown input 'nope'"),
        ({"artifacts.files.table_b.filename": "a.csv"}, [], "'a.csv' is already used by"),
        ({}, ["artifacts.files.manifest"], "missing the 'manifest' artifact"),
        ({"artifacts.profiles.small": ["table_a"]}, [], "every profile must contain 'manifest'"),
        ({"artifacts.profiles.small": ["table_a", "manifest", "ghost"]}, [], "unknown artifact 'ghost'"),
        ({"artifacts.profiles.small": []}, [], "expected a non-empty list"),
        ({"artifacts.layers.geo.artifact": "ghost"}, [], "unknown artifact 'ghost'"),
        ({"artifacts.layers.geo.table": "ghost"}, [], "unknown table 'ghost'"),
        ({"artifacts.layers.attrs.name": "geo"}, [], "layer name 'geo' is used twice"),
        ({"artifacts.layers.geo.geometry_type": "None"}, [], "must be None or one of"),  # the string, not None
        ({"artifacts.layers.geo.geometry_type": "Circle"}, [], "must be None or one of"),
        ({"manifest.keys.meta.keys.name.name": "other"}, [], "differs from its key"),
        ({"manifest.keys.meta.free_form": True}, [], "a free-form key cannot declare nested keys"),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_from_dict_raises_given_an_invalid_definition(make_definition, changes, deleted, expected):
    with pytest.raises(SpecificationError) as error:
        ProductSpecification.from_dict(make_definition(changes, deleted))
    assert any(expected in problem for problem in error.value.problems)


def test_from_dict_raises_listing_all_problems(make_definition):
    definition = make_definition(
        {"version": "v1", "checks.c1.severity": "fatal", "artifacts.files.table_a.table": "nope"}
    )
    with pytest.raises(SpecificationError) as error:
        ProductSpecification.from_dict(definition)
    assert len(error.value.problems) == 3
    assert str(error.value).startswith("Invalid product specification:\n  - ")


def replace_artifact(spec, key, **changes):
    return {**spec.artifacts, key: dataclasses.replace(spec.artifacts[key], **changes)}


@pytest.mark.parametrize(
    "changes, expected",
    [
        (lambda s: {"version": "v1"}, "version"),
        (lambda s: {"tables": {**s.tables, "x": s.tables["t_a"]}}, "key differs from name"),
        (lambda s: {"parameters": (*s.parameters, dataclasses.replace(s.parameters[0], group="mode"))}, "name 'n'"),
        (
            lambda s: {"parameters": (dataclasses.replace(s.parameters[0], source="NOPE"), *s.parameters[1:])},
            "unknown reference",
        ),
        (lambda s: {"artifacts": replace_artifact(s, "table_a", table="nope")}, "unknown table 'nope'"),
        (lambda s: {"artifacts": replace_artifact(s, "table_b", requires_input="nope")}, "unknown input 'nope'"),
        (
            lambda s: {
                "artifacts": {k: v for k, v in s.artifacts.items() if k != "manifest"},
                "profiles": {"small": ("table_a",)},
            },
            "missing the 'manifest' artifact",
        ),
        (lambda s: {"profiles": {**s.profiles, "small": ("table_a",)}}, "every profile must contain"),
        (
            lambda s: {"profiles": {**s.profiles, "small": ("table_a", "manifest", "ghost")}},
            "unknown artifact 'ghost'",
        ),
        (
            lambda s: {"layers": {**s.layers, "attrs": dataclasses.replace(s.layers["attrs"], name="geo")}},
            "layer name",
        ),
        pytest.param(
            lambda s: {"artifacts": replace_artifact(s, "table_b", filename="a.csv")},
            "already used by",
            marks=pytest.mark.xfail(
                strict=True, reason="_problems() records a filename only inside the duplicate branch"
            ),
        ),
    ],
    ids=[
        "version",
        "key",
        "parameter name",
        "source",
        "table",
        "input",
        "no manifest",
        "profile",
        "ghost",
        "layer",
        "filename",
    ],
)
def test_construction_raises_given_a_broken_reference_however_it_is_built(spec, changes, expected):
    with pytest.raises(SpecificationError) as error:
        dataclasses.replace(spec, **changes(spec))
    assert any(expected in problem for problem in error.value.problems)


@pytest.mark.parametrize(
    "lookup, argument",
    [
        ("parameter", ("ray", "nope")),
        ("table", ("nope",)),
        ("artifact", ("nope",)),
        ("layer", ("nope",)),
        ("check", ("nope",)),
        ("filename", ("nope",)),
    ],
)
def test_lookup_raises_a_key_error_listing_the_available_ids(spec, lookup, argument):
    with pytest.raises(KeyError, match=r"Unknown .* Available: "):
        getattr(spec, lookup)(*argument)


def test_column_raises_a_key_error_listing_the_available_names(spec):
    with pytest.raises(KeyError, match="Available: id, v"):
        spec.table("t_a").column("nope")


@pytest.mark.parametrize(
    "profile, present_inputs, expected",
    [
        ("small", set(), ["table_a", "manifest"]),
        ("big", set(), ["table_a", "manifest", "container"]),  # table_b needs the references
        ("big", {"dem"}, ["table_a", "manifest", "container"]),
        ("big", {"dem", "refs"}, ["table_a", "table_b", "manifest", "container"]),
    ],
)
def test_artifacts_for_returns_the_artifacts_of_the_profile_given_the_inputs(spec, profile, present_inputs, expected):
    assert [a.key for a in spec.artifacts_for(profile, present_inputs)] == expected


def test_artifacts_for_raises_given_an_unknown_profile(spec):
    with pytest.raises(KeyError, match="Unknown profile 'nope'"):
        spec.artifacts_for("nope", set())


@pytest.mark.parametrize(
    "artifact, present_inputs, expected",
    [("container", set(), ["geo"]), ("container", {"refs"}, ["geo", "attrs"]), ("table_a", {"refs"}, [])],
)
def test_layers_for_returns_the_layers_of_the_artifact_given_the_inputs(spec, artifact, present_inputs, expected):
    assert [layer.key for layer in spec.layers_for(artifact, present_inputs)] == expected


def test_layers_for_raises_given_an_unknown_artifact(spec):
    with pytest.raises(KeyError, match="Unknown artifact 'nope'"):
        spec.layers_for("nope", set())
