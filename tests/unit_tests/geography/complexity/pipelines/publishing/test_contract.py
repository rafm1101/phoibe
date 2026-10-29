import yaml

from phoibe.geography.complexity.rix.pipelines.publishing.contract import render_contract


def test_render_contract_returns_the_sections_of_the_product(spec):
    contract = render_contract(spec)
    assert list(contract) == [
        "product",
        "version",
        "description",
        "inputs",
        "parameters",
        "checks",
        "outputs",
        "profiles",
        "references",
    ]
    assert (contract["product"], contract["version"], contract["description"]) == ("Demo", "1.2.3", "A demo product.")
    assert contract["profiles"] == {
        "small": ["table_a", "manifest"],
        "big": ["table_a", "table_b", "manifest", "container"],
    }
    assert contract["references"]["REF"]["title"] == "A reference"


def test_render_contract_returns_only_the_notes_and_conditions_that_exist(spec):
    assert render_contract(spec)["inputs"] == {
        "dem": {
            "required": True,
            "description": "DEM.",
            "type": "array",
            "notes": ["a note"],
            "conditions": ["a rule"],
        },
        "refs": {"required": False, "description": "References."},
    }


def test_render_contract_returns_the_value_of_locked_parameters_only(spec):
    assert render_contract(spec)["parameters"] == {
        "ray": {
            "n": {"description": "Count.", "dtype": "int", "unit": None, "locked": True, "value": 3, "source": "REF"},
            "r": {"description": "Radius.", "dtype": "float", "unit": "km", "locked": False, "range": [0, 10]},
            "crs": {"description": "CRS.", "dtype": "str", "unit": None, "locked": False},
        },
        "mode": {"method": {"description": "Method.", "dtype": "str", "unit": None, "locked": False}},
    }


def test_render_contract_returns_the_checks_with_their_consequence(spec):
    assert render_contract(spec)["checks"] == [
        {"name": "c1", "description": "A check.", "on_failure": "warn", "parameters": {"max": 0.1}},
        {"name": "c2", "description": "Another check.", "on_failure": "abort"},
    ]


def test_render_contract_returns_the_files_and_columns_of_the_outputs(spec):
    outputs = render_contract(spec)["outputs"]
    assert list(outputs) == ["manifest", "table_a", "table_b", "container"]
    assert outputs["manifest"] == {"file": "manifest.yaml", "columns": []}
    assert outputs["table_a"] == {
        "file": "a.csv",
        "columns": [
            {"name": "id", "type": None, "unit": None, "description": "Identifier."},
            {"name": "v", "type": "float", "unit": "m", "description": "Value.", "range": [0, 1]},
        ],
    }
    assert outputs["table_b"]["requires_input"] == "refs"
    assert outputs["table_b"]["columns"][0]["values"] == {0: "No", 1: "Yes"}


def test_render_contract_returns_the_layers_of_a_container_with_their_requirement(spec):
    container = render_contract(spec)["outputs"]["container"]
    assert (container["file"], container["columns"], "requires_input" in container) == ("c.gpkg", [], False)
    assert list(container["layers"]) == ["geo", "attrs"]
    geo, attrs = container["layers"]["geo"], container["layers"]["attrs"]
    assert (geo["name"], geo["geometry_type"], [c["name"] for c in geo["columns"]], "requires_input" in geo) == (
        "geo",
        "Point",
        ["id", "geometry"],
        False,
    )
    assert (attrs["geometry_type"], attrs["requires_input"]) == (None, "refs")


def test_render_contract_returns_only_plain_values(spec):
    contract = render_contract(spec)
    assert yaml.safe_load(yaml.safe_dump(contract)) == contract


def test_render_contract_returns_plain_values_given_the_product_specification(product_spec):
    contract = render_contract(product_spec)
    assert yaml.safe_load(yaml.safe_dump(contract)) == contract


def test_render_contract_returns_the_same_contract_for_every_call(spec):
    assert render_contract(spec) == render_contract(spec)
