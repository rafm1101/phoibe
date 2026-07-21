import pytest

from phoibe.geography.complexity.rix.schema import PRODUCT_DEFINITION_TRIX


def _iter_parameter_entries(definition):
    """Yield (path, entry) for every parameter definition dict."""
    for group_name, group in definition["parameters"].items():
        for param_name, entry in group.items():
            yield f"parameters.{group_name}.{param_name}", entry


def _iter_column_entries(definition):
    """Yield (path, entry) for every column definition across rix_table, trix_table, and all geopackage layers."""
    schema = definition["schema"]
    for table_name in ("rix_table", "trix_table"):
        for column_name, entry in schema[table_name]["columns"].items():
            yield f"schema.{table_name}.columns.{column_name}", entry
    for layer_name, layer in schema["geopackage"]["layers"].items():
        for column_name, entry in layer["columns"].items():
            yield f"schema.geopackage.layers.{layer_name}.columns.{column_name}", entry


def test_product_definition_has_version_string():
    assert isinstance(PRODUCT_DEFINITION_TRIX["version"], str)


def test_product_definition_references_include_tr6_and_riley99():
    references = PRODUCT_DEFINITION_TRIX["references"]
    assert "TR6" in references
    assert "riley99" in references


@pytest.mark.parametrize("group", ["ray", "slope", "sampler"])
def test_product_definition_parameters_are_grouped_as_documented(group):
    assert group in PRODUCT_DEFINITION_TRIX["parameters"]


@pytest.mark.parametrize(
    "group, name",
    [("ray", "n_angles"), ("ray", "R_km"), ("slope", "slope_critical")],
)
def test_product_definition_tr6_parameters_are_locked_with_source(group, name):
    parameter = PRODUCT_DEFINITION_TRIX["parameters"][group][name]
    assert parameter["source"] == "TR6"
    assert parameter["locked"] is True


def test_product_definition_no_locked_parameter_also_declares_a_range():
    violations = [
        path
        for path, entry in _iter_parameter_entries(PRODUCT_DEFINITION_TRIX)
        if entry.get("locked") is True and "range" in entry
    ]
    assert violations == []


def test_product_definition_no_column_declares_both_range_and_values():
    violations = [
        path for path, entry in _iter_column_entries(PRODUCT_DEFINITION_TRIX) if "range" in entry and "values" in entry
    ]
    assert violations == []


def test_product_definition_no_unit_is_an_empty_string():
    violations = [path for path, entry in _iter_parameter_entries(PRODUCT_DEFINITION_TRIX) if entry.get("unit") == ""]
    violations += [path for path, entry in _iter_column_entries(PRODUCT_DEFINITION_TRIX) if entry.get("unit") == ""]
    assert violations == []


def test_product_definition_geometry_type_is_python_none_or_a_real_type_name():
    layers = PRODUCT_DEFINITION_TRIX["schema"]["geopackage"]["layers"]
    for layer_name, layer in layers.items():
        geometry_type = layer["geometry_type"]
        assert geometry_type is None or isinstance(geometry_type, str)
        assert geometry_type != "None", f"{layer_name}.geometry_type is the string 'None', not Python None"


def test_product_definition_artifact_filenames_cover_every_profile_member():
    artifacts = PRODUCT_DEFINITION_TRIX["artifacts"]
    referenced = {name for members in artifacts["profiles"].values() for name in members}
    assert referenced <= set(artifacts["filenames"].keys())


def test_product_definition_identifier_columns_use_their_own_key_as_output_name():
    violations = [
        path
        for path, entry in _iter_column_entries(PRODUCT_DEFINITION_TRIX)
        if (column_key := path.rsplit(".", 1)[-1]) in ("site_id", "reference_id")
        and "name" in entry
        and entry["name"] != column_key
    ]
    assert violations == []
