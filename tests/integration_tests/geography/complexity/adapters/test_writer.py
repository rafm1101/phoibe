import dataclasses

import pandas as pd
import pyogrio
import pytest
import yaml

from phoibe.geography.complexity.rix.adapters.writer import FileResultWriter
from phoibe.geography.complexity.rix.pipelines import errors, ports

REFERENCES = {"ids": ("r1", "r2", "r3"), "xs": (500, -500, 0), "ys": (500, -500, 500)}


@pytest.fixture
def spec(product_spec):
    """The specification with a geopackage that does not need references, as in the current definition."""
    geopackage = dataclasses.replace(product_spec.artifacts["geopackage"], requires_input=None)
    return dataclasses.replace(product_spec, artifacts={**product_spec.artifacts, "geopackage": geopackage})


@pytest.fixture
def writer(spec):
    return FileResultWriter(spec)


@pytest.fixture
def make_result(make_analyzer, make_field, make_locations, spec):
    def make(references=True, crs=None, **run_kwargs):
        references = make_locations(**REFERENCES, crs=crs) if references else None
        return make_analyzer(spec).run(make_field(), make_locations(crs=crs), references, **run_kwargs)

    return make


def layer_names(spec, result):
    return {layer.key: layer.name for layer in spec.layers_for("geopackage", result.record.present_inputs)}


def test_file_result_writer_is_a_result_writer(writer):
    assert isinstance(writer, ports.ResultWriter)


def test_write_artifacts_returns_the_csv_of_the_summary_and_writes_no_manifest(writer, make_result, tmp_path):
    result = make_result(references=False)
    written = writer.write_artifacts(result, tmp_path, "summary")

    assert sorted(path.name for path in tmp_path.iterdir()) == ["rix_table.csv"]
    assert (written.profile, dict(written.files), dict(written.table_records)) == (
        "summary",
        {"rix_table": "rix_table.csv"},
        {"rix_table": 3},
    )
    read_back = pd.read_csv(tmp_path / "rix_table.csv")
    assert written.table_columns["rix_table"] == tuple(read_back.columns)  # reported as read from the file
    pd.testing.assert_frame_equal(read_back, result.summary.reset_index(), check_dtype=False)


def test_write_artifacts_returns_the_pairs_as_one_row_each_given_references(writer, make_result, spec, tmp_path):
    result = make_result()
    written = writer.write_artifacts(result, tmp_path, "summary")
    trix = pd.read_csv(tmp_path / "trix_table.csv")
    assert (sorted(written.files), written.table_records["trix_table"], len(trix)) == (
        ["rix_table", "trix_table"],
        9,
        9,
    )
    assert (
        list(trix.columns) == list(written.table_columns["trix_table"]) == list(spec.table("trix_table").column_names)
    )
    assert list(trix[["site_id", "reference_id"]].iloc[0]) == ["a", "r1"]


def test_write_artifacts_returns_the_layers_of_the_geopackage_as_read_back_from_the_file(
    writer, make_result, spec, tmp_path
):
    result = make_result()
    written = writer.write_artifacts(result, tmp_path, "full")

    names = layer_names(spec, result)
    path = tmp_path / "rix_details.gpkg"
    assert {layer[0] for layer in pyogrio.list_layers(path)} == set(names.values())
    for key, name in names.items():
        info = pyogrio.read_info(path, layer=name)
        assert (written.layer_geometry_types[key], written.layer_records[key]) == (
            info["geometry_type"],
            info["features"],
        )
        assert written.layer_columns[key] == tuple(info["fields"])
    assert {key: (written.layer_geometry_types[key], written.layer_records[key]) for key in names} == {
        "locations_site": ("Point", 3),
        "locations_reference": ("Point", 3),
        "ruggedness": ("LineString", 6),
        "trix_table": (None, 9),  # a table without geometry
    }


def test_write_artifacts_returns_the_crs_of_the_locations_in_their_layers(writer, make_result, spec, tmp_path):
    result = make_result(crs="EPSG:32633")
    writer.write_artifacts(result, tmp_path, "full")
    names = layer_names(spec, result)
    assert pyogrio.read_info(tmp_path / "rix_details.gpkg", layer=names["locations_site"])["crs"] == "EPSG:32633"


def test_write_artifacts_returns_no_stale_layer_of_an_earlier_run(writer, make_result, spec, tmp_path):
    writer.write_artifacts(make_result(references=True), tmp_path, "full")
    result = make_result(references=False)
    written = writer.write_artifacts(result, tmp_path, "full")
    names = {layer[0] for layer in pyogrio.list_layers(tmp_path / "rix_details.gpkg")}
    assert names == {"locations_site", "ruggedness"} and set(written.layer_columns) == {"locations_site", "ruggedness"}


def test_write_artifacts_creates_the_target_directory(writer, make_result, tmp_path):
    target = tmp_path / "new" / "nested"
    writer.write_artifacts(make_result(references=False), target, "summary")
    assert (target / "rix_table.csv").is_file()


def test_write_artifacts_raises_before_writing_given_a_result_without_a_table_of_the_profile(
    writer, make_result, tmp_path
):
    result = make_result(include_steep_segments=False)
    target = tmp_path / "out"
    with pytest.raises(errors.ContractViolation, match="no table 'ruggedness'"):
        writer.write_artifacts(result, target, "full")
    assert not target.exists()


def test_write_artifacts_raises_given_a_geometry_layer_whose_table_has_no_geometry(writer, make_result, tmp_path):
    result = make_result(references=False)
    flat = pd.DataFrame(result.tables["locations_site"].drop(columns="geometry"))
    broken = dataclasses.replace(result, tables={**result.tables, "locations_site": flat})
    with pytest.raises(errors.ContractViolation, match="has no geometry"):
        writer.write_artifacts(broken, tmp_path, "full")


def test_write_artifacts_raises_a_port_error_given_a_target_that_is_a_file(writer, make_result, tmp_path):
    target = tmp_path / "file"
    target.write_text("not a directory")
    with pytest.raises(errors.PortError, match="Could not write artifacts"):
        writer.write_artifacts(make_result(references=False), target, "summary")


def test_write_manifest_returns_the_yaml_of_the_manifest_in_its_order(writer, tmp_path):
    manifest = {
        "contract": {"product": "Zürich"},
        "meta": {"n": 1},
        "artifacts": {"files": {"manifest": "summary.yaml"}},
    }
    writer.write_manifest(manifest, tmp_path)
    text = (tmp_path / "summary.yaml").read_text(encoding="utf-8")
    assert "Zürich" in text  # not escaped
    assert yaml.safe_load(text) == manifest and list(yaml.safe_load(text)) == ["contract", "meta", "artifacts"]


def test_write_manifest_raises_a_port_error_given_a_missing_directory(writer, tmp_path):
    with pytest.raises(errors.PortError, match="Could not write the manifest"):
        writer.write_manifest({"meta": {}}, tmp_path / "missing")


def test_publish_returns_the_files_of_the_profile_and_the_manifest_that_lists_them(
    make_analyzer, make_field, make_locations, spec, tmp_path
):
    analyzer = make_analyzer(spec, writer=FileResultWriter(spec))
    result = analyzer.run(make_field(), make_locations(), make_locations(**REFERENCES))

    written = analyzer.publish(result, tmp_path, "full")

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "rix_details.gpkg",
        "rix_table.csv",
        "summary.yaml",
        "trix_table.csv",
    ]
    manifest = yaml.safe_load((tmp_path / "summary.yaml").read_text())
    assert manifest["artifacts"]["files"] == {**written.files, "manifest": "summary.yaml"}
    assert manifest["artifacts"]["layer_records"] == dict(written.layer_records)
