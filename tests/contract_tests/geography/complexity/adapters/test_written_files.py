"""D-A2: what is on disk equals the specification. Read independently of the writer's own read-back."""

import pandas as pd
import pyogrio
import pytest

from phoibe.geography.complexity.rix.pipelines.publishing.contract_check import written_artifact_problems


@pytest.mark.parametrize("profile", ["summary", "full"])
@pytest.mark.parametrize("with_references", [True, False])
def test_files_match_the_specification(
    product_spec, analyzer, make_field, make_locations, references, tmp_path, profile, with_references
):
    spec = product_spec
    result = analyzer.run(
        make_field(), make_locations(), references if with_references else None, include_steep_segments=True
    )
    written = analyzer.publish(result, tmp_path, profile=profile)
    present = result.record.present_inputs

    expected = [a for a in spec.artifacts_for(profile, present) if a.key != "manifest"]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [a.filename for a in expected] + [spec.filename("manifest")]
    )
    for artifact in expected:
        path = tmp_path / artifact.filename
        if artifact.table is not None:
            assert list(pd.read_csv(path).columns) == list(spec.table(artifact.table).column_names)
        for layer in spec.layers_for(artifact.key, present):
            info = pyogrio.read_info(path, layer=layer.name)
            attributes = [
                c for c in spec.table(layer.table).column_names if layer.geometry_type is None or c != "geometry"
            ]
            assert list(info["fields"]) == attributes
            assert info["geometry_type"] == layer.geometry_type
    assert written_artifact_problems(spec, written, present) == []


def test_flat_terrain_gives_an_empty_but_typed_layer(
    product_spec, analyzer, make_field, make_locations, references, tmp_path
):
    """No steep segment: the layer exists, has no record, and still has the geometry type of the specification."""
    result = analyzer.run(make_field(a=0.0), make_locations(), references)
    assert len(result.steep_segments) == 0

    written = analyzer.publish(result, tmp_path, profile="full")

    assert written.layer_records["ruggedness"] == 0
    assert written.layer_geometry_types["ruggedness"] == "LineString"
    layer = product_spec.layer("ruggedness")
    assert (
        pyogrio.read_info(tmp_path / product_spec.artifact(layer.artifact).filename, layer=layer.name)["geometry_type"]
        == "LineString"
    )
