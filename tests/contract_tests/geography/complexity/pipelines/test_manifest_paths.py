"""D-A1: the keys of the manifest equal the keys declared in the product definition."""

import numpy as np
import pytest
import xarray
import yaml

from phoibe.geography.complexity.rix.computations.fieldsampler import Field

rio = pytest.mark.rio

PROVENANCE = "spatial_context.source_dem."  # free-form keys supplied by the caller, their structure is not fixed yet


def walk(node, leaves, prefix=""):
    for key, value in node.items():
        path = f"{prefix}{key}"
        yield path
        if isinstance(value, dict) and value and path not in leaves:
            yield from walk(value, leaves, f"{path}.")


def under_null(manifest, path):
    """True if an ancestor of `path` is an explicit null, e.g. the extent of a DEM without georeference."""
    node = manifest
    for part in path.split(".")[:-1]:
        node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            return True
    return False


def key_differences(spec, manifest, exempt_null=True):
    declared = set(spec.manifest.paths())
    actual = set(walk(manifest, set(spec.manifest.paths(leaves_only=True))))
    undeclared = sorted(p for p in actual if p not in declared and not p.startswith(PROVENANCE))
    missing = sorted(p for p in declared - actual if not (exempt_null and under_null(manifest, p)))
    return undeclared, missing


@pytest.mark.parametrize("profile", ["summary", "full"])
@pytest.mark.parametrize("with_references", [True, False])
def test_manifest_keys_match_the_definition(
    product_spec, analyzer, make_field, make_locations, references, tmp_path, profile, with_references
):
    result = analyzer.run(
        make_field(), make_locations(), references if with_references else None, dem_metadata={"name": "demo"}
    )
    analyzer.publish(result, tmp_path, profile=profile)
    manifest = yaml.safe_load((tmp_path / product_spec.filename("manifest")).read_text())

    undeclared, missing = key_differences(product_spec, manifest)
    assert undeclared == [], f"in the manifest but not in the definition: {undeclared}"
    assert missing == [], f"in the definition but not in the manifest: {missing}"


def test_a_dem_without_georeference_gives_null_for_crs_extent_and_resolution(
    product_spec, analyzer, make_field, make_locations, tmp_path
):
    analyzer.publish(analyzer.run(make_field(), make_locations()), tmp_path, profile="summary")
    dem = yaml.safe_load((tmp_path / product_spec.filename("manifest")).read_text())["spatial_context"]["source_dem"]
    assert (dem["crs"], dem["extent"], dem["resolution"]) == (None, None, None)  # no `rio` accessor, nothing to read


def test_the_comparison_finds_an_undeclared_and_a_missing_key(product_spec, analyzer, result, tmp_path):
    analyzer.publish(result, tmp_path, profile="summary")
    manifest = yaml.safe_load((tmp_path / product_spec.filename("manifest")).read_text())
    manifest["run"]["surprise"] = 1
    del manifest["run"]["n_sites"]
    assert key_differences(product_spec, manifest) == (["run.surprise"], ["run.n_sites"])


@pytest.fixture
def make_rio_field():
    import rioxarray  # noqa: F401  # registers the .rio accessor for the whole process

    def make(crs="EPSG:32633", half=2000, step=50):
        x = np.arange(-half, half + 1, step, dtype=float)
        da = xarray.DataArray(x[None, :] + 0 * x[:, None], coords={"x": x, "y": x}, dims=("y", "x"))
        return Field.from_array(da if crs is None else da.rio.write_crs(crs))

    return make


@rio
def test_manifest_keys_match_the_definition_given_a_dem_with_georeference(
    product_spec, analyzer, make_rio_field, make_locations, tmp_path
):
    analyzer.publish(analyzer.run(make_rio_field(), make_locations(crs="EPSG:32633")), tmp_path, profile="summary")
    manifest = yaml.safe_load((tmp_path / product_spec.filename("manifest")).read_text())
    assert key_differences(product_spec, manifest, exempt_null=False) == ([], [])  # extent and resolution exist now
    dem = manifest["spatial_context"]["source_dem"]
    assert (dem["crs"], dem["resolution"]["dx"], dem["extent"]["west"]) == ("EPSG:32633", 50.0, -2025.0)


@rio
def test_a_dem_without_crs_gives_a_null_crs_only(product_spec, analyzer, make_rio_field, make_locations, tmp_path):
    analyzer.publish(analyzer.run(make_rio_field(crs=None), make_locations()), tmp_path, profile="summary")
    dem = yaml.safe_load((tmp_path / product_spec.filename("manifest")).read_text())["spatial_context"]["source_dem"]
    assert dem["crs"] is None
    assert dem["extent"] is not None and dem["resolution"] is not None  # from the coordinates; null only without `rio`
