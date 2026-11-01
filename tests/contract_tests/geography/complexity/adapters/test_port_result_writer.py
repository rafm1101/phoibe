"""T4: every ResultWriter implementation passes the same suite. Add new writers to the `writer` fixture.

The protocol itself has no behaviour. This suite holds each implementation to what the protocol promises:
exactly the artifacts of the profile, never the manifest, names from the specification, the manifest as given.
"""

import dataclasses

import pytest
import yaml

from phoibe.geography.complexity.rix.adapters.writer import FileResultWriter
from phoibe.geography.complexity.rix.pipelines.ports import ResultWriter
from phoibe.geography.complexity.rix.pipelines.publishing.contract_check import written_artifact_problems


@pytest.fixture(params=["files", "memory"])
def writer(request, product_spec, make_writer):
    return FileResultWriter(product_spec) if request.param == "files" else make_writer()


def manifest_written(writer, target, spec):
    if isinstance(writer, FileResultWriter):
        return yaml.safe_load((target / spec.filename("manifest")).read_text())
    return writer.manifest


def has_manifest(writer, target, spec):
    if isinstance(writer, FileResultWriter):
        return (target / spec.filename("manifest")).exists()
    return writer.manifest is not None


def test_is_a_result_writer(writer):
    assert isinstance(writer, ResultWriter)


@pytest.mark.parametrize("profile", ["summary", "full"])
@pytest.mark.parametrize("with_references", [True, False])
def test_writes_what_the_specification_lists_and_no_manifest(
    writer, product_spec, analyzer, make_field, make_locations, references, tmp_path, profile, with_references
):
    result = analyzer.run(make_field(), make_locations(), references if with_references else None)
    present = result.record.present_inputs

    written = writer.write_artifacts(result, tmp_path, profile)

    expected = {a.key for a in product_spec.artifacts_for(profile, present) if a.key != "manifest"}
    assert (written.profile, set(written.files)) == (profile, expected)
    assert all(written.files[key] == product_spec.filename(key) for key in expected)  # names from the specification
    assert "manifest" not in written.files and not has_manifest(writer, tmp_path, product_spec)
    assert written_artifact_problems(product_spec, written, present) == []


def test_writes_the_same_again_into_the_same_target(writer, result, tmp_path):
    assert writer.write_artifacts(result, tmp_path, "full") == writer.write_artifacts(result, tmp_path, "full")


def test_writes_the_manifest_as_given_and_decides_nothing_about_it(writer, product_spec, tmp_path):
    manifest = {"contract": {"product": "x"}, "run": {"n_sites": 3, "findings": []}, "anything": [1, 2.5, None, "text"]}
    writer.write_manifest(manifest, tmp_path)
    assert manifest_written(writer, tmp_path, product_spec) == manifest


def test_the_manifest_does_not_change_the_artifacts_written_before(writer, result, tmp_path):
    before = writer.write_artifacts(result, tmp_path, "summary")
    writer.write_manifest({"meta": {}}, tmp_path)
    assert dataclasses.asdict(writer.write_artifacts(result, tmp_path, "summary")) == dataclasses.asdict(before)
