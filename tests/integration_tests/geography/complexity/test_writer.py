import types

import pandas as pd
import pytest
import yaml

from phoibe.geography.complexity.rix.writer import RIXWriter, WriterProfile


class _FakeGeoFrame:
    def __init__(self, n=2):
        self._n = n
        self.written_layers = []

    def to_file(self, path, layer, driver):
        self.written_layers.append(layer)

    def __len__(self):
        return self._n


def _make_result(with_trix=False, index_name="site_id"):
    summary = pd.DataFrame({"rix": [0.1, 0.2], "n_rays": [4, 4]}, index=pd.Index(["a", "b"], name=index_name))
    trix_table = None
    if with_trix:
        trix_table = pd.DataFrame(
            {"transferability": [2], "distance": [1.2], "trix": [5.5], "A": [8.0], "B": [12.0]},
            index=pd.MultiIndex.from_tuples([("a", "ref1")], names=["site_id", "reference_id"]),
        )
    meta = {
        "meta": {"name": "T-RIX assessment", "version": "1.0", "description": "d", "created_at": "2026-01-01"},
        "parameters": {
            "ray": {"n_angles": 4, "R_km": 0.5, "dr_km": 0.05, "crs": "EPSG:32633"},
            "slope": {"slope_critical": 0.1},
        },
        "spatial_context": {"source_dem": {}, "source_ray": {}, "alignment": {}},
        "run": {
            "n_sites": 2,
            "n_references": 1 if with_trix else 0,
            "computed": ["rix_site"] + (["rix_reference", "trix"] if with_trix else []),
            "diagnostics": {"n_sites_with_nans": 0, "transferability_counts": ({2: 1} if with_trix else {})},
        },
    }
    locations_site = _FakeGeoFrame(2)
    locations_reference = _FakeGeoFrame(1) if with_trix else None
    steep_segments = _FakeGeoFrame(0)

    return types.SimpleNamespace(
        summary=summary,
        trix_table=trix_table,
        meta=meta,
        locations_site=locations_site,
        locations_reference=locations_reference,
        steep_segments=steep_segments,
    )


def test_rix_table_csv_roundtrips_through_pandas(tmp_path):
    result = _make_result()
    RIXWriter(result, profile=WriterProfile.SUMMARY).write(directory=tmp_path)

    read_back = pd.read_csv(tmp_path / "rix_table.csv")
    assert list(read_back["site_id"]) == ["a", "b"]
    assert list(read_back["rix"]) == [0.1, 0.2]


def test_rix_table_csv_uses_locations_site_identifier_name(tmp_path):
    result = _make_result(index_name="wea_id")
    RIXWriter(result, profile=WriterProfile.SUMMARY).write(directory=tmp_path)

    read_back = pd.read_csv(tmp_path / "rix_table.csv")
    assert "wea_id" in read_back.columns


def test_manifest_contains_expected_top_level_sections(tmp_path):
    result = _make_result(with_trix=True)
    RIXWriter(result, profile=WriterProfile.SUMMARY).write(directory=tmp_path, project_name="demo-project")

    manifest = yaml.safe_load((tmp_path / "summary.yaml").read_text())
    assert manifest["project_name"] == "demo-project"
    for key in ("meta", "parameters", "spatial_context", "run", "artifacts"):
        assert key in manifest


def test_manifest_lists_trix_file_only_when_trix_table_present(tmp_path):
    result_without = _make_result(with_trix=False)
    RIXWriter(result_without, profile=WriterProfile.SUMMARY).write(directory=tmp_path / "no_trix")
    manifest_without = yaml.safe_load((tmp_path / "no_trix" / "summary.yaml").read_text())
    assert "trix_table" not in manifest_without["artifacts"]["files"]

    result_with = _make_result(with_trix=True)
    RIXWriter(result_with, profile=WriterProfile.SUMMARY).write(directory=tmp_path / "with_trix")
    manifest_with = yaml.safe_load((tmp_path / "with_trix" / "summary.yaml").read_text())
    assert "trix_table" in manifest_with["artifacts"]["files"]


@pytest.mark.filterwarnings("ignore:'crs' was not provided.")
def test_full_profile_writes_geopackage_records_into_manifest(tmp_path):
    result = _make_result(with_trix=True)
    RIXWriter(result, profile=WriterProfile.FULL).write(directory=tmp_path)

    manifest = yaml.safe_load((tmp_path / "summary.yaml").read_text())
    records = manifest["artifacts"]["geopackage_records_per_layer"]
    assert records["locations_site"] == 2
    assert records["locations_reference"] == 1
    assert records["trix"] == 1


def test_full_profile_omits_locations_reference_layer_when_absent(tmp_path):
    result = _make_result(with_trix=False)
    assert result.locations_reference is None

    RIXWriter(result, profile=WriterProfile.FULL).write(directory=tmp_path)

    manifest = yaml.safe_load((tmp_path / "summary.yaml").read_text())
    records = manifest["artifacts"]["geopackage_records_per_layer"]
    assert "locations_reference" not in records
    assert "locations_site" in records
