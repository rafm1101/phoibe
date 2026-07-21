import types

import pandas as pd
import pytest

from phoibe.geography.complexity.rix.writer import RIXWriter, WriterProfile


def _make_result(with_trix=False):
    summary = pd.DataFrame({"rix": [0.1, 0.2]}, index=pd.Index(["a", "b"], name="site_id"))
    trix_table = None
    if with_trix:
        trix_table = pd.DataFrame(
            {"transferability": [2], "distance": [1.0], "trix": [5.0], "A": [8.0], "B": [12.0]},
            index=pd.MultiIndex.from_tuples([("a", "ref1")], names=["site_id", "reference_id"]),
        )
    meta = {
        "meta": {"name": "T-RIX assessment", "version": "1.0", "description": "d", "created_at": "now"},
        "parameters": {
            "ray": {"n_angles": 4, "R_km": 0.5, "dr_km": 0.05, "crs": None},
            "slope": {"slope_critical": 0.1},
        },
        "spatial_context": {"source_dem": {}, "source_ray": {}, "alignment": {}},
        "run": {"n_sites": 2, "n_references": 1 if with_trix else 0, "computed": [], "diagnostics": {}},
    }

    class _FakeGeoFrame:
        def __init__(self, crs="EPSG:4326"):
            crs = crs

        def to_file(self, path, layer, driver):
            pass

        def __len__(self):
            return 2

    return types.SimpleNamespace(
        summary=summary,
        trix_table=trix_table,
        meta=meta,
        locations_site=_FakeGeoFrame(),
        locations_reference=_FakeGeoFrame() if with_trix else None,
        steep_segments=_FakeGeoFrame(),
    )


class RIXWriterContract:
    """Contracts that any RIXWriter.write() call must satisfy."""

    def test_manifest_always_written(self, written_dir):
        assert (written_dir / "summary.yaml").exists()

    def test_rix_table_always_written(self, written_dir):
        assert (written_dir / "rix_table.csv").exists()

    def test_manifest_has_project_name_at_top_level(self, written_dir):
        import yaml

        manifest = yaml.safe_load((written_dir / "summary.yaml").read_text())
        assert "project_name" in manifest
        assert "project_name" not in manifest.get("meta", {})


class TestSummaryProfileNoTrix(RIXWriterContract):
    @pytest.fixture
    def written_dir(self, tmp_path):
        writer = RIXWriter(_make_result(with_trix=False), profile=WriterProfile.SUMMARY)
        writer.write(directory=tmp_path)
        return tmp_path

    def test_trix_csv_not_written_given_no_trix_table(self, written_dir):
        assert not (written_dir / "trix.csv").exists()

    def test_geopackage_not_written_for_summary_profile(self, written_dir):
        assert not (written_dir / "rix_details.gpkg").exists()


class TestSummaryProfileWithTrix(RIXWriterContract):
    @pytest.fixture
    def written_dir(self, tmp_path):
        writer = RIXWriter(_make_result(with_trix=True), profile=WriterProfile.SUMMARY)
        writer.write(directory=tmp_path)
        return tmp_path

    def test_trix_csv_written(self, written_dir):
        assert (written_dir / "trix.csv").exists()


@pytest.mark.filterwarnings("ignore:'crs' was not provided.")
class TestFullProfile(RIXWriterContract):
    @pytest.fixture
    def written_dir(self, tmp_path):
        writer = RIXWriter(_make_result(with_trix=True), profile=WriterProfile.FULL)
        writer.write(directory=tmp_path)
        return tmp_path

    def test_geopackage_referenced_in_manifest(self, written_dir):
        import yaml

        manifest = yaml.safe_load((written_dir / "summary.yaml").read_text())
        assert "geopackage" in manifest["artifacts"]["files"]
