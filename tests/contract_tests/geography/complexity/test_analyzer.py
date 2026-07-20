import numpy as np
import pytest
import xarray

from phoibe.geography.complexity.rix.analyzer import ResultSummary, TRIXAnalyzer


def _make_config(n_angles=4, R_km=0.5, dr_km=0.05, slope_critical=0.1, crs=None):
    return {
        "name": "T-RIX assessment",
        "version": "1.0",
        "description": "test config",
        "parameters": {
            "n_angles": n_angles,
            "R_km": R_km,
            "dr_km": dr_km,
            "slope_critical": slope_critical,
            "crs": crs,
        },
        "sampler": {"interpolation_method": "linear"},
    }


def _make_planar_dem():
    """Synthetic DEM, elevation = x [m], covering +-2000m around the origin."""
    x = np.arange(-2000, 2001, 50, dtype=float)
    y = np.arange(-2000, 2001, 50, dtype=float)
    xx, yy = np.meshgrid(x, y)
    z = xx.copy()
    return xarray.DataArray(data=z, coords={"x": x, "y": y}, dims=("y", "x"), name="dem")


def _make_locations(ids, xs, ys, crs=None):
    import geopandas as gpd
    import shapely.geometry

    return gpd.GeoDataFrame(
        {"site_id": ids},
        geometry=[shapely.geometry.Point(x, y) for x, y in zip(xs, ys, strict=True)],
        crs=crs,
    )


class TRIXAnalyzerResultContract:
    """Contracts that any TRIXAnalyzer.run() result must satisfy."""

    def test_returns_result_summary(self, result):
        assert isinstance(result, ResultSummary)

    def test_summary_has_required_columns(self, result):
        required = {"rix", "rix_std", "rix_min", "rix_max", "n_rays", "slope_critical"}
        assert required.issubset(set(result.summary.columns))

    def test_summary_row_count_matches_sites(self, result):
        assert len(result.summary) == len(result.locations_site)

    def test_rix_values_in_valid_range(self, result):
        assert result.summary["rix"].between(0.0, 1.0).all()

    def test_ruggedness_roses_row_count_matches_sites(self, result):
        assert len(result.ruggedness_roses) == len(result.locations_site)

    def test_steep_segments_has_required_columns(self, result):
        required = {"theta", "segment_id", "geometry"}
        assert required.issubset(set(result.steep_segments.columns))

    def test_meta_contains_required_top_level_keys(self, result):
        for key in ("meta", "parameters", "spatial_context", "run"):
            assert key in result.meta

    def test_meta_ray_parameters_include_crs(self, result):
        assert "crs" in result.meta["parameters"]["ray"]

    def test_run_diagnostics_present_without_crash(self, result):
        assert "n_sites_with_nans" in result.meta["run"]["diagnostics"]
        assert "transferability_counts" in result.meta["run"]["diagnostics"]


class TestTRIXAnalyzerSiteOnly(TRIXAnalyzerResultContract):
    """Contracts of the single-collection path w/o locations_reference."""

    @pytest.fixture
    def result(self):
        analyzer = TRIXAnalyzer(config=_make_config())
        locations_site = _make_locations(["a", "b", "c"], [0, 100, -100], [0, 0, 100])
        return analyzer.run(dem=_make_planar_dem(), locations_site=locations_site)

    def test_trix_table_is_none(self, result):
        assert result.trix_table is None

    def test_locations_reference_is_none(self, result):
        assert result.locations_reference is None


class TestTRIXAnalyzerWithReference(TRIXAnalyzerResultContract):
    """Contracts of the pairwise TRIX path w/ locations_reference provided."""

    @pytest.fixture
    def result(self):
        analyzer = TRIXAnalyzer(config=_make_config())
        locations_site = _make_locations(["a", "b"], [0, 100], [0, 0])
        locations_reference = _make_locations(["ref1", "ref2"], [500, -500], [500, -500])
        return analyzer.run(
            dem=_make_planar_dem(), locations_site=locations_site, locations_reference=locations_reference
        )

    def test_trix_table_is_not_none(self, result):
        assert result.trix_table is not None

    def test_trix_table_has_required_columns(self, result):
        required = {"transferability", "distance", "trix", "A", "B"}
        assert required.issubset(set(result.trix_table.columns))

    def test_trix_table_row_count_is_cartesian_product(self, result):
        assert len(result.trix_table) == len(result.locations_site) * len(result.locations_reference)

    def test_transferability_values_are_valid_codes(self, result):
        assert result.trix_table["transferability"].isin([0, 1, 2]).all()
