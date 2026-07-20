import numpy as np
import pytest
import xarray
import yaml

from phoibe.geography.complexity.rix.analyzer import TRIXAnalyzer
from phoibe.geography.complexity.rix.config import ANALYZER_DEFAULTS


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
    """Synthetic DEM w/o rio CRS."""
    x = np.arange(-2000, 2001, 50, dtype=float)
    y = np.arange(-2000, 2001, 50, dtype=float)
    xx, yy = np.meshgrid(x, y)
    return xarray.DataArray(data=xx.copy(), coords={"x": x, "y": y}, dims=("y", "x"), name="dem")


def _make_locations(ids=None, xs=(0, 100, -100), ys=(0, 0, 100), crs=None, index_name=None):
    import geopandas as gpd
    import shapely.geometry

    points = [shapely.geometry.Point(x, y) for x, y in zip(xs, ys, strict=True)]
    if ids is not None:
        gdf = gpd.GeoDataFrame({"site_id": ids}, geometry=points, crs=crs)
    else:
        gdf = gpd.GeoDataFrame(geometry=points, crs=crs)
        if index_name is not None:
            gdf.index = gdf.index.rename(index_name)
    return gdf


def test_run_succeeds_wo_given_locations_reference():
    analyzer = TRIXAnalyzer(config=_make_config())
    locations_site = _make_locations(ids=["a", "b", "c"])
    result = analyzer.run(dem=_make_planar_dem(), locations_site=locations_site)

    assert result.locations_reference is None
    assert result.trix_table is None
    assert len(result.summary) == 3


def test_run_has_empty_transferability_counts_wo_given_locations_reference():
    analyzer = TRIXAnalyzer(config=_make_config())
    result = analyzer.run(dem=_make_planar_dem(), locations_site=_make_locations(ids=["a", "b", "c"]))

    assert result.meta["run"]["diagnostics"]["transferability_counts"] == {}


def test_run_sums_transferability_counts_to_total_pairs_given_locations_reference():
    analyzer = TRIXAnalyzer(config=_make_config())
    locations_site = _make_locations(ids=["a", "b"], xs=[0, 100], ys=[0, 0])
    locations_reference = _make_locations(ids=["ref1", "ref2", "ref3"], xs=[500, -500, 0], ys=[500, -500, 500])
    result = analyzer.run(
        dem=_make_planar_dem(), locations_site=locations_site, locations_reference=locations_reference
    )

    counts = result.meta["run"]["diagnostics"]["transferability_counts"]
    assert sum(counts.values()) == len(locations_site) * len(locations_reference)


def test_run_passes_wo_passed_dem_crs_metadata_given_no_rioxarray():
    analyzer = TRIXAnalyzer(config=_make_config())
    result = analyzer.run(dem=_make_planar_dem(), locations_site=_make_locations(ids=["a", "b", "c"]))

    source_dem = result.meta["spatial_context"]["source_dem"]
    assert source_dem["extent"] == []
    assert source_dem["resolution"] == []


def test_locations_site_reindexed_given_site_id_column_present():
    analyzer = TRIXAnalyzer(config=_make_config())
    locations_site = _make_locations(ids=["a", "b", "d"])
    result = analyzer.run(dem=_make_planar_dem(), locations_site=locations_site)

    assert result.locations_site.index.name == "site_id"
    assert list(result.locations_site.index) == ["a", "b", "d"]


def test_locations_site_index_preserved_given_no_site_id_column():
    analyzer = TRIXAnalyzer(config=_make_config())
    locations_site = _make_locations(ids=None, index_name="wea_id")
    result = analyzer.run(dem=_make_planar_dem(), locations_site=locations_site)

    assert result.locations_site.index.name == "wea_id"


def test_run_raises_on_crs_mismatch_between_site_and_reference():
    analyzer = TRIXAnalyzer(config=_make_config())
    locations_site = _make_locations(ids=["a", "b", "d"], crs="EPSG:32633")
    locations_reference = _make_locations(ids=["ref1"], xs=[500], ys=[500], crs="EPSG:4326")

    with pytest.raises(ValueError, match="CRS mismatch"):
        analyzer.run(dem=_make_planar_dem(), locations_site=locations_site, locations_reference=locations_reference)


def test_run_raises_on_duplicate_site_index():
    analyzer = TRIXAnalyzer(config=_make_config())
    locations_site = _make_locations(ids=["a", "a", "a"])

    with pytest.raises(ValueError, match="unique"):
        analyzer.run(dem=_make_planar_dem(), locations_site=locations_site)


def test_from_config_loads_yaml_and_runs(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.dump({"parameters": {"n_angles": 4, "R_km": 0.5, "dr_km": 0.05, "slope_critical": 0.1}})
    )

    analyzer = TRIXAnalyzer.from_config(config_path)
    result = analyzer.run(dem=_make_planar_dem(), locations_site=_make_locations(ids=["a"], xs=[0], ys=[0]))

    assert len(result.summary) == 1


def test_from_config_rejects_unknown_parameter(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.dump({"parameters": {"n_angles": 4, "R_km": 0.5, "dr_km": 0.05, "not_a_real_param": 1}})
    )

    with pytest.raises(ValueError, match="unknown parameter"):
        TRIXAnalyzer.from_config(config_path)


@pytest.mark.parametrize(
    "config",
    [
        {"parameters": {"n_angles": 4, "R_km": 0.5}},
        {"parameters": {"crs": 4326, "dr_km": 0.001}, "sampler": {"interpolation_method": "nearest"}},
    ],
)
def test_from_config_fills_missing_keys_w_analyzer_defaults(tmp_path, config):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config))
    analyzer = TRIXAnalyzer.from_config(config_path)
    keys_top_level = {"name", "version", "description", "parameters", "sampler"}
    keys_parameters = {"n_angles", "R_km", "dr_km", "slope_critical", "crs"}
    keys_sampler = {"interpolation_method"}

    assert keys_top_level.issubset(analyzer._config.keys())
    assert keys_parameters == set(analyzer._config["parameters"].keys())
    assert keys_sampler == set(analyzer._config["sampler"].keys())

    assert all(
        analyzer._config["parameters"][key] == ANALYZER_DEFAULTS["parameters"][key]
        for key in keys_parameters
        if key not in config.get("parameters", {}).keys()
    )
    assert all(
        analyzer._config["sampler"][key] == ANALYZER_DEFAULTS["sampler"][key]
        for key in keys_sampler
        if key not in config.get("sampler", {}).keys()
    )
