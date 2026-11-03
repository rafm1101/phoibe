"""The numbers of 1.1 survive the refactoring: RIX and TRIX of a synthetic dataset equal frozen references.

The references come from an independent oracle (`reference/make_reference.py`, it does not import the package).
The central computation did not change between 1.1 and 2.0, so one set of numbers serves both. The differences the
changelog 2.0.0 documents are pinned on purpose: `elevation_std` is gone, a z coordinate does not change a distance,
NaN along a ray stays in the length (policy "mask").
"""

import importlib.util
import math
import pathlib

import geopandas as gpd
import pandas as pd
import pytest
import shapely.geometry as sg
import xarray
import yaml

from phoibe.geography.complexity.rix import main as main_module
from phoibe.geography.complexity.rix.adapters.writer import FileResultWriter
from phoibe.geography.complexity.rix.computations.fieldsampler import Field

REFERENCE = pathlib.Path(__file__).parent  # / "reference"
TOLERANCE = {"check_exact": False, "rtol": 1e-9, "atol": 1e-12}


def load_oracle():
    spec = importlib.util.spec_from_file_location("make_reference", REFERENCE / "make_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = load_oracle()


def reference_tables():
    return (
        pd.read_csv(REFERENCE / "rix_table.csv"),
        pd.read_csv(REFERENCE / "trix_table.csv"),
        yaml.safe_load((REFERENCE / "diagnostics.yaml").read_text(encoding="utf-8")),
    )


def locations(collection):
    points = [sg.Point(x, y) if z is None else sg.Point(x, y, z) for x, y, z in collection.values()]
    return gpd.GeoDataFrame({"site_id": list(collection)}, geometry=points, crs=oracle.CRS)


@pytest.fixture
def published(product_spec, make_analyzer, tmp_path):
    """Run the analyzer with the defaults of the product on the dataset and publish the summary."""
    x, y, z = oracle.build_dem()
    field = Field.from_array(xarray.DataArray(z, coords={"x": x, "y": y}, dims=("y", "x")))
    assert field.info.crs is None, "rioxarray was imported in this process: run with -m 'not rio'"
    analyzer = make_analyzer(
        product_spec, writer=FileResultWriter(product_spec), n_angles=72, R_km=3.5, dr_km=0.05, slope_critical=0.033
    )
    result = analyzer.run(field, locations(oracle.SITES), locations(oracle.REFERENCES))
    analyzer.publish(result, tmp_path, "summary")
    return result, tmp_path


def test_the_reference_files_are_what_the_oracle_computes():
    expected_rix, expected_trix, expected_diagnostics = oracle.compute()
    rix, trix, diagnostics = reference_tables()
    pd.testing.assert_frame_equal(rix, expected_rix, **TOLERANCE)
    pd.testing.assert_frame_equal(trix, expected_trix, **TOLERANCE)
    assert diagnostics == expected_diagnostics


def test_the_dataset_tells_apart_what_a_regression_must_tell_apart():
    rix, trix, diagnostics = reference_tables()
    assert sorted(trix["transferability"].unique()) == [0, 1, 2]  # every class of transferability
    assert (trix["A"] == 1.5).any() and (trix["B"] == 3.0).any()  # both limits on their floors, and some above them
    assert (trix["A"] > 1.5).any() and (trix["B"] > 3.0).any()
    assert rix["rix"].between(0.05, 0.95).sum() >= 4 and rix["rix"].nunique() == len(
        rix
    )  # values that differ, not 0 or 1
    assert (
        diagnostics["n_locations_with_nans"]["locations_site"] >= 1
        and diagnostics["nan_count"]["locations_reference"] > 0
    )
    s2 = oracle.SITES["S2"]
    assert s2[2] is not None
    flat = math.hypot(s2[0] - oracle.REFERENCES["R1"][0], s2[1] - oracle.REFERENCES["R1"][1]) / 1000
    assert math.sqrt(flat**2 + ((s2[2] - 0.0) / 1000) ** 2) - flat > 0.1  # a 3D distance would be a different number


def test_the_rix_table_equals_the_reference(published):
    _, directory = published
    actual, expected = pd.read_csv(directory / "rix_table.csv"), reference_tables()[0]
    assert list(actual.columns) == list(expected.columns) and "elevation_std" not in actual.columns  # removed in 2.0.0
    pd.testing.assert_frame_equal(actual, expected, **TOLERANCE)


def test_the_trix_table_equals_the_reference(published):
    _, directory = published
    actual, expected = pd.read_csv(directory / "trix_table.csv"), reference_tables()[1]
    assert list(actual.columns) == list(expected.columns)
    assert actual["transferability"].tolist() == expected["transferability"].tolist()  # exactly, not within a tolerance
    pd.testing.assert_frame_equal(actual, expected, **TOLERANCE)


def test_the_counts_of_the_run_equal_the_reference(published):
    result, _ = published
    expected = reference_tables()[2]
    collections = result.record.collections
    assert {name: collections[name].nan_count for name in collections} == expected["nan_count"]
    assert {name: collections[name].n_locations_with_nans for name in collections} == expected["n_locations_with_nans"]
    assert dict(result.record.transferability_counts) == expected["transferability_counts"]


def test_a_z_coordinate_changes_no_distance(published):
    """Version 1.1 measured in 3D for points with a z coordinate, 2.0 in the x/y plane (changelog 2.0.0)."""
    _, directory = published
    trix = pd.read_csv(directory / "trix_table.csv").set_index(["site_id", "reference_id"])
    (x, y, z), (xr, yr, _) = oracle.SITES["S2"], oracle.REFERENCES["R1"]
    assert trix.loc[("S2", "R1"), "distance"] == pytest.approx(math.hypot(x - xr, y - yr) / 1000, rel=1e-12)
    assert abs(math.sqrt(math.hypot(x - xr, y - yr) ** 2 + z**2) / 1000 - trix.loc[("S2", "R1"), "distance"]) > 0.1


@pytest.mark.rio
def test_the_command_line_equals_the_reference_given_files(tmp_path, capsys):
    oracle.export(tmp_path / "input")
    output = tmp_path / "out"
    argv = [
        "run",
        "--dem",
        str(tmp_path / "input" / "DEM.tif"),
        "--sites",
        str(tmp_path / "input" / "sites.gpkg"),
        "--references",
        str(tmp_path / "input" / "references.gpkg"),
        "--output",
        str(output),
    ]

    assert main_module.main(argv) == 0

    rix, trix, diagnostics = reference_tables()
    pd.testing.assert_frame_equal(pd.read_csv(output / "rix_table.csv"), rix, **TOLERANCE)
    pd.testing.assert_frame_equal(pd.read_csv(output / "trix_table.csv"), trix, **TOLERANCE)
    manifest = yaml.safe_load((output / "summary.yaml").read_text(encoding="utf-8"))["run"]["diagnostics"]
    assert (manifest["nan_count"], manifest["n_locations_with_nans"], manifest["transferability_counts"]) == (
        diagnostics["nan_count"],
        diagnostics["n_locations_with_nans"],
        diagnostics["transferability_counts"],
    )
    assert (
        f"Assessed {len(oracle.SITES)} site(s) against {len(oracle.REFERENCES)} reference location(s)"
        in capsys.readouterr().out
    )
