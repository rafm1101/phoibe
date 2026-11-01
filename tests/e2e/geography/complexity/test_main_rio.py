"""`main.main(argv)` with real files: a GeoTIFF as DEM, point files as sites and references, a YAML configuration."""

import geopandas as gpd
import numpy as np
import pytest
import shapely.geometry as sg
import xarray
import yaml

from phoibe.geography.complexity.rix import adapters
from phoibe.geography.complexity.rix import main as main_module
from phoibe.geography.complexity.rix.adapters.writer import FileResultWriter

pytestmark = pytest.mark.rio

ORIGIN = (500_000.0, 5_000_000.0)
CRS = "EPSG:32633"


@pytest.fixture
def files(tmp_path):
    import rioxarray  # noqa: F401  # registers the .rio accessor for the whole process

    class Files:
        def dem(self, name="dem.tif", half=4000, step=50, slope=0.01, crs=CRS):
            offsets = np.arange(-half, half + 1, step, dtype=float)
            x, y = ORIGIN[0] + offsets, ORIGIN[1] + offsets[::-1]  # y decreases with the row, as in a GeoTIFF
            da = xarray.DataArray(
                slope * (x[None, :] - ORIGIN[0]) + 0 * y[:, None], coords={"y": y, "x": x}, dims=("y", "x")
            )
            path = tmp_path / name
            da.rio.write_crs(crs).rio.to_raster(path)
            return path

        def locations(self, name, ids, offsets, crs=CRS, geometries=None, layer=None):
            geometries = geometries or [sg.Point(ORIGIN[0] + dx, ORIGIN[1] + dy) for dx, dy in offsets]
            path = tmp_path / name
            gpd.GeoDataFrame({"site_id": list(ids)}, geometry=geometries, crs=crs).to_file(
                path, driver="GPKG", layer=layer
            )
            return path

        def config(self, text):
            path = tmp_path / "run.yaml"
            path.write_text(text, encoding="utf-8")
            return path

    return Files()


@pytest.fixture
def run(tmp_path, files):
    """Arguments of a run with a DEM, two sites and an output directory."""
    output = tmp_path / "out"
    base = [
        "run",
        "--dem",
        str(files.dem()),
        "--sites",
        str(files.locations("sites.gpkg", ("a", "b"), [(0, 0), (200, 100)])),
        "--output",
        str(output),
    ]
    return base, output


def manifest(output):
    return yaml.safe_load((output / "summary.yaml").read_text(encoding="utf-8"))


def test_main_returns_zero_and_writes_the_summary_given_the_defaults(run, capsys):
    argv, output = run
    assert main_module.main(argv) == 0
    assert sorted(path.name for path in output.iterdir()) == ["rix_table.csv", "summary.yaml"]
    text = capsys.readouterr().out
    assert "Assessed 2 site(s)" in text and "Wrote profile 'summary'" in text and "Findings" not in text
    assert (
        manifest(output)["run"]["n_sites"],
        manifest(output)["run"]["project_name"],
        manifest(output)["artifacts"]["profile"],
    ) == (2, "", "summary")


def test_main_returns_zero_and_writes_every_file_of_the_full_profile_given_references_and_a_configuration(
    files, tmp_path
):
    config = files.config(
        "project_name: demo\nparameters:\n  n_angles: 8\n  R_km: 1.0\n  dr_km: 0.05\n  slope_critical: 0.1\n"
    )
    argv = [
        "run",
        "--dem",
        str(files.dem()),
        "--sites",
        str(files.locations("sites.gpkg", ("a", "b"), [(0, 0), (200, 100)])),
        "--references",
        str(files.locations("refs.gpkg", ("r1", "r2", "r3"), [(500, 500), (-500, -500), (0, 500)])),
        "--output",
        str(tmp_path / "out"),
        "--config",
        str(config),
        "--profile",
        "full",
    ]
    assert main_module.main(argv) == 0
    output = tmp_path / "out"
    assert sorted(path.name for path in output.iterdir()) == [
        "rix_details.gpkg",
        "rix_table.csv",
        "summary.yaml",
        "trix.csv",
    ]
    content = manifest(output)
    assert (content["run"]["n_sites"], content["run"]["n_references"], content["run"]["project_name"]) == (2, 3, "demo")
    assert (
        content["parameters"]["ray"]["n_angles"] == 8
        and "locked_parameter_changed" in content["run"]["diagnostics"]["finding_counts"]
    )


def test_main_returns_the_project_name_and_the_dem_description_of_the_command_line(run, files):
    argv, output = run
    argv += [
        "--config",
        str(files.config("project_name: from the file\n")),
        "--project-name",
        "from the line",
        "--dem-meta",
        "name=demo",
        "--dem-meta",
        "source=test",
    ]
    assert main_module.main(argv) == 0
    content = manifest(output)
    assert content["run"]["project_name"] == "from the line"  # the command line wins
    dem = content["spatial_context"]["source_dem"]
    assert (dem["name"], dem["source"], dem["crs"], dem["resolution"]) == (
        "demo",
        "test",
        CRS,
        {"dx": 50.0, "dy": -50.0},
    )


def test_main_returns_zero_given_the_layers_of_files_with_several_layers(files, tmp_path):
    files.locations("sites.gpkg", ("x",), [(900, 900)], layer="other")
    files.locations("sites.gpkg", ("a", "b"), [(0, 0), (200, 100)], layer="wanted")
    files.locations("refs.gpkg", ("y",), [(900, -900)], layer="other")
    files.locations("refs.gpkg", ("r1", "r2", "r3"), [(500, 500), (-500, -500), (0, 500)], layer="wanted")
    argv = [
        "run",
        "--dem",
        str(files.dem()),
        "--sites",
        str(tmp_path / "sites.gpkg"),
        "--sites-layer",
        "wanted",
        "--references",
        str(tmp_path / "refs.gpkg"),
        "--references-layer",
        "wanted",
        "--output",
        str(tmp_path / "out"),
        "-vv",
    ]
    assert main_module.main(argv) == 0
    content = manifest(tmp_path / "out")
    assert (content["run"]["n_sites"], content["run"]["n_references"]) == (2, 3)


@pytest.mark.parametrize(
    "case",
    ["polygon", "unknown parameter", "duplicate identifiers"],
)
def test_main_returns_one_and_writes_nothing_given_a_problem_with_the_inputs(case, files, tmp_path, caplog):
    sites = {
        "polygon": lambda: files.locations("sites.gpkg", ("a",), [], geometries=[sg.box(0, 0, 1, 1)]),
        "unknown parameter": lambda: files.locations("sites.gpkg", ("a",), [(0, 0)]),
        "duplicate identifiers": lambda: files.locations("sites.gpkg", ("a", "a"), [(0, 0), (100, 0)]),
    }[case]()
    argv = ["run", "--dem", str(files.dem()), "--sites", str(sites), "--output", str(tmp_path / "out")]
    if case == "unknown parameter":
        argv += ["--config", str(files.config("parameters:\n  nonsense: 1\n"))]
    assert main_module.main(argv) == 1
    assert not (tmp_path / "out").exists() and caplog.records[-1].levelname == "ERROR"


def test_main_returns_three_given_a_dem_that_does_not_exist(run, tmp_path, caplog):
    argv, output = run
    argv[argv.index("--dem") + 1] = str(tmp_path / "missing.tif")
    assert main_module.main(argv) == 3
    assert "Could not read a DEM" in caplog.text and not output.exists()


def test_main_returns_three_given_an_output_that_is_a_file(run, tmp_path):
    argv, output = run
    output.write_text("in the way")
    assert main_module.main(argv) == 3


def test_main_returns_four_and_writes_no_manifest_given_a_writer_that_violates_the_contract(run, monkeypatch):
    class Deviating(FileResultWriter):
        def write_artifacts(self, result, target, profile):
            written = super().write_artifacts(result, target, profile)
            return written.__class__(
                **{**written.__dict__, "table_columns": {"rix_table": written.table_columns["rix_table"][:-1]}}
            )

    monkeypatch.setattr(adapters, "FileResultWriter", Deviating)
    argv, output = run
    assert main_module.main(argv) == 4
    assert not (output / "summary.yaml").exists()


def test_main_returns_seventy_given_a_defect(run, monkeypatch, caplog):
    def fail(path, **kwargs):
        raise RuntimeError("a defect")

    monkeypatch.setattr(adapters, "read_field", fail)
    argv, _ = run
    assert main_module.main(argv) == 70 and "Unexpected error" in caplog.text
