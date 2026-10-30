import geopandas as gpd
import pytest
import shapely.geometry as sg

from phoibe.geography.complexity.rix.adapters.locations_reader import read_locations
from phoibe.geography.complexity.rix.pipelines import errors


def points(ids=("a", "b"), crs="EPSG:2056", **columns):
    return gpd.GeoDataFrame(
        {"site_id": list(ids), **columns}, geometry=[sg.Point(k, k) for k in range(len(ids))], crs=crs
    )


def test_read_locations_returns_the_columns_and_the_crs_as_they_are(tmp_path):
    points(height=[10.0, 20.0]).to_file(tmp_path / "p.gpkg", driver="GPKG")
    locations = read_locations(tmp_path / "p.gpkg")
    assert (list(locations.columns), locations.crs.to_string(), list(locations["site_id"])) == (
        ["site_id", "height", "geometry"],
        "EPSG:2056",
        ["a", "b"],
    )


def test_read_locations_returns_the_first_layer_by_default_and_another_by_name(tmp_path):
    points(("first",)).to_file(tmp_path / "p.gpkg", layer="first", driver="GPKG")
    points(("second1", "second2")).to_file(tmp_path / "p.gpkg", layer="second", driver="GPKG")
    assert list(read_locations(tmp_path / "p.gpkg")["site_id"]) == ["first"]
    assert list(read_locations(tmp_path / "p.gpkg", layer="second")["site_id"]) == ["second1", "second2"]


def test_read_locations_returns_the_locations_of_any_vector_format(tmp_path):
    points(crs="EPSG:4326").to_file(tmp_path / "p.geojson", driver="GeoJSON")
    assert len(read_locations(tmp_path / "p.geojson")) == 2


def test_read_locations_returns_points_with_a_z_coordinate(tmp_path):
    gpd.GeoDataFrame({"site_id": ["a"]}, geometry=[sg.Point(0, 0, 5)], crs="EPSG:2056").to_file(
        tmp_path / "p.gpkg", driver="GPKG"
    )
    assert read_locations(tmp_path / "p.gpkg").geometry.iloc[0].has_z


def test_read_locations_raises_a_port_error_given_a_missing_file(tmp_path):
    with pytest.raises(errors.PortError, match="Could not read locations"):
        read_locations(tmp_path / "missing.gpkg")


def test_read_locations_raises_a_port_error_given_a_missing_layer(tmp_path):
    points().to_file(tmp_path / "p.gpkg", driver="GPKG")
    with pytest.raises(errors.PortError, match="Could not read locations"):
        read_locations(tmp_path / "p.gpkg", layer="missing")


def test_read_locations_raises_given_a_file_without_locations(tmp_path):
    (tmp_path / "empty.geojson").write_text('{"type": "FeatureCollection", "features": []}')
    with pytest.raises(errors.InputRequirementError, match="contains no locations"):
        read_locations(tmp_path / "empty.geojson")


@pytest.mark.parametrize(
    "geometries, expected",
    [
        ([sg.LineString([(0, 0), (1, 1)])], ["LineString"]),
        ([sg.Point(0, 0), sg.Polygon([(0, 0), (1, 0), (1, 1)])], ["Polygon"]),
        ([sg.MultiPoint([(0, 0), (1, 1)])], ["MultiPoint"]),
        ([sg.Polygon([(0, 0), (1, 0), (1, 1)]), sg.LineString([(0, 0), (1, 1)])], ["LineString", "Polygon"]),  # sorted
    ],
    ids=["line", "point and polygon", "multipoint", "two kinds"],
)
def test_read_locations_raises_given_anything_but_points_and_names_what_it_found(tmp_path, geometries, expected):
    gpd.GeoDataFrame({"site_id": range(len(geometries))}, geometry=geometries, crs="EPSG:2056").to_file(
        tmp_path / "x.geojson", driver="GeoJSON"
    )
    with pytest.raises(errors.InputRequirementError, match="point locations only") as error:
        read_locations(tmp_path / "x.geojson")
    assert str(expected) in str(error.value)


def test_read_locations_raises_given_a_location_without_geometry(tmp_path):
    (tmp_path / "null.geojson").write_text(
        '{"type": "FeatureCollection", "features": ['
        '{"type": "Feature", "properties": {"a": 1}, "geometry": null},'
        '{"type": "Feature", "properties": {"a": 2}, "geometry": {"type": "Point", "coordinates": [1, 2]}}]}'
    )
    with pytest.raises(errors.InputRequirementError, match="missing geometry"):
        read_locations(tmp_path / "null.geojson")
