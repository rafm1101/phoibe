import logging

import numpy as np
import pyproj
import pytest
import shapely.geometry

from phoibe.geography.complexity.rix.computations.geometry import RayGeometry

UTM = pyproj.CRS.from_epsg(32633)
WGS84 = pyproj.CRS.from_epsg(4326)
COMPASS = {0.0: (0.0, 1.0), 90.0: (1.0, 0.0), 180.0: (0.0, -1.0), 270.0: (-1.0, 0.0)}  # theta: (x, y) direction


@pytest.mark.parametrize(
    "R_km, dr_km, expected_match",
    [
        (np.nan, 0.1, "R_km contains NaN or infinite values"),
        (np.inf, 0.1, "R_km contains NaN or infinite values"),
        (-7, 0.1, "R_km contains negative values"),
        (13, np.nan, "dr_km contains NaN or infinite values"),
        (13, -1, "dr_km contains non-positive values"),
        (13, 0.0, "dr_km contains non-positive values"),
        (13, 23, "dr_km must stay below R_km"),
        (13, 13, "dr_km must stay below R_km"),
    ],
)
def test_from_compass_regular_raises_given_invalid_inputs(origin, R_km, dr_km, expected_match):
    with pytest.raises(ValueError, match=expected_match):
        RayGeometry.from_compass_regular(location=origin, theta=270, R_km=R_km, dr_km=dr_km)


@pytest.mark.parametrize("R_km, dr_km, expected_spacing, expected_last", [(2, 0.5, 500, 2000), (5, 0.1, 100, 5000)])
def test_from_compass_regular_returns_equidistant_points(origin, R_km, dr_km, expected_spacing, expected_last):
    ray = RayGeometry.from_compass_regular(location=origin, theta=270, R_km=R_km, dr_km=dr_km)
    assert ray.r_m[0] == 0
    assert ray.r_m[-1] == pytest.approx(expected_last)
    np.testing.assert_allclose(np.diff(ray.r_m), expected_spacing)
    np.testing.assert_allclose(np.hypot(np.diff(ray.xs), np.diff(ray.ys)), expected_spacing)


@pytest.mark.parametrize("R_km, dr_km", [(0.3, 0.1), (0.7, 0.1), (3.5, 0.05), (1.0, 0.1), (0.1, 0.01)])
def test_from_compass_regular_returns_the_full_grid_given_a_multiple_without_warning(caplog, origin, R_km, dr_km):
    with caplog.at_level(logging.WARNING):
        ray = RayGeometry.from_compass_regular(location=origin, theta=0.0, R_km=R_km, dr_km=dr_km)
    assert len(ray.r_m) == round(R_km / dr_km) + 1
    assert ray.r_m[-1] == pytest.approx(R_km * 1000)
    assert caplog.records == []


def test_from_compass_regular_returns_a_truncated_grid_and_warns_given_a_remainder(caplog, origin):
    with caplog.at_level(logging.WARNING):
        ray = RayGeometry.from_compass_regular(location=origin, theta=0.0, R_km=10.0, dr_km=3.0)
    np.testing.assert_allclose(ray.r_m, [0, 3000, 6000, 9000])
    assert [record.levelno for record in caplog.records] == [logging.WARNING]
    assert "Ray for 0.0 truncated" in caplog.records[0].message


@pytest.mark.parametrize("location", [shapely.geometry.Point(0.0, 0.0), shapely.geometry.Point(100.0, -200.0)])
@pytest.mark.parametrize("theta, direction", COMPASS.items())
def test_from_compass_returns_points_in_the_compass_direction(location, theta, direction):
    r_m = np.array([0.0, 100.0, 250.0])
    ray = RayGeometry.from_compass(location=location, theta=theta, r_m=r_m, crs=None)
    np.testing.assert_allclose(ray.xs, location.x + direction[0] * r_m, atol=1e-9)
    np.testing.assert_allclose(ray.ys, location.y + direction[1] * r_m, atol=1e-9)
    np.testing.assert_allclose(ray.r_m, r_m)
    assert ray.theta == theta


@pytest.mark.parametrize("theta", [45.0, 135.0])
def test_from_compass_regular_returns_the_points_of_from_compass_given_an_equivalent_grid(origin, theta):
    regular = RayGeometry.from_compass_regular(location=origin, theta=theta, R_km=1.0, dr_km=0.25)
    explicit = RayGeometry.from_compass(location=origin, theta=theta, r_m=regular.r_m, crs=None)
    np.testing.assert_allclose(explicit.xs, regular.xs)
    np.testing.assert_allclose(explicit.ys, regular.ys)
    assert explicit.theta == regular.theta


@pytest.mark.parametrize(
    "r_m, expected_match",
    [
        ([0.0, 100.0, 50.0], "contains non-positive values"),
        ([0.0, 100.0, 100.0], "contains non-positive values"),
        ([-50.0, 0.0, 100.0], "contains negative values"),
    ],
)
def test_from_compass_raises_given_r_m_not_strictly_increasing_or_negative(origin, r_m, expected_match):
    with pytest.raises(ValueError, match=expected_match):
        RayGeometry.from_compass(location=origin, theta=0.0, r_m=np.array(r_m), crs=None)


def test_ray_geometry_raises_given_non_increasing_r_m_on_direct_construction(origin):
    with pytest.raises(ValueError, match="contains non-positive values"):
        RayGeometry(
            location=origin,
            crs=None,
            theta=0.0,
            r_m=np.array([100.0, 50.0]),
            xs=np.array([0.0, 0.0]),
            ys=np.array([100.0, 50.0]),
        )


@pytest.mark.parametrize("crs, expected", [(None, None), ("EPSG:32633", UTM), (32633, UTM)])
def test_from_compass_returns_the_crs_as_pyproj_crs(origin, crs, expected):
    result = RayGeometry.from_compass(location=origin, theta=0.0, r_m=[0.0, 1.0], crs=crs).crs
    assert result == expected
    assert result is None or isinstance(result, pyproj.CRS)  # pyproj also calls a string equal


@pytest.mark.parametrize(
    "ray_crs, target_crs, expected_fragments",
    [
        (None, None, ["Ray-CRS None", "DEM-CRS None"]),
        (None, UTM, ["Ray-CRS None", "DEM-CRS EPSG:32633"]),
        (UTM, None, ["Ray-CRS EPSG:32633", "DEM-CRS None"]),
        (UTM, UTM, ["same CRS EPSG:32633"]),
    ],
    ids=["both missing", "ray crs missing", "target missing", "equal"],
)
def test_to_crs_returns_the_same_ray_given_a_missing_or_equal_crs(origin, ray_crs, target_crs, expected_fragments):
    ray = RayGeometry.from_compass_regular(location=origin, theta=0.0, R_km=1.0, dr_km=0.5, crs=ray_crs)
    result, message = ray.to_crs(target_crs)
    assert result is ray
    assert all(fragment in message for fragment in expected_fragments)


def test_to_crs_returns_a_warning_given_an_equal_geographic_crs(origin):
    ray = RayGeometry.from_compass_regular(location=origin, theta=0.0, R_km=1.0, dr_km=0.5, crs=WGS84)
    _, message = ray.to_crs(WGS84)
    assert "No guarantee unless sites are presented in a metric CRS." in message


def test_to_crs_returns_a_transformed_ray_given_a_different_crs():
    origin = shapely.geometry.Point(500_000.0, 0.0)  # the central meridian of UTM zone 33N is 15°E
    ray = RayGeometry.from_compass_regular(location=origin, theta=0.0, R_km=1.0, dr_km=0.5, crs=UTM)

    result, message = ray.to_crs(WGS84)

    assert result is not ray
    assert result.crs == WGS84
    assert result.location.x == pytest.approx(15.0, abs=1e-9)
    assert result.location.y == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(result.xs, 15.0, atol=1e-9)
    xs_back, ys_back = pyproj.Transformer.from_crs(WGS84, UTM, always_xy=True).transform(result.xs, result.ys)
    np.testing.assert_allclose(xs_back, ray.xs, atol=1e-6)
    np.testing.assert_allclose(ys_back, ray.ys, atol=1e-6)
    np.testing.assert_allclose(result.r_m, ray.r_m)
    assert result.theta == ray.theta
    assert "Transformed ray CRS EPSG:32633 to DEM CRS EPSG:4326" in message
