import logging

import numpy as np
import pytest

from phoibe.geography.complexity.rix.geometry import RayGeometry


@pytest.mark.parametrize(
    "theta, R_km, dr_km, expected_match",
    [
        (270, np.nan, 0.1, "R_km contains NaN or infinite values"),
        (270, np.inf, 0.1, "R_km contains NaN or infinite values"),
        (270, -7, 0.1, "R_km contains negative values"),
        (270, 13, np.nan, "dr_km contains NaN or infinite values"),
        (270, 13, -1, "dr_km contains non-positive values"),
        (270, 13, 0.0, "dr_km contains non-positive values"),
        (270, 13, 23, "dr_km must not exceed R_km"),
    ],
)
def test_raygeometry_from_compass_regular_rejects_invalid_inputs(origin, theta, R_km, dr_km, expected_match):
    with pytest.raises(ValueError, match=expected_match):
        RayGeometry.from_compass_regular(location=origin, theta=theta, R_km=R_km, dr_km=dr_km)


@pytest.mark.parametrize(
    "theta, R_km, dr_km, expected_spacing, expected_last", [(270, 2, 0.5, 500, 2000), (270, 5, 0.1, 100, 5000)]
)
def test_raygeometry_returns_correct_spacing(origin, theta, R_km, dr_km, expected_spacing, expected_last):
    ray = RayGeometry.from_compass_regular(location=origin, theta=theta, R_km=R_km, dr_km=dr_km)
    grid = ray.r_m
    assert np.allclose(np.diff(grid), expected_spacing)
    assert np.allclose(grid[0], 0)
    assert np.allclose(grid[-1], expected_last)
    assert np.allclose(np.abs(np.diff(ray.xs)) + np.abs(np.diff(ray.ys)), expected_spacing)


@pytest.mark.parametrize("theta, R_km, dr_km", [(0, 1.0, 0.5), (270, 1.0, 0.5)])
def test_raygeometry_returns_correct_orientation(origin, theta, R_km, dr_km):
    ray = RayGeometry.from_compass_regular(location=origin, theta=theta, R_km=R_km, dr_km=dr_km)
    xs, ys = ray.xs, ray.ys
    assert np.all(ys >= origin.y - 1e-7)
    assert np.all(xs <= origin.x + 1e-7)
    assert np.allclose(ys, origin.y) or np.allclose(xs, origin.x)


@pytest.mark.parametrize(
    "theta, R_km, dr_km, expected_grid",
    [(0, 10.0, 3.0, [0, 3000, 6000, 9000])],
)
def test_raygeometry_truncates_with_warning(caplog, origin, theta, R_km, dr_km, expected_grid):
    ray = RayGeometry.from_compass_regular(location=origin, theta=theta, R_km=R_km, dr_km=dr_km)

    with caplog.at_level(logging.WARNING):
        r = ray.r_m

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert np.allclose(r, expected_grid)
    assert len(warnings) == 1
    assert "Ray for 0.0 truncated as" in warnings[0].message


@pytest.mark.parametrize(
    "theta, r_m, expected_xs, expected_ys",
    [
        (90.0, np.array([0.0, 100.0, 250.0]), [0.0, 100.0, 250.0], [0.0, 0.0, 0.0]),
        (180.0, np.array([0.0, 100.0, 250.0]), [0.0, 0.0, 0.0], [0.0, -100.0, -250.0]),
    ],
)
def test_from_compass_computes_correct_cartesian_coordinates(origin, theta, r_m, expected_xs, expected_ys):
    ray = RayGeometry.from_compass(location=origin, theta=theta, r_m=r_m, crs=None)

    np.testing.assert_allclose(ray.xs, expected_xs, atol=1e-9)
    np.testing.assert_allclose(ray.ys, expected_ys, atol=1e-9)
    assert np.allclose(ray.r_m, r_m)


@pytest.mark.parametrize("theta, R_km, dr_km", [(45.0, 1.0, 0.25), (135.0, 1.0, 0.25)])
def test_from_compass_matches_from_compass_regular_for_equivalent_grid(origin, theta, R_km, dr_km):
    regular = RayGeometry.from_compass_regular(location=origin, theta=theta, R_km=R_km, dr_km=dr_km)
    explicit = RayGeometry.from_compass(location=origin, theta=theta, r_m=regular.r_m, crs=None)

    np.testing.assert_allclose(explicit.xs, regular.xs)
    np.testing.assert_allclose(explicit.ys, regular.ys)
    assert explicit.theta == regular.theta


def test_from_compass_raises_valueerror_given_non_increasing_r_m(origin):
    with pytest.raises(ValueError, match="contains non-positive values"):
        RayGeometry.from_compass(location=origin, theta=0.0, r_m=np.array([0.0, 100.0, 50.0]), crs=None)


def test_from_compass_raises_valueerror_given_negative_radius(origin):
    with pytest.raises(ValueError, match="contains negative values"):
        RayGeometry.from_compass(location=origin, theta=0.0, r_m=np.array([-50.0, 0.0, 100.0]), crs=None)


def test_direct_construction_raises_valueerror_given_non_increasing_r_m(origin):
    with pytest.raises(ValueError, match="contains non-positive values"):
        RayGeometry(
            location=origin,
            crs=None,
            theta=0.0,
            r_m=np.array([100.0, 50.0]),
            xs=np.array([0.0, 0.0]),
            ys=np.array([100.0, 50.0]),
        )


def test_to_crs_returns_self_unchanged_given_both_crs_are_none(origin):
    ray = RayGeometry.from_compass_regular(location=origin, theta=0.0, R_km=1.0, dr_km=0.5)
    result, message = ray.to_crs(None)

    assert result is ray
    assert message == "Assume all coordinates are in the same CRS. Ray-CRS None. DEM-CRS None. "


def test_to_crs_returns_self_unchanged_given_ray_crs_is_none(origin, crs_utm33n):
    ray = RayGeometry.from_compass_regular(location=origin, theta=0.0, R_km=1.0, dr_km=0.5)
    result, message = ray.to_crs(crs_utm33n)

    assert result is ray
    assert "Ray-CRS None." in message
    assert f"DEM-CRS {crs_utm33n.to_string()}." in message


def test_to_crs_returns_self_unchanged_given_target_crs_is_none(origin, crs_utm33n):
    ray = RayGeometry.from_compass_regular(location=origin, theta=0.0, R_km=1.0, dr_km=0.5, crs=crs_utm33n)
    result, message = ray.to_crs(None)

    assert result is ray
    assert f"Ray-CRS {crs_utm33n.to_string()}." in message
    assert "DEM-CRS None." in message


def test_to_crs_returns_self_unchanged_given_crs_already_matches(origin, crs_utm33n):
    ray = RayGeometry.from_compass_regular(location=origin, theta=0.0, R_km=1.0, dr_km=0.5, crs=crs_utm33n)
    result, message = ray.to_crs(crs_utm33n)

    assert result is ray
    assert message == f"All coordinates are in the same CRS {crs_utm33n.to_string()}. "


def test_to_crs_warns_about_distortion_given_matching_crs_is_geographic(origin, crs_wgs84):
    ray = RayGeometry.from_compass_regular(location=origin, theta=0.0, R_km=1.0, dr_km=0.5, crs=crs_wgs84)
    _, message = ray.to_crs(crs_wgs84)

    assert "No guarantee unless sites are presented in a metric CRS." in message


def test_to_crs_transforms_and_returns_new_ray_given_differing_crs(origin, crs_wgs84, crs_utm33n):
    ray = RayGeometry.from_compass_regular(location=origin, theta=0.0, R_km=1.0, dr_km=0.5, crs=crs_utm33n)
    result, message = ray.to_crs(crs_wgs84)

    assert result is not ray
    assert result.crs == crs_wgs84
    assert result.theta == ray.theta
    np.testing.assert_allclose(result.r_m, ray.r_m)
    assert message == f"Transformed ray CRS {crs_utm33n.to_string()} to DEM CRS {crs_wgs84.to_string()} for sampling."
