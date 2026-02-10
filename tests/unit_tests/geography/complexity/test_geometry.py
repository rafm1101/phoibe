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
    assert np.all(ys >= origin.northing - 1e-7)
    assert np.all(xs <= origin.easting + 1e-7)
    assert np.allclose(ys, origin.northing) or np.allclose(xs, origin.easting)


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
