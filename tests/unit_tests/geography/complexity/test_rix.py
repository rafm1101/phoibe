import numpy as np
import pytest

from phoibe.geography.complexity.rix import RayGeometry

# from phoibe.geography.complexity.rix import RayProfile

# from phoibe.geography.complexity.rix import compute_radial_rix


@pytest.mark.parametrize(
    "theta, R_km, dr_km, expected_match",
    [
        (270, np.nan, 0.1, "R_km contains NaN or infinite values"),
        (270, np.inf, 0.1, "R_km contains NaN or infinite values"),
        (270, -7, 0.1, "R_km contains negative values"),
        (270, 13, np.nan, "dr_km contains NaN or infinite values"),
        (270, 13, -1, "dr_km contains negative values"),
        (270, 13, 23, "dr_km must not exceed R_km"),
    ],
)
def test_raygeometry_rejects_invalid_inputs(origin, theta, R_km, dr_km, expected_match):
    with pytest.raises(ValueError, match=expected_match):
        RayGeometry(origin, theta, R_km, dr_km)


@pytest.mark.parametrize(
    "theta, R_km, dr_km, expected_spacing, expected_last",
    [(270, 2, 0.5, 500, 2000), (270, 5, 0.1, 100, 5000)],
)
def test_raygeometry_returns_correct_spacing(origin, theta, R_km, dr_km, expected_spacing, expected_last):
    ray_geometry = RayGeometry(origin, theta, R_km, dr_km)
    grid = ray_geometry.r_m
    assert np.allclose(np.diff(grid), expected_spacing)
    assert np.allclose(grid[-1], expected_last)


@pytest.mark.parametrize(
    "theta, R_km, dr_km, expected_spacing, expected_last",
    [(0, 1.0, 0.5, 500, 2000), (270, 1.0, 0.5, 100, 5000)],
)
def test_raygeometry_returns_correct_orientation(origin, theta, R_km, dr_km, expected_spacing, expected_last):
    ray_geometry = RayGeometry(origin, theta, R_km, dr_km)
    xs, ys = ray_geometry.xs, ray_geometry.ys
    assert np.all(ys >= origin.northing - 1e-7)
    assert np.all(xs <= origin.easting)
    assert np.allclose(ys, origin.northing) or np.allclose(xs, origin.easting)
