import logging

import numpy as np
import pytest

from phoibe.geography.complexity.rix import NaNPolicy
from phoibe.geography.complexity.rix import RayGeometry
from phoibe.geography.complexity.rix import RayProfile
from phoibe.geography.complexity.rix import RegularRayProfile

# from phoibe.geography.complexity.rix import compute_radial_rix


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
        RayGeometry.from_compass_regular(origin, theta, R_km, dr_km)


@pytest.mark.parametrize(
    "theta, R_km, dr_km, expected_spacing, expected_last",
    [(270, 2, 0.5, 500, 2000), (270, 5, 0.1, 100, 5000)],
)
def test_raygeometry_returns_correct_spacing(origin, theta, R_km, dr_km, expected_spacing, expected_last):
    ray = RayGeometry.from_compass_regular(origin, theta, R_km, dr_km)
    grid = ray.r_m
    assert np.allclose(np.diff(grid), expected_spacing)
    assert np.allclose(grid[0], 0)
    assert np.allclose(grid[-1], expected_last)


@pytest.mark.parametrize(
    "theta, R_km, dr_km, expected_spacing, expected_last",
    [(0, 1.0, 0.5, 500, 2000), (270, 1.0, 0.5, 100, 5000)],
)
def test_raygeometry_returns_correct_orientation(origin, theta, R_km, dr_km, expected_spacing, expected_last):
    ray = RayGeometry.from_compass_regular(origin, theta, R_km, dr_km)
    xs, ys = ray.xs, ray.ys
    assert np.all(ys >= origin.northing - 1e-7)
    assert np.all(xs <= origin.easting)
    assert np.allclose(ys, origin.northing) or np.allclose(xs, origin.easting)


@pytest.mark.parametrize(
    "theta, R_km, dr_km, expected_grid",
    [(0, 10.0, 3.0, [0, 3000, 6000, 9000])],
)
def test_raygeometry_truncates_with_warning(caplog, origin, theta, R_km, dr_km, expected_grid):
    ray = RayGeometry.from_compass_regular(origin, theta, R_km, dr_km)

    with caplog.at_level(logging.WARNING):
        r = ray.r_m

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert np.allclose(r, expected_grid)
    assert len(warnings) == 1
    assert "Ray for 0.0 truncated as" in warnings[0].message


class DummySampler:
    def __init__(self, z):
        self._z = np.asarray(z, dtype=float)

    def sample(self, xs, ys):
        assert len(xs) == len(self._z)
        return self._z


@pytest.fixture
def dummy_sampler(request):
    z = request.param
    return DummySampler(z=z)


@pytest.fixture
def profile_sampler():
    z = [0, 1, 1, 1, 0, 0, 0, 2, 2, 1, 0]
    return DummySampler(z=z)


@pytest.fixture
def invalid_profile_sampler(request):
    if request.param == "single-intern":
        z = [0, 1, 1, np.nan, 1, 0]
    return DummySampler(z=z)


@pytest.mark.parametrize(
    "ray_01km, dummy_sampler, nan_policy, expected_n_slopes",
    [
        (0.025, [0, 1, 2, 3, 4], NaNPolicy.ERROR, 4),
    ],
    indirect=["ray_01km", "dummy_sampler"],
)
def test_ray_profile_contracts_lengths(ray_01km, dummy_sampler, nan_policy, expected_n_slopes):
    ray_profile = RegularRayProfile(ray_01km, dummy_sampler, nan_policy)
    assert len(ray_profile.slopes) == expected_n_slopes


@pytest.mark.parametrize(
    "ray_01km, nan_policy, critical_slope, expected_mask, expected_rix",
    [
        (0.01, NaNPolicy.ERROR, 0.2, [False] * 10, 0.0),
        (0.01, NaNPolicy.ERROR, 0.199, [False, False, False, False, False, False, True, False, False, False], 0.1),
        (0.01, NaNPolicy.ERROR, 0.099, [True, False, False, True, False, False, True, False, True, True], 0.5),
        (0.01, NaNPolicy.TRUNCATE, 0.2, [False] * 10, 0.0),
        (0.01, NaNPolicy.TRUNCATE, 0.199, [False, False, False, False, False, False, True, False, False, False], 0.1),
        (0.01, NaNPolicy.TRUNCATE, 0.099, [True, False, False, True, False, False, True, False, True, True], 0.5),
    ],
    indirect=["ray_01km"],
)
def test_ray_profile_returns_correct_intermediate_values_given_valid_profile(
    ray_01km, profile_sampler, nan_policy, critical_slope, expected_mask, expected_rix
):
    ray_profile = RegularRayProfile(ray_01km, profile_sampler, nan_policy)
    assert np.allclose(ray_profile.slopes, [0.1, 0.0, 0.0, -0.1, 0.0, 0.0, 0.2, 0.0, -0.1, -0.1])

    mask = ray_profile.steep_mask(critical_slope)
    assert np.all(mask == expected_mask)
    assert np.isclose(ray_profile.rix(critical_slope), expected_rix)


@pytest.mark.parametrize(
    "ray_01km, invalid_profile_sampler, nan_policy, critical_slope, expected_slopes, expected_mask, expected_rix",
    [
        (0.02, "single-intern", NaNPolicy.TRUNCATE, 0.05, [0.05, 0.0], [False] * 2, 0.0),
        (0.02, "single-intern", NaNPolicy.TRUNCATE, 0.049, [0.05, 0.0], [True, False], 0.5),
        (0.02, "single-intern", NaNPolicy.MASK, 0.05, [0.05, 0.0, np.nan, np.nan, -0.05], [False] * 5, 0.0),
        (
            0.02,
            "single-intern",
            NaNPolicy.MASK,
            0.049,
            [0.05, 0.0, np.nan, np.nan, -0.05],
            [True] + [False] * 3 + [True],
            0.4,
        ),
    ],
    indirect=["ray_01km", "invalid_profile_sampler"],
)
def test_ray_profile_returns_correct_intermediate_values(
    ray_01km, invalid_profile_sampler, nan_policy, critical_slope, expected_slopes, expected_mask, expected_rix
):
    ray_profile = RegularRayProfile(ray_01km, invalid_profile_sampler, nan_policy)
    assert np.allclose(ray_profile.slopes, expected_slopes, equal_nan=True)

    mask = ray_profile.steep_mask(critical_slope)
    assert np.all(mask == expected_mask)
    assert np.isclose(ray_profile.rix(critical_slope), expected_rix)


class DummyProfile(RayProfile):
    def __init__(self, slopes, segment_lengths):
        self._slopes = np.asarray(slopes, dtype=float)
        self._segment_lengths = np.asarray(segment_lengths, dtype=float)

    @property
    def slopes(self):
        return self._slopes

    @property
    def segment_lengths(self):
        return self._segment_lengths


@pytest.fixture
def dummy_profile(request):
    slopes = request.param[0]
    segment_lengths = request.param[1]
    return DummyProfile(slopes, segment_lengths)


# @pytest.mark.parametrize(
#     "dummy_profile, slope_critical1, slope_critical2",
#     [
#         (([1, 0, -3, 3, 1], [2, 1, 3, 2, 1]), 0.25, 0.5),
#     ],
#     indirect=["dummy_profile"],
# )
# def test_ray_profile_rix_bounds(dummy_profile, slope_critical1, slope_critical2):
#     rix1 = dummy_profile.rix(slope_critical1)
#     rix2 = dummy_profile.rix(slope_critical2)
#     assert 0 <= rix1 <= 1
#     assert rix2 <= rix1
