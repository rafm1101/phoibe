import numpy as np
import pytest

from phoibe.geography.complexity.rix import analyse
from phoibe.geography.complexity.rix.profiles import NaNPolicy
from phoibe.geography.complexity.rix.profiles import RayProfile
from phoibe.geography.complexity.rix.profiles import RegularRayProfile


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
    ray_profile = RegularRayProfile.create(ray_01km, dummy_sampler, nan_policy)
    slope_values = analyse.slopes(ray_profile)
    assert len(slope_values) == expected_n_slopes
    assert len(ray_profile.r_m) == expected_n_slopes + 1
    assert len(ray_profile.z) == expected_n_slopes + 1


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
    ray_profile = RegularRayProfile.create(ray_01km, profile_sampler, nan_policy)
    slope_values = analyse.slopes(ray_profile)
    assert np.allclose(slope_values, [0.1, 0.0, 0.0, -0.1, 0.0, 0.0, 0.2, 0.0, -0.1, -0.1])

    mask = analyse.steep_mask(ray_profile, critical_slope)
    assert np.all(mask == expected_mask)
    ruggedness = analyse.rix(ray_profile, critical_slope)
    assert np.isclose(ruggedness, expected_rix)


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
    ray_profile = RegularRayProfile.create(ray_01km, invalid_profile_sampler, nan_policy)
    slope_values = analyse.slopes(ray_profile)
    assert np.allclose(slope_values, expected_slopes, equal_nan=True)

    mask = analyse.steep_mask(ray_profile, critical_slope)
    assert np.all(mask == expected_mask)
    ruggedness = analyse.rix(ray_profile, critical_slope)
    assert np.isclose(ruggedness, expected_rix)


@pytest.mark.parametrize(
    "ray_01km, invalid_profile_sampler, nan_policy, critical_slope, expected_slopes, expected_mask, expected_rix",
    [
        (0.02, "single-intern", NaNPolicy.TRUNCATE, 0.05, [0.05, 0.0], [False, False], 0.0),
        (0.02, "single-intern", NaNPolicy.TRUNCATE, 0.049, [0.05, 0.0], [True, False], 0.5),
        (
            0.02,
            "single-intern",
            NaNPolicy.MASK,
            0.05,
            [0.05, 0.0, np.nan, np.nan, -0.05],
            [False, False, False, False, False],
            0.0,
        ),
        (
            0.02,
            "single-intern",
            NaNPolicy.MASK,
            0.049,
            [0.05, 0.0, np.nan, np.nan, -0.05],
            [True, False, False, False, True],
            0.4,
        ),
    ],
    indirect=["ray_01km", "invalid_profile_sampler"],
)
def test_profile_analysis_with_nan_handling(
    ray_01km, invalid_profile_sampler, nan_policy, critical_slope, expected_slopes, expected_mask, expected_rix
):
    profile = RegularRayProfile.create(ray=ray_01km, sampler=invalid_profile_sampler, nan_policy=nan_policy)
    slope_arr = analyse.slopes(profile)
    assert np.allclose(slope_arr, expected_slopes, equal_nan=True)

    mask = analyse.steep_mask(profile, critical_slope)
    assert np.all(mask == expected_mask)

    ruggedness = analyse.rix(profile, critical_slope)
    assert np.isclose(ruggedness, expected_rix)


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
