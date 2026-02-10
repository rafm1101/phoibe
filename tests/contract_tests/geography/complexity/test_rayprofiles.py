import numpy as np
import pytest
import shapely

from phoibe.geography.complexity.rix import analyse
from phoibe.geography.complexity.rix.profiles import LevelCrossingRayProfile
from phoibe.geography.complexity.rix.profiles import NaNPolicy
from phoibe.geography.complexity.rix.profiles import RegularRayProfile


@pytest.fixture
def make_discrete_profile(ray_north):
    def _make(sampler):
        return RegularRayProfile.create(ray=ray_north, sampler=sampler, nan_policy=NaNPolicy.ERROR)

    return _make


@pytest.fixture
def make_levelcrossing_profile(ray_north):
    def _make(sampler):
        return LevelCrossingRayProfile(
            ray_geometry=ray_north, sampler=sampler, levels=[0, 5, 10], nan_policy=NaNPolicy.ERROR
        )

    return _make


class RayProfileContract:
    """Contracts for any `RayProfile` and any profile."""

    def test_verify_valid_instances(self, profile):
        assert isinstance(analyse.slopes(profile), np.ndarray)
        assert isinstance(analyse.segment_lengths(profile), np.ndarray)
        assert isinstance(analyse.steep_ray_segments(profile, 1.0), list)
        assert isinstance(analyse.steep_mask(profile, 1.0), np.ndarray)
        assert isinstance(analyse.rix(profile, 1.0), float)

    def test_verify_lengths_are_consistent(self, profile):
        slopes = analyse.slopes(profile)
        assert len(analyse.segment_lengths(profile)) == len(slopes)
        assert len(analyse.steep_mask(profile, 1.0)) == len(slopes)
        assert len(analyse.steep_ray_segments(profile, 1.0)) <= len(slopes)

    def test_segments_are_valid_linestrings(self, profile):
        for segment in analyse.steep_ray_segments(profile, 1.0):
            assert isinstance(segment, shapely.LineString)
            assert len(segment.coords) >= 2

    def test_verify_valid_values(self, profile):
        assert np.all(analyse.segment_lengths(profile) > 0)
        dr = np.diff(profile.r_m)
        assert np.all(dr > 0)
        dz = np.diff(profile.z)
        assert np.allclose(analyse.slopes(profile), dz / dr)
        ruggedness = analyse.rix(profile, 0.3)
        assert np.isnan(ruggedness) or (0.0 <= ruggedness <= 1.0)

    def test_rix_is_decreasing_given_critical_slope(self, profile):
        ruggedness_value_low = analyse.rix(profile, 0.1)
        ruggedness_value_high = analyse.rix(profile, 10.0)
        assert ruggedness_value_low >= ruggedness_value_high


class RayProfileContractFlatProfile(RayProfileContract):
    """Contracts for any `RayProfile` and flat profiles."""

    def test_rix_is_zero_given_flat_profile(self, profile):
        assert np.isclose(analyse.rix(profile, 0.0), 0.0)

    def test_no_segments_given_flat_profile(self, profile):
        assert len(analyse.steep_ray_segments(profile, 1.0)) == 0


class RayProfileContractLinearProfile(RayProfileContract):
    """Contracts for any `RayProfile` and non-flat profiles."""

    def test_rix_is_positive_given_linear_profile(self, profile):
        assert analyse.rix(profile, 0.009) > 0


class TestDiscreteRayProfileFlat(RayProfileContractFlatProfile):
    @pytest.fixture
    def profile(self, make_discrete_profile, flat_sampler):
        return make_discrete_profile(flat_sampler)


class TestDiscreteRayProfileLinear(RayProfileContractLinearProfile):
    @pytest.fixture
    def profile(self, make_discrete_profile, linear_sampler):
        return make_discrete_profile(linear_sampler)


# class TestLevelCrossingRayProfileFlat(RayProfileContractFlatProfile):
#     @pytest.fixture
#     def profile(self, make_levelcrossing_profile, flat_sampler):
#         return make_levelcrossing_profile(flat_sampler)


# class TestLevelCrossingRayProfileLinear(RayProfileContractLinearProfile):
#     @pytest.fixture
#     def profile(self, make_levelcrossing_profile, linear_sampler):
#         return make_levelcrossing_profile(linear_sampler)
