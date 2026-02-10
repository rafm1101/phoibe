import numpy as np
import pytest
import shapely

from phoibe.geography.complexity.rix import analyse
from phoibe.geography.complexity.rix.profiles import NaNPolicy
from phoibe.geography.complexity.rix.profiles import RayProfile


@pytest.fixture
def make_regular_profile(ray_north):
    def _make(sampler):
        return RayProfile.create_regular(ray=ray_north, sampler=sampler, nan_policy=NaNPolicy.ERROR)

    return _make


@pytest.fixture
def make_levelcrossing_profile(ray_north):
    def _make(sampler):
        return RayProfile.create_levelcrossing(
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
        assert isinstance(analyse.ruggedness(profile, 1.0), float)

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
        ruggedness = analyse.ruggedness(profile, 0.3)
        assert np.isnan(ruggedness) or (0.0 <= ruggedness <= 1.0)

    def test_rix_is_decreasing_given_critical_slope(self, profile):
        ruggedness_value_low = analyse.ruggedness(profile, 0.1)
        ruggedness_value_high = analyse.ruggedness(profile, 10.0)
        assert ruggedness_value_low >= ruggedness_value_high


class RayProfileContractFlatProfile(RayProfileContract):
    """Contracts for any `RayProfile` and flat profiles."""

    def test_rix_is_zero_given_flat_profile(self, profile):
        assert np.isclose(analyse.ruggedness(profile, 0.0), 0.0)

    def test_no_segments_given_flat_profile(self, profile):
        assert len(analyse.steep_ray_segments(profile, 1.0)) == 0


class RayProfileContractLinearProfile(RayProfileContract):
    """Contracts for any `RayProfile` and non-flat profiles."""

    def test_rix_is_positive_given_linear_profile(self, profile):
        assert analyse.ruggedness(profile, 0.009) > 0


class TestRayProfileRegularFlat(RayProfileContractFlatProfile):
    @pytest.fixture
    def profile(self, make_regular_profile, flat_sampler):
        return make_regular_profile(flat_sampler)


class TestRayProfileRegularLinear(RayProfileContractLinearProfile):
    @pytest.fixture
    def profile(self, make_regular_profile, linear_sampler):
        return make_regular_profile(linear_sampler)


class TestRayProfileLevelCrossing:

    @pytest.fixture
    def level_crossing_profile(self, ray_north, linear_sampler):
        return RayProfile.create_levelcrossing(
            ray=ray_north, sampler=linear_sampler, levels=[0, 5, 7.5, 10], nan_policy=NaNPolicy.ERROR
        )

    def test_level_crossing_includes_levels(self, level_crossing_profile):
        levels = [0, 5, 7.5, 10]
        for level in levels:
            assert any(np.isclose(level_crossing_profile.z, level))

    def test_level_crossing_profile_builds_valid_profile(self, level_crossing_profile):
        assert len(level_crossing_profile.r_m) > 0
        assert len(level_crossing_profile.z) > 0
        assert len(level_crossing_profile.r_m) == len(level_crossing_profile.z)

        dr = np.diff(level_crossing_profile.r_m)
        assert np.all(dr > 0)
