import dataclasses

import numpy as np
import pytest
import shapely

from phoibe.geography.complexity.rix.rays import LevelCrossingRayProfile
from phoibe.geography.complexity.rix.rays import NaNPolicy
from phoibe.geography.complexity.rix.rays import RayGeometry
from phoibe.geography.complexity.rix.rays import RegularRayProfile


@dataclasses.dataclass(frozen=True)
class Location:
    easting: float
    northing: float


@pytest.fixture
def origin():
    return Location(easting=0.0, northing=0.0)


@pytest.fixture
def ray_north(origin):
    ray = RayGeometry.from_compass_regular(location=origin, theta=0.0, R_km=1.0, dr_km=0.1)
    return ray


class LinearSampler:
    def sample(self, xs, ys):
        return np.linspace(0.0, 10.0, len(xs))


class FlatSampler:
    def __init__(self, value):
        self.value = value

    def sample(self, xs, ys):
        return np.full(len(xs), self.value)


@pytest.fixture
def linear_sampler():
    return LinearSampler()


@pytest.fixture
def flat_sampler():
    return FlatSampler(value=5.0)


@pytest.fixture
def make_discrete_profile(ray_north):
    def _make(sampler):
        return RegularRayProfile(ray=ray_north, sampler=sampler, nan_policy=NaNPolicy.ERROR)

    return _make


@pytest.fixture
def make_levelcrossing_profile(ray_north):
    def _make(sampler):
        return LevelCrossingRayProfile(ray=ray_north, sampler=sampler, levels=[0, 5, 10], nan_policy=NaNPolicy.ERROR)

    return _make


class RayProfileContract:
    """Contracts for any `RayProfile` and any profile."""

    def test_verify_valid_instances(self, profile):
        assert isinstance(profile.slopes, np.ndarray)
        assert isinstance(profile.segment_lengths, np.ndarray)
        assert isinstance(profile.steep_ray_segments(1.0), list)
        assert isinstance(profile.steep_mask(1.0), np.ndarray)
        assert isinstance(profile.rix(1.0), float)

    def test_verify_lengths_are_consistent(self, profile):
        assert len(profile.slopes) == len(profile.segment_lengths)
        assert len(profile.slopes) == len(profile.steep_mask(1.0))
        assert len(profile.steep_ray_segments(1.0)) <= len(profile.slopes)

    def test_segments_are_consistent(self, profile):
        for segment in profile.steep_ray_segments(1.0):
            assert isinstance(segment, shapely.LineString)
            assert len(segment.coords) >= 2

    def test_verify_valid_values(self, profile):
        assert np.all(profile.segment_lengths > 0)
        dr = np.diff(profile.r_m)
        assert np.all(dr > 0)
        dz = np.diff(profile.z)
        assert np.allclose(profile.slopes, dz / dr)
        rix = profile.rix(slope_critical=0.3)
        assert 0.0 <= rix <= 1.0

    def test_rix_is_decreasing_given_critical_slope(self, profile):
        rix_low = profile.rix(slope_critical=0.1)
        rix_high = profile.rix(slope_critical=10.0)
        assert rix_low >= rix_high


class RayProfileContractFlatProfile(RayProfileContract):
    """Contracts for any `RayProfile` and flat profiles."""

    def test_rix_is_zero_given_flat_profile(self, profile):
        assert np.isclose(profile.rix(slope_critical=0.0), 0.0)

    def test_no_segments_given_flat_profile(self, profile):
        assert len(profile.steep_ray_segments(1.0)) == 0


class RayProfileContractLinearProfile(RayProfileContract):
    """Contracts for any `RayProfile` and non-flat profiles."""

    def test_rix_is_positive_given_linear_profile(self, profile):
        assert profile.rix(slope_critical=0.009) > 0


class TestDiscreteRayProfileFlat(RayProfileContractFlatProfile):
    @pytest.fixture
    def profile(self, make_discrete_profile, flat_sampler):
        return make_discrete_profile(flat_sampler)


class TestDiscreteRayProfileLinear(RayProfileContractLinearProfile):
    @pytest.fixture
    def profile(self, make_discrete_profile, linear_sampler):
        return make_discrete_profile(linear_sampler)


class TestLevelCrossingRayProfileFlat(RayProfileContractFlatProfile):
    @pytest.fixture
    def profile(self, make_levelcrossing_profile, flat_sampler):
        return make_levelcrossing_profile(flat_sampler)


class TestLevelCrossingRayProfileLinear(RayProfileContractLinearProfile):
    @pytest.fixture
    def profile(self, make_levelcrossing_profile, linear_sampler):
        return make_levelcrossing_profile(linear_sampler)
