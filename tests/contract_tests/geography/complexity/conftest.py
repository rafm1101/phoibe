import dataclasses

import numpy as np
import pytest

from phoibe.geography.complexity.rix.geometry import RayGeometry
from phoibe.geography.complexity.rix.profiles import NaNPolicy
from phoibe.geography.complexity.rix.profiles import RayProfile
from phoibe.geography.complexity.rix.results import RadialRixResult
from phoibe.geography.complexity.rix.results import RayResult


@dataclasses.dataclass(frozen=True)
class Location:
    easting: float
    northing: float


@pytest.fixture
def origin():
    return Location(easting=0.0, northing=0.0)


class DummySampler:
    def __init__(self, z):
        self._z = np.asarray(z, dtype=float)

    def sample(self, xs, ys):
        return self._z.copy()


@pytest.fixture
def flat_sampler():
    """Sampler producing flat terrain (constant elevation)."""
    return DummySampler(z=[5.0] * 11)


@pytest.fixture
def linear_sampler():
    """Sampler producing linear slope."""
    return DummySampler(z=np.linspace(0, 10, 11))


@pytest.fixture
def steep_sampler():
    """Sampler producing very steep terrain."""
    return DummySampler(z=[0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])


@pytest.fixture
def ray_north(origin):
    """Ray pointing North with 1km length, 0.1km spacing."""
    return RayGeometry.from_compass_regular(location=origin, theta=0.0, R_km=1.0, dr_km=0.1)


@pytest.fixture
def make_ray_result(ray_north):
    """Factory for creating RayResult instances."""

    def _make(sampler, slope_critical=0.3, nan_policy=NaNPolicy.ERROR):
        profile = RayProfile.create_regular(ray=ray_north, sampler=sampler, nan_policy=nan_policy)
        return RayResult(profile=profile, slope_critical=slope_critical)

    return _make


@pytest.fixture
def make_radial_result(origin):
    """Factory for creating RadialRixResult instances."""

    def _make(n_angles=8, slope_critical=0.3, sampler_factory=None):
        if sampler_factory is None:

            def sampler_factory():
                return DummySampler(z=np.linspace(0, 10, 11))

        angles = np.linspace(0, 360, n_angles, endpoint=False)
        ray_results = []

        for theta in angles:
            ray = RayGeometry.from_compass_regular(location=origin, theta=theta, R_km=1.0, dr_km=0.1)
            profile = RayProfile.create_regular(ray=ray, sampler=sampler_factory(), nan_policy=NaNPolicy.ERROR)
            ray_results.append(RayResult(profile=profile, slope_critical=slope_critical))

        return RadialRixResult(rays=tuple(ray_results))

    return _make
