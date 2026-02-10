import dataclasses

import numpy as np
import pytest
import shapely.geometry

from phoibe.geography.complexity.rix.profiles import NaNPolicy
from phoibe.geography.complexity.rix.profiles import RayGeometry
from phoibe.geography.complexity.rix.profiles import RegularRayProfile
from phoibe.geography.complexity.rix.results import RadialRixResult
from phoibe.geography.complexity.rix.results import RayResult


@dataclasses.dataclass(frozen=True)
class Location:
    easting: float
    northing: float


class DummySampler:
    def __init__(self, z):
        self._z = np.asarray(z, dtype=float)

    def sample(self, xs, ys):
        return self._z.copy()


def _make_ray_result(z_values, slope_critical=0.3, theta=0.0):
    """Helper to create minimal RayResult for unit testing."""
    origin = Location(easting=0.0, northing=0.0)
    ray = RayGeometry.from_compass_regular(location=origin, theta=theta, R_km=1.0, dr_km=0.1)
    sampler = DummySampler(z=z_values)
    profile = RegularRayProfile.create(ray=ray, sampler=sampler, nan_policy=NaNPolicy.ERROR)

    return RayResult(profile=profile, slope_critical=slope_critical)


@pytest.fixture
def ray_result(request, origin):
    z_values = request.param[0]
    dr_km = request.param[1]
    theta = request.param[2]
    nan_policy = request.param[3]
    slope_critical = request.param[4]
    ray = RayGeometry.from_compass_regular(location=origin, theta=theta, R_km=1.0, dr_km=dr_km)
    sampler = DummySampler(z=z_values)
    profile = RegularRayProfile.create(ray=ray, sampler=sampler, nan_policy=nan_policy)
    return RayResult(profile=profile, slope_critical=slope_critical)


def _make_radial_result(n_rays=8, z_values=None, slope_critical=0.3):
    """Helper to create RadialRixResult for testing."""
    if z_values is None:
        z_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    origin = Location(easting=0.0, northing=0.0)
    angles = np.linspace(0, 360, n_rays, endpoint=False)

    ray_results = []
    for theta in angles:
        ray = RayGeometry.from_compass_regular(location=origin, theta=theta, R_km=1.0, dr_km=0.1)
        sampler = DummySampler(z=z_values)
        profile = RegularRayProfile.create(ray=ray, sampler=sampler, nan_policy=NaNPolicy.ERROR)
        ray_results.append(RayResult(profile=profile, slope_critical=slope_critical))

    return RadialRixResult(rays=tuple(ray_results))


@pytest.fixture
def radial_result(request, origin):
    n_rays = request.param[0]
    z_values = request.param[1]
    dr_km = request.param[2]
    slope_critical = request.param[3]

    angles = np.linspace(0, 360, n_rays, endpoint=False)
    ray_results = []
    for theta in angles:
        ray = RayGeometry.from_compass_regular(location=origin, theta=theta, R_km=1.0, dr_km=dr_km)
        sampler = DummySampler(z=z_values)
        profile = RegularRayProfile.create(ray=ray, sampler=sampler, nan_policy=NaNPolicy.ERROR)
        ray_results.append(RayResult(profile=profile, slope_critical=slope_critical))

    return RadialRixResult(rays=tuple(ray_results))


@pytest.mark.parametrize("ray_result", [([0, 1, 2], 0.5, 45.0, NaNPolicy.ERROR, 0.03)], indirect=["ray_result"])
def test_ray_result_passes_theta(ray_result):
    assert ray_result.theta == 45.0


@pytest.mark.parametrize(
    "ray_result, expected_ruggedness, expected_steep_length_m, expected_total_length_m, expected_max_abs",
    [
        (([0, 100, 200, 300, 400], 0.25, 45.0, NaNPolicy.ERROR, 0.3), 1.0, 1e3, 1e3, 0.4),
        (([5, 5, 5, 5, 5], 0.25, 45.0, NaNPolicy.ERROR, 0.1), 0.0, 0, 1e3, 0.0),
        (([100, 100, 1100, 1200, 400], 0.25, 45.0, NaNPolicy.ERROR, 3.0), 0.5, 5e2, 1e3, 4.0),
    ],
    indirect=["ray_result"],
)
def test_ray_result_computes_values(
    ray_result, expected_ruggedness, expected_steep_length_m, expected_total_length_m, expected_max_abs
):
    assert np.isclose(ray_result.ruggedness, expected_ruggedness)
    assert np.isclose(ray_result.steep_length_m, expected_steep_length_m)
    assert np.isclose(ray_result.total_length_m, expected_total_length_m)
    assert np.isclose(ray_result.max_abs_slope, expected_max_abs)


@pytest.mark.parametrize(
    "ray_result, expected_ruggedness, expected_steep_length_m, expected_total_length_m, expected_max_abs",
    [
        (([0, np.nan, np.nan, 300, 400], 0.25, 45.0, NaNPolicy.MASK, 0.3), 0.25, 250, 1e3, 0.4),
        (([5, 5, np.nan, np.nan, 5], 0.25, 45.0, NaNPolicy.MASK, 0.1), 0.0, 0, 1e3, 0.0),
        (([100, 100, 1100, np.nan, 400], 0.25, 45.0, NaNPolicy.MASK, 3.0), 0.25, 250, 1e3, 4.0),
    ],
    indirect=["ray_result"],
)
def test_ray_result_computes_values_given_nan(
    ray_result, expected_ruggedness, expected_steep_length_m, expected_total_length_m, expected_max_abs
):
    assert np.isclose(ray_result.ruggedness, expected_ruggedness)
    assert np.isclose(ray_result.steep_length_m, expected_steep_length_m)
    assert np.isclose(ray_result.total_length_m, expected_total_length_m)
    assert np.isclose(ray_result.max_abs_slope, expected_max_abs)


@pytest.mark.parametrize(
    "ray_result",
    [([0, 100, 200, 300, 400], 0.25, 45.0, NaNPolicy.ERROR, 0.3)],
    indirect=["ray_result"],
)
def test_ray_result_steep_segments_are_valid_linestrings(ray_result):
    assert len(ray_result.steep_segments) > 0
    for segment in ray_result.steep_segments:
        assert isinstance(segment, shapely.geometry.LineString)
        assert len(segment.coords) >= 2


@pytest.mark.parametrize(
    "ray_result",
    [([0, 100, 200, 300, 400], 0.25, 45.0, NaNPolicy.ERROR, 0.3)],
    indirect=["ray_result"],
)
def test_ray_result_describe_contains_all_metrics(ray_result):
    description = ray_result.describe()
    required_keys = {"theta", "ruggedness", "total_length_m", "steep_length_m", "max_abs_slope", "n_steep_segments"}
    assert required_keys.issubset(description.keys())


@pytest.mark.parametrize(
    "ray_result",
    [([0, 100, 200, 300, 400], 0.25, 45.0, NaNPolicy.ERROR, 0.3)],
    indirect=["ray_result"],
)
def test_ray_result_caches_properties(ray_result):
    ruggedness1 = ray_result.ruggedness
    steep_length_m1 = ray_result.steep_length_m
    ruggedness2 = ray_result.ruggedness
    steep_length_m2 = ray_result.steep_length_m
    assert ruggedness1 == ruggedness2
    assert steep_length_m1 == steep_length_m2


def test_radial_result_raises_valueerrer_given_empty_rays():
    with pytest.raises(ValueError, match="at least one ray"):
        RadialRixResult(rays=tuple())


@pytest.mark.parametrize("ray_result", [([0, 1, 2], 0.5, 45.0, NaNPolicy.ERROR, 0.3)], indirect=["ray_result"])
def test_radial_result_rejects_given_mixed_slope_critical(ray_result):
    other_ray_result = RayResult(profile=ray_result.profile, slope_critical=0.7)
    with pytest.raises(ValueError, match="same slope_critical"):
        RadialRixResult(rays=(ray_result, other_ray_result))


@pytest.mark.parametrize("radial_result", [(4, [0, 1, 2], 0.5, 0.3)], indirect=["radial_result"])
def test_radial_result_rix_is_mean_of_rays(radial_result):
    manual_mean = np.mean([ray.ruggedness for ray in radial_result.rays])
    assert np.isclose(radial_result.rix, manual_mean)


@pytest.mark.parametrize("radial_result", [(8, [0, 1, 2], 0.5, 0.3)], indirect=["radial_result"])
def test_radial_result_angles_are_unique(radial_result):
    angles = radial_result.angles
    assert len(angles) == len(set(angles))


@pytest.mark.parametrize("radial_result", [(72, [0, 1, 2], 0.5, 0.3)], indirect=["radial_result"])
def test_radial_result_retrieves_correct_ray(radial_result):
    theta = radial_result.angles[35]
    ray = radial_result.ray(theta)
    assert ray.theta == theta


@pytest.mark.parametrize("radial_result", [(8, [0, 1, 2], 0.5, 0.3)], indirect=["radial_result"])
def test_radial_result_directional_stats_order(radial_result):
    directional_ruggedness = radial_result.directional_stats()
    angles = radial_result.angles
    for angle, ruggedness in zip(angles, directional_ruggedness):
        ray = radial_result.ray(angle)
        assert np.isclose(ruggedness, ray.ruggedness)


@pytest.mark.parametrize("radial_result", [(8, [0, 1, 2], 0.5, 0.3)], indirect=["radial_result"])
def test_radial_result_describe_statistics(radial_result):
    description = radial_result.describe()
    rix_values = [ray.ruggedness for ray in radial_result.rays]
    assert np.isclose(description["rix_mean"], np.mean(rix_values))
    assert np.isclose(description["rix_std"], np.std(rix_values))
    assert np.isclose(description["rix_min"], np.min(rix_values))
    assert np.isclose(description["rix_max"], np.max(rix_values))
    assert description["n_rays"] == len(radial_result.rays)
