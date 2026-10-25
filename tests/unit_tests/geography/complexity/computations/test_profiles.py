import numpy as np
import pyproj
import pytest
import shapely.geometry

from phoibe.geography.complexity.rix.computations.fieldsampler import FieldInfo
from phoibe.geography.complexity.rix.computations.geometry import RayGeometry
from phoibe.geography.complexity.rix.computations.profiles import NaNPolicy, RayProfile, _compute_level_crossings

UTM = pyproj.CRS.from_epsg(32633)
WGS84 = pyproj.CRS.from_epsg(4326)
NAN = np.nan


@pytest.fixture
def ray(origin):
    """Six points, 20 m apart."""
    return RayGeometry.from_compass_regular(location=origin, theta=0.0, R_km=0.1, dr_km=0.02)


def test_create_regular_returns_one_value_per_ray_point(ray, make_sampler):
    profile = RayProfile.create_regular(ray, make_sampler([0, 1, 2, 3, 4, 5]), NaNPolicy.ERROR)
    assert len(profile.r_m) == len(profile.z) == 6
    np.testing.assert_allclose(profile.r_m, ray.r_m)


@pytest.mark.parametrize("policy", list(NaNPolicy))
def test_create_regular_returns_the_sampled_values_given_no_nan(ray, make_sampler, policy):
    profile = RayProfile.create_regular(ray, make_sampler([0, 1, 2, 3, 4, 5]), policy)
    np.testing.assert_allclose(profile.z, [0, 1, 2, 3, 4, 5])


@pytest.mark.parametrize(
    "policy, expected_r_m, expected_z",
    [
        (NaNPolicy.TRUNCATE, [0, 20, 40], [0, 1, 1]),  # cut before the first NaN
        (NaNPolicy.MASK, [0, 20, 40, 60, 80, 100], [0, 1, 1, NAN, 1, 0]),
    ],
)
def test_create_regular_returns_the_profile_of_the_nan_policy(ray, make_sampler, policy, expected_r_m, expected_z):
    profile = RayProfile.create_regular(ray, make_sampler([0, 1, 1, NAN, 1, 0]), policy)
    np.testing.assert_allclose(profile.r_m, expected_r_m)
    np.testing.assert_allclose(profile.z, expected_z)


def test_create_regular_returns_the_sampling_counts_given_a_truncated_profile(ray, make_sampler):
    profile = RayProfile.create_regular(ray, make_sampler([0, 1, 1, NAN, 1, 0]), NaNPolicy.TRUNCATE)
    assert (profile.sampling_info.n_points, profile.sampling_info.nan_count) == (6, 1)  # of the sampling, not the cut


@pytest.mark.parametrize("policy, expected_length", [("mask", 6), ("truncate", 3)])
def test_create_regular_returns_the_profile_given_the_policy_as_a_string(ray, make_sampler, policy, expected_length):
    profile = RayProfile.create_regular(ray, make_sampler([0, 1, 1, NAN, 1, 0]), policy)
    assert len(profile.z) == expected_length


@pytest.mark.parametrize(
    "z, policy, expected_match",
    [
        ([0, 1, 1, NAN, 1, 0], NaNPolicy.ERROR, "NaNs encountered"),
        ([NAN, 1, 1, 1, 1, 0], NaNPolicy.TRUNCATE, "contains no valid numbers"),
    ],
)
def test_create_regular_raises_given_nan_that_the_policy_does_not_allow(ray, make_sampler, z, policy, expected_match):
    with pytest.raises(ValueError, match=expected_match):
        RayProfile.create_regular(ray, make_sampler(z), policy)


@pytest.mark.parametrize("ray_crs, expected_crs", [(None, None), (UTM, "EPSG:32633")])
def test_create_regular_returns_the_sampling_info_of_the_sampler(origin, make_sampler, ray_crs, expected_crs):
    field_info = FieldInfo("EPSG:32633", (0.0, 0.0, 10.0, 10.0), (30.0, -30.0))
    ray = RayGeometry.from_compass_regular(location=origin, theta=0.0, R_km=0.1, dr_km=0.02, crs=ray_crs)

    sampler = make_sampler([0, 1, NAN, 3, 4, 5], field_info=field_info)
    info = RayProfile.create_regular(ray, sampler, NaNPolicy.MASK).sampling_info

    assert (info.field_info, info.crs, info.n_points, info.nan_count) == (field_info, expected_crs, 6, 1)


def test_create_regular_samples_at_the_reprojected_points_given_another_sampler_crs(make_sampler):
    ray = RayGeometry.from_compass_regular(
        location=shapely.geometry.Point(500_000.0, 0.0), theta=0.0, R_km=0.1, dr_km=0.02, crs=UTM
    )
    sampler = make_sampler([0, 1, 2, 3, 4, 5], crs=WGS84)

    profile = RayProfile.create_regular(ray, sampler, NaNPolicy.MASK)

    xs, ys = sampler.calls[0]
    np.testing.assert_allclose(xs, 15.0, atol=1e-9)  # the central meridian of UTM zone 33N
    assert np.all(np.diff(ys) > 0) and ys[0] == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(profile.ray.xs, ray.xs)  # the profile keeps the ray in its own CRS
    assert "Transformed" in profile.sampling_info.message


def test_ray_profile_raises_given_values_and_distances_of_different_length(ray, make_sampler):
    info = RayProfile.create_regular(ray, make_sampler([0, 1, 2, 3, 4, 5]), NaNPolicy.MASK).sampling_info
    with pytest.raises(ValueError, match="same length"):
        RayProfile(ray_=ray, r_m=ray.r_m, z=np.zeros(3), sampling_info=info)


def test_ray_profile_raises_given_non_increasing_distances(ray, make_sampler):
    info = RayProfile.create_regular(ray, make_sampler([0, 1, 2, 3, 4, 5]), NaNPolicy.MASK).sampling_info
    with pytest.raises(ValueError, match="non-positive"):
        RayProfile(ray_=ray, r_m=ray.r_m[::-1], z=np.zeros(6), sampling_info=info)


@pytest.fixture
def ray_40m(origin):
    return RayGeometry.from_compass_regular(location=origin, theta=0.0, R_km=0.04, dr_km=0.01)


def test_create_levelcrossing_returns_the_start_the_levels_and_the_end(ray_40m, make_sampler):
    profile = RayProfile.create_levelcrossing(
        ray_40m, make_sampler([0, 10, 20, 30, 40]), levels=np.array([15.0, 25.0]), nan_policy=NaNPolicy.MASK
    )
    np.testing.assert_allclose(profile.r_m, [0, 15, 25, 40])
    np.testing.assert_allclose(profile.z, [0, 15, 25, 40])
    np.testing.assert_allclose(profile.ray.ys, [0, 15, 25, 40], atol=1e-9)  # the ray follows the crossings


def test_create_levelcrossing_returns_the_sampling_counts_of_the_regular_points(ray_40m, make_sampler):
    profile = RayProfile.create_levelcrossing(
        ray_40m, make_sampler([0, 10, NAN, 30, 40]), levels=np.array([5.0]), nan_policy=NaNPolicy.MASK
    )
    assert (profile.sampling_info.n_points, profile.sampling_info.nan_count) == (5, 1)


@pytest.mark.parametrize(
    "z, r, levels, expected_z, expected_r",
    [
        ([0, 10], [0, 10], [0, 3, 7.5, 10], [0, 3, 7.5, 10], [0, 3, 7.5, 10]),
        ([10, 0], [0, 10], [0, 3, 7.5, 10], [10, 7.5, 3, 0], [0, 2.5, 7, 10]),
        ([0, 10, 15], [0, 4, 10], [0, 5, 10, 12, 18], [0, 5, 10, 12, 15], [0, 2, 4, 6.4, 10]),
        ([0, 10, 0], [0, 5, 10], [0, 4, 8, 12], [0, 4, 8, 8, 4, 0], [0, 2, 4, 6, 8, 10]),
        ([0, 10, 10, 10, 0], [0, 10, 20, 30, 40], [0, 10], [0, 10, 10, 10, 0], [0, 10, 20, 30, 40]),
        ([0, 8, 5, 8, 0], [0, 10, 20, 30, 40], [0, 10], [0, 0], [0, 40]),
        ([0, 20, 20, 20, 0], [0, 10, 20, 30, 40], [0, 10], [0, 10, 10, 0], [0, 5, 35, 40]),
    ],
)
def test_compute_level_crossings_returns_the_crossings_of_the_levels(z, r, levels, expected_z, expected_r):
    r_crossings, z_crossings = _compute_level_crossings(z=z, r=r, levels=levels)
    np.testing.assert_allclose(r_crossings, expected_r)
    np.testing.assert_allclose(z_crossings, expected_z)
