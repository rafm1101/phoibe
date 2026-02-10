import dataclasses

import numpy as np
import pytest

from phoibe.geography.complexity.rix import analyse
from phoibe.geography.complexity.rix.geometry import RayGeometry
from phoibe.geography.complexity.rix.profiles import RayProfile

# from phoibe.geography.complexity.rix import compute_radial_rix


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


@pytest.fixture
def dummy_profile(request, ray_north):
    ray = ray_north
    r_m = np.asarray(request.param[0])
    z = np.asarray(request.param[1])
    return RayProfile(ray_=ray, r_m=r_m, z=z)


@pytest.mark.parametrize(
    "dummy_profile, expected_length",
    [
        (([0.0], [5.0]), 1),
    ],
    indirect=["dummy_profile"],
)
def test_slopes_returns_nan_given_single_point(dummy_profile, expected_length):
    result = analyse.slopes(dummy_profile)
    assert len(result) == expected_length
    assert np.isnan(result[0])


@pytest.mark.parametrize(
    "dummy_profile, expected_length, expected_slopes",
    [
        (([0.0, 100.0], [0.0, 10.0]), 1, [0.1]),
        (([0.0, 1.0, 2.0], [5.0, 5.0, 5.0]), 2, [0, 0]),
        (([0.0, 100.0, 300], [0.0, 10.0, 0.0]), 2, [0.1, -0.05]),
    ],
    indirect=["dummy_profile"],
)
def test_slopes_returns_correct_values(dummy_profile, expected_length, expected_slopes):
    result = analyse.slopes(dummy_profile)
    assert len(result) == expected_length
    assert np.allclose(result, expected_slopes)


@pytest.mark.parametrize("dummy_profile", [([0.0], [5.0])], indirect=["dummy_profile"])
def test_rix_handles_empty_profile(dummy_profile):
    result = analyse.rix(dummy_profile, slope_critical=0.3)
    assert np.isnan(result)


@pytest.mark.parametrize(
    "dummy_profile, slope_critical, expected_rix",
    [
        (([0.0, 100.0], [0.0, 10.0]), 0.0, 1.0),
        (([0.0, 100.0], [0.0, 10.0]), 0.3, 0.0),
        (([0.0, 1.0, 2.0], [5.0, 5.0, 5.0]), 0.0, 0.0),
        (([0.0, 1.0, 2.0], [0.0, 10.0, 20.0]), 6.0, 1.0),
    ],
    indirect=["dummy_profile"],
)
def test_rix_returns_correct_values(dummy_profile, slope_critical, expected_rix):
    result = analyse.rix(dummy_profile, slope_critical)
    assert np.isclose(result, expected_rix)


@pytest.mark.parametrize("dummy_profile", [([0.0, 1.0], [np.nan, np.nan])], indirect=["dummy_profile"])
def test_steep_mask_with_all_nan_slopes(dummy_profile):
    result = analyse.steep_mask(dummy_profile, slope_critical=0.3)
    assert len(result) == 1
    assert not result[0]


# @pytest.mark.parametrize(
#     "dummy_profile, slope_critical, expected_indices",
#     [
#         (([0, 0, 0], [0, 1, 2]), 1.0, []),
#         (([0, 10, 20], [0, 1, 2]), 5.0, [(0, 2)]),
#         (([0, 10, 10, 20, 20, 20], [0, 1, 2, 3, 4, 5]), 5.0, [(0, 1), (2, 3)]),
#     ],
#     indirect=["dummy_profile"]
# )
# def test_steep_segment_indices(dummy_profile, slope_critical, expected_indices):
#     """steep_segment_indices() finds correct contiguous steep runs."""
#     result = radial_ruggedness.steep_segment_indices(dummy_profile, slope_critical)
#     breakpoint()
#     assert result == expected_indices


# ============================================================================
# Helper for minimal profile creation
# ============================================================================


# @pytest.mark.parametrize(
#     "theta, R_km, dr_km, expected_match",
#     [
#         (270, np.nan, 0.1, "R_km contains NaN or infinite values"),
#         (270, np.inf, 0.1, "R_km contains NaN or infinite values"),
#         (270, -7, 0.1, "R_km contains negative values"),
#         (270, 13, np.nan, "dr_km contains NaN or infinite values"),
#         (270, 13, -1, "dr_km contains non-positive values"),
#         (270, 13, 0.0, "dr_km contains non-positive values"),
#         (270, 13, 23, "dr_km must not exceed R_km"),
#     ],
# )
# def test_raygeometry_from_compass_regular_rejects_invalid_inputs(origin, theta, R_km, dr_km, expected_match):
#     with pytest.raises(ValueError, match=expected_match):
#         RayGeometry.from_compass_regular(origin, theta, R_km, dr_km)


# @pytest.mark.parametrize(
#     "theta, R_km, dr_km, expected_spacing, expected_last",
#     [(270, 2, 0.5, 500, 2000), (270, 5, 0.1, 100, 5000)],
# )
# def test_raygeometry_returns_correct_spacing(origin, theta, R_km, dr_km, expected_spacing, expected_last):
#     ray = RayGeometry.from_compass_regular(origin, theta, R_km, dr_km)
#     grid = ray.r_m
#     assert np.allclose(np.diff(grid), expected_spacing)
#     assert np.allclose(grid[0], 0)
#     assert np.allclose(grid[-1], expected_last)


# @pytest.mark.parametrize(
#     "theta, R_km, dr_km, expected_spacing, expected_last",
#     [(0, 1.0, 0.5, 500, 2000), (270, 1.0, 0.5, 100, 5000)],
# )
# def test_raygeometry_returns_correct_orientation(origin, theta, R_km, dr_km, expected_spacing, expected_last):
#     ray = RayGeometry.from_compass_regular(origin, theta, R_km, dr_km)
#     xs, ys = ray.xs, ray.ys
#     assert np.all(ys >= origin.northing - 1e-7)
#     assert np.all(xs <= origin.easting)
#     assert np.allclose(ys, origin.northing) or np.allclose(xs, origin.easting)


# @pytest.mark.parametrize(
#     "theta, R_km, dr_km, expected_grid",
#     [(0, 10.0, 3.0, [0, 3000, 6000, 9000])],
# )
# def test_raygeometry_truncates_with_warning(caplog, origin, theta, R_km, dr_km, expected_grid):
#     ray = RayGeometry.from_compass_regular(origin, theta, R_km, dr_km)

#     with caplog.at_level(logging.WARNING):
#         r = ray.r_m

#     warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
#     assert np.allclose(r, expected_grid)
#     assert len(warnings) == 1
#     assert "Ray for 0.0 truncated as" in warnings[0].message


# class DummySampler:
#     def __init__(self, z):
#         self._z = np.asarray(z, dtype=float)

#     def sample(self, xs, ys):
#         assert len(xs) == len(self._z)
#         return self._z


# @pytest.fixture
# def dummy_sampler(request):
#     z = request.param
#     return DummySampler(z=z)


# @pytest.fixture
# def profile_sampler():
#     z = [0, 1, 1, 1, 0, 0, 0, 2, 2, 1, 0]
#     return DummySampler(z=z)


# @pytest.fixture
# def invalid_profile_sampler(request):
#     if request.param == "single-intern":
#         z = [0, 1, 1, np.nan, 1, 0]
#     return DummySampler(z=z)


# @pytest.mark.parametrize(
#     "ray_01km, dummy_sampler, nan_policy, expected_n_slopes",
#     [
#         (0.025, [0, 1, 2, 3, 4], NaNPolicy.ERROR, 4),
#     ],
#     indirect=["ray_01km", "dummy_sampler"],
# )
# def test_ray_profile_contracts_lengths(ray_01km, dummy_sampler, nan_policy, expected_n_slopes):
#     ray_profile = RayProfile(ray_01km, dummy_sampler, nan_policy)
#     assert len(ray_profile.slopes) == expected_n_slopes


# @pytest.mark.parametrize(
#     "ray_01km, nan_policy, critical_slope, expected_mask, expected_rix",
#     [
#         (0.01, NaNPolicy.ERROR, 0.2, [False] * 10, 0.0),
#         (0.01, NaNPolicy.ERROR, 0.199, [False, False, False, False, False, False, True, False, False, False], 0.1),
#         (0.01, NaNPolicy.ERROR, 0.099, [True, False, False, True, False, False, True, False, True, True], 0.5),
#         (0.01, NaNPolicy.TRUNCATE, 0.2, [False] * 10, 0.0),
#         (0.01, NaNPolicy.TRUNCATE, 0.199, [False, False, False, False, False, False, True, False, False, False], 0.1),
#         (0.01, NaNPolicy.TRUNCATE, 0.099, [True, False, False, True, False, False, True, False, True, True], 0.5),
#     ],
#     indirect=["ray_01km"],
# )
# def test_ray_profile_returns_correct_intermediate_values_given_valid_profile(
#     ray_01km, profile_sampler, nan_policy, critical_slope, expected_mask, expected_rix
# ):
#     ray_profile = RayProfile(ray_01km, profile_sampler, nan_policy)
#     assert np.allclose(ray_profile.slopes, [0.1, 0.0, 0.0, -0.1, 0.0, 0.0, 0.2, 0.0, -0.1, -0.1])

#     mask = ray_profile.steep_mask(critical_slope)
#     assert np.all(mask == expected_mask)
#     assert np.isclose(ray_profile.rix(critical_slope), expected_rix)


# @pytest.mark.parametrize(
#     "ray_01km, invalid_profile_sampler, nan_policy, critical_slope, expected_slopes, expected_mask, expected_rix",
#     [
#         (0.02, "single-intern", NaNPolicy.TRUNCATE, 0.05, [0.05, 0.0], [False] * 2, 0.0),
#         (0.02, "single-intern", NaNPolicy.TRUNCATE, 0.049, [0.05, 0.0], [True, False], 0.5),
#         (0.02, "single-intern", NaNPolicy.MASK, 0.05, [0.05, 0.0, np.nan, np.nan, -0.05], [False] * 5, 0.0),
#         (
#             0.02,
#             "single-intern",
#             NaNPolicy.MASK,
#             0.049,
#             [0.05, 0.0, np.nan, np.nan, -0.05],
#             [True] + [False] * 3 + [True],
#             0.4,
#         ),
#     ],
#     indirect=["ray_01km", "invalid_profile_sampler"],
# )
# def test_ray_profile_returns_correct_intermediate_values(
#     ray_01km, invalid_profile_sampler, nan_policy, critical_slope, expected_slopes, expected_mask, expected_rix
# ):
#     ray_profile = RayProfile(ray_01km, invalid_profile_sampler, nan_policy)
#     assert np.allclose(ray_profile.slopes, expected_slopes, equal_nan=True)

#     mask = ray_profile.steep_mask(critical_slope)
#     assert np.all(mask == expected_mask)
#     assert np.isclose(ray_profile.rix(critical_slope), expected_rix)


# class DummyProfile(RayProfile):
#     def __init__(self, slopes, segment_lengths):
#         self._slopes = np.asarray(slopes, dtype=float)
#         self._segment_lengths = np.asarray(segment_lengths, dtype=float)

#     @property
#     def slopes(self):
#         return self._slopes

#     @property
#     def segment_lengths(self):
#         return self._segment_lengths


# @pytest.fixture
# def dummy_profile(request):
#     slopes = request.param[0]
#     segment_lengths = request.param[1]
#     return DummyProfile(slopes, segment_lengths)


# # @pytest.mark.parametrize(
# #     "dummy_profile, slope_critical1, slope_critical2",
# #     [
# #         (([1, 0, -3, 3, 1], [2, 1, 3, 2, 1]), 0.25, 0.5),
# #     ],
# #     indirect=["dummy_profile"],
# # )
# # def test_ray_profile_rix_bounds(dummy_profile, slope_critical1, slope_critical2):
# #     rix1 = dummy_profile.rix(slope_critical1)
# #     rix2 = dummy_profile.rix(slope_critical2)
# #     assert 0 <= rix1 <= 1
# #     assert rix2 <= rix1
