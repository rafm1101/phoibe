import numpy as np
import pytest

from phoibe.geography.complexity.rix.computations import evaluate

PROFILE_R_M = list(range(0, 110, 10))
PROFILE_Z = [0, 1, 1, 1, 0, 0, 0, 2, 2, 1, 0]


@pytest.mark.parametrize(
    "r_m, z, expected",
    [
        ([0, 100], [0, 10], [0.1]),
        ([0, 1, 2], [5, 5, 5], [0, 0]),
        ([0, 100, 300], [0, 10, 0], [0.1, -0.05]),
    ],
)
def test_slopes_returns_height_difference_per_distance(make_profile, r_m, z, expected):
    np.testing.assert_allclose(evaluate.slopes(make_profile(r_m, z)), expected)


@pytest.mark.parametrize(
    "slope_critical, expected_mask",
    [
        (0.2, [False] * 10),
        (0.199, [False] * 6 + [True] + [False] * 3),
        (0.099, [True, False, False, True, False, False, True, False, True, True]),
    ],
)
def test_steep_mask_returns_the_segments_above_the_critical_slope(make_profile, slope_critical, expected_mask):
    profile = make_profile(np.arange(11) * 10, [0, 1, 1, 1, 0, 0, 0, 2, 2, 1, 0])
    assert evaluate.steep_mask(profile, slope_critical).tolist() == expected_mask


@pytest.mark.parametrize(
    "r_m, z, slope_critical, expected",
    [
        ([0, 100], [0, 10], 0.0, 1.0),
        ([0, 100], [0, 10], 0.3, 0.0),
        ([0, 1, 2], [5, 5, 5], 0.0, 0.0),
        ([0, 1, 2], [0, 10, 20], 6.0, 1.0),
        ([0, 100, 400], [0, 0, 150], 0.1, 0.75),
        (PROFILE_R_M, PROFILE_Z, 0.2, 0.0),
        (PROFILE_R_M, PROFILE_Z, 0.199, 0.1),
        (PROFILE_R_M, PROFILE_Z, 0.099, 0.5),
    ],
)
def test_ruggedness_returns_the_steep_share_of_the_length(make_profile, r_m, z, slope_critical, expected):
    assert evaluate.ruggedness(make_profile(r_m, z), slope_critical) == pytest.approx(expected)


def test_ruggedness_returns_nan_given_profile_without_a_segment(make_profile):
    """The check `rix_nan` relies on this."""
    assert np.isnan(evaluate.ruggedness(make_profile([0.0], [5.0]), slope_critical=0.3))


@pytest.mark.parametrize(
    "r_m, z, expected",
    [
        ([0, 1, 2], [0, 0, 0], []),
        ([0, 1, 2], [0, 10, 20], [(0, 2)]),
        ([0, 1, 2, 3, 4, 5], [0, 10, 10, 20, 20, 20], [(0, 1), (2, 3)]),
        ([0, 1, 2, 3], [0, 0, 0, 10], [(2, 3)]),
    ],
)
def test_steep_segment_indices_returns_the_runs_of_contiguous_steep_segments(make_profile, r_m, z, expected):
    assert evaluate.steep_segment_indices(make_profile(r_m, z), slope_critical=5.0) == expected


def test_steep_ray_segments_returns_linestrings_along_the_ray(make_profile):
    profile = make_profile([0, 1000, 2000, 3000, 4000], [0, 0, 500, 1000, 1000])
    segments = evaluate.steep_ray_segments(profile, slope_critical=0.1)
    assert [list(segment.coords) for segment in segments] == [[(0.0, 1000.0), (0.0, 2000.0), (0.0, 3000.0)]]


def test_steep_ray_segments_splits_at_flat_stretches(make_profile):
    profile = make_profile([0, 1000, 2000, 3000, 4000], [0, 500, 500, 1000, 1500])
    segments = evaluate.steep_ray_segments(profile, slope_critical=0.1)
    assert [list(segment.coords) for segment in segments] == [
        [(0.0, 0.0), (0.0, 1000.0)],
        [(0.0, 2000.0), (0.0, 3000.0), (0.0, 4000.0)],
    ]
