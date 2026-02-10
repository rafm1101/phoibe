import numpy as np
import pytest

from phoibe.geography.complexity.rix import analyse
from phoibe.geography.complexity.rix.geometry import RayGeometry
from phoibe.geography.complexity.rix.profiles import RayProfile


@pytest.fixture
def dummy_profile(request, origin):
    r_m = np.asarray(request.param[0])
    z = np.asarray(request.param[1])
    ray = RayGeometry.from_compass(location=origin, theta=0, r_m=r_m)
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
    result = analyse.ruggedness(dummy_profile, slope_critical=0.3)
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
    result = analyse.ruggedness(dummy_profile, slope_critical)
    assert np.isclose(result, expected_rix)


@pytest.mark.parametrize("dummy_profile", [([0.0, 1.0], [np.nan, np.nan])], indirect=["dummy_profile"])
def test_steep_mask_with_all_nan_slopes(dummy_profile):
    result = analyse.steep_mask(dummy_profile, slope_critical=0.3)
    assert len(result) == 1
    assert not result[0]


@pytest.mark.parametrize(
    "dummy_profile, slope_critical, expected_indices",
    [
        (([0, 1, 2], [0, 0, 0]), 1.0, []),
        (([0, 1, 2], [0, 10, 20]), 5.0, [(0, 2)]),
        (([0, 1, 2, 3, 4, 5], [0, 10, 10, 20, 20, 20]), 5.0, [(0, 1), (2, 3)]),
    ],
    indirect=["dummy_profile"],
)
def test_steep_segment_indices(dummy_profile, slope_critical, expected_indices):
    """steep_segment_indices() finds correct contiguous steep runs."""
    result = analyse.steep_segment_indices(dummy_profile, slope_critical)
    assert result == expected_indices
