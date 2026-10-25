import numpy as np
import pytest

from phoibe.geography.complexity.rix.computations.trix import (
    assess_transferability,
    compute_trix,
    compute_trix_limit_distances,
    evaluate_transferability_limits,
    pairwise_distances_km,
)


def test_compute_trix_returns_sites_in_rows_and_references_in_columns():
    result = compute_trix([0.0, 1.0], [100.0, 200.0], [0.5], [150.0])
    assert result.shape == (2, 1)
    np.testing.assert_allclose(result, [[27.5], [72.5]])


@pytest.mark.parametrize(
    "rix_site, elevation_site, rix_reference, elevation_reference, expected",
    [
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0, 45.0),
        (0.2, 100.0, 0.4, 100.0, 27.0),
        (0.0, 0.0, 0.0, 100.0, 10.0),
        (0.0, 100.0, 0.0, 0.0, 10.0),
    ],
)
def test_compute_trix_returns_hand_computed_values(
    rix_site, elevation_site, rix_reference, elevation_reference, expected
):
    result = compute_trix(rix_site, elevation_site, rix_reference, elevation_reference)
    np.testing.assert_allclose(result, [[expected]])


def test_compute_trix_rejects_two_dimensional_input():
    with pytest.raises(TypeError, match="scalar or 1D"):
        compute_trix(np.zeros((2, 2)), [0.0], [0.0], [0.0])


@pytest.mark.parametrize(
    "trix, decimals, expected_a, expected_b",
    [
        (0.0, 1, 8.5, 15.0),
        (10.0, 1, 7.6, 13.6),
        (10.0, None, 7.63, 13.6),
        (82.0, None, 1.5, 3.52),
        (100.0, 1, 1.5, 3.0),
        (0.5, None, 8.4565, 14.93),
        (0.5, 1, 8.5, 14.9),
    ],
)
def test_compute_trix_limit_distances_returns_hand_computed_values(trix, decimals, expected_a, expected_b):
    a, b = compute_trix_limit_distances(np.array([trix]), decimals=decimals)
    np.testing.assert_allclose(a, [expected_a])
    np.testing.assert_allclose(b, [expected_b])


def test_compute_trix_limit_distances_keeps_monotonicity_a_smaller_b():
    a, b = compute_trix_limit_distances(np.linspace(0, 500, 200), decimals=None)
    assert np.all(a <= b)


@pytest.mark.parametrize(
    "distance, expected",
    [(1.0, 2), (1.5, 2), (2.0, 1), (3.0, 1), (3.1, 0)],
)
def test_evaluate_transferability_limits_class_by_limits(distance, expected):
    result = evaluate_transferability_limits(np.array([distance]), np.array([1.5]), np.array([3.0]))
    assert result.tolist() == [expected]
    assert result.dtype.kind == "i"


@pytest.mark.parametrize(
    "distance, a, b",
    [(np.nan, 1.5, 3.0), (1.0, np.nan, 3.0), (1.0, 1.5, np.nan)],
)
def test_evaluate_transferability_limits_returns_zero_given_distance_or_a_limit_is_nan(distance, a, b):
    result = evaluate_transferability_limits(np.array([distance]), np.array([a]), np.array([b]))
    assert result.tolist() == [0]


def test_assess_transferability_returns_different_class_given_rounding_of_the_limits():
    """ADR 0004: TRIX 0.5 gives A = 8.4565 km, rounded 8.5 km. A pair at 8.48 km switches class."""
    kwargs = {
        "rix_site": [0.0],
        "elevation_site": [5.0],
        "rix_reference": [0.0],
        "elevation_reference": [0.0],
        "xy_site": [[0.0, 0.0]],
        "xy_reference": [[8480.0, 0.0]],
    }
    assert assess_transferability(**kwargs, decimals=None).transferability.tolist() == [[1]]
    assert assess_transferability(**kwargs, decimals=1).transferability.tolist() == [[2]]


def test_assess_transferability_disgrants_transfer_given_rix_nan():
    result = assess_transferability(
        rix_site=[np.nan],
        elevation_site=[0.0],
        rix_reference=[0.0],
        elevation_reference=[0.0],
        xy_site=[[0.0, 0.0]],
        xy_reference=[[1000.0, 0.0]],
        decimals=1,
    )
    assert np.isnan(result.trix).all() and np.isnan(result.A).all() and np.isnan(result.B).all()
    assert result.transferability.tolist() == [[0]]


def test_assess_transferability_equals_a_pairwise_loop():
    rix_s, z_s, xy_s = [0.1, 0.3], [100.0, 400.0], [[0.0, 0.0], [5000.0, 0.0]]
    rix_r, z_r, xy_r = [0.2, 0.6, 0.0], [120.0, 300.0, 900.0], [[1000.0, 0.0], [0.0, 6000.0], [9000.0, 9000.0]]

    result = assess_transferability(
        rix_site=rix_s,
        elevation_site=z_s,
        rix_reference=rix_r,
        elevation_reference=z_r,
        xy_site=xy_s,
        xy_reference=xy_r,
        decimals=1,
    )

    for i in range(2):
        for j in range(3):
            trix = 0.9 * 50 * (rix_s[i] + rix_r[j]) + 0.1 * abs(z_s[i] - z_r[j])
            a = round(max(8.5 - 0.087 * trix, 1.5), 1)
            b = round(max(15.0 - 0.14 * trix, 3.0), 1)
            distance = np.hypot(xy_s[i][0] - xy_r[j][0], xy_s[i][1] - xy_r[j][1]) / 1000
            expected = 2 if distance <= a else 1 if distance <= b else 0
            assert result.trix[i, j] == pytest.approx(trix)
            assert result.A[i, j] == pytest.approx(a)
            assert result.B[i, j] == pytest.approx(b)
            assert result.distance_km[i, j] == pytest.approx(distance)
            assert result.transferability[i, j] == expected


def test_pairwise_distances_km_returns_euclidean_in_kilometres():
    result = pairwise_distances_km([[0.0, 0.0], [3000.0, 0.0]], [[3000.0, 4000.0], [0.0, 0.0], [6000.0, 8000.0]])
    assert result.shape == (2, 3)
    np.testing.assert_allclose(result[0], [5.0, 0.0, 10.0])


@pytest.mark.parametrize("bad", [[0.0, 0.0], [[0.0, 0.0, 10.0]]])
def test_pairwise_distances_reject_anything_but_xy_pairs(bad):
    with pytest.raises(ValueError, match="shape"):
        pairwise_distances_km(bad, [[0.0, 0.0]])


@pytest.mark.parametrize(
    "override",
    [
        {"elevation_site": [0.0, 0.0]},
        {"xy_site": [[0.0, 0.0], [1.0, 1.0]]},
        {"elevation_reference": [0.0, 0.0]},
        {"xy_reference": [[0.0, 0.0], [1.0, 1.0]]},
    ],
)
def test_assess_transferability_rejects_mismatching_sizes(override):
    kwargs = {
        "rix_site": [0.0],
        "elevation_site": [0.0],
        "rix_reference": [0.0],
        "elevation_reference": [0.0],
        "xy_site": [[0.0, 0.0]],
        "xy_reference": [[1.0, 1.0]],
        "decimals": 1,
    }
    with pytest.raises(ValueError, match="same"):
        assess_transferability(**{**kwargs, **override})
