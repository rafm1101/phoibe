import numpy as np
import pytest

from phoibe.geography.complexity.rix.trix import (
    _ensure_1D,
    compute_trix,
    compute_trix_limit_distances,
    evaluate_transferability_limits,
)


class TestComputeTrix:
    """Unit tests for compute_trix business logic."""

    def test_recover_given_reference_values(self):
        rix_site = [0.0, 1.0]
        elevation_site = [100.0, 200.0]
        rix_wind = [0.5]
        elevation_wind = [150.0]

        result = compute_trix(rix_site, elevation_site, rix_wind, elevation_wind)

        expected_table = np.array([[27.5], [72.5]])
        assert result.shape == (2, 1)
        np.testing.assert_allclose(result, expected_table)

    def test_return_scalar_given_scalar_inputs(self):
        result = compute_trix(0.2, 100.0, 0.4, 100.0)
        assert result.shape == (1, 1)
        np.testing.assert_allclose(result, [[27.0]])

    def test_use_additive_not_multiplicative_rix_combination(self):
        result = compute_trix([0.0], [0.0], [1.0], [0.0])
        np.testing.assert_allclose(result, [[45.0]])

    def test_return_zero_given_zero_rix_and_zero_elevation(self):
        result = compute_trix([0.0], [0.0], [0.0], [0.0])
        np.testing.assert_allclose(result, [[0.0]])

    def test_output_shape_is_outer_product_of_site_and_wind(self):
        rix_site = [0.1, 0.2, 0.3]
        elevation_site = [10.0, 20.0, 30.0]
        rix_wind = [0.4, 0.5]
        elevation_wind = [40.0, 50.0]

        result = compute_trix(rix_site, elevation_site, rix_wind, elevation_wind)
        assert result.shape == (3, 2)


class TestComputeTrixLimitDistances:
    """Unit tests for compute_trix_limit_distances."""

    def test_return_unclipped_intercepts_given_zero_trix(self):
        A, B = compute_trix_limit_distances(np.array([0.0]))
        np.testing.assert_allclose(A, [8.5])
        np.testing.assert_allclose(B, [15.0])

    def test_monotonicity_in_range_for_a_floors_before_b(self):
        A, B = compute_trix_limit_distances(np.array([82.0]), decimals=None)
        np.testing.assert_allclose(A, [1.5])
        np.testing.assert_allclose(B, [3.52])

    def test_both_floors_active_given_large_trix(self):
        A, B = compute_trix_limit_distances(np.array([100.0]))
        np.testing.assert_allclose(A, [1.5])
        np.testing.assert_allclose(B, [3.0])

    def test_a_never_exceeds_b_given_representative_range(self):
        trix_values = np.linspace(0, 500, 200)
        A, B = compute_trix_limit_distances(trix_values, decimals=None)
        assert np.all(A <= B)

    def test_default_rounds_to_one_decimal(self):
        A, B = compute_trix_limit_distances(np.array([10.0]))
        np.testing.assert_allclose(A, [7.6])
        np.testing.assert_allclose(B, [13.6])

    def test_decimals_none_disables_rounding(self):
        A, B = compute_trix_limit_distances(np.array([10.0]), decimals=None)
        np.testing.assert_allclose(A, [7.63])
        np.testing.assert_allclose(B, [13.6])


class TestEvaluateTransferabilityLimits:
    """Unit tests for evaluate_transferability_limits."""

    def test_distance_below_a_is_fully_transferable(self):
        result = evaluate_transferability_limits(np.array([1.0]), np.array([1.5]), np.array([3.0]))
        assert result.tolist() == [2]

    def test_distance_equal_to_a_is_still_fully_transferable(self):
        result = evaluate_transferability_limits(np.array([1.5]), np.array([1.5]), np.array([3.0]))
        assert result.tolist() == [2]

    def test_distance_between_a_and_b_is_conditionally_transferable(self):
        result = evaluate_transferability_limits(np.array([2.0]), np.array([1.5]), np.array([3.0]))
        assert result.tolist() == [1]

    def test_distance_equal_to_b_is_still_conditionally_transferable(self):
        result = evaluate_transferability_limits(np.array([3.0]), np.array([1.5]), np.array([3.0]))
        assert result.tolist() == [1]

    def test_distance_above_b_is_not_transferable(self):
        result = evaluate_transferability_limits(np.array([3.1]), np.array([1.5]), np.array([3.0]))
        assert result.tolist() == [0]

    def test_mixed_array_classifies_each_pair_independently(self):
        distances = np.array([1.0, 2.0, 5.0])
        A = np.array([1.5, 1.5, 1.5])
        B = np.array([3.0, 3.0, 3.0])
        result = evaluate_transferability_limits(distances, A, B)
        assert result.tolist() == [2, 1, 0]

    def test_result_dtype_is_int(self):
        result = evaluate_transferability_limits(np.array([1.0]), np.array([1.5]), np.array([3.0]))
        assert np.issubdtype(result.dtype, np.integer)


class TestEnsure1D:
    """Unit tests for the private _ensure_1D helper."""

    def test_promote_to_length_one_array_given_scalar(self):
        result = _ensure_1D(3.0)
        assert result.shape == (1,)
        np.testing.assert_allclose(result, [3.0])

    def test_promote_to_array_given_list(self):
        result = _ensure_1D([1, 2, 3])
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_pass_1d_array(self):
        result = _ensure_1D(np.array([1.0, 2.0]))
        assert result.shape == (2,)

    def test_raise_type_error_given_2d_array(self):
        with pytest.raises(TypeError, match="Argument must be a scalar or 1D array"):
            _ensure_1D(np.array([[1.0, 2.0], [3.0, 4.0]]))
