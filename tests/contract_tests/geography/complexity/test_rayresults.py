import numpy as np
import pytest

from phoibe.geography.complexity.rix.results import RayResult


class RayResultContract:
    """Contracts that any RayResult must satisfy."""

    def test_verify_required_attributes(self, ray_result):
        assert hasattr(ray_result, "theta")
        assert hasattr(ray_result, "ruggedness")
        assert hasattr(ray_result, "total_length_m")
        assert hasattr(ray_result, "steep_length_m")
        assert hasattr(ray_result, "max_abs_slope")
        assert hasattr(ray_result, "n_steep_segments")
        assert hasattr(ray_result, "steep_segments")

    def test_verify_valid_instances(self, ray_result):
        assert isinstance(ray_result.theta, float)
        assert isinstance(ray_result.ruggedness, float)
        assert isinstance(ray_result.total_length_m, float)
        assert isinstance(ray_result.steep_length_m, float)
        assert isinstance(ray_result.max_abs_slope, (float, type(np.nan)))
        assert isinstance(ray_result.n_steep_segments, int)
        assert isinstance(ray_result.steep_segments, tuple)

    def test_verify_lengths_consistent(self, ray_result):
        assert ray_result.n_steep_segments == len(ray_result.steep_segments)

    def test_verify_ruggedness_in_valid_range(self, ray_result):
        ruggedness = ray_result.ruggedness
        assert np.isnan(ruggedness) or (0.0 <= ruggedness <= 1.0)

    def test_verify_properties(self, ray_result):
        assert ray_result.steep_length_m <= ray_result.total_length_m + 1e-7

    def test_verify_describe_returns_dict(self, ray_result):
        description = ray_result.describe()
        assert isinstance(description, dict)
        assert "theta" in description
        assert "ruggedness" in description
        assert "total_length_m" in description

    def test_cached_properties_are_stable(self, ray_result):
        ruggedness1 = ray_result.ruggedness
        ruggedness2 = ray_result.ruggedness
        assert ruggedness1 == ruggedness2 or (np.isnan(ruggedness1) and np.isnan(ruggedness2))

        steep_length1 = ray_result.steep_length_m
        steep_length2 = ray_result.steep_length_m
        assert steep_length1 == steep_length2


class RayResultContractFlatProfile(RayResultContract):

    def test_ruggedness_is_zero(self, ray_result):
        assert np.isclose(ray_result.ruggedness, 0.0)

    def test_steep_length_is_zero(self, ray_result):
        assert np.isclose(ray_result.steep_length_m, 0.0)

    def test_no_steep_segments(self, ray_result):
        assert ray_result.n_steep_segments == 0
        assert len(ray_result.steep_segments) == 0


class RayResultContractSteepProfile(RayResultContract):

    def test_ruggedness_is_positive(self, ray_result):
        assert ray_result.ruggedness > 0

    def test_steep_length_is_positive(self, ray_result):
        assert ray_result.steep_length_m > 0

    def test_has_steep_segments(self, ray_result):
        assert ray_result.n_steep_segments > 0


class TestRayResultFlatProfile(RayResultContractFlatProfile):
    @pytest.fixture
    def ray_result(self, make_ray_result, flat_sampler):
        return make_ray_result(flat_sampler, slope_critical=0.0)


class TestRayResultLinearProfile(RayResultContract):
    @pytest.fixture
    def ray_result(self, make_ray_result, linear_sampler):
        return make_ray_result(linear_sampler, slope_critical=0.009)

    def test_linear_has_expected_ruggedness(self, ray_result):
        assert ray_result.ruggedness > 0


class TestRayResultSteep(RayResultContractSteepProfile):
    @pytest.fixture
    def ray_result(self, make_ray_result, steep_sampler):
        return make_ray_result(steep_sampler, slope_critical=0.49)


class RadialRixResultContract:

    def test_verify_required_attributes(self, radial_result):
        assert hasattr(radial_result, "rix")
        assert hasattr(radial_result, "n_rays")
        assert hasattr(radial_result, "angles")
        assert hasattr(radial_result, "slope_critical")

    def test_verify_valid_instances(self, radial_result):
        assert isinstance(radial_result.rix, float)
        assert isinstance(radial_result.n_rays, int)
        assert isinstance(radial_result.angles, np.ndarray)
        assert isinstance(radial_result.slope_critical, float)

    def test_verify_lengths_consistent(self, radial_result):
        assert len(radial_result.angles) == radial_result.n_rays
        assert len(radial_result.rays) == radial_result.n_rays
        assert len(radial_result.ruggednesses) == radial_result.n_rays

    def test_angles_are_sorted(self, radial_result):
        angles = radial_result.angles
        assert np.all(angles[:-1] <= angles[1:])

    def test_verify_ray_access_returns_valid_ray(self, radial_result):
        theta = radial_result.angles[0]
        ray_result = radial_result.ray(theta)
        assert isinstance(ray_result, RayResult)
        assert ray_result.theta == theta

    def test_verify_ray_access_fails_for_invalid_angle(self, radial_result):
        with pytest.raises(KeyError, match="No ray found"):
            radial_result.ray(999.0)

    def test_verify_rix_in_valid_range(self, radial_result):
        rix = radial_result.rix
        assert np.isnan(rix) or (0.0 <= rix <= 1.0)
        ruggednesses = radial_result.ruggednesses
        assert np.all((ruggednesses >= 0.0) & (ruggednesses <= 1.0))

    def test_verify_describe_returns_dict(self, radial_result):
        description = radial_result.describe()
        assert isinstance(description, dict)
        assert "rix_mean" in description
        assert "rix_std" in description
        assert "n_rays" in description


class RadialRixResultContractFlatProfile(RadialRixResultContract):

    def test_verify_ruggedness_is_zero(self, radial_result):
        assert np.isclose(radial_result.rix, 0.0)
        directional_rix = radial_result.directional_stats()
        assert np.allclose(directional_rix, 0.0)


class TestRadialRixResultMultipleAngles(RadialRixResultContract):
    @pytest.fixture(params=[4, 8, 72])
    def radial_result(self, request, make_radial_result):
        n_angles = request.param
        return make_radial_result(n_angles=n_angles, slope_critical=0.3)
