import numpy as np
import pyproj
import pytest
import shapely.geometry
import xarray

from phoibe.geography.complexity.rix.computations.fieldsampler import Field, RegularGridXYSampler
from phoibe.geography.complexity.rix.computations.rix import compute_regular_rix

ORIGIN = shapely.geometry.Point(0.0, 0.0)


@pytest.fixture
def make_sampler():
    """Sampler of the plane z = a x + b y on a 10 m grid from -half to +half."""

    def make(a=1.0, b=0.0, half=1200):
        x = np.arange(-half, half + 1, 10)
        da = xarray.DataArray(a * x[None, :] + b * x[:, None], coords={"x": x, "y": x}, dims=("y", "x"))
        return RegularGridXYSampler(Field.from_array(da), "linear")

    return make


def rix(sampler, location=ORIGIN, **kwargs):
    parameters = {"n_angles": 8, "R_km": 1.0, "dr_km": 0.1, "crs": None, "slope_critical": 0.5}
    return compute_regular_rix(location=location, sampler=sampler, **{**parameters, **kwargs})


@pytest.mark.parametrize(
    "n_angles, expected",
    [(1, [0.0]), (4, [0.0, 90.0, 180.0, 270.0]), (8, [0, 45, 90, 135, 180, 225, 270, 315])],
)
def test_compute_regular_rix_returns_rays_spread_evenly_over_the_circle(make_sampler, n_angles, expected):
    assert rix(make_sampler(), n_angles=n_angles).angles.tolist() == expected  # 360° itself is the ray of 0°


@pytest.mark.parametrize(
    "slope_critical, expected_rix",
    [(0.5, 6 / 8), (0.8, 2 / 8), (1.1, 0.0)],
)
def test_compute_regular_rix_returns_the_share_of_steep_rays_given_a_plane(make_sampler, slope_critical, expected_rix):
    assert rix(make_sampler(a=1.0), slope_critical=slope_critical).rix == pytest.approx(expected_rix)


def test_compute_regular_rix_returns_zero_given_a_flat_field(make_sampler):
    assert rix(make_sampler(a=0.0)).rix == 0.0


def test_compute_regular_rix_returns_rays_with_the_given_grid_crs_and_slope(make_sampler):
    radial = rix(make_sampler(), R_km=0.5, dr_km=0.05, crs="EPSG:32633", slope_critical=0.7)
    np.testing.assert_allclose(radial.rays[0].profile.r_m, np.arange(11) * 50.0)
    assert radial.rays[0].profile.ray.crs == pyproj.CRS.from_epsg(32633)
    assert radial.slope_critical == 0.7


def test_compute_regular_rix_returns_rays_starting_at_the_location(make_sampler):
    radial = rix(make_sampler(a=1.0, b=2.0), location=shapely.geometry.Point(100.0, 50.0))
    assert radial.z == pytest.approx(200.0)  # z = x + 2 y at the location
    assert all(ray.profile.z[0] == pytest.approx(200.0) for ray in radial.rays)


def test_compute_regular_rix_returns_the_nan_count_given_rays_leaving_the_field(make_sampler):
    radial = rix(make_sampler(half=500), n_angles=4)  # 5 of 11 points lie outside along each ray
    assert radial.runs.nan_count == 4 * 5


@pytest.mark.parametrize("nan_policy", ["mask", "truncate"])
def test_compute_regular_rix_returns_the_profile_of_the_policy_given_rays_leaving_the_field(make_sampler, nan_policy):
    radial = rix(make_sampler(half=500), n_angles=4, nan_policy=nan_policy)
    expected_points = {"mask": 11, "truncate": 6}[nan_policy]
    assert all(len(ray.profile.z) == expected_points for ray in radial.rays)


def test_compute_regular_rix_raises_given_rays_leaving_the_field_and_policy_error(make_sampler):
    with pytest.raises(ValueError, match="NaNs encountered"):
        rix(make_sampler(half=500), n_angles=4, nan_policy="error")
