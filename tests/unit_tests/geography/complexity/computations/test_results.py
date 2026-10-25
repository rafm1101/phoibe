import numpy as np
import pytest

from phoibe.geography.complexity.rix.computations.fieldsampler import FieldInfo
from phoibe.geography.complexity.rix.computations.results import RadialRuggedness, RayProfileMeta

STEEP = [0, 50, 100]
FLAT = [0, 0, 0]
FIELD = FieldInfo(crs="EPSG:32633", extent=(0.0, 0.0, 10.0, 10.0), resolution=(30.0, -30.0))


@pytest.mark.parametrize(
    "z, slope_critical, expected",
    [
        (
            [0, 100, 200, 300, 400],
            0.3,
            {"ruggedness": 1.0, "steep_length_m": 1000, "total_length_m": 1000, "max_abs_slope": 0.4},
        ),
        ([5, 5, 5, 5, 5], 0.1, {"ruggedness": 0.0, "steep_length_m": 0, "total_length_m": 1000, "max_abs_slope": 0.0}),
        (
            [100, 100, 1100, 1200, 400],
            3.0,
            {"ruggedness": 0.5, "steep_length_m": 500, "total_length_m": 1000, "max_abs_slope": 4.0},
        ),
    ],
)
def test_ray_ruggedness_returns_the_metrics_of_its_profile(make_ray, z, slope_critical, expected):
    ray = make_ray(z, slope_critical=slope_critical, dr_m=250.0)
    assert {name: getattr(ray, name) for name in expected} == pytest.approx(expected)


def test_describe_of_a_ray_returns_the_metrics_of_its_profile(make_ray):
    ray = make_ray([100, 100, 1100, 1200, 400], theta=45.0, slope_critical=3.0, dr_m=250.0)
    assert ray.describe() == pytest.approx(
        {
            "theta": 45.0,
            "ruggedness": 0.5,
            "total_length_m": 1000,
            "steep_length_m": 500,
            "max_abs_slope": 4.0,
            "n_steep_segments": 2,
        }
    )


def test_steep_length_m_returns_zero_given_a_profile_without_a_segment(make_ray):
    assert make_ray([5.0]).steep_length_m == 0.0


@pytest.mark.parametrize("field_info", [FieldInfo(None, None, None), FIELD])
def test_meta_of_a_ray_returns_the_field_and_the_sampling_info_of_its_profile(make_ray, field_info):
    meta = make_ray([0, 1, 2], nan_count=1, field_info=field_info).meta
    assert (meta.crs_dem, meta.extent, meta.resolution, meta.nan_count) == (
        field_info.crs,
        field_info.extent,
        field_info.resolution,
        1,
    )


def test_from_runs_returns_the_field_of_the_runs_and_the_summed_nan_count(make_profile):
    runs = {
        theta: make_profile(
            [0, 100], [0, 1], theta=theta, nan_count=n, field_info=FIELD, crs="EPSG:32633"
        ).sampling_info
        for theta, n in ((0.0, 1), (10.0, 2), (20.0, 0))
    }
    meta = RayProfileMeta.from_runs(runs)
    assert (meta.crs_dem, meta.extent, meta.resolution, meta.nan_count) == (
        "EPSG:32633",
        FIELD.extent,
        FIELD.resolution,
        3,
    )
    assert meta.crs_ray == ["EPSG:32633"]
    assert len(meta.messages) == 3


@pytest.mark.parametrize(
    "fields", [[], [FIELD, FieldInfo("EPSG:2056", None, None)]], ids=["no runs", "different fields"]
)
def test_from_runs_returns_no_field_given_none_or_different_fields(make_profile, fields):
    runs = {
        float(k): make_profile([0, 100], [0, 1], theta=float(k), field_info=f).sampling_info
        for k, f in enumerate(fields)
    }
    meta = RayProfileMeta.from_runs(runs)
    assert (meta.crs_dem, meta.extent, meta.resolution) == (None, None, None)


def test_radial_ruggedness_raises_given_no_rays():
    with pytest.raises(ValueError, match="at least one ray"):
        RadialRuggedness(rays=())


def test_radial_ruggedness_raises_given_mixed_slope_critical(make_ray):
    with pytest.raises(ValueError, match="same slope_critical"):
        RadialRuggedness(rays=(make_ray(STEEP, slope_critical=0.1), make_ray(STEEP, theta=10.0, slope_critical=0.7)))


def test_radial_ruggedness_raises_given_mixed_crs(make_ray):
    with pytest.raises(ValueError, match="same crs"):
        RadialRuggedness(rays=(make_ray(STEEP, crs="EPSG:32633"), make_ray(STEEP, theta=10.0, crs="EPSG:2056")))


def test_ruggednesses_returns_the_values_in_the_order_of_the_sorted_angles(make_ray):
    rays = (make_ray(FLAT, theta=20.0), make_ray(STEEP, theta=0.0), make_ray([0, 0, 50], theta=10.0))
    radial = RadialRuggedness(rays=rays)
    assert radial.angles.tolist() == [0.0, 10.0, 20.0]
    assert radial.ruggednesses.tolist() == pytest.approx([1.0, 0.5, 0.0])


def test_rix_returns_the_mean_over_the_rays(make_radial):
    assert make_radial(STEEP, FLAT, FLAT).rix == pytest.approx(1 / 3)


def test_describe_of_a_radial_returns_the_statistics_over_the_rays(make_radial):
    expected = {
        "rix": 1 / 3,
        "rix_std": np.sqrt(2) / 3,
        "rix_min": 0.0,
        "rix_max": 1.0,
        "n_rays": 3,
        "slope_critical": 0.1,
    }
    assert make_radial(STEEP, FLAT, FLAT).describe() == pytest.approx(expected)


def test_z_returns_the_centre_elevation(make_radial):
    assert make_radial([10, 11, 12], [10, 20, 30]).z == 10.0


@pytest.mark.parametrize(
    "theta",
    [10.0, 10.0 - 1e-6, 10.0 + 1e-6],
)
def test_ray_returns_the_ray_within_the_tolerance(make_radial, theta):
    assert make_radial(FLAT, STEEP, FLAT).ray(theta).ruggedness == pytest.approx(1.0)  # the ray at 10° is the steep one


@pytest.mark.parametrize("theta", [7.0, 10.001])
def test_ray_raises_given_no_ray_within_the_tolerance(make_radial, theta):
    with pytest.raises(KeyError):
        make_radial(FLAT, STEEP, FLAT).ray(theta)


def test_runs_of_a_radial_return_the_nan_count_summed_over_the_rays(make_radial):
    assert make_radial(([0, 1], 1, 2), ([0, 1], 2, 2), ([0, 1], 0, 2)).runs.nan_count == 3
