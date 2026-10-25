import numpy as np
import pyproj
import pytest

from phoibe.geography.complexity.rix.computations.checks import (
    CENTRE_ELEVATION,
    CRS_SITE_VS_REFERENCE,
    CRS_VS_CONFIG,
    N_ANGLE,
    NAN_COUNT,
    RIX_NAN,
    UNIQUE_IDS,
    centre_elevation_findings,
    crs_config_findings,
    crs_consistency_findings,
    n_angle_findings,
    nan_findings,
    rix_findings,
    unique_id_findings,
)
from phoibe.geography.complexity.rix.computations.results import RadialRuggedness, RayRuggedness


@pytest.fixture
def make_radial(make_profile):
    """Radial result with rays at 0°, 10°, ...; a spec is `z` or `(z, nan_count, n_points)`."""

    def make(*specs):
        rays = []
        for k, spec in enumerate(specs):
            z, nan_count, n_points = spec if isinstance(spec, tuple) else (spec, 0, None)
            profile = make_profile(np.arange(len(z)) * 100.0, z, theta=10.0 * k, nan_count=nan_count, n_points=n_points)
            rays.append(RayRuggedness(profile, slope_critical=0.1))
        return RadialRuggedness(rays=tuple(rays))

    return make


def test_nan_findings_returns_nothing_given_no_nan(make_radial):
    assert nan_findings(make_radial([0, 1], [0, 1])) == ()


def test_nan_findings_returns_a_finding_per_ray_above_the_threshold(make_radial):
    radial = make_radial(([0, 1], 2, 10), ([0, 1], 1, 10), ([0, 1], 0, 10))
    findings = nan_findings(radial, max_fraction=0.1)
    assert [(f.check, f.theta) for f in findings] == [(NAN_COUNT, 0.0)]


def test_nan_findings_returns_a_finding_for_any_nan_given_the_default_threshold(make_radial):
    findings = nan_findings(make_radial(([0, 1], 1, 100)))
    assert [(f.check, f.theta) for f in findings] == [(NAN_COUNT, 0.0)]


def test_nan_findings_returns_the_count_of_sampling_given_a_truncated_ray(make_radial):
    findings = nan_findings(make_radial(([0, 1], 3, 10)))
    assert "3 of 10" in findings[0].detail


def test_nan_findings_returns_nothing_given_no_sampled_points(make_radial):
    assert nan_findings(make_radial(([0, 1], 0, 0))) == ()


def test_centre_elevation_findings_returns_nothing_given_equal_centres(make_radial):
    assert centre_elevation_findings(make_radial([10, 1], [10, 1], [10, 1])) == ()


def test_centre_elevation_findings_returns_only_the_deviating_ray(make_radial):
    radial = make_radial([10, 1], [10, 1], [10, 1], [10, 1], [15, 1])
    findings = centre_elevation_findings(radial)
    assert [(f.check, f.theta) for f in findings] == [(CENTRE_ELEVATION, 40.0)]


def test_centre_elevation_findings_returns_nothing_given_a_deviation_equal_to_atol(make_radial):
    assert centre_elevation_findings(make_radial([10, 1], [10, 1], [10.5, 1]), atol=0.5) == ()


def test_centre_elevation_findings_returns_no_ray_finding_given_a_nan_centre(make_radial):
    findings = centre_elevation_findings(make_radial([10, 1], [10, 1], [np.nan, 1]))
    assert [f for f in findings if f.theta is not None] == []


@pytest.mark.parametrize(
    "zs",
    [
        [[np.nan, 1], [np.nan, 1], [np.nan, 1]],
        [[10, 1], [10, 1], [np.nan, 1]],
    ],
)
def test_centre_elevation_findings_returns_one_location_finding_given_a_missing_centre(make_radial, zs):
    findings = centre_elevation_findings(make_radial(*zs))
    assert [(f.check, f.theta) for f in findings] == [(CENTRE_ELEVATION, None)]


def test_rix_findings_returns_nothing_given_finite_ruggedness(make_radial):
    assert rix_findings(make_radial([0, 1], [0, 1])) == ()


def test_rix_findings_returns_one_finding_per_location_however_many_rays_are_affected(make_radial):
    radial = make_radial([5], [5], [5], [0, 1], [0, 1])  # three rays without a segment
    findings = rix_findings(radial)
    assert [(f.check, f.theta) for f in findings] == [(RIX_NAN, None)]
    assert "3 of 5" in findings[0].detail


def test_rix_findings_returns_at_most_five_directions_in_the_detail(make_radial):
    detail = rix_findings(make_radial(*[[5]] * 7))[0].detail
    assert "40°, ..." in detail
    assert "50°" not in detail


@pytest.mark.parametrize("n_angles, expected_count", [(0, 1), (1, 0), (72, 0), (360, 0), (361, 1)])
def test_n_angle_findings_returns_a_finding_outside_the_inclusive_range(n_angles, expected_count):
    findings = n_angle_findings(n_angles, (1, 360))
    assert [f.check for f in findings] == [N_ANGLE] * expected_count


@pytest.mark.parametrize(
    "crs_site, crs_reference, expected_count",
    [
        (pyproj.CRS.from_epsg(2056), pyproj.CRS.from_epsg(2056), 0),
        (pyproj.CRS.from_epsg(2056), pyproj.CRS.from_user_input("urn:ogc:def:crs:EPSG::2056"), 0),  # other notation
        (None, None, 0),
        (pyproj.CRS.from_epsg(2056), pyproj.CRS.from_epsg(32633), 1),
        (pyproj.CRS.from_epsg(2056), None, 1),
        (None, pyproj.CRS.from_epsg(2056), 1),
    ],
)
def test_crs_consistency_findings_returns_a_finding_given_different_crs(crs_site, crs_reference, expected_count):
    findings = crs_consistency_findings(crs_site, crs_reference)
    assert [f.check for f in findings] == [CRS_SITE_VS_REFERENCE] * expected_count


@pytest.mark.parametrize(
    "crs_site, crs_config, expected_count",
    [
        (pyproj.CRS.from_epsg(2056), "EPSG:2056", 0),
        (pyproj.CRS.from_epsg(2056), "EPSG:32633", 1),
        (pyproj.CRS.from_epsg(2056), None, 0),  # nothing to compare with
        (None, "EPSG:2056", 0),  # unknown
    ],
)
def test_crs_config_findings_returns_a_finding_given_a_different_configured_crs(crs_site, crs_config, expected_count):
    findings = crs_config_findings(crs_site, crs_config)
    assert [f.check for f in findings] == [CRS_VS_CONFIG] * expected_count


def test_unique_id_findings_returns_nothing_given_unique_ids():
    assert unique_id_findings(["a", "b", "c"]) == ()


def test_unique_id_findings_returns_one_finding_with_at_most_three_examples():
    findings = unique_id_findings(["a", "a", "b", "b", "c", "c", "d", "d"])
    assert [f.check for f in findings] == [UNIQUE_IDS]
    assert "4 identifier(s)" in findings[0].detail
    assert "'c'" in findings[0].detail and "'d'" not in findings[0].detail
