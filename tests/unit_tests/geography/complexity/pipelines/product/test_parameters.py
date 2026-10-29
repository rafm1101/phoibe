import typing

import pytest

from phoibe.geography.complexity.rix.computations import INTERPOLATION_METHODS
from phoibe.geography.complexity.rix.pipelines.product.parameters import RunParameters, SampleParameters


def run_parameters(**changes):
    return RunParameters(
        **{**{"n_angles": 72, "R_km": 3.5, "dr_km": 0.05, "slope_critical": 0.033, "crs": None}, **changes}
    )


def test_run_parameters_returns_the_given_values():
    parameters = run_parameters(crs="EPSG:2056")
    assert (parameters.n_angles, parameters.R_km, parameters.dr_km, parameters.crs) == (72, 3.5, 0.05, "EPSG:2056")


@pytest.mark.parametrize("dr_km", [0.0, -1.0, 3.5, 4.0, float("nan")])  # not within (0, R_km)
def test_run_parameters_raises_given_a_step_outside_zero_and_the_radius(dr_km):
    with pytest.raises(ValueError, match="dr_km must lie within"):
        run_parameters(dr_km=dr_km)


@pytest.mark.parametrize("slope_critical", [0.0, -0.033])
def test_run_parameters_raises_given_a_slope_that_is_not_positive(slope_critical):
    with pytest.raises(ValueError, match="slope_critical must be positive"):
        run_parameters(slope_critical=slope_critical)


def test_run_parameters_raises_given_a_nan_slope():
    with pytest.raises(ValueError, match="slope_critical"):
        run_parameters(slope_critical=float("nan"))


@pytest.mark.parametrize("method", typing.get_args(INTERPOLATION_METHODS))
def test_sample_parameters_returns_every_method_of_the_fieldsampler(method):
    assert SampleParameters(interpolation_method=method).interpolation_method == method


def test_sample_parameters_raises_given_an_unknown_method():
    with pytest.raises(ValueError, match="Unknown interpolation_method"):
        SampleParameters(interpolation_method="cubic")
