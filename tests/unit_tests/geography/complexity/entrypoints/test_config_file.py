import pytest

from phoibe.geography.complexity.rix.entrypoints.config_file import (
    RunConfiguration,
    default_configuration,
    load_configuration,
)
from phoibe.geography.complexity.rix.pipelines.errors import ConfigurationError
from phoibe.geography.complexity.rix.pipelines.requirements import REQUIRED_FIXED_PARAMETERS

FIXED = [name for _, name in REQUIRED_FIXED_PARAMETERS]


def configuration(spec, **parameters):
    return load_configuration({"parameters": parameters}, spec)


def test_default_configuration_returns_the_defaults_of_the_specification(product_spec):
    configuration = default_configuration(product_spec)
    default = lambda name: next(p.value for p in product_spec.parameters if p.name == name)  # noqa: E731
    run, sampler = configuration.parameters, configuration.sampler_parameters
    assert (run.n_angles, run.R_km, run.dr_km, run.slope_critical, run.crs) == tuple(
        default(name) for name in ("n_angles", "R_km", "dr_km", "slope_critical", "crs")
    )
    assert (sampler.interpolation_method, configuration.project_name) == (default("interpolation_method"), "")


def test_load_configuration_returns_the_supplied_values_and_the_defaults_for_the_rest(product_spec):
    loaded = load_configuration(
        {"project_name": "demo", "parameters": {"n_angles": 36, "crs": "EPSG:2056"}}, product_spec
    )
    default = default_configuration(product_spec)
    assert (loaded.parameters.n_angles, loaded.parameters.crs, loaded.project_name) == (36, "EPSG:2056", "demo")
    assert (loaded.parameters.R_km, loaded.parameters.dr_km, loaded.sampler_parameters) == (
        default.parameters.R_km,
        default.parameters.dr_km,
        default.sampler_parameters,
    )
    assert isinstance(loaded, RunConfiguration)


@pytest.mark.parametrize(
    "source", [{}, {"parameters": None}, {"parameters": {}}], ids=["empty", "no parameters", "empty parameters"]
)
def test_load_configuration_returns_the_defaults_given_no_parameters(product_spec, source):
    assert load_configuration(source, product_spec) == default_configuration(product_spec)


def test_load_configuration_returns_a_changed_locked_parameter_without_complaint(product_spec):
    """A deviation from TR6 is a finding of the run, not an error of the configuration. So is an unlikely anglecount."""
    loaded = configuration(product_spec, R_km=5.0, n_angles=0, slope_critical=0.3)
    assert (loaded.parameters.R_km, loaded.parameters.n_angles, loaded.parameters.slope_critical) == (5.0, 0, 0.3)


def test_load_configuration_returns_a_float_given_an_integer_and_none_given_a_default_of_none(product_spec):
    loaded = configuration(product_spec, R_km=4, crs=None)
    assert (loaded.parameters.R_km, loaded.parameters.crs) == (4, None)


@pytest.mark.parametrize(
    "source, expected",
    [
        ({"unknown": 1}, "Unknown key(s) ['unknown']"),
        ({"project_name": 3}, "project_name must be text"),
        ({"parameters": [1, 2]}, "parameters must be a mapping"),
        ({"parameters": {"nonsense": 1}}, "unknown parameter 'nonsense'"),
        ({"parameters": {"n_angles": "36"}}, "'n_angles' must be of type int"),
        ({"parameters": {"n_angles": True}}, "'n_angles' must be of type int"),  # a bool is no number here
        ({"parameters": {"n_angles": 36.0}}, "'n_angles' must be of type int"),
        ({"parameters": {"crs": 2056}}, "'crs' must be of type str"),
        ({"parameters": {"n_angles": None}}, "'n_angles' must not be empty"),
        ({"parameters": {"dr_km": 10.0}}, "'dr_km' is 10.0, allowed range is"),
        ({"parameters": {"dr_km": 0}}, "'dr_km' is 0, allowed range is"),
        ({"parameters": {"R_km": 0.04}}, "dr_km must lie within"),  # the step of 0.05 km does not fit into the radius
        ({"parameters": {"slope_critical": 0.0}}, "slope_critical must be positive"),
        ({"parameters": {"interpolation_method": "cubic"}}, "Unknown interpolation_method"),
    ],
)
def test_load_configuration_raises_given_an_invalid_configuration(product_spec, source, expected):
    with pytest.raises(
        ConfigurationError,
        match=expected.replace("[", r"\[").replace("]", r"\]").replace("(", r"\(").replace(")", r"\)"),
    ):
        load_configuration(source, product_spec)


@pytest.mark.parametrize("name", FIXED)
def test_load_configuration_raises_given_a_parameter_the_product_fixes(product_spec, name):
    with pytest.raises(
        ConfigurationError, match=f"parameter '{name}' is fixed by the product and cannot be configured"
    ):
        configuration(product_spec, **{name: "anything"})


def test_load_configuration_raises_listing_all_problems_and_the_configurable_parameters(product_spec):
    with pytest.raises(ConfigurationError) as error:
        configuration(product_spec, nonsense=1, **{FIXED[0]: 1})
    message = str(error.value)
    assert "unknown parameter 'nonsense'" in message and "fixed by the product" in message
    assert "Configurable: n_angles, R_km, dr_km, slope_critical, crs, interpolation_method" in message
