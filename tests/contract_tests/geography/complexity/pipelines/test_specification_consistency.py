"""Code and specification agree (7.8): what the code offers as settable is what the specification defines."""

import dataclasses

from phoibe.geography.complexity.rix.pipelines.product import RunParameters, SampleParameters
from phoibe.geography.complexity.rix.pipelines.requirements import REQUIRED_FIXED_PARAMETERS


def test_run_and_sample_parameters_are_the_settable_parameters_of_the_specification(product_spec):
    fields = {field.name for cls in (RunParameters, SampleParameters) for field in dataclasses.fields(cls)}
    fixed = {name for _, name in REQUIRED_FIXED_PARAMETERS}  # read by the code from the specification, never settable
    assert fields == {parameter.name for parameter in product_spec.parameters} - fixed


def test_the_parameters_of_the_run_are_defined_once_across_the_groups(product_spec):
    names = [parameter.name for parameter in product_spec.parameters]
    assert len(names) == len(set(names))


def test_the_configuration_accepts_every_settable_parameter_with_its_default(product_spec):
    from phoibe.geography.complexity.rix.entrypoints import default_configuration, load_configuration

    settable = {field.name for cls in (RunParameters, SampleParameters) for field in dataclasses.fields(cls)}
    defaults = {p.name: p.value for p in product_spec.parameters if p.name in settable}
    assert set(defaults) == settable
    assert load_configuration({"parameters": defaults}, product_spec) == default_configuration(product_spec)


def test_the_configuration_rejects_every_parameter_the_code_reads_as_fixed(product_spec):
    import pytest

    from phoibe.geography.complexity.rix.entrypoints import load_configuration
    from phoibe.geography.complexity.rix.pipelines.errors import ConfigurationError

    for _, name in REQUIRED_FIXED_PARAMETERS:
        with pytest.raises(ConfigurationError, match=f"'{name}' is fixed by the product"):
            load_configuration(
                {"parameters": {name: next(p.value for p in product_spec.parameters if p.name == name)}}, product_spec
            )
