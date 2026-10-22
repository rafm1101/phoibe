"""What the pipeline requires of a product specification.

The code raises checks, uses fixed parameters and fills tables whose names are defined in the specification. A name
missing there would surface as a lookup error in the middle of a run. `check_specification_complete` verifies it once,
when the analyzer is created, and reports everything that is missing at the same time.
"""

from __future__ import annotations

from .. import computations
from . import checks, product, results

REQUIRED_CHECKS = (
    computations.checks.N_ANGLE,
    computations.checks.NAN_COUNT,
    computations.checks.CENTRE_ELEVATION,
    computations.checks.CRS_SITE_VS_REFERENCE,
    computations.checks.CRS_VS_CONFIG,
    computations.checks.UNIQUE_IDS,
    checks.LOCKED_PARAMETER_CHANGED,
)
"""Checks whose findings the pipeline can raise. Each needs a severity in `checks`."""

REQUIRED_FIXED_PARAMETERS = (("trix", "decimals"), ("sampler", "nan_policy"))
"""Parameters (group, name) the code takes from the specification instead of a literal. They must be locked."""

REQUIRED_TABLES = (
    results.RIX_TABLE,
    results.TRIX_TABLE,
    results.LOCATIONS_SITE_TABLE,
    results.LOCATIONS_REFERENCE_TABLE,
    results.RUGGEDNESS_TABLE,
)
REQUIRED_INPUTS = (checks.DEM, checks.LOCATIONS_SITE, checks.LOCATIONS_REFERENCE)


def specification_problems(spec: product.ProductSpecification) -> list[str]:
    """Everything the pipeline needs but the specification does not define."""
    problems = [
        f"check {name!r} is raised by the pipeline but not defined in 'checks'."
        for name in REQUIRED_CHECKS
        if name not in spec.checks
    ]
    parameters = {(parameter.group, parameter.name): parameter for parameter in spec.parameters}
    for group, name in REQUIRED_FIXED_PARAMETERS:
        parameter = parameters.get((group, name))
        if parameter is None:
            problems.append(f"parameter '{group}.{name}' is used by the pipeline but not defined in 'parameters'.")
        elif not parameter.locked:
            problems.append(f"parameter '{group}.{name}' must be locked, it is used as a fixed value.")
    problems += [
        f"table {name!r} is built by the pipeline but not defined in 'tables'."
        for name in REQUIRED_TABLES
        if name not in spec.tables
    ]
    problems += [
        f"input {name!r} is used by the pipeline but not defined in 'inputs'."
        for name in REQUIRED_INPUTS
        if name not in spec.inputs
    ]
    return problems


def check_specification_complete(spec: product.ProductSpecification) -> None:
    """Raise `SpecificationError` listing everything the pipeline needs but the specification lacks."""
    problems = specification_problems(spec)
    if problems:
        raise product.SpecificationError(problems)
