"""What the pipeline requires of a product specification.

The code raises checks, uses fixed parameters and fills tables whose names are defined in the specification. A name
missing there would surface as a lookup error in the middle of a run. `check_specification_complete` verifies it once,
when the analyzer is created, and reports everything that is missing at the same time.
"""

from __future__ import annotations

from ..computations.checks import (
    CENTRE_ELEVATION,
    CRS_SITE_VS_REFERENCE,
    CRS_VS_CONFIG,
    N_ANGLE,
    NAN_COUNT,
    UNIQUE_IDS,
)
from .checks.findings import LOCKED_PARAMETER_CHANGED
from .checks.inputs import DEM, LOCATIONS_REFERENCE, LOCATIONS_SITE
from .product.specification import ProductSpecification, SpecificationError
from .results.tables import (
    LOCATIONS_REFERENCE_TABLE,
    LOCATIONS_SITE_TABLE,
    RIX_TABLE,
    RUGGEDNESS_TABLE,
    TRIX_TABLE,
)

REQUIRED_CHECKS = (
    N_ANGLE,
    NAN_COUNT,
    CENTRE_ELEVATION,
    CRS_SITE_VS_REFERENCE,
    CRS_VS_CONFIG,
    UNIQUE_IDS,
    LOCKED_PARAMETER_CHANGED,
)
"""Checks whose findings the pipeline can raise. Each needs a severity in `checks`."""

REQUIRED_FIXED_PARAMETERS = (("trix", "decimals"), ("sampler", "nan_policy"))
"""Parameters (group, name) the code takes from the specification instead of a literal. They must be locked."""

REQUIRED_TABLES = (RIX_TABLE, TRIX_TABLE, LOCATIONS_SITE_TABLE, LOCATIONS_REFERENCE_TABLE, RUGGEDNESS_TABLE)
REQUIRED_INPUTS = (DEM, LOCATIONS_SITE, LOCATIONS_REFERENCE)


def specification_problems(spec: ProductSpecification) -> list[str]:
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


def check_specification_complete(spec: ProductSpecification) -> None:
    """Raise `SpecificationError` listing everything the pipeline needs but the specification lacks."""
    problems = specification_problems(spec)
    if problems:
        raise SpecificationError(problems)
