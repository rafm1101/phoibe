"""Handling of the inputs before any computation: names, normalisation and input checks."""

from __future__ import annotations

import typing
from collections.abc import Collection

from ... import computations
from .. import errors, product
from . import findings

if typing.TYPE_CHECKING:
    import geopandas as gpd

DEM = "dem"
LOCATIONS_SITE = "locations_site"
LOCATIONS_REFERENCE = "locations_reference"
"""Names of the inputs. They must be keys of `inputs` in the product specification."""

ID_COLUMN = "site_id"
"""Column holding the identifier of a location. It becomes the index. Must be a column of the location tables."""


def check_required_inputs(spec: product.ProductSpecification, present: Collection[str]) -> None:
    """Every required input of the specification must be present, and nothing else may be supplied.

    Raises
    ------
    InputRequirementError
        If a required input is missing, or an input is not defined in the specification.
    """
    missing = [name for name, item in spec.inputs.items() if item.required and name not in present]
    if missing:
        raise errors.InputRequirementError(f"Required input(s) missing: {', '.join(missing)}.")
    unknown = sorted(name for name in present if name not in spec.inputs)
    if unknown:
        raise errors.InputRequirementError(f"Input(s) not defined in the specification: {', '.join(unknown)}.")


def normalise_locations(locations: gpd.GeoDataFrame, id_column: str = ID_COLUMN) -> gpd.GeoDataFrame:
    """Copy of the locations whose index is the location identifier.

    If the identifier is a column, it becomes the index. Otherwise the existing index is taken as the identifier.
    """
    locations = locations.copy()
    if id_column in locations.columns:
        locations = locations.set_index(id_column)
    return locations


def input_findings(
    spec: product.ProductSpecification,
    locations_site: gpd.GeoDataFrame,
    locations_reference: gpd.GeoDataFrame | None,
    crs_config: str | None,
) -> tuple[findings.LocatedFinding, ...]:
    """Run the input rules on normalised locations and attach the severities of the specification."""
    found = list(
        findings.locate(spec, computations.checks.unique_id_findings(locations_site.index), collection=LOCATIONS_SITE)
    )
    found += findings.locate(
        spec, computations.checks.crs_config_findings(locations_site.crs, crs_config), collection=LOCATIONS_SITE
    )
    if locations_reference is not None:
        found += findings.locate(
            spec, computations.checks.unique_id_findings(locations_reference.index), collection=LOCATIONS_REFERENCE
        )
        found += findings.locate(
            spec, computations.checks.crs_consistency_findings(locations_site.crs, locations_reference.crs)
        )
    return tuple(found)
