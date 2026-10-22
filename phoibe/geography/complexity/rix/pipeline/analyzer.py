"""DRAFT of the use case: assess sites and, optionally, reference locations (RIX and TRIX).

It only fixes the order of the steps (U2). Rules live in `domain`, tables in `tables`, names, severities and fixed
parameters in the product specification. Not runnable yet: it imports `RunParameters` and `SampleParameters` from
`definitions.config` (their final home is open), and needs the change of `SamplingInfo` (`n_points`) in `profiles.py`.

Compared with the previous version it
- takes a `Field`, i.e. the caller creates the field from the data it has (no DEM reading here),
- takes fixed values (`decimals`, `nan_policy`) from the specification, never from a literal,
- builds a mapping of tables keyed by their specification name, so publishing is generic,
- skips the steep segments on request, which saves memory for the summary profile.

Results stay in memory. Writing them is a separate, optional step: `publish`.
"""

from __future__ import annotations

import datetime
import logging
import typing
from collections.abc import Callable, Hashable

import numpy as np
import pandas as pd
import pipeline

from ..computations.fieldsampler import Field, RegularGridXYSampler
from ..computations.results import RadialRuggedness
from ..computations.rix import compute_regular_rix
from ..computations.trix import assess_transferability
from .checks.findings import (
    location_findings,
    locked_parameter_findings,
    log_findings,
    raise_on_abort,
)
from .checks.inputs import (
    LOCATIONS_REFERENCE,
    LOCATIONS_SITE,
    check_required_inputs,
    input_findings,
    normalise_locations,
)
from .errors import ContractViolation
from .ports import ResultWriter, Target, WrittenArtifacts
from .product.specification import ProductSpecification
from .publishing.contract_check import check_written_artifacts, tables_missing_for
from .publishing.manifest import build_manifest
from .results import tables
from .results.records import build_run_record, fixed_parameter, parameter_values
from .results.result_summary import ResultSummary

if typing.TYPE_CHECKING:
    import geopandas as gpd

LOGGER = logging.getLogger(__name__)

_DEM = "dem"


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.UTC)


class TRIXAnalyzer:
    """Compute RIX (and optionally TRIX) for one or two collections of locations.

    Parameters
    ----------
    spec
        Specification of the product: names, severities of checks, fixed parameters, expected artifacts.
    parameters
        Parameters of the assessment.
    sampler_parameters
        Parameters of the field sampler.
    writer
        Needed only for `publish`. Without it, results stay in memory.
    clock
        Source of the timestamp of a run. Replace it in tests.

    Notes
    -----
    CRS responsibility lies with the caller: the field and the locations must share one projected metric CRS.
    Deviations are reported as findings.
    """

    def __init__(
        self,
        spec: ProductSpecification,
        parameters: pipeline.RunParameters,
        sampler_parameters: pipeline.SampleParameters,
        writer: ResultWriter | None = None,
        clock: Callable[[], datetime.datetime] = _utc_now,
    ) -> None:
        self._spec = spec
        self._parameters = parameters
        self._sampler_parameters = sampler_parameters
        self._writer = writer
        self._clock = clock

    def run(
        self,
        field: Field,
        locations_site: gpd.GeoDataFrame,
        locations_reference: gpd.GeoDataFrame | None = None,
        *,
        dem_metadata: dict | None = None,
        project_name: str = "",
        include_steep_segments: bool = True,
    ) -> ResultSummary:
        """Run the full assessment and return the result. Nothing is written.

        Parameters
        ----------
        field
            Elevation field to sample from, e.g. from `Field.from_array(dem)`.
        locations_site
            Point locations. The column `site_id`, if present, or else the index is the identifier. Must be unique.
        locations_reference
            Optional second collection for the pairwise TRIX computation.
        dem_metadata
            Free-form description of the DEM, reported in the manifest.
        project_name
            Label of the run, reported in the manifest.
        include_steep_segments
            Whether to build the table of steep segments. Needed for the "full" profile only.

        Raises
        ------
        InputRequirementError
            If a required input is missing.
        CheckAborted
            If a finding belongs to a check with severity "abort".
        """
        spec = self._spec
        present = {_DEM, LOCATIONS_SITE} | ({LOCATIONS_REFERENCE} if locations_reference is not None else set())
        check_required_inputs(spec, present)

        sites = normalise_locations(locations_site)
        references = None if locations_reference is None else normalise_locations(locations_reference)
        values = parameter_values(spec, self._parameters, self._sampler_parameters)

        findings = [
            *input_findings(spec, sites, references, crs_config=self._parameters.crs),
            *locked_parameter_findings(spec, values),
        ]
        raise_on_abort(findings)

        sampler = RegularGridXYSampler(field, method=self._sampler_parameters.interpolation_method)
        rix = {LOCATIONS_SITE: self._compute_rix(sampler, sites)}
        if references is not None:
            rix[LOCATIONS_REFERENCE] = self._compute_rix(sampler, references)
        for collection, results in rix.items():
            findings.extend(location_findings(spec, collection, results))
        raise_on_abort(findings)
        log_findings(findings)

        result_tables: dict[str, pd.DataFrame] = {
            tables.RIX_TABLE: tables.rix_table(spec, rix[LOCATIONS_SITE]),
            tables.LOCATIONS_SITE_TABLE: tables.locations_table(spec, tables.LOCATIONS_SITE_TABLE, sites),
        }
        if include_steep_segments:
            result_tables[tables.RUGGEDNESS_TABLE] = tables.steep_segments(spec, rix[LOCATIONS_SITE])

        transferability_counts = None
        if references is not None:
            result_tables[tables.LOCATIONS_REFERENCE_TABLE] = tables.locations_table(
                spec, tables.LOCATIONS_REFERENCE_TABLE, references
            )
            assessment = assess_transferability(
                rix_site=_rix(rix[LOCATIONS_SITE]),
                elevation_site=_elevation(rix[LOCATIONS_SITE]),
                rix_reference=_rix(rix[LOCATIONS_REFERENCE]),
                elevation_reference=_elevation(rix[LOCATIONS_REFERENCE]),
                xy_site=_xy(sites),
                xy_reference=_xy(references),
                decimals=fixed_parameter(spec, "trix", "decimals"),
            )
            result_tables[tables.TRIX_TABLE] = tables.trix_table(
                spec, assessment, list(rix[LOCATIONS_SITE]), list(rix[LOCATIONS_REFERENCE])
            )
            transferability_counts = tables.transferability_counts(spec, result_tables[tables.TRIX_TABLE])

        record = build_run_record(
            created_at=self._clock(),
            project_name=project_name,
            parameters=values,
            field_info=field.info,
            dem_provenance=dem_metadata,
            results=rix,
            transferability_counts=transferability_counts,
            findings=findings,
        )
        return ResultSummary(
            tables=result_tables,
            ruggedness_roses=tables.rose(rix[LOCATIONS_SITE]),
            record=record,
        )

    def publish(self, result: ResultSummary, target: Target, profile: str = "summary") -> WrittenArtifacts:
        """Write a result: data files first, then the contract check, the manifest last (DA3, DA4).

        Raises
        ------
        ContractViolation
            If the result lacks a table of the profile, or what was written does not match the specification.
            No manifest is written then, so the artifact counts as unpublished.
        PortError
            If the writer fails technically.
        """
        if self._writer is None:
            raise RuntimeError("TRIXAnalyzer was created without a writer.")
        present = result.record.present_inputs
        missing = tables_missing_for(self._spec, result.tables.keys(), profile, present)
        if missing:
            raise ContractViolation(
                [f"the result has no table {name!r}, which profile {profile!r} requires." for name in missing]
            )
        written = self._writer.write_artifacts(result, target, profile)
        check_written_artifacts(self._spec, written, present)
        self._writer.write_manifest(build_manifest(self._spec, result.record, written), target)
        return written

    def _compute_rix(
        self, sampler: RegularGridXYSampler, locations: gpd.GeoDataFrame
    ) -> dict[Hashable, RadialRuggedness]:
        """RIX of every location, keyed by location identifier."""
        parameters = self._parameters
        nan_policy = fixed_parameter(self._spec, "sampler", "nan_policy")
        results = {}
        for location_id, row in locations.iterrows():
            LOGGER.debug("Computing RIX for location_id=%s", location_id)
            results[location_id] = compute_regular_rix(
                location=row.geometry,
                sampler=sampler,
                n_angles=parameters.n_angles,
                R_km=parameters.R_km,
                dr_km=parameters.dr_km,
                crs=locations.crs,
                slope_critical=parameters.slope_critical,
                nan_policy=nan_policy,
            )
        return results


def _rix(results: dict[Hashable, RadialRuggedness]) -> np.ndarray:
    return np.array([radial_rugs.rix for radial_rugs in results.values()], dtype=float)


def _elevation(results: dict[Hashable, RadialRuggedness]) -> np.ndarray:
    return np.array([radial_rugs.z for radial_rugs in results.values()], dtype=float)


def _xy(locations: gpd.GeoDataFrame) -> np.ndarray:
    return np.column_stack([locations.geometry.x, locations.geometry.y])
