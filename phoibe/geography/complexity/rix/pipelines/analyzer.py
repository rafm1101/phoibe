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

from ..computations.fieldsampler import Field, RegularGridXYSampler
from ..computations.results import RadialRuggedness
from ..computations.rix import compute_regular_rix
from ..computations.trix import assess_transferability
from . import checks, errors, ports, product, publishing, results
from .results import tables

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
        spec: product.ProductSpecification,
        parameters: product.RunParameters,
        sampler_parameters: product.SampleParameters,
        writer: ports.ResultWriter | None = None,
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
    ) -> results.ResultSummary:
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
        present = {_DEM, checks.LOCATIONS_SITE} | (
            {checks.LOCATIONS_REFERENCE} if locations_reference is not None else set()
        )
        checks.check_required_inputs(spec, present)

        sites = checks.normalise_locations(locations_site)
        references = None if locations_reference is None else checks.normalise_locations(locations_reference)
        values = results.records.parameter_values(spec, self._parameters, self._sampler_parameters)

        findings = [
            *checks.input_findings(spec, sites, references, crs_config=self._parameters.crs),
            *checks.parameter_findings(spec, values),
            *checks.locked_parameter_findings(spec, values),
        ]
        checks.raise_on_abort(findings)

        sampler = RegularGridXYSampler(field, method=self._sampler_parameters.interpolation_method)
        rix = {checks.LOCATIONS_SITE: self._compute_rix(sampler, sites)}
        if references is not None:
            rix[checks.LOCATIONS_REFERENCE] = self._compute_rix(sampler, references)
        for collection, ruggednesses in rix.items():
            findings.extend(checks.location_findings(spec, collection, ruggednesses))
        checks.raise_on_abort(findings)
        checks.log_findings(findings)

        result_tables: dict[str, pd.DataFrame] = {
            results.RIX_TABLE: tables.rix_table(spec, rix[checks.LOCATIONS_SITE]),
            results.LOCATIONS_SITE_TABLE: tables.locations_table(spec, results.LOCATIONS_SITE_TABLE, sites),
        }
        if include_steep_segments:
            result_tables[results.RUGGEDNESS_TABLE] = tables.steep_segments(spec, rix[checks.LOCATIONS_SITE])

        transferability_counts = None
        if references is not None:
            result_tables[results.LOCATIONS_REFERENCE_TABLE] = tables.locations_table(
                spec, results.LOCATIONS_REFERENCE_TABLE, references
            )
            assessment = assess_transferability(
                rix_site=_rix(rix[checks.LOCATIONS_SITE]),
                elevation_site=_elevation(rix[checks.LOCATIONS_SITE]),
                rix_reference=_rix(rix[checks.LOCATIONS_REFERENCE]),
                elevation_reference=_elevation(rix[checks.LOCATIONS_REFERENCE]),
                xy_site=_xy(sites),
                xy_reference=_xy(references),
                decimals=results.records.fixed_parameter(spec, "trix", "decimals"),
            )
            result_tables[results.TRIX_TABLE] = tables.trix_table(
                spec, assessment, list(rix[checks.LOCATIONS_SITE]), list(rix[checks.LOCATIONS_REFERENCE])
            )
            transferability_counts = tables.transferability_counts(spec, result_tables[results.TRIX_TABLE])

        record = results.records.build_run_record(
            created_at=self._clock(),
            project_name=project_name,
            parameters=values,
            field_info=field.info,
            dem_provenance=dem_metadata,
            results=rix,
            transferability_counts=transferability_counts,
            findings=findings,
        )
        return results.ResultSummary(
            tables=result_tables,
            ruggedness_roses=tables.rose(rix[checks.LOCATIONS_SITE]),
            record=record,
        )

    def publish(
        self, result: results.ResultSummary, target: ports.Target, profile: str = "summary"
    ) -> ports.WrittenArtifacts:
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
        missing = publishing.tables_missing_for(self._spec, result.tables.keys(), profile, present)
        if missing:
            raise errors.ContractViolation(
                [f"the result has no table {name!r}, which profile {profile!r} requires." for name in missing]
            )
        written = self._writer.write_artifacts(result, target, profile)
        publishing.check_written_artifacts(self._spec, written, present)
        self._writer.write_manifest(publishing.build_manifest(self._spec, result.record, written), target)
        return written

    def _compute_rix(
        self, sampler: RegularGridXYSampler, locations: gpd.GeoDataFrame
    ) -> dict[Hashable, RadialRuggedness]:
        """RIX of every location, keyed by location identifier."""
        parameters = self._parameters
        nan_policy = results.records.fixed_parameter(self._spec, "sampler", "nan_policy")
        rix_results = {}
        for location_id, row in locations.iterrows():
            LOGGER.debug("Computing RIX for location_id=%s", location_id)
            rix_results[location_id] = compute_regular_rix(
                location=row.geometry,
                sampler=sampler,
                n_angles=parameters.n_angles,
                R_km=parameters.R_km,
                dr_km=parameters.dr_km,
                crs=locations.crs,
                slope_critical=parameters.slope_critical,
                nan_policy=nan_policy,
            )
        return rix_results


def _rix(results: dict[Hashable, RadialRuggedness]) -> np.ndarray:
    return np.array([radial_rugs.rix for radial_rugs in results.values()], dtype=float)


def _elevation(results: dict[Hashable, RadialRuggedness]) -> np.ndarray:
    return np.array([radial_rugs.z for radial_rugs in results.values()], dtype=float)


def _xy(locations: gpd.GeoDataFrame) -> np.ndarray:
    return np.column_stack([locations.geometry.x, locations.geometry.y])
