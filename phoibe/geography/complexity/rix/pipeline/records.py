"""Typed record of one run: what was used, what was found, what was processed.

It replaces the untyped `meta` dict. It is built once by the use case from facts (results, findings, the field), so
every aggregation is written once and a result can be published later with the record of the run that produced it.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime
import typing
from collections.abc import Hashable, Mapping, Sequence

from ..computations.fieldsampler import FieldInfo
from .findings import LocatedFinding
from .inputs import DEM, LOCATIONS_REFERENCE, LOCATIONS_SITE
from .specification import ProductSpecification

if typing.TYPE_CHECKING:
    from ..computations.results import RadialRuggedness

MAX_FINDINGS_PER_CHECK = 100
"""Located findings kept per check. Counts always cover all findings."""


@dataclasses.dataclass(frozen=True)
class CollectionRecord:
    """Sampling summary of one collection of locations, e.g. the sites."""

    n_locations: int
    """Total number of locations evaluated."""
    n_locations_with_nans: int
    """Locations with at least one NaN on any ray. A fact, independent of the thresholds of the NaN check."""
    nan_count: int
    """NaN values summed over all rays of all locations."""
    ray_crs: tuple[str | None, ...]
    """Distinct CRS of the rays, in order of appearance."""
    messages: tuple[str, ...]
    """Distinct messages from aligning ray and field, in order of appearance."""

    @classmethod
    def from_results(cls, results: Mapping[Hashable, RadialRuggedness]) -> CollectionRecord:
        infos = [[ray.profile.sampling_info for ray in radial.rays] for radial in results.values()]
        nan_per_location = [sum(int(info.nan_count) for info in location) for location in infos]
        ray_crs = dict.fromkeys(info.crs for location in infos for info in location)
        messages = dict.fromkeys(str(info.message) for location in infos for info in location if info.message)
        return cls(
            n_locations=len(results),
            n_locations_with_nans=sum(count > 0 for count in nan_per_location),
            nan_count=sum(nan_per_location),
            ray_crs=tuple(ray_crs),
            messages=tuple(messages),
        )


@dataclasses.dataclass(frozen=True)
class RunRecord:
    """Everything the manifest reports about a run, as typed values."""

    created_at: datetime.datetime
    """Timezone-aware."""
    project_name: str
    """Name of the evaluated project."""
    parameters: Mapping[str, typing.Any]
    """Value of every parameter used in this run, by parameter name."""
    field_info: FieldInfo
    """Information about the underlying field sampled from."""
    dem_provenance: Mapping[str, typing.Any]
    """Free-form description of the DEM as supplied by the user: name, source, description."""
    collections: Mapping[str, CollectionRecord]
    """Keyed by input name: "locations_site" and, if supplied, "locations_reference"."""
    transferability_counts: Mapping[int, int] | None
    """Number of pairs per transferability state. None if TRIX was not computed."""
    findings: tuple[LocatedFinding, ...]
    """At most `MAX_FINDINGS_PER_CHECK` per check."""
    finding_counts: Mapping[str, int]
    """All findings per check."""

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware.")
        if LOCATIONS_SITE not in self.collections:
            raise ValueError(f"collections must contain {LOCATIONS_SITE!r}.")

    @property
    def present_inputs(self) -> frozenset[str]:
        """Names of the inputs this run was given."""
        return frozenset({DEM, *self.collections})

    @property
    def computed(self) -> tuple[str, ...]:
        """What has been computed."""
        computed = ["rix_site"]
        if LOCATIONS_REFERENCE in self.collections:
            computed.append("rix_reference")
        if self.transferability_counts is not None:
            computed.append("trix")
        return tuple(computed)

    @property
    def n_findings_omitted(self) -> int:
        """Findings counted but not kept in `findings`."""
        return sum(self.finding_counts.values()) - len(self.findings)


def fixed_parameter(spec: ProductSpecification, group: str, name: str) -> typing.Any:
    """Value of a parameter that the specification locks. Code uses it instead of a literal.

    Raises
    ------
    ValueError
        If the parameter is not locked, i.e. its value is not fixed by the specification.
    """
    parameter = spec.parameter(group=group, name=name)
    if not parameter.locked:
        raise ValueError(f"Parameter '{group}.{name}' is not fixed by the specification.")
    return parameter.value


def parameter_values(spec: ProductSpecification, *parameter_sets: typing.Any) -> dict[str, typing.Any]:
    """Flat mapping parameter name -> value from parameter dataclasses, e.g. `RunParameters`, `SampleParameters`."""
    names = [parameter.name for parameter in spec.parameters]
    if len(set(names)) != len(names):
        raise ValueError("Parameter names must be unique across groups of the specification.")
    values = {parameter.name: parameter.value for parameter in spec.parameters if parameter.locked}
    supplied: dict[str, typing.Any] = {}
    for parameter_set in parameter_sets:
        for name, value in dataclasses.asdict(parameter_set).items():
            if name in supplied:
                raise ValueError(f"Parameter {name!r} is defined twice.")
            supplied[name] = value
    return values | supplied


def build_run_record(
    *,
    created_at: datetime.datetime,
    project_name: str,
    parameters: Mapping[str, typing.Any],
    field_info: FieldInfo,
    dem_provenance: Mapping[str, typing.Any] | None,
    results: Mapping[str, Mapping[Hashable, RadialRuggedness]],
    transferability_counts: Mapping[int, int] | None,
    findings: Sequence[LocatedFinding],
    max_findings_per_check: int = MAX_FINDINGS_PER_CHECK,
) -> RunRecord:
    """Assemble the record of a run.

    Parameters
    ----------
    results
        Results of all locations, by collection name (input name).
    findings
        All findings of the run. They are capped per check; the counts are not.
    """
    counts = collections.Counter(item.finding.check for item in findings)
    seen: collections.Counter[str] = collections.Counter()
    kept = []
    for item in findings:
        if seen[item.finding.check] < max_findings_per_check:
            kept.append(item)
            seen[item.finding.check] += 1
    return RunRecord(
        created_at=created_at,
        project_name=project_name,
        parameters=dict(parameters),
        field_info=field_info,
        dem_provenance=dict(dem_provenance or {}),
        collections={name: CollectionRecord.from_results(collection) for name, collection in results.items()},
        transferability_counts=(
            None if transferability_counts is None else {int(k): int(v) for k, v in transferability_counts.items()}
        ),
        findings=tuple(kept),
        finding_counts=dict(counts),
    )
