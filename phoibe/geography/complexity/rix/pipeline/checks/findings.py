"""Findings with their consequence.

The domain reports facts (`Finding`). Here the product specification decides what each means: its check carries the
severity, "abort" or "warn", and the thresholds of the check (DP3).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import typing
from collections.abc import Hashable, Iterable, Mapping, Sequence

from ...computations.checks import (
    CENTRE_ELEVATION,
    N_ANGLE,
    NAN_COUNT,
    Finding,
    centre_elevation_findings,
    n_angle_findings,
    nan_findings,
)
from .. import errors
from ..product.specification import ProductSpecification, Severity

if typing.TYPE_CHECKING:
    from ...computations.results import RadialRuggedness

LOGGER = logging.getLogger(__name__)

LOCKED_PARAMETER_CHANGED = "locked_parameter_changed"


@dataclasses.dataclass(frozen=True)
class LocatedFinding:
    """A finding with the severity of its check and the place where it occurred."""

    finding: Finding
    """Some observation that occurred during a run."""
    severity: Severity
    """Severity given the check information."""
    collection: str | None = None
    """Input the finding belongs to, e.g. "locations_site". None if it concerns the run as a whole."""
    location_id: Hashable | None = None
    """Identifier of the location. None if it concerns a collection or the run."""

    def __str__(self) -> str:
        where = _where(self.collection, self.location_id, self.finding.theta)
        return f"[{self.severity}] {self.finding.check}{where}: {self.finding.detail}"


def locate(
    spec: ProductSpecification,
    findings: Iterable[Finding],
    *,
    collection: str | None = None,
    location_id: Hashable | None = None,
) -> tuple[LocatedFinding, ...]:
    """Attach the severity of each finding's check and the place of the finding."""
    return tuple(
        LocatedFinding(
            finding=finding, severity=spec.check(finding.check).severity, collection=collection, location_id=location_id
        )
        for finding in findings
    )


def check_parameter(spec: ProductSpecification, check: str, name: str, default: typing.Any) -> typing.Any:
    """Threshold or tolerance of a check from the specification, or `default` if the check does not define it."""
    return spec.check(check).parameters.get(name, default)


def location_findings(
    spec: ProductSpecification, collection: str, results: Mapping[Hashable, RadialRuggedness]
) -> tuple[LocatedFinding, ...]:
    """Findings of all locations of one collection. Works for sites and reference locations alike."""
    max_fraction = check_parameter(spec, NAN_COUNT, "max_fraction", 0.0)
    atol = check_parameter(spec, CENTRE_ELEVATION, "atol", 1e-6)
    located: list[LocatedFinding] = []
    for location_id, radial in results.items():
        found = (*nan_findings(radial, max_fraction=max_fraction), *centre_elevation_findings(radial, atol=atol))
        located.extend(locate(spec, found, collection=collection, location_id=location_id))
    return tuple(located)


def locked_parameter_findings(
    spec: ProductSpecification, values: Mapping[str, typing.Any]
) -> tuple[LocatedFinding, ...]:
    """Locked parameters whose value in this run differs from the one fixed by the specification.

    The run is not stopped by default: the result is delivered, but visibly not conformant to the source.
    Parameters that are absent from `values` are not reported.
    """
    found = [
        Finding(
            LOCKED_PARAMETER_CHANGED,
            f"{parameter.group}.{parameter.name} is {values[parameter.name]!r} instead of the locked "
            f"{parameter.value!r}.",
        )
        for parameter in spec.parameters
        if parameter.locked and parameter.name in values and _differs(values[parameter.name], parameter.value)
    ]
    return locate(spec, found)


def parameter_findings(spec: ProductSpecification, values: Mapping[str, typing.Any]) -> tuple[LocatedFinding, ...]:
    """Findings about the parameter of a run: locked parameters that were changed, and the number of angles."""
    found = list(locked_parameter_findings(spec, values))
    if "n_angles" in values:
        valid_range = check_parameter(spec, N_ANGLE, "range", (1, 360))
        found += locate(spec, n_angle_findings(values["n_angles"], valid_range))
    return tuple(found)


def raise_on_abort(findings: Sequence[LocatedFinding]) -> None:
    """Stop the run if any finding has severity "abort"."""
    aborting = [item for item in findings if item.severity == "abort"]
    if aborting:
        raise errors.CheckAborted(aborting)


def log_findings(findings: Iterable[LocatedFinding], logger: logging.Logger = LOGGER) -> None:
    """Log one message per location and check, so findings are visible without opening the manifest."""
    groups: dict[tuple, list[LocatedFinding]] = {}
    for item in findings:
        groups.setdefault((item.severity, item.collection, item.location_id, item.finding.check), []).append(item)
    for (severity, collection, location_id, check), items in groups.items():
        logger.log(
            logging.ERROR if severity == "abort" else logging.WARNING,
            "Check '%s'%s: %d finding(s). First: %s",
            check,
            _where(collection, location_id, None),
            len(items),
            items[0].finding.detail,
        )


def _where(collection: str | None, location_id: Hashable | None, theta: float | None) -> str:
    parts = []
    if collection is not None:
        parts.append(collection)
    if location_id is not None:
        parts.append(f"id={location_id!r}")
    if theta is not None:
        parts.append(f"theta={theta:.1f}")
    return f" at {', '.join(parts)}" if parts else ""


def _is_number(value: typing.Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _differs(actual: typing.Any, fixed: typing.Any) -> bool:
    if _is_number(actual) and _is_number(fixed):
        return not math.isclose(actual, fixed, rel_tol=1e-12, abs_tol=0.0)
    return actual != fixed
