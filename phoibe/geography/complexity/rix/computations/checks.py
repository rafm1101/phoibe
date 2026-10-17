"""Rules that report anomalies as findings.

All functions are pure: they take plain values or result objects and return `Finding` objects. They never raise
for a data problem and never decide what a finding means for the run; that is the use case's job, driven by the
severity in the product specification (DP2, DP3).

Input rules take plain values (CRS objects, identifiers), not GeoDataFrames, so no GIS library is needed here.
The `check` of a finding is the name of a check in the product specification.
"""

from __future__ import annotations

import collections
import dataclasses
import typing
from collections.abc import Hashable, Iterable, Sequence

import numpy as np
import pyproj

if typing.TYPE_CHECKING:
    from .results import RadialRuggedness

N_ANGLE = "n_angle"
NAN_COUNT = "nan_count"
CENTRE_ELEVATION = "centre_elevation"
CRS_SITE_VS_REFERENCE = "crs_site_vs_reference"
CRS_VS_CONFIG = "crs_vs_config"
UNIQUE_IDS = "unique_ids"


@dataclasses.dataclass(frozen=True)
class Finding:
    """An anomaly found by a check. It carries facts only, no consequence."""

    check: str
    """Name of the check in the product specification."""
    detail: str
    """What was found."""
    theta: float | None = None
    """Ray direction [°] if the finding concerns one ray, None if it concerns the location or the inputs."""


def nan_findings(radial: RadialRuggedness, max_fraction: float = 0.0) -> tuple[Finding, ...]:
    """Rays that encountered more than `max_nan_count` NaN values while sampling.

    The count is the one recorded while sampling, so it is also reported for rays that were truncated afterwards.
    With the default, any NaN is reported (out-of-bounds or data gap).
    """
    findings = []
    for ray in radial.rays:
        info = ray.profile.sampling_info
        nan_count, n_points = int(info.nan_count), int(info.n_points)
        fraction = nan_count / n_points if n_points else 0.0
        if fraction > max_fraction:
            findings.append(
                Finding(
                    NAN_COUNT,
                    f"{nan_count} of {n_points} sampled values are NaN along the ray (allowed: {max_fraction}).",
                    ray.theta,
                )
            )
    return tuple(findings)


def centre_elevation_findings(radial: RadialRuggedness, atol: float = 1e-6) -> tuple[Finding, ...]:
    """Rays whose elevation at the origin deviates from the others.

    All rays of a location start at the same point, so their first value must agree. Rays are compared with the
    median, so that a single odd ray is flagged and not all of them. NaN values are skipped, because
    `nan_findings` reports them.

    Parameters
    ----------
    radial
        Result of one location.
    atol
        Tolerated deviation [m]. The same point is sampled the same way, so this only guards against rounding.
    """
    thetas = [ray.theta for ray in radial.rays]
    z0 = np.array([ray.profile.z[0] if len(ray.profile.z) else np.nan for ray in radial.rays], dtype=float)
    finite = np.isfinite(z0)
    if not finite.any():
        return ()
    reference = float(np.median(z0[finite]))
    return tuple(
        Finding(
            CENTRE_ELEVATION,
            f"Elevation at the centre {z:.3f} m differs from the median {reference:.3f} m of all rays.",
            theta,
        )
        for theta, z, valid in zip(thetas, z0, finite, strict=True)
        if valid and abs(z - reference) > atol
    )


def n_angle_findings(n_angles: int, valid_range: Sequence[float]) -> tuple[Finding, ...]:
    """The number of ray directions must lie within `valid_range`."""
    lower, upper = valid_range
    if lower <= n_angles <= upper:
        return ()
    return (Finding(N_ANGLE, f"n_angles is {n_angles}, allowed range is [{lower:g}, {upper:g}]."),)


def crs_consistency_findings(crs_site: pyproj.CRS | None, crs_reference: pyproj.CRS | None) -> tuple[Finding, ...]:
    """The sites and the reference locations must share one CRS."""
    if crs_site == crs_reference:
        return ()
    return (
        Finding(
            CRS_SITE_VS_REFERENCE,
            f"CRS of the sites ({_name(crs_site)}) differs from the CRS of the references ({_name(crs_reference)}).",
        ),
    )


def crs_config_findings(crs_site: pyproj.CRS | None, crs_config: str | None) -> tuple[Finding, ...]:
    """The CRS of the sites should agree with the configured one. Nothing is reported if either is unknown."""
    if crs_config is None or crs_site is None:
        return ()
    if crs_site == pyproj.CRS.from_user_input(crs_config):
        return ()
    return (
        Finding(CRS_VS_CONFIG, f"CRS of the sites ({_name(crs_site)}) differs from the configured CRS ({crs_config})."),
    )


def unique_id_findings(ids: Iterable[Hashable]) -> tuple[Finding, ...]:
    """Location identifiers must be unique."""
    counts = collections.Counter(ids)
    duplicated = [identifier for identifier, count in counts.items() if count > 1]
    if not duplicated:
        return ()
    examples = ", ".join(repr(identifier) for identifier in duplicated[:3])
    return (Finding(UNIQUE_IDS, f"{len(duplicated)} identifier(s) occur more than once, e.g. {examples}."),)


def _name(crs: typing.Any) -> str:
    if crs is None:
        return "none"
    return crs.to_string() if hasattr(crs, "to_string") else str(crs)
