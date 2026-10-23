import dataclasses
import numbers

import numpy as np
import scipy.spatial.distance
from numpy.typing import NDArray


def compute_trix(
    rix_site: NDArray, elevation_site: NDArray, rix_reference: NDArray, elevation_reference: NDArray
) -> NDArray:
    """Compute the TRIX representativity measure according to FGW TR6 Rev12.

    Parameters
    ----------
    rix_site
        RIX values of the site(s).
    elevation_site
        Elevation values of the site(s).
    rix_reference
        RIX values of the reference/wind data base locations.
    elevation_reference
        Elevation values of the reference/wind data base locations.

    Returns
    -------
    trix
        T-RIX representativity measure.
    """
    rix_site = _ensure_1D(rix_site)
    elevation_site = _ensure_1D(elevation_site)
    rix_reference = _ensure_1D(rix_reference)
    elevation_reference = _ensure_1D(elevation_reference)

    rix_representativity = 0.5 * 100 * np.add.outer(rix_site, rix_reference)
    elevation_representativity = np.abs(np.subtract.outer(elevation_site, elevation_reference))
    trix = 0.9 * rix_representativity + 0.1 * elevation_representativity
    return trix


def compute_trix_limit_distances(trix: NDArray, decimals: int | None = 1) -> tuple[NDArray, NDArray]:
    """Compute the distances A [km] and B [km] from T-RIX values.

    Parameters
    ----------
    trix
        Pre-computed matrix of T-RIX values for sites and reference locations.
    decimals
        Number of decimals to round to.

    Returns
    -------
    A, B
        Matrices containing the limit distances A and B.

    Notes
    -----
    1. A shall not be surpassed, B must not be surpassed.
    """
    A = np.maximum(-0.087 * trix + 8.5, np.full_like(trix, fill_value=1.5))
    B = np.maximum(-0.14 * trix + 15.0, np.full_like(trix, fill_value=3.0))

    if decimals is not None:
        A = np.round(A, decimals=decimals)
        B = np.round(B, decimals=decimals)
    return A, B


def evaluate_transferability_limits(distances: NDArray, A: NDArray, B: NDArray) -> NDArray:
    """Apply the limits of transferability due to complexity.

    Parameters
    ----------
    distances
        Pairwise distances of locations to evaluate.
    A
        Limit distances for flow models.
    B
        Limit distances for flow models while accounting for additional uncertainties.

    Returns
    -------
    result
        Transferability matrix.

    Notes
    -----
    1. Encoding:
       - 2: Unconditional transferability (below limits A).
       - 1: Transferrable subject to additional uncertainties (below limits B).
       - 0: No transfer (above limits B).
    2. Transfers are not established if not granted, e.g. in case of distance or threshold is NaN.
       Catch comparisons with NaN as they yield `False` and thus an unconditional 2.
    """
    result: NDArray = 2 * np.ones_like(distances, dtype=int)

    result = np.where(distances > A, 1, result)
    result = np.where(distances > B, 0, result)
    result = np.where(np.isnan(distances) | np.isnan(A) | np.isnan(B), 0, result)
    return result


def _ensure_1D(a: NDArray | numbers.Real, dtype: str = "float") -> NDArray:
    """Returns `a` as 1D array."""
    result = np.atleast_1d(np.asarray(a, dtype=dtype))
    if result.ndim > 1:
        raise TypeError(f"Argument must be a scalar or 1D array. Received {a}.")
    return result


@dataclasses.dataclass(frozen=True)
class TrixAssessment:
    """Pairwise TRIX assessment of sites against reference locations.

    All matrices have the sites in rows and the reference locations in columns.
    """

    trix: NDArray
    """T-RIX representativity measure."""
    A: NDArray
    """Limit distance [km] up to which a transfer is unconditionally possible."""
    B: NDArray
    """Limit distance [km] up to which a transfer is possible with additional uncertainties."""
    distance_km: NDArray
    """Distance [km] between site and reference location."""
    transferability: NDArray
    """Encoded transferability, see `evaluate_transferability_limits`."""


def pairwise_distances_km(xy_site: NDArray, xy_reference: NDArray) -> NDArray:
    """Pairwise Euclidean distances [km] between points given in a metric CRS.

    Parameters
    ----------
    xy_site
        Coordinates [m] of shape (n_sites, 2).
    xy_reference
        Coordinates [m] of shape (n_references, 2).

    Returns
    -------
    NDArray
        Distances [km] of shape (n_sites, n_references).
    """
    xy_site = np.asarray(xy_site, dtype=float)
    xy_reference = np.asarray(xy_reference, dtype=float)
    for name, xy in (("xy_site", xy_site), ("xy_reference", xy_reference)):
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"{name} must have shape (n, 2), got {xy.shape}.")
    return scipy.spatial.distance.cdist(xy_site, xy_reference, metric="euclidean") / 1000


def assess_transferability(
    *,
    rix_site: NDArray,
    elevation_site: NDArray,
    rix_reference: NDArray,
    elevation_reference: NDArray,
    xy_site: NDArray,
    xy_reference: NDArray,
    decimals: int | None,
) -> TrixAssessment:
    """Assess the transferability of every site to every reference location.

    Parameters
    ----------
    rix_site, elevation_site
        RIX and elevation of the sites, one value per site.
    rix_reference, elevation_reference
        RIX and elevation of the reference locations, one value per location.
    xy_site, xy_reference
        Coordinates [m] of shape (n, 2) in a projected metric CRS.
    decimals
        Number of decimals to which the limit distances A and B are rounded. None: no rounding.

    Returns
    -------
    TrixAssessment
        TRIX, limit distances, distances and the encoded transferability.
    """
    n_site, n_reference = len(np.atleast_1d(rix_site)), len(np.atleast_1d(rix_reference))

    if len(np.atleast_1d(elevation_site)) != n_site or len(np.atleast_2d(xy_site)) != n_site:
        raise ValueError("rix_site, elevation_site and xy_site must describe the same number of sites.")
    if len(np.atleast_1d(elevation_reference)) != n_reference or len(np.atleast_2d(xy_reference)) != n_reference:
        raise ValueError("rix_reference, elevation_reference and xy_reference must describe the same locations.")

    trix = compute_trix(rix_site, elevation_site, rix_reference, elevation_reference)
    A, B = compute_trix_limit_distances(trix, decimals=decimals)
    distance_km = pairwise_distances_km(xy_site, xy_reference)
    transferability = evaluate_transferability_limits(distances=distance_km, A=A, B=B)

    return TrixAssessment(trix=trix, A=A, B=B, distance_km=distance_km, transferability=transferability)
