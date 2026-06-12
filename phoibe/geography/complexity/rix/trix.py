import numbers

import numpy as np
from numpy.typing import NDArray


def compute_trix(rix_site: NDArray, elevation_site: NDArray, rix_wind: NDArray, elevation_wind: NDArray) -> NDArray:
    """Compute the TRIX representativity measure according to FGW TR6 Rev12.

    Parameters
    ----------
    rix_site
        RIX values of the site(s).
    elevation_site
        Elevation values of the site(s).
    rix_wind
        RIX values of the reference/wind data base locations.
    elevation_wind
        Elevation values of the reference/wind data base locations.

    Returns
    -------
    trix
        T-RIX representativity measure.
    """
    rix_site = _ensure_1D(rix_site)
    elevation_site = _ensure_1D(elevation_site)
    rix_wind = _ensure_1D(rix_wind)
    elevation_wind = _ensure_1D(elevation_wind)

    rix_representativity = 0.5 * 100 * np.add.outer(rix_site, rix_wind)
    elevation_representativity = np.abs(np.subtract.outer(elevation_site, elevation_wind))
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
        Matrices containing the distance measures A and B.

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


def _ensure_1D(a: NDArray | numbers.Real, dtype: str = "float") -> NDArray:
    """Returns `a` as 1D array."""
    result = np.atleast_1d(np.asarray(a, dtype=dtype))
    if result.ndim > 1:
        raise TypeError(f"Argument must be a scalar or 1D array. Received {a}.")
    return result
