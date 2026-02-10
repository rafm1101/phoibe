import dataclasses
import logging

import ergaleiothiki.kiklos.circle
import numpy as np
from ergaleiothiki.perdix import LocationCCS
from ergaleiothiki.tididi.validate_numerics import _validate_non_negative
from ergaleiothiki.tididi.validate_numerics import _validate_notna_finite
from ergaleiothiki.tididi.validate_numerics import _validate_positive
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class RayGeometry:
    """Geometric representation of a ray embedded in 2D world.

    A `RayGeometry` defines a directed ray originating from `location` in direction `theta`, and provides
    its supporting points as distances from the origin `r_m` together with their respective coordinates in
    the 2D world given by `xs` and `ys`.

    Creation
    --------
    from_compass_regular
        Create a regular grid starting from `location` in direction `theta` spacing `dr_km` of length `R_km`.
    from_compass
        Create any grid starting from `location` in direction `theta` given in increasing sequence of distances `r_m`.
    """

    location: LocationCCS
    """Origin of the ray in world coordinates."""
    theta: float
    """Direction of the ray [°] with 0° facing North and angles increasing clockwise."""
    r_m: NDArray[np.floating]
    """Strictly increasing distances [m] from ray origin. Each entry corresponds to one supporting point."""
    xs: NDArray[np.floating]
    """X coordinates of the ray supporting points in the world space."""
    ys: NDArray[np.floating]
    """Y coordinates of the ray supporting points in the world space."""

    def __post_init__(self):
        if len(self.r_m) > 1:
            dr = np.diff(self.r_m)
            _validate_positive(dr, "delta r_m")

    @classmethod
    def from_compass_regular(cls, location, theta, R_km, dr_km):
        """Generate `RayGeometry` as a equidistant grid.

        Parameters
        ----------
        location
            Origin of the ray in real world Cartesian coordinates.
        theta
            Direction of the ray [°] with 0° facing North and angles increasing clockwise.
        R_km
            Length [km] of the ray.
        dr_km
            Stepsize [km] of the grid.

        Returns
        -------
        RayGeometry
            Immutable representation of the regular grid starting at `location` and heading in direction `theta`.
        """
        _validate_notna_finite(R_km, "R_km")
        _validate_non_negative(R_km, "R_km")
        _validate_notna_finite(dr_km, "dr_km")
        _validate_positive(dr_km, "dr_km")
        if dr_km >= R_km:
            raise ValueError("dr_km must not exceed R_km.")

        n_steps = int(np.floor(R_km / dr_km))
        r_m = np.arange(n_steps + 1, dtype=float) * dr_km * 1000

        remainder = R_km * 1000 - r_m[-1]
        if remainder > 1e-3:
            msg = "Ray for %.1f truncated as %.3f not multiple of %.3f. Last point at %.3fm."
            LOGGER.warning(msg, theta, R_km, dr_km, r_m[-1])

        return cls._from_compass_r_m(location, theta, r_m)

    @classmethod
    def from_compass(cls, location, theta, r_m):
        """Generate `RayGeometry` as any grid.

        Parameters
        ----------
        location
            Origin of the ray in real world Cartesian coordinates.
        theta
            Direction of the ray [°] with 0° facing North and angles increasing clockwise.
        r_m
            Gridpoint distances [m] from the origin. Must be strictly increasing.

        Returns
        -------
        RayGeometry
            Immutable representation of the regular grid starting at `location` and heading in direction `theta`.
        """
        return cls._from_compass_r_m(location, theta, r_m)

    @classmethod
    def _from_compass_r_m(cls, location, theta, r_m):
        r_m = np.asarray(r_m, dtype=float)
        _validate_positive(np.diff(r_m), "dr_m")

        dx, dy = ergaleiothiki.kiklos.circle.compass_polar_to_cartesian(theta, r_m)
        xs = location.easting + dx
        ys = location.northing + dy
        return cls(location=location, theta=theta, r_m=r_m, xs=xs, ys=ys)
