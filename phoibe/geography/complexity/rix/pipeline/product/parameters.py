import dataclasses

from ...computations import INTERPOLATION_METHODS


@dataclasses.dataclass(frozen=True)
class RunParameters:
    """Run parameters for the analyzer."""

    n_angles: int
    """Number of angles for which rays are assessed."""
    R_km: float
    """Radius of the circle where RIX is computed."""
    dr_km: float
    """Stepsize along each ray."""
    slope_critical: float
    """Slope threshold for a step to be considered steep."""
    crs: str | None
    """CRS of the given sites."""

    def __post_init__(self):
        if not 0 < self.dr_km < self.R_km:
            raise ValueError(f"dr_km must lie within (0, R_km), got {self.dr_km}.")
        if self.slope_critical <= 0:
            raise ValueError("slope_critical must be positive.")


@dataclasses.dataclass(frozen=True)
class SampleParameters:
    """Sample parameters for the fieldsampler."""

    interpolation_method: INTERPOLATION_METHODS

    def __post_init__(self):
        if self.interpolation_method not in ("linear", "nearest"):
            raise ValueError(f"Unknown interpolation_method {self.interpolation_method!r}.")
