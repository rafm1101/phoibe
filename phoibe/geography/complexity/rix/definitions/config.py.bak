import dataclasses
import enum
import typing

from ..pipeline.product.schema import PRODUCT_DEFINITION_TRIX
from .interface import _get_parameter

INTERPOLATION_METHODS = typing.Literal["linear", "nearest"]


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


class InterpolationMethods(enum.StrEnum):
    LINEAR = "linear"
    NEAREST = "nearest"


@dataclasses.dataclass(frozen=True)
class SampleParameters:
    """Sample parameters for the fieldsampler."""

    interpolation_method: INTERPOLATION_METHODS

    def __post_init__(self):
        if self.interpolation_method not in ("linear", "nearest"):
            raise ValueError(f"Unknown interpolation_method {self.interpolation_method!r}.")


@dataclasses.dataclass(frozen=True)
class AnalyzerConfig:
    """Sample parameters for the fieldsampler."""

    name: str
    version: str
    description: str
    parameters: RunParameters
    sampler: SampleParameters


ANALYZER_DEFAULTS = AnalyzerConfig(
    name=_get_parameter(PRODUCT_DEFINITION_TRIX, "name"),
    version=_get_parameter(PRODUCT_DEFINITION_TRIX, "version"),
    description=_get_parameter(PRODUCT_DEFINITION_TRIX, "description"),
    parameters=RunParameters(
        n_angles=_get_parameter(PRODUCT_DEFINITION_TRIX, "parameters", "ray", "n_angles", "value"),
        R_km=_get_parameter(PRODUCT_DEFINITION_TRIX, "parameters", "ray", "R_km", "value"),
        dr_km=_get_parameter(PRODUCT_DEFINITION_TRIX, "parameters", "ray", "dr_km", "value"),
        slope_critical=_get_parameter(PRODUCT_DEFINITION_TRIX, "parameters", "slope", "slope_critical", "value"),
        crs=_get_parameter(PRODUCT_DEFINITION_TRIX, "parameters", "ray", "crs", "value"),
    ),
    sampler=SampleParameters(
        interpolation_method=_get_parameter(
            PRODUCT_DEFINITION_TRIX, "parameters", "sampler", "interpolation_method", "value"
        ),
    ),
)


WRITER_DEFAULTS: dict = {
    "filenames": {
        file: _get_parameter(PRODUCT_DEFINITION_TRIX, "artifacts", "files", file, "filename")
        for file in ["manifest", "rix_table", "trix_table", "geopackage"]
    },
    "gpkg_layers": {
        layer: _get_parameter(PRODUCT_DEFINITION_TRIX, "artifacts", "layers", layer)
        for layer in ["locations_site", "locations_reference", "ruggedness", "trix_table"]
    },
}

COPDEM_METADATA: dict = {
    "name": "Copernicus DEM GLO-30",
    "source": "Copernicus",
    "description": "Downloaded 2026.",
}
