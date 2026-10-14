import dataclasses
import typing

from ..pipeline.schema import PRODUCT_DEFINITION_TRIX
from .interface import _get_parameter


@dataclasses.dataclass(frozen=True)
class RunParameters:
    """Run parameters for the analyzer."""

    n_angles: int
    R_km: float
    dr_km: float
    slope_critical: float
    crs: str | None


@dataclasses.dataclass(frozen=True)
class SampleParameters:
    """Sample parameters for the fieldsampler."""

    interpolation_method: str


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

INTERPOLATION_METHODS = typing.Literal["linear", "nearest"]

WRITER_DEFAULTS: dict = {
    "filenames": {
        file: _get_parameter(PRODUCT_DEFINITION_TRIX, "artifacts", "filenames", file)
        for file in ["manifest", "rix_table", "trix_table", "geopackage"]
    },
    "gpkg_layers": {
        layer: _get_parameter(PRODUCT_DEFINITION_TRIX, "artifacts", "geopackage_layers", layer)
        for layer in ["locations_site", "locations_reference", "ruggedness", "trix"]
    },
}

COPDEM_METADATA: dict = {
    "name": "Copernicus DEM GLO-30",
    "source": "Copernicus",
    "description": "Downloaded 2026.",
}
