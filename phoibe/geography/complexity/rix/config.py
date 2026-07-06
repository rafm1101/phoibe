import typing

from .keys import _get_parameter
from .schema import PRODUCT_DEFINITION_TRIX

ANALYZER_DEFAULTS: dict = {
    "name": _get_parameter(PRODUCT_DEFINITION_TRIX, "name"),
    "meta": {
        "version": _get_parameter(PRODUCT_DEFINITION_TRIX, "version"),
        "description": _get_parameter(PRODUCT_DEFINITION_TRIX, "description"),
    },
    "parameters": {
        "n_angles": _get_parameter(PRODUCT_DEFINITION_TRIX, "parameters", "ray", "n_angles", "value"),
        "R_km": _get_parameter(PRODUCT_DEFINITION_TRIX, "parameters", "ray", "R_km", "value"),
        "dr_km": _get_parameter(PRODUCT_DEFINITION_TRIX, "parameters", "ray", "dr_km", "value"),
        "slope_critical": _get_parameter(PRODUCT_DEFINITION_TRIX, "parameters", "slope", "slope_critical", "value"),
        "crs": _get_parameter(PRODUCT_DEFINITION_TRIX, "parameters", "ray", "crs", "value"),
    },
    "sampling": {
        "method": _get_parameter(PRODUCT_DEFINITION_TRIX, "parameters", "sampler", "interpolation_method", "value"),
    },
}

INTERPOLATION_METHODS = typing.Literal["linear", "nearest"]
