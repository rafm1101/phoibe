import dataclasses

ANALYZER_DEFAULTS: dict = {
    "name": "T-RIX assessment",
    "meta": {
        "version": "1.0",
        "description": "TRIX, TR6 Rev.12",
    },
    "parameters": {
        "n_angles": 72,
        "R_km": 3.5,
        "dr_km": 0.05,
        "slope_critical": 0.3,
        "crs": None,
    },
    "sampling": {
        "method": "linear",
    },
}


@dataclasses.dataclass
class ColumnKeys:
    boiler: str = "plate"
