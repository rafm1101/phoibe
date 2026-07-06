from .keys import ColumnKeys

COLUMN_KEYS = ColumnKeys()

PRODUCT_DEFINITION_TRIX = {
    "name": "T-RIX assessment",
    "description": "TRIX, TR6 Rev.12",
    "version": "1.0",
    "references": {
        "TR6": {
            "title": "Technische Richtlinien für Windenergieanlagen, "
            "Teil 6: Bestimmung von Windpotenzial und Energieerträgen",
            "author": "FGW e.V.",
            "description": "Technische Richtlinie 6. Defines locked parameters `slope_critical` and `R_km`.",
            "year": "2021",
        },
        "riley99": {
            "title": "T-RIX A new criteria to classify orographic complexity for wind energy yield assessment",
            "authors": "Riley, S.J., De Gloria, S.D., & Elliot, R.",
            "description": "Description of the ruggedness index method.",
            "journal": "Intermountain Journal of Science 5 (1-4)",
            "year": "1999",
        },
        "fgw23": {
            "title": "T-RIX A new criteria to classify orographic complexity for wind energy yield assessment",
            "authors": "FGW e.V., Adler, D., Albrecht, C., Cordes, J., Gaupp, T., & Richter-Rose, M.",
            "description": "Comparison of various methods and summary of TR6 conform configuration.",
            "meeting": "31. Windenergietage",
            "publisher": "Zenodo",
            "year": "2023",
        },
    },
    "parameters": {
        "ray": {
            "n_angles": {
                "description": "Number of ray directions, evenly spaced over 360°, North included.",
                "unit": None,
                "value": 72,
                "dtype": "int",
                "source": "TR6",
                "locked": True,
            },
            "R_km": {
                "description": "Ray radius.",
                "unit": "km",
                "value": 3.5,
                "source": "TR6",
                "locked": True,
            },
            "dr_km": {
                "description": "Sampling step size along a ray.",
                "unit": "km",
                "value": 0.05,
                "range": [0.001, 3.5],
            },
            # TODO: Clarify if this is needed for any check/validation.
            "crs": {
                "description": ".",
                "unit": None,
                "value": None,
            },
        },
        "slope": {
            "slope_critical": {
                "description": "Slope threshold distinguishing steep and flat.",
                "unit": "m/m",
                "value": 0.033,
                "source": "TR6",
                "locked": True,
            },
        },
        "sampler": {
            "interpolation_method": {
                "description": "Sampling method of the site elevations from the raster DEM.",
                "value": "linear",
            },
        },
    },
    "inputs": {
        "dem": {
            "description": "Digital elevation model.",
            "type": "xarray.DataArray",
            "required": True,
            "notes": [
                "If CRS is not exposed, all coordinates shall be in the same CRS.",
                "CRS may differ from the site's CRS.",
            ],
            "conditions": [
                "coordinates 'x' and 'y' present",
                "if CRS is relevant, `rio` and `crs` must be an attribute each",
                "resolution at least 30m",
            ],
        },
        "locations_site": {
            "description": "Coordinates of the assessed sites.",
            "required": True,
            "notes": [
                "If CRS is not projected metric, results may be distorted and not interpretable.",
            ],
            "conditions": [
                "projected metric CRS",
                "if CRS is relevant, `rio` and `crs` must be an attribute each",
            ],
        },
        "locations_reference": {
            "description": "Coordinates of the wind data base sites.",
            "required": False,
            "notes": ["If absent, TRIX and rix_details are omitted."],
            "conditions": [
                "projected metric CRS",
                "if CRS is relevant, `rio` and `crs` must be an attribute each",
                "CRS must agree with the one of `locations_site`",
            ],
        },
    },
    "artifacts": {
        "profiles": {
            "summary": ["rix_summary", "trix", "manifest"],
            "full": ["rix_summary", "trix", "manifest", "geopackage"],
        },
    },
    "schema": {
        "manifest": {
            "description": "Run manifest: metadata, config, artifact references, field definitions",
        },
        "rix_summary": {
            "description": "Table of the RIX and elevation values, and summary statistics thereof.",
            "columns": {
                COLUMN_KEYS.site_id: {
                    "description": "Site identifier of the assessed site.",
                    "unit": None,
                },
                COLUMN_KEYS.elevation: {
                    "description": "Site elevation as determined from the DEM. Mean of the origins of all rays.",
                    "unit": "m",
                },
                COLUMN_KEYS.elevation_std: {
                    "description": "Standard deviation of `elevation`.",
                    "unit": "m",
                },
                COLUMN_KEYS.rix: {
                    "description": "Mean ruggedness index across all ray directions. "
                    "Proportion of the total length of all steep segments among the total length of all rays.",
                    "unit": None,
                },
                COLUMN_KEYS.rix_std: {
                    "description": "Standard deviation of ruggedness across ray directions.",
                    "unit": None,
                },
                COLUMN_KEYS.rix_min: {
                    "description": "Minimal ruggedness across ray directions.",
                    "unit": None,
                },
                COLUMN_KEYS.rix_max: {
                    "description": "Maximal ruggedness across ray directions.",
                    "unit": None,
                },
                COLUMN_KEYS.n_rays: {
                    "description": "Number of rays directions evaluated.",
                    "unit": None,
                },
                COLUMN_KEYS.slope_critical: {
                    "description": "Slope threshold distinguishing steep and flat",
                    "unit": "m/m",
                },
            },
        },
        "trix": {
            "description": "Table of representativity measures and related distances.",
            "columns": {
                COLUMN_KEYS.site_id: {
                    "description": "Site identifier of the assessed site.",
                    "unit": None,
                },
                COLUMN_KEYS.reference_id: {
                    "description": "Site identifier of the wind data base site.",
                    "unit": None,
                },
                COLUMN_KEYS.transferability: {
                    "description": "Representability of the wind measurement at the wind date base site. "
                    "for the assessed site.",
                    "unit": None,
                    "dtype": "int",
                    "values": {
                        0: "Not representable.",
                        1: "Conditionally representable with additional uncertainty.",
                        2: "Representable.",
                    },
                },
                COLUMN_KEYS.distance: {
                    "description": "Distance between the assessed site and the wind data base site.",
                    "unit": "km",
                },
                COLUMN_KEYS.trix: {
                    "description": "T-RIX measure between the assessed site and the wind data base site.",
                    "unit": "km",
                },
                COLUMN_KEYS.A: {
                    "description": "Distance threshold for full representability at the given T-RIX value.",
                    "unit": "km",
                },
                COLUMN_KEYS.B: {
                    "description": "Distance threshold for conditional representability at the given T-RIX value.",
                    "unit": "km",
                },
            },
        },
        "geopackage": {
            "layers": {
                "locations_site": {
                    "description": "Coordinates of the assessed sites.",
                    "columns": {
                        COLUMN_KEYS.site_id: {
                            "description": "Site identifier of the assessed site.",
                            "unit": None,
                        },
                        COLUMN_KEYS.geometry: {
                            "description": "Coordinates of the assessed sites. CRS as presented by the user.",
                        },
                    },
                    "geometry_column": COLUMN_KEYS.geometry,
                    "geometry_type": "Point",
                },
                "locations_reference": {
                    "description": "Coordinates of the wind data base sites.",
                    "columns": {
                        COLUMN_KEYS.site_id: {
                            "description": "Site identifier of the wind data base site.",
                            "unit": None,
                        },
                        COLUMN_KEYS.geometry: {
                            "description": "Coordinates of the wind data base sites. CRS as presented by the user.",
                        },
                    },
                    "geometry_column": COLUMN_KEYS.geometry,
                    "geometry_type": "Point",
                },
                "ruggedness": {
                    "description": "Steep segments of the assessed sites on each ray.",
                    "columns": {
                        COLUMN_KEYS.site_id: {
                            "description": "Site identifier of the wind data base site.",
                            "unit": None,
                        },
                        COLUMN_KEYS.theta: {
                            "description": "Direction of the ray.",
                            "unit": "°",
                            "dtype": "float",
                            "range": [0, 360],
                        },
                        COLUMN_KEYS.segment_id: {
                            "description": "Identifier of the steep segment on the given ray of the given site.",
                            "unit": None,
                        },
                        COLUMN_KEYS.geometry: {
                            "description": "Coordinates of segments start/end. CRS as presented by the user.",
                        },
                    },
                    "geometry_column": COLUMN_KEYS.geometry,
                    "geometry_type": "LineString",
                },
                COLUMN_KEYS.trix: {
                    "description": "Table of representativity measures and related distances.",
                    "columns": {
                        COLUMN_KEYS.site_id: {
                            "description": "Site identifier of the assessed site.",
                            "unit": None,
                        },
                        COLUMN_KEYS.reference_id: {
                            "description": "Site identifier of the wind data base site.",
                            "unit": None,
                        },
                        COLUMN_KEYS.transferability: {
                            "description": "Representability of the wind measurement at the wind date base site. "
                            "for the assessed site.",
                            "unit": None,
                            "dtype": "int",
                            "values": {
                                0: "Not representable.",
                                1: "Conditionally representable with additional uncertainty.",
                                2: "Representable.",
                            },
                        },
                        COLUMN_KEYS.distance: {
                            "description": "Distance between the assessed site and the wind data base site.",
                            "unit": "km",
                        },
                        COLUMN_KEYS.trix: {
                            "description": "T-RIX measure between the assessed site and the wind data base site.",
                            "unit": "km",
                        },
                        COLUMN_KEYS.A: {
                            "description": "Distance threshold for full representability at the given T-RIX value.",
                            "unit": "km",
                        },
                        COLUMN_KEYS.B: {
                            "description": "Distance threshold for conditional representability at the given T-RIX "
                            "value.",
                            "unit": "km",
                        },
                    },
                    "geometry_column": COLUMN_KEYS.geometry,
                    "geometry_type": None,
                },
            }
        },
    },
}
