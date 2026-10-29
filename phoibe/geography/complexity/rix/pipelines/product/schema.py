from .manifest_definition import MANIFEST_DEFINITION
from .specification import ProductSpecification

PRODUCT_DEFINITION_TRIX = {
    "name": "T-RIX assessment",
    "description": "TRIX, TR6 Rev.12",
    "version": "2.0.0",
    "references": {
        "TR6": {
            "title": "Technische Richtlinien für Windenergieanlagen, "
            "Teil 6: Bestimmung von Windpotenzial und Energieerträgen",
            "author": "FGW e.V.",
            "description": "Technische Richtlinie 6. Defines locked parameters `slope_critical` and `R_km`.",
            "year": "2021",
        },
        "riley99": {
            "title": "A Terrain Ruggedness that Quantifies Topographic Heterogeneity",
            "authors": "Riley, S.J., De Gloria, S.D., & Elliot, R.",
            "description": "Description of the ruggedness index method.",
            "journal": "Intermountain Journal of Science 5 (1--4)",
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
        "implementation": {
            "title": "T-RIX assessment, implementation.",
            "description": "Choices of this implementation that fix the meaning of numbers, but are not prescribed by "
            "other sources.",
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
                "range": [0.00001, 3.5],
            },
            "crs": {
                "description": "CRS for validation against data source CRS. To be set by the user.",
                "unit": None,
                "value": None,
                "dtype": "str",
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
            "nan_policy": {
                "description": "Handling of NaN values encountered while sampling a ray: "
                "keep (mask), cut ray at the first one (truncate) or fail (error).",
                "value": "mask",
                "dtype": "str",
                "source": "implementation",
                "locked": True,
            },
        },
        "trix": {
            "decimals": {
                "description": "Number of decimals T-RIX is rounded to and presented in the table. "
                "Comparison after rounding.",
                "unit": None,
                "value": 1,
                "source": "TR6",
                "locked": True,
            }
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
    "checks": {
        "crs_site_vs_reference": {
            "description": "Coordinates of sites and references are provided in the same CRS.",
            "severity": "warn",
            "parameters": {},
        },
        "crs_vs_config": {
            "description": "CRS of the sites agrees with configured CRS..",
            "severity": "warn",
            "parameters": {},
        },
        "unique_ids": {
            "description": "Provided locations are identifyable uniquely.",
            "severity": "abort",
            "parameters": {},
        },
        "centre_elevation": {
            "description": "When sampling rays, all rays' origin shall agree.",
            "severity": "warn",
            "parameters": {},
        },
        "locked_parameter_changed": {
            "description": "Check if chosen parameters comply with TR6.",
            "severity": "warn",
            "parameters": {},
        },
        "nan_count": {
            "description": "Number of NaN encountered during sampling.",
            "severity": "warn",
            "parameters": {"max_fraction": 0.05},
        },
        "rix_nan": {
            "description": "The RIX of a site is NaN, so TRIX cannot be established.",
            "severity": "warn",
        },
        "n_angle": {
            "description": "Number of angles for the assessment.",
            "severity": "abort",
            "parameters": {"range": [1, 360]},
        },
    },
    "artifacts": {
        "profiles": {
            "summary": ["rix_table", "trix_table", "manifest"],
            "full": ["rix_table", "trix_table", "manifest", "geopackage"],
        },
        "files": {
            "manifest": {"filename": "summary.yaml"},
            "rix_table": {"filename": "rix_table.csv", "table": "rix_table", "requires_input": None},
            "trix_table": {
                "filename": "trix_table.csv",
                "table": "trix_table",
                "requires_input": "locations_reference",
            },
            "geopackage": {"filename": "rix_details.gpkg", "table": None, "requires_input": None},
        },
        "layers": {
            "locations_site": {
                "artifact": "geopackage",
                "table": "locations_site",
                "name": "locations_site",
                "geometry_type": "Point",
                "requires_input": None,
            },
            "locations_reference": {
                "artifact": "geopackage",
                "table": "locations_reference",
                "name": "locations_reference",
                "geometry_type": "Point",
                "requires_input": "locations_reference",
            },
            "ruggedness": {
                "artifact": "geopackage",
                "table": "ruggedness",
                "name": "ruggedness",
                "geometry_type": "LineString",
                "requires_input": None,
            },
            "trix_table": {
                "artifact": "geopackage",
                "table": "trix_table",
                "name": "trix_table",
                "geometry_type": None,
                "requires_input": "locations_reference",
            },
        },
    },
    "manifest": MANIFEST_DEFINITION,
    # {
    #     "description": "Run manifest: metadata, config, artifact references, field definitions",
    #     "keys": {
    #         "project_name": {"name": "project_name", "description": "Project name if provided."},
    #         "meta": {"name": "meta", "description": "What is processed."},
    #         "parameters": {"name": "parameters", "description": "Configuration of the analysis."},
    #         "spatial_context": {"name": "spatial_context", "description": "Context about data sources."},
    #         "run": {
    #             "name": "run",
    #             "description": "Context about the run: What has been evaluated and computed, summary statistics.",
    #             "keys": {
    #                 "n_sites": {"name": "n_sites", "description": "Number of assessed sites."},
    #                 "n_references": {"name": "n_references", "description": "Number of reference sites."},
    #                 "computed": {
    #                     "name": "computed",
    #                     "description": "What has been computed.",
    #                     "values": {
    #                         "rix_site": "RIX at assessed sites",
    #                         "rix_reference": "RIX at reference sites",
    #                         "trix": "T-RIX and distances",
    #                     },
    #                 },
    #                 "diagnostics": {
    #                     "name": "diagnostics",
    #                     "description": "Summary statistics about the results.",
    #                     "keys": {
    #                         "n_sites_with_nans": {
    #                             "name": "n_sites_with_nans",
    #                             "description": "Number of assessed sites that experienced NaN in their rays.",
    #                         },
    #                         "transferability_counts": {
    #                             "name": "transferability_counts",
    #                             "description": "Number of pairs for each transferability state.",
    #                         },
    #                     },
    #                 },
    #             },
    #         },
    #         "artifacts": {"name": "artifacts", "description": "Output files."},
    #     },
    # },
    "tables": {
        "rix_table": {
            "description": "Table of the RIX and elevation values, and summary statistics thereof.",
            "columns": {
                "site_id": {
                    "name": "site_id",
                    "description": "Site identifier of the assessed site.",
                    "unit": None,
                },
                "elevation": {
                    "name": "elevation",
                    "description": "Site elevation as determined from the DEM. Mean of the origins of all rays.",
                    "unit": "m",
                },
                # "elevation_std": {
                #     "name": "elevation_std",
                #     "description": "Standard deviation of `elevation`.",
                #     "unit": "m",
                # },
                "rix": {
                    "name": "rix",
                    "description": "Mean ruggedness index across all ray directions. "
                    "Proportion of the total length of all steep segments among the total length of all rays.",
                    "unit": None,
                },
                "rix_std": {
                    "name": "rix_std",
                    "description": "Standard deviation of ruggedness across ray directions.",
                    "unit": None,
                },
                "rix_min": {
                    "name": "rix_min",
                    "description": "Minimal ruggedness across ray directions.",
                    "unit": None,
                },
                "rix_max": {
                    "name": "rix_max",
                    "description": "Maximal ruggedness across ray directions.",
                    "unit": None,
                },
                "n_rays": {
                    "name": "n_rays",
                    "description": "Number of rays directions evaluated.",
                    "unit": None,
                },
                "slope_critical": {
                    "name": "slope_critical",
                    "description": "Slope threshold distinguishing steep and flat",
                    "unit": "m/m",
                },
            },
        },
        "trix_table": {
            "description": "Table of representativity measures and related distances.",
            "columns": {
                "site_id": {
                    "description": "Site identifier of the assessed site.",
                    "unit": None,
                },
                "reference_id": {
                    "name": "reference_id",
                    "description": "Site identifier of the wind data base site.",
                    "unit": None,
                },
                "transferability": {
                    "name": "transferability",
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
                "distance": {
                    "name": "distance",
                    "description": "Distance between the assessed site and the wind data base site.",
                    "unit": "km",
                },
                "trix": {
                    "name": "trix",
                    "description": "T-RIX measure between the assessed site and the wind data base site.",
                    "unit": None,
                },
                "A": {
                    "name": "A",
                    "description": "Distance threshold for full representability at the given T-RIX value.",
                    "unit": "km",
                },
                "B": {
                    "name": "B",
                    "description": "Distance threshold for conditional representability at the given T-RIX value.",
                    "unit": "km",
                },
            },
        },
        "locations_site": {
            "description": "Coordinates of the assessed sites.",
            "columns": {
                "site_id": {
                    "description": "Site identifier of the assessed site.",
                    "unit": None,
                },
                "geometry": {
                    "description": "Coordinates of the assessed sites. CRS as presented by the user.",
                },
            },
            # "geometry_column": "geometry",
            # "geometry_type": "Point",
        },
        "locations_reference": {
            "description": "Coordinates of the wind data base sites.",
            "columns": {
                "site_id": {
                    "description": "Site identifier of the wind data base site.",
                    "unit": None,
                },
                "geometry": {
                    "description": "Coordinates of the wind data base sites. CRS as presented by the user.",
                },
            },
            # "geometry_column": "geometry",
            # "geometry_type": "Point",
        },
        "ruggedness": {
            "description": "Steep segments of the assessed sites on each ray.",
            "columns": {
                "site_id": {
                    "description": "Site identifier of the assessed site.",
                    "unit": None,
                },
                "theta": {
                    "name": "theta",
                    "description": "Direction of the ray.",
                    "unit": "°",
                    "dtype": "float",
                    "range": [0, 360],
                },
                "segment_id": {
                    "name": "segment_id",
                    "description": "Identifier of the steep segment on the given ray of the given site.",
                    "unit": None,
                },
                "geometry": {
                    "name": "geometry",
                    "description": "Coordinates of segments start/end. CRS as presented by the user.",
                },
            },
            # "geometry_column": "geometry",
            # "geometry_type": "LineString",
        },
    },
}

PRODUCT_SPECIFICATION = ProductSpecification.from_dict(PRODUCT_DEFINITION_TRIX)
