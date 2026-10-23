"""Value of the "manifest" entry of the product definition, matching what `pipelines.manifest.build_manifest` writes.

Replace `PRODUCT_DEFINITION_TRIX["manifest"]` with `MANIFEST_DEFINITION`. Where a key holds a mapping whose keys
depend on the run (parameter groups, collections, checks) it is described as one key, not enumerated.
"""

MANIFEST_DEFINITION = {
    "description": "Run manifest: the contract of the product version and a description of this run.",
    "keys": {
        "contract": {
            "description": "Contract of this product version, generated from the specification: inputs, parameters, "
            "checks, outputs, profiles, references. Identical for all runs of one version.",
        },
        "meta": {
            "description": "What is processed.",
            "keys": {
                "name": {"description": "Name of the product."},
                "version": {"description": "Version of the product."},
                "description": {"description": "Description of the product."},
                "created_at": {"description": "Time of the run, UTC, ISO 8601."},
            },
        },
        "parameters": {
            "description": "Value of every parameter used in this run, by group and parameter name. "
            "Includes the values fixed by the product.",
        },
        "spatial_context": {
            "description": "Context about data sources.",
            "keys": {
                "source_dem": {
                    "description": "The DEM. Free-form provenance as supplied by the user (e.g. name, source), "
                    "plus what is read from the data.",
                    "keys": {
                        "crs": {"description": "CRS of the DEM. Null if the DEM carries none."},
                        "extent": {
                            "description": "Extent of the DEM. Null if the DEM carries no georeferencing."
                            "(no `rio` accessor)",
                            "keys": {
                                "west": {"description": "Left edge."},
                                "south": {"description": "Bottom edge."},
                                "east": {"description": "Right edge."},
                                "north": {"description": "Top edge."},
                            },
                        },
                        "resolution": {
                            "description": "Cell size of the DEM. Null if the DEM carries no georeferencing."
                            "(no `rio` accessor)",
                            "keys": {
                                "dx": {"description": "Cell size in x direction."},
                                "dy": {"description": "Cell size in y direction. Negative if y decreases with row."},
                            },
                        },
                    },
                },
                "source_ray": {
                    "description": "The rays.",
                    "keys": {"crs": {"description": "Distinct CRS of the rays, as a list."}},
                },
                "alignment": {
                    "description": "Alignment of rays and DEM.",
                    "keys": {"messages": {"description": "Distinct messages from aligning the CRS of rays and DEM."}},
                },
            },
        },
        "run": {
            "description": "Context about the run: what has been evaluated and computed, summary statistics.",
            "keys": {
                "project_name": {"description": "Project name if provided."},
                "n_sites": {"description": "Number of assessed sites."},
                "n_references": {"description": "Number of reference sites."},
                "computed": {
                    "description": "What has been computed.",
                    "values": {
                        "rix_site": "RIX at assessed sites",
                        "rix_reference": "RIX at reference sites",
                        "trix": "T-RIX and distances",
                    },
                },
                "diagnostics": {
                    "description": "Summary statistics about the results.",
                    "keys": {
                        "nan_count": {
                            "description": "NaN values encountered while sampling, per collection of locations.",
                        },
                        "n_locations_with_nans": {
                            "description": "Locations with at least one NaN on any ray, per collection of locations.",
                        },
                        "transferability_counts": {
                            "description": "Number of pairs for each transferability state.",
                        },
                        "finding_counts": {"description": "Number of findings per check."},
                        "findings_omitted": {"description": "Findings counted but not listed in `findings`."},
                    },
                },
                "findings": {
                    "description": "List of findings, capped per check. Each entry has: check, severity, collection, "
                    "location_id, theta, detail.",
                },
            },
        },
        "artifacts": {
            "description": "Output files.",
            "keys": {
                "profile": {"description": "Output profile that was written."},
                "files": {"description": "Filename of every artifact, by artifact key. Includes the manifest."},
                "table_records": {"description": "Number of rows of every table artifact, by artifact key."},
                "layer_records": {"description": "Number of records of every layer, by layer key."},
            },
        },
    },
}
