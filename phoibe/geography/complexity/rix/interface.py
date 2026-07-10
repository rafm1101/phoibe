import dataclasses
import typing

from .schema import PRODUCT_DEFINITION_TRIX


def _get_parameter(definition: dict, *path: str, strict: bool = True) -> typing.Any:
    """Navigate nested dict.

    Parameters
    ----------
    dictionary
        Nested dictionary
    *path
        Path of keys.

    Returns
    -------
    node
        Value of the requested field.

    Raises
    ------
    KeyError
        In case a full path on miss.
    """
    node = definition
    for n, key in enumerate(path):
        try:
            node = node[key]
        except KeyError:
            if not strict and n == len(path) - 1:
                return None
            else:
                raise KeyError(f"PRODUCT_DEFINITION missing expected key {key}: {' -> '.join(path)}") from None
    return node


@dataclasses.dataclass(frozen=True)
class Keys:
    """Keys employed at publsihed artifacts. Changes require version bumps."""

    """Joint table columns:"""
    site_id: str = _get_parameter(PRODUCT_DEFINITION_TRIX, "schema", "rix_summary", "columns", "site_id", "name")
    """Id of the site to assess."""
    geometry: str = _get_parameter(
        PRODUCT_DEFINITION_TRIX, "schema", "geopackage", "layers", "locations_site", "geometry_column"
    )
    """Coordinates of geometric objects."""

    """Summary:"""
    elevation: str = _get_parameter(PRODUCT_DEFINITION_TRIX, "schema", "rix_summary", "columns", "elevation", "name")
    """Site elevation."""
    elevation_std: str = _get_parameter(
        PRODUCT_DEFINITION_TRIX, "schema", "rix_summary", "columns", "elevation_std", "name"
    )
    """Site elevation standard deviation (if measured in different occasions)."""
    n_rays: str = _get_parameter(PRODUCT_DEFINITION_TRIX, "schema", "rix_summary", "columns", "n_rays", "name")
    """Number of rays evaluated."""
    rix: str = _get_parameter(PRODUCT_DEFINITION_TRIX, "schema", "rix_summary", "columns", "rix", "name")
    """RIX value."""
    rix_std: str = _get_parameter(PRODUCT_DEFINITION_TRIX, "schema", "rix_summary", "columns", "rix_std", "name")
    """Standard deviation of directional rix values."""
    rix_min: str = _get_parameter(PRODUCT_DEFINITION_TRIX, "schema", "rix_summary", "columns", "rix_min", "name")
    """Minimum of directional rix values."""
    rix_max: str = _get_parameter(PRODUCT_DEFINITION_TRIX, "schema", "rix_summary", "columns", "rix_max", "name")
    """Maximum of directional rix values."""
    slope_critical: str = _get_parameter(
        PRODUCT_DEFINITION_TRIX, "schema", "rix_summary", "columns", "slope_critical", "name"
    )
    """Slope threshold considered as steepness."""

    """TRIX evaluation:"""
    reference_id: str = _get_parameter(
        PRODUCT_DEFINITION_TRIX, "schema", "trix_table", "columns", "reference_id", "name"
    )
    """ID of reference site"""
    transferability: str = _get_parameter(
        PRODUCT_DEFINITION_TRIX, "schema", "trix_table", "columns", "transferability", "name"
    )
    """Transferability according to the distance thresholds."""
    distance: str = _get_parameter(PRODUCT_DEFINITION_TRIX, "schema", "trix_table", "columns", "distance", "name")
    """Distance [km] between site and reference."""
    trix: str = _get_parameter(PRODUCT_DEFINITION_TRIX, "schema", "trix_table", "columns", "trix", "name")
    """T-RIX."""
    A: str = _get_parameter(PRODUCT_DEFINITION_TRIX, "schema", "trix_table", "columns", "A", "name")
    """Distance A [km]."""
    B: str = _get_parameter(PRODUCT_DEFINITION_TRIX, "schema", "trix_table", "columns", "B", "name")
    """Distance B [km]."""

    """Ray evaluation results:"""
    ruggedness: str = "ruggedness"
    """Ruggedness along a single ray."""
    total_length_m: str = "total_length_m"
    """Total length of a ray."""
    steep_length_m: str = "steep_length_m"
    """Total length of the steep parts of a ray."""
    max_abs_slope: str = "max_abs_slope"
    """Maximum absolute slope encountered along a ray."""
    n_steep_segments: str = "n_steep_segments"
    """Number of steep segments encountered along a ray."""

    """Steep segments:"""
    segment_id: str = _get_parameter(
        PRODUCT_DEFINITION_TRIX, "schema", "geopackage", "layers", "ruggedness", "columns", "segment_id", "name"
    )
    """Ientifier of a (steep) ray segment."""
    theta: str = _get_parameter(
        PRODUCT_DEFINITION_TRIX, "schema", "geopackage", "layers", "ruggedness", "columns", "theta", "name"
    )
    """Ray angle."""

    """Writer keys:"""
    manifest: str = "manifest"
    """Manifest file key."""
    rix_summary: str = "rix_summary"
    """Rix table file key."""
    trix_table: str = "trix_table"
    """T-RIX table file key."""
    geopackage: str = "geopackage"
    """Geopackage file key."""
    locations_site_layer: str = "locations_site"
    """Assessed locations layer key."""
    locations_reference_layer: str = "locations_reference"
    """Wind data base locations layer key."""
    ruggedness_layer: str = "ruggedness"
    """Ruggedness layer key."""
    trix_layer: str = "trix"
    """T-RIX layer key."""

    """DEM dimension identifiers:"""
    x: str = "x"
    """Zonal dimension."""
    y: str = "y"
    """Meridonal dimension."""

    """Summary/top-level keys:"""
    project_name: str = _get_parameter(PRODUCT_DEFINITION_TRIX, "schema", "manifest", "keys", "project_name", "name")
    """Project name."""
    meta: str = _get_parameter(PRODUCT_DEFINITION_TRIX, "schema", "manifest", "keys", "meta", "name")
    """Context about the process."""
    parameters: str = _get_parameter(PRODUCT_DEFINITION_TRIX, "schema", "manifest", "keys", "parameters", "name")
    """Context about the parameters."""
    spatial_context: str = _get_parameter(
        PRODUCT_DEFINITION_TRIX, "schema", "manifest", "keys", "spatial_context", "name"
    )
    """Context about the data sources evaluated."""
    run: str = _get_parameter(PRODUCT_DEFINITION_TRIX, "schema", "manifest", "keys", "run", "name")
    """Information about the run."""
    artifacts: str = _get_parameter(PRODUCT_DEFINITION_TRIX, "schema", "manifest", "keys", "artifacts", "name")
    """Information about the artifacts stored."""

    """Metainformation/internal keys:"""
    created_at: str = "created_at"
    """meta: Creation timestamp."""
    source_dem: str = "source_dem"
    """spatial_context: Context about the DEM."""
    source_ray: str = "source_ray"
    """spatial_context: Context about the rays."""
    n_sites: str = _get_parameter(
        PRODUCT_DEFINITION_TRIX, "schema", "manifest", "keys", "run", "keys", "n_sites", "name"
    )
    """run: Number of assessed sites."""
    n_references: str = _get_parameter(
        PRODUCT_DEFINITION_TRIX, "schema", "manifest", "keys", "run", "keys", "n_references", "name"
    )
    """run: Number of assessed references."""
    computed: str = _get_parameter(
        PRODUCT_DEFINITION_TRIX, "schema", "manifest", "keys", "run", "keys", "computed", "name"
    )
    """run: What has been computed."""
    diagnostics: str = _get_parameter(
        PRODUCT_DEFINITION_TRIX, "schema", "manifest", "keys", "run", "keys", "diagnostics", "name"
    )
    """run: Summary statistics."""
    n_sites_with_nans: str = _get_parameter(
        PRODUCT_DEFINITION_TRIX,
        "schema",
        "manifest",
        "keys",
        "run",
        "keys",
        "diagnostics",
        "keys",
        "n_sites_with_nans",
        "name",
    )
    """run.diagnostics: Summary statistics."""
    transferability_counts: str = _get_parameter(
        PRODUCT_DEFINITION_TRIX,
        "schema",
        "manifest",
        "keys",
        "run",
        "keys",
        "diagnostics",
        "keys",
        "transferability_counts",
        "name",
    )
    """run.diagnostics: Summary statistics."""
    dem: str = "dem"
    """Top-level key holding DEM information."""
    rays: str = "rays"
    """Top-level key holding ray information."""
    alignment: str = "internal_alignment"
    """Context about the alignment of DEM and rays."""
    crs_dem: str = "crs_dem"
    """CRS of the digital elevation model."""
    crs_ray: str = "crs_ray"
    """CRS of the ray's coordinates."""
    extent_dem: str = "extent_dem"
    """Spatial extent of the digital elevation model."""
    resolution_dem: str = "resolution_dem"
    """Spatial resolution of the digital elevation model."""
    message: str = "message"
    """Messages appearing during processing."""
    nan_count: str = "nan_count"
    """Number of NaNs appeared during sampling."""
