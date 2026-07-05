import dataclasses
import typing

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

INTERPOLATION_METHODS = typing.Literal["linear", "nearest"]


@dataclasses.dataclass(frozen=True)
class ColumnKeys:
    """Keys employed at various occasions."""

    """DEM dimension identifiers:"""
    x: str = "x"
    """Zonal dimension."""
    y: str = "y"
    """Meridonal dimension."""

    """Metainformation:"""
    created_at: str = "created_at"
    """Creation timestamp."""
    spatial_context: str = "spatial_context" ""
    """Information about data sources evaluated."""
    source_dem: str = "source_dem"
    """Context about the DEM."""
    source_ray: str = "source_ray"
    """Context about the rays."""
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
    segment_id: str = "segment_id"
    """Ientifier of a (steep) ray segment."""
    site_id: str = "location_id"
    """Id of the site to assess."""
    theta: str = "theta"
    """Ray angle."""

    """Summary:"""
    elevation: str = "elevation"
    """Site elevation."""
    elevation_std: str = "elevation_std"
    """Site elevation standard deviation (if measured in different occasions)."""
    n_rays: str = "n_rays"
    """Number of rays evaluated."""
    rix: str = "rix"
    """RIX value."""
    rix_std: str = "rix_std"
    """Standard deviation of directional rix values."""
    rix_min: str = "rix_min"
    """Minimum of directional rix values."""
    rix_max: str = "rix_max"
    """Maximum of directional rix values."""
    slope_critical: str = "slope_critical"
    """Slope threshold considered as steepness."""

    """TRIX evaluation:"""
    # site_id: str = "site_id"
    # """ID of assessed site"""
    reference_id: str = "reference_id"
    """ID of reference site"""
    transferability: str = "transferability"
    """Transferability according to the distance thresholds."""
    distance: str = "distance"
    """Distance [km] between site and reference."""
    trix: str = "trix"
    """T-RIX."""
    A: str = "A"
    """Distance A [km]."""
    B: str = "B"
    """Distance B [km]."""
    geometry: str = "geometry"
    """Coordinates of geometric objects."""
