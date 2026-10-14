import numpy as np
import pandas as pd

from ..computations import trix
from ..computations.results import RadialRuggedness
from ..definitions.interface import Keys
from .schema import PRODUCT_SPECIFICATION
from .specification import TableSpecification

KEYS = Keys()


def steep_segments_geodataframe(radial_ruggedness: RadialRuggedness):
    """Generate a GeoDataFrame holding all steep segments.

    Returns
    -------
    geopandas.GeoDataFrame
        Geodataframe where each row contains the geometry of one steep segment.

    Raises
    ------
    ImportError
        If geopandas is not installed.
    """
    try:
        import geopandas as gpd
    except ImportError as exception:
        raise ImportError("RadialRixResult.steep_segments_geodataframe() requires geopandas.") from exception

    records = []
    for ray in radial_ruggedness.rays:
        for i, segment in enumerate(ray.steep_segments):
            records.append(
                {
                    PRODUCT_SPECIFICATION.tables["ruggedness"].column("theta").name: ray.theta,
                    PRODUCT_SPECIFICATION.tables["ruggedness"].column("segment_id").name: i,
                    PRODUCT_SPECIFICATION.tables["ruggedness"].column("geometry").name: segment,
                }
            )
    return gpd.GeoDataFrame(
        records,
        columns=[
            PRODUCT_SPECIFICATION.tables["ruggedness"].column("theta").name,
            PRODUCT_SPECIFICATION.tables["ruggedness"].column("segment_id").name,
            PRODUCT_SPECIFICATION.tables["ruggedness"].column("geometry").name,
        ],
        geometry=PRODUCT_SPECIFICATION.tables["ruggedness"].column("geometry").name,
        crs=radial_ruggedness.rays[0].profile.ray.crs,
    )


def _build_rix_rose(radial_results: dict[object, RadialRuggedness]) -> pd.DataFrame:
    """Build summary DataFrame and detail GeoDataFrame from radial results."""
    rix_rose_rows = []

    for location_id, radial_rix in radial_results.items():
        rix_rose = pd.Series(index=radial_rix.angles, data=radial_rix.ruggednesses, name=location_id)
        rix_rose_rows.append(rix_rose)

    rix_roses = pd.DataFrame(rix_rose_rows)

    return rix_roses


def _build_summary(
    radial_results: dict[object, RadialRuggedness],
    spec: TableSpecification,
    index_name: object = None,
) -> pd.DataFrame:
    """Build summary DataFrame from radial results."""
    summary_rows = []

    for location_id, radial_rix in radial_results.items():
        description = radial_rix.describe()
        summary_rows.append(
            {
                spec.column("site_id").name: location_id,
                spec.column("elevation").name: radial_rix.z[0],
                spec.column("elevation_std").name: radial_rix.z[1],
                spec.column("rix").name: radial_rix.rix,
                spec.column("rix_std").name: description[spec.column("rix_std").name],
                spec.column("rix_min").name: description[spec.column("rix_min").name],
                spec.column("rix_max").name: description[spec.column("rix_max").name],
                spec.column("n_rays").name: radial_rix.n_rays,
                spec.column("slope_critical").name: radial_rix.slope_critical,
            }
        )
    index = pd.Index(radial_results.keys(), name=index_name)
    summary = pd.DataFrame(summary_rows, index=index)
    return summary


def _compute_trix(
    summary_site: pd.DataFrame, summary_reference: pd.DataFrame, spec: TableSpecification
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute pairwise TRIX as Cartesian product of summary_site × summary_reference.

    Returns
    -------
    trix_result, A, B
        Pairwise trix values and limit distance matrices A and B with sites in rows
        and reference locations in columns.
    """
    trix_records = trix.compute_trix(
        rix_site=np.array(summary_site[spec.column("rix").name]),
        elevation_site=np.array(summary_site[spec.column("elevation").name]),
        rix_reference=np.array(summary_reference[spec.column("rix").name]),
        elevation_wind=np.array(summary_reference[spec.column("elevation").name]),
    )
    index = summary_site.index.rename(spec.column("site_id").name)
    columns = summary_reference.index.rename(spec.column("reference_id").name)

    trix_result = pd.DataFrame(data=trix_records, index=index, columns=columns)

    _A, _B = trix.compute_trix_limit_distances(trix=trix_records, decimals=1)
    A = pd.DataFrame(data=_A, index=index, columns=columns)
    B = pd.DataFrame(data=_B, index=index, columns=columns)

    return trix_result, A, B
