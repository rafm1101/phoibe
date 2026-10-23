import typing
import warnings
from collections.abc import Hashable, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd

from ...computations import trix
from ...computations.results import RadialRuggedness
from ..product.schema import PRODUCT_SPECIFICATION
from ..product.specification import ProductSpecification, TableSpecification

RIX_TABLE = "rix_table"
TRIX_TABLE = "trix_table"
LOCATIONS_SITE_TABLE = "locations_site"
LOCATIONS_REFERENCE_TABLE = "locations_reference"
RUGGEDNESS_TABLE = "ruggedness"

SITE_ID = "site_id"
REFERENCE_ID = "reference_id"
GEOMETRY = "geometry"


def rix_table(spec: ProductSpecification, results: Mapping[Hashable, RadialRuggedness]) -> pd.DataFrame:
    """RIX and elevation of every site, indexed by the site identifier."""
    table = spec.table(RIX_TABLE)
    ids = pd.Index(list(results), name=table.column(SITE_ID).name)
    radial_rugs = list(results.values())
    zs = [radial_rug.z for radial_rug in radial_rugs]
    ruggedness = [radial_rug.ruggednesses for radial_rug in radial_rugs]
    quantities = {
        "elevation": zs,
        "rix": [radial_rug.rix for radial_rug in radial_rugs],
        "rix_std": [float(np.std(values)) for values in ruggedness],
        "rix_min": [float(np.min(values)) for values in ruggedness],
        "rix_max": [float(np.max(values)) for values in ruggedness],
        "n_rays": [radial_rug.n_rays for radial_rug in radial_rugs],
        "slope_critical": [radial_rug.slope_critical for radial_rug in radial_rugs],
    }
    return _to_df(table, ids, quantities)


def trix_table(
    spec: ProductSpecification,
    assessment: trix.TrixAssessment,
    site_ids: Sequence[Hashable],
    reference_ids: Sequence[Hashable],
) -> pd.DataFrame:
    """One row per pair of site and reference location, indexed by both identifiers."""
    table = spec.table(TRIX_TABLE)
    index = pd.MultiIndex.from_product(
        [list(site_ids), list(reference_ids)],
        names=[table.column(SITE_ID).name, table.column(REFERENCE_ID).name],
    )
    quantities = {
        "transferability": assessment.transferability.ravel(),
        "distance": assessment.distance_km.ravel(),
        "trix": assessment.trix.ravel(),
        "A": assessment.A.ravel(),
        "B": assessment.B.ravel(),
    }
    return _to_df(table, index, quantities)


def transferability_counts(spec: ProductSpecification, table: pd.DataFrame) -> dict[int, int]:
    """Number of pairs per transferability state."""
    column = spec.table(TRIX_TABLE).column("transferability").name
    counts = table[column].value_counts().sort_index()
    return {int(str(state)): int(count) for state, count in counts.items()}


def locations_table(spec: ProductSpecification, name: str, locations: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Identifier and geometry of the locations. Other columns of the input are not part of the product."""
    table = spec.table(name)
    geometry = table.column(GEOMETRY).name
    index = locations.index.rename(table.column(SITE_ID).name)
    frame = _to_df(table, index, {geometry: locations.geometry.values})
    return gpd.GeoDataFrame(frame, geometry=geometry, crs=locations.crs)


def steep_segments(spec: ProductSpecification, results: Mapping[Hashable, RadialRuggedness]) -> gpd.GeoDataFrame:
    """Steep segments of all sites, one row per segment, indexed by the site identifier."""
    table = spec.table(RUGGEDNESS_TABLE)
    geometry = table.column(GEOMETRY).name
    ids, thetas, segment_ids, segments = [], [], [], []
    crs = None
    for location_id, radial in results.items():
        for ray in radial.rays:
            crs = ray.profile.ray.crs if crs is None else crs
            for segment_id, segment in enumerate(ray.steep_segments):
                ids.append(location_id)
                thetas.append(ray.theta)
                segment_ids.append(segment_id)
                segments.append(segment)
    index = pd.Index(ids, name=table.column(SITE_ID).name)
    frame = _to_df(table, index, {"theta": thetas, "segment_id": segment_ids, geometry: segments})
    return gpd.GeoDataFrame(frame, geometry=geometry, crs=crs)


def rose(results: Mapping[Hashable, RadialRuggedness]) -> pd.DataFrame:
    """Ruggedness rose: one row per site, one column per ray direction. Not an output of the product."""
    rows = [
        pd.Series(index=radial.angles, data=radial.ruggednesses, name=location_id)
        for location_id, radial in results.items()
    ]
    return pd.DataFrame(rows)


def _to_df(table: TableSpecification, index: pd.Index, quantities: Mapping[str, typing.Any]) -> pd.DataFrame:
    """Frame with the columns of `table`, in order. The identifier columns form the index."""
    index_names: list[str] = []
    for name in index.names:
        if not isinstance(name, str):
            raise ValueError(f"Table '{table.name}': Each index level must be named after a column, got {name!r}.")
        table.column(name)
        index_names.append(name)
    columns = [name for name in table.column_names if name not in index_names]
    missing = [name for name in columns if name not in quantities]
    if missing:
        raise ValueError(f"Table '{table.name}' lists column(s) this builder cannot provide: {', '.join(missing)}.")
    return pd.DataFrame({name: quantities[name] for name in columns}, index=index)


@warnings.deprecated("`steep_segments_geodataframe` will be removed.")
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


@warnings.deprecated("`_build_rix_rose` will be removed.")
def _build_rix_rose(radial_results: dict[object, RadialRuggedness]) -> pd.DataFrame:
    """Build summary DataFrame and detail GeoDataFrame from radial results."""
    rix_rose_rows = []

    for location_id, radial_rix in radial_results.items():
        rix_rose = pd.Series(index=radial_rix.angles, data=radial_rix.ruggednesses, name=location_id)
        rix_rose_rows.append(rix_rose)

    rix_roses = pd.DataFrame(rix_rose_rows)

    return rix_roses


@warnings.deprecated("`_build_summary` will be removed.")
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
                spec.column("elevation").name: radial_rix.z,
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


@warnings.deprecated("`_compute_trix` will be removed.")
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
        elevation_reference=np.array(summary_reference[spec.column("elevation").name]),
    )
    index = summary_site.index.rename(spec.column("site_id").name)
    columns = summary_reference.index.rename(spec.column("reference_id").name)

    trix_result = pd.DataFrame(data=trix_records, index=index, columns=columns)

    _A, _B = trix.compute_trix_limit_distances(trix=trix_records, decimals=1)
    A = pd.DataFrame(data=_A, index=index, columns=columns)
    B = pd.DataFrame(data=_B, index=index, columns=columns)

    return trix_result, A, B
