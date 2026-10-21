"""Outward result model of the assessment (D2)."""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Mapping

from .records import RunRecord

if typing.TYPE_CHECKING:
    import geopandas as gpd
    import pandas as pd


@dataclasses.dataclass(frozen=True)
class ResultSummary:
    """Container for a completed RIX analysis run.

    Kept in memory as it is returned by the use case. It can be published afterwards without recomputing.
    The tables are keyed by their name in the product specification, so a writer needs no hard-coded mapping.
    """

    tables: Mapping[str, pd.DataFrame | gpd.GeoDataFrame]
    """Output tables by name in the specification. Optional ones are absent, e.g. "trix_table" without references."""
    ruggedness_roses: pd.DataFrame
    """Ruggedness rose for all sites. Not an output of the product, for inspection."""
    record: RunRecord
    """Typed information about the run: parameters, field, findings, counts."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "tables", types.MappingProxyType(dict(self.tables)))

    @property
    def summary(self) -> pd.DataFrame:
        """RIX assessment of the sites."""
        return self.tables["rix_table"]

    @property
    def locations_site(self) -> gpd.GeoDataFrame:
        """Assessed sites: identifier and geometry."""
        return self.tables["locations_site"]

    @property
    def locations_reference(self) -> gpd.GeoDataFrame | None:
        """Assessed reference locations: identifier and geometry."""
        return self.tables.get("locations_reference")

    @property
    def trix_table(self) -> pd.DataFrame | None:
        """Pairwise TRIX metrics and the limit distances A and B."""
        return self.tables.get("trix_table")

    @property
    def steep_segments(self) -> gpd.GeoDataFrame | None:
        """Steep segments of the sites. Absent if they were not requested."""
        return self.tables.get("ruggedness")


# @dataclasses.dataclass(frozen=True)
# class ResultSummary:
#     """Immutable container for a completed RIX analysis run.

#     Kept in memory as it is returned by the use case. It can be published afterwards without recomputing.
#     """

#     locations_site: gpd.GeoDataFrame
#     """Assessed sites."""
#     locations_reference: gpd.GeoDataFrame | None
#     """Assessed reference locations (wind data base)."""
#     summary: pd.DataFrame
#     """Sites' RIX assessment."""
#     ruggedness_roses: pd.DataFrame
#     """Ruggedness rose for all sites."""
#     trix_table: pd.DataFrame | None
#     """Table with all pairwise metrics and the threshold matrices A and B required for transferability."""
#     steep_segments: gpd.GeoDataFrame
#     """Steep segments of sites as LineStrings."""
#     record: RunRecord
#     """Typed information about the run: parameters, field, findings, counts."""
