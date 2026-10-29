import dataclasses

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely.geometry

from phoibe.geography.complexity.rix.computations.results import RadialRuggedness
from phoibe.geography.complexity.rix.computations.trix import TrixAssessment
from phoibe.geography.complexity.rix.pipelines.results import tables

STEEP = [10, 60, 110]  # slope 0.5, steep for the default critical slope 0.1, centre elevation 10
FLAT = [10, 10, 10]


def spec_with_columns(spec, table_name, change):
    table = spec.tables[table_name]
    new_table = dataclasses.replace(table, columns=change(table.columns))
    return dataclasses.replace(spec, tables={**spec.tables, table_name: new_table})


def columns_without_ids(spec, table_name, ids=("site_id", "reference_id")):
    return [name for name in spec.table(table_name).column_names if name not in ids]


def test_rix_table_returns_one_row_per_site_with_the_quantities_of_its_rays(product_spec, make_radial):
    results = {"s1": make_radial(STEEP, FLAT, FLAT), "s2": make_radial(FLAT, FLAT)}
    table = tables.rix_table(product_spec, results)
    assert (table.index.name, list(table.index)) == ("site_id", ["s1", "s2"])
    assert list(table.columns) == columns_without_ids(product_spec, "rix_table")
    expected = {
        "elevation": 10.0,
        "rix": 1 / 3,
        "rix_std": np.sqrt(2) / 3,
        "rix_min": 0.0,
        "rix_max": 1.0,
        "n_rays": 3,
        "slope_critical": 0.1,
    }
    assert table.loc["s1"].to_dict() == pytest.approx(expected)
    assert table.loc["s2"].to_dict() == pytest.approx(
        {
            "elevation": 10.0,
            "rix": 0.0,
            "rix_std": 0.0,
            "rix_min": 0.0,
            "rix_max": 0.0,
            "n_rays": 2,
            "slope_critical": 0.1,
        }
    )


def test_rix_table_returns_no_column_the_definition_dropped(product_spec, make_radial):
    spec = spec_with_columns(
        product_spec, "rix_table", lambda columns: tuple(c for c in columns if c.name != "rix_std")
    )
    table = tables.rix_table(spec, {"s1": make_radial(FLAT, FLAT)})
    assert "rix_std" not in table.columns and "rix_max" in table.columns


def test_rix_table_raises_given_a_column_the_builder_cannot_provide(product_spec, make_radial):
    spec = spec_with_columns(
        product_spec, "rix_table", lambda columns: (*columns, dataclasses.replace(columns[-1], name="extra"))
    )
    with pytest.raises(ValueError, match="cannot provide: extra"):
        tables.rix_table(spec, {"s1": make_radial(FLAT, FLAT)})


def test_trix_table_returns_one_row_per_pair_with_sites_in_the_outer_level(product_spec):
    matrices = {
        name: np.arange(6.0).reshape(2, 3) * factor
        for name, factor in (("trix", 1), ("A", 10), ("B", 100), ("distance_km", 1000))
    }
    assessment = TrixAssessment(transferability=np.arange(6).reshape(2, 3) % 3, **matrices)
    table = tables.trix_table(product_spec, assessment, ["s1", "s2"], ["r1", "r2", "r3"])

    assert table.index.names == ["site_id", "reference_id"]
    assert list(table.columns) == columns_without_ids(product_spec, "trix_table")
    for i, site in enumerate(["s1", "s2"]):
        for j, reference in enumerate(["r1", "r2", "r3"]):
            row = table.loc[(site, reference)]
            assert (row["trix"], row["A"], row["B"], row["distance"], row["transferability"]) == (
                matrices["trix"][i, j],
                matrices["A"][i, j],
                matrices["B"][i, j],
                matrices["distance_km"][i, j],
                assessment.transferability[i, j],
            )


def test_transferability_counts_returns_the_number_of_pairs_per_state_that_occurs(product_spec):
    table = pd.DataFrame({"transferability": [2, 2, 1, 2, 0, 2]})
    assert list(tables.transferability_counts(product_spec, table).items()) == [(0, 1), (1, 1), (2, 4)]  # by state
    assert tables.transferability_counts(product_spec, table.iloc[:2]) == {2: 2}  # states without a pair are absent


@pytest.mark.parametrize("name", [tables.LOCATIONS_SITE_TABLE, tables.LOCATIONS_REFERENCE_TABLE])
def test_locations_table_returns_the_identifier_and_the_geometry_only(product_spec, name):
    locations = gpd.GeoDataFrame(
        {"height": [1, 2]}, geometry=gpd.points_from_xy([0, 1], [0, 1]), index=["a", "b"], crs="EPSG:2056"
    )
    table = tables.locations_table(product_spec, name, locations)
    assert isinstance(table, gpd.GeoDataFrame)
    assert (table.index.name, list(table.index), list(table.columns), table.crs) == (
        "site_id",
        ["a", "b"],
        ["geometry"],
        locations.crs,
    )
    assert table.geometry.tolist() == locations.geometry.tolist()


def test_steep_segments_returns_one_row_per_steep_segment(product_spec, make_radial):
    results = {"s1": make_radial([0, 50, 50, 100, 150]), "s2": make_radial([0, 0, 0])}  # two steep runs, none
    result = tables.steep_segments(product_spec, results)
    assert (list(result.index), result["segment_id"].tolist(), result["theta"].tolist()) == (
        ["s1", "s1"],
        [0, 1],
        [0.0, 0.0],
    )
    assert [list(segment.coords) for segment in result.geometry] == [[(0, 0), (0, 100)], [(0, 200), (0, 300), (0, 400)]]
    assert isinstance(result.geometry.iloc[0], shapely.geometry.LineString)


def test_steep_segments_returns_the_columns_of_the_definition_given_no_steep_segment(product_spec, make_radial):
    result = tables.steep_segments(product_spec, {"s1": make_radial(FLAT)})
    assert (len(result), result.index.name, list(result.columns)) == (
        0,
        "site_id",
        columns_without_ids(product_spec, "ruggedness"),
    )
    assert isinstance(result, gpd.GeoDataFrame) and result.geometry.name == "geometry"


def test_steep_segments_returns_the_direction_of_the_ray(product_spec, make_ray):
    result = tables.steep_segments(product_spec, {"s1": RadialRuggedness(rays=(make_ray(STEEP, theta=45.0),))})
    assert (result["theta"].tolist(), result["segment_id"].tolist()) == ([45.0], [0])


@pytest.mark.parametrize("crs", [None, "EPSG:32633"])
def test_steep_segments_returns_the_crs_of_the_rays(product_spec, make_ray, crs):
    result = tables.steep_segments(product_spec, {"s1": RadialRuggedness(rays=(make_ray(STEEP, crs=crs),))})
    assert (result.crs is None) if crs is None else (result.crs.to_string() == crs)


def test_rose_returns_the_ruggedness_per_site_and_direction(make_radial):
    rose = tables.rose({"s1": make_radial(STEEP, FLAT, FLAT), "s2": make_radial(FLAT, FLAT, STEEP)})
    assert list(rose.index) == ["s1", "s2"] and rose.columns.tolist() == [0.0, 10.0, 20.0]
    assert rose.to_numpy().tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
