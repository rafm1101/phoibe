import pytest

from phoibe.geography.complexity.rix.pipelines.results.result_summary import ResultSummary


def summary(**tables):
    return ResultSummary(tables=tables, ruggedness_roses="rose", record="record")


@pytest.mark.parametrize(
    "attribute, table",
    [
        ("summary", "rix_table"),
        ("locations_site", "locations_site"),
        ("locations_reference", "locations_reference"),
        ("trix_table", "trix_table"),
        ("steep_segments", "ruggedness"),
    ],
)
def test_attributes_return_the_table_of_their_name_in_the_specification(attribute, table):
    assert getattr(summary(**{table: "the table"}), attribute) == "the table"


@pytest.mark.parametrize("attribute", ["locations_reference", "trix_table", "steep_segments"])
def test_optional_attributes_return_none_given_no_table(attribute):
    assert getattr(summary(rix_table="t", locations_site="s"), attribute) is None


def test_summary_raises_given_no_rix_table():
    with pytest.raises(KeyError):
        _ = summary(locations_site="s").summary


def test_tables_return_a_read_only_copy():
    tables = {"rix_table": "t"}
    result = ResultSummary(tables=tables, ruggedness_roses="rose", record="record")
    tables["other"] = "x"
    assert "other" not in result.tables
    with pytest.raises(TypeError):
        result.tables["rix_table"] = "changed"
