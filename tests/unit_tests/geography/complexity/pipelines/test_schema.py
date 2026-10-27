import pytest

from phoibe.geography.complexity.rix.pipelines.product.parameters import RunParameters
from phoibe.geography.complexity.rix.pipelines.product.schema import PRODUCT_SPECIFICATION as SPEC


@pytest.mark.parametrize(
    "group, name, value",
    [("ray", "n_angles", 72), ("ray", "R_km", 3.5), ("slope", "slope_critical", 0.033), ("trix", "decimals", 1)],
)
def test_parameter_returns_the_locked_tr6_value(group, name, value):
    parameter = SPEC.parameter(group, name)
    assert (parameter.value, parameter.locked, parameter.source) == (value, True, "TR6")


@pytest.mark.xfail(strict=True, reason="the definition says 'truncate', ADR 0005 and the changelog say 'mask'")
def test_parameter_returns_mask_as_the_fixed_nan_policy():
    parameter = SPEC.parameter("sampler", "nan_policy")
    assert (parameter.value, parameter.locked) == ("mask", True)


def test_run_parameters_accepts_the_defaults_of_the_definition():
    names = (("ray", "n_angles"), ("ray", "R_km"), ("ray", "dr_km"), ("slope", "slope_critical"), ("ray", "crs"))
    RunParameters(**{name: SPEC.parameter(group, name).value for group, name in names})


def test_parameters_returns_no_locked_parameter_with_a_range():
    assert [p.name for p in SPEC.parameters if p.locked and p.valid_range is not None] == []


def test_tables_return_no_column_with_both_range_and_values():
    violations = [
        f"{table.name}.{column.name}"
        for table in SPEC.tables.values()
        for column in table.columns
        if column.valid_range is not None and column.values is not None
    ]
    assert violations == []


def test_column_returns_the_coding_of_the_transferability():
    column = SPEC.table("trix_table").column("transferability")
    assert (column.dtype, sorted(column.values)) == ("int", [0, 1, 2])  # as returned by evaluate_transferability_limits


def test_manifest_returns_the_sections_in_the_order_of_adr_0008():
    assert [key.name for key in SPEC.manifest.keys] == [
        "contract",
        "meta",
        "parameters",
        "spatial_context",
        "run",
        "artifacts",
    ]
