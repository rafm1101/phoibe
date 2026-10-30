import dataclasses

import numpy as np
import pytest
import xarray

from phoibe.geography.complexity.rix.computations import Field, FieldInfo, checks
from phoibe.geography.complexity.rix.pipelines import TRIXAnalyzer, errors, ports, product
from phoibe.geography.complexity.rix.pipelines import analyzer as analyzer_module
from phoibe.geography.complexity.rix.pipelines.checks import LOCKED_PARAMETER_CHANGED

# from phoibe.geography.complexity.rix.pipelines.ports import WrittenArtifacts
# from phoibe.geography.complexity.rix.pipelines.product import SpecificationError

REFERENCES = {"ids": ("r1", "r2", "r3"), "xs": (500, -500, 0), "ys": (500, -500, 500)}


def test_run_returns_the_tables_of_the_sites_given_no_references(make_analyzer, make_field, make_locations):
    result = make_analyzer().run(make_field(), make_locations())
    assert set(result.tables) == {"rix_table", "locations_site", "ruggedness"}
    assert len(result.summary) == 3 and result.locations_reference is None and result.trix_table is None
    assert (result.record.transferability_counts, result.record.computed) == (None, ("rix_site",))
    assert result.ruggedness_roses.shape == (3, 4)  # a row per site, a column per ray


def test_run_returns_the_tables_of_the_pairs_given_references(make_analyzer, make_field, make_locations):
    result = make_analyzer().run(
        make_field(), make_locations(ids=("a", "b"), xs=(0, 100), ys=(0, 0)), make_locations(**REFERENCES)
    )
    assert set(result.tables) == {"rix_table", "locations_site", "ruggedness", "locations_reference", "trix_table"}
    assert len(result.trix_table) == 2 * 3 and sum(result.record.transferability_counts.values()) == 2 * 3
    assert (result.record.computed, result.record.present_inputs) == (
        ("rix_site", "rix_reference", "trix"),
        frozenset({"dem", "locations_site", "locations_reference"}),
    )


def test_run_returns_the_rix_and_the_elevation_of_a_plane(make_analyzer, make_field, make_locations):
    result = make_analyzer().run(
        make_field(a=1.0), make_locations()
    )  # z = x: the rays along x are steep, the others flat
    assert result.summary["rix"].tolist() == pytest.approx([0.5] * 3)
    assert result.summary["rix_std"].tolist() == pytest.approx([0.5] * 3)
    assert (result.summary["rix_min"].tolist(), result.summary["rix_max"].tolist()) == ([0.0] * 3, [1.0] * 3)
    assert result.summary["elevation"].tolist() == pytest.approx([0.0, 100.0, -100.0])


def test_run_returns_the_trix_of_a_pair_as_computed_by_hand(make_analyzer, make_field, make_locations):
    """RIX 0.5 at both:
    TRIX = 0.9 * 50 * (0.5 + 0.5) + 0.1 * |0 - 500| = 95.
    A = max(8.5 - 8.265, 1.5),
    B = max(1.7, 3).
    """
    result = make_analyzer().run(
        make_field(), make_locations(("a",), (0,), (0,)), make_locations(("r1",), (500,), (500,))
    )
    row = result.trix_table.loc[("a", "r1")]
    assert result.trix_table.index.names == ["site_id", "reference_id"]
    assert (row["trix"], row["A"], row["B"], row["transferability"]) == pytest.approx((95.0, 1.5, 3.0, 2))
    assert row["distance"] == pytest.approx(707.1067811865476 / 1000)


def test_run_returns_the_steep_segments_only_on_request(make_analyzer, make_field, make_locations):
    analyzer = make_analyzer()
    with_segments = analyzer.run(make_field(), make_locations())
    without_segments = analyzer.run(make_field(), make_locations(), include_steep_segments=False)
    assert len(with_segments.steep_segments) == 3 * 2  # per site the two steep rays, each one segment
    assert without_segments.steep_segments is None and "ruggedness" not in without_segments.tables


def test_run_returns_the_record_of_the_run(make_analyzer, make_field, make_locations):
    result = make_analyzer().run(make_field(), make_locations(), dem_metadata={"name": "demo"}, project_name="p")
    record = result.record
    assert (record.project_name, dict(record.dem_provenance), record.parameters["n_angles"]) == (
        "p",
        {"name": "demo"},
        4,
    )
    assert (record.created_at.isoformat(), record.field_info) == (
        "2026-10-24T12:00:00+00:00",
        FieldInfo(None, None, None),
    )


def test_run_returns_the_identifier_as_the_index_of_all_tables(make_analyzer, make_field, make_locations):
    result = make_analyzer().run(make_field(), make_locations(ids=("a", "b", "d")))
    assert [(table.index.name, list(table.index)) for table in (result.summary, result.locations_site)] == [
        ("site_id", ["a", "b", "d"])
    ] * 2


def test_run_returns_the_existing_index_as_site_id_given_no_identifier_column(
    make_analyzer, make_field, make_locations
):
    """Changed since 1.x: the identifier is always called as in the specification, here `site_id`, not `wea_id`."""
    result = make_analyzer().run(make_field(), make_locations(ids=None, index_name="wea_id"))
    assert (result.locations_site.index.name, list(result.locations_site.index)) == ("site_id", [0, 1, 2])
    assert result.summary.index.name == "site_id"


def test_run_returns_no_finding_given_the_locked_parameters_of_the_specification(
    make_analyzer, make_field, make_locations
):
    result = make_analyzer(n_angles=72, R_km=3.5, slope_critical=0.033).run(
        make_field(a=0.01, half=4000), make_locations()
    )
    assert (result.record.findings, dict(result.record.finding_counts)) == ((), {})
    assert result.summary["rix"].tolist() == [0.0] * 3


def test_run_returns_a_finding_per_locked_parameter_that_was_changed(make_analyzer, make_field, make_locations):
    record = make_analyzer().run(make_field(), make_locations()).record  # 4 rays of 0.5 km, slope 0.1
    assert dict(record.finding_counts) == {LOCKED_PARAMETER_CHANGED: 3}
    assert sorted(item.finding.detail.split(" ")[0] for item in record.findings) == [
        "ray.R_km",
        "ray.n_angles",
        "slope.slope_critical",
    ]


def test_run_returns_the_crs_finding_given_another_configured_crs(make_analyzer, make_field, make_locations):
    sites = make_locations(crs="EPSG:32633")
    assert (
        dict(make_analyzer(crs="EPSG:2056").run(make_field(), sites).record.finding_counts)[checks.CRS_VS_CONFIG] == 1
    )
    assert checks.CRS_VS_CONFIG not in make_analyzer(crs="EPSG:32633").run(make_field(), sites).record.finding_counts


def test_run_returns_the_rix_by_the_critical_slope_of_the_parameters(make_analyzer, make_field, make_locations):
    field = make_field(a=0.05)  # between the critical slopes 0.033 and 0.1
    assert make_analyzer(slope_critical=0.1).run(field, make_locations()).summary["rix"].tolist() == [0.0] * 3
    assert make_analyzer(slope_critical=0.03).run(field, make_locations()).summary["rix"].tolist() == pytest.approx(
        [0.5] * 3
    )


@pytest.mark.parametrize("method, expected", [("linear", 10.0), ("nearest", 0.0)])
def test_run_returns_the_elevation_by_the_interpolation_method_of_the_parameters(
    make_analyzer, make_field, make_locations, method, expected
):
    off_the_grid = make_locations(("a",), (10,), (0,))  # 10 m next to the grid node at 0 m
    assert make_analyzer(method=method).run(make_field(), off_the_grid).summary["elevation"].tolist() == [expected]


def test_run_returns_the_crs_of_the_locations_as_the_crs_of_the_rays(make_analyzer, make_field, make_locations):
    record = make_analyzer().run(make_field(), make_locations(crs="EPSG:32633")).record
    assert record.collections["locations_site"].ray_crs == ("EPSG:32633",)


def test_run_returns_the_rounding_of_the_limits_given_by_the_specification(make_analyzer, make_locations):
    """ADR 0004 end to end: TRIX 0.5 gives A = 8.4565km, rounded 8.5km.
    The pair at 8.48km is unconditional only with it."""
    x = np.arange(-9000, 9001, 50, dtype=float)
    field = Field.from_array(
        xarray.DataArray(x[None, :] / 1696.0 + 0 * x[:, None], coords={"x": x, "y": x}, dims=("y", "x"))
    )
    result = make_analyzer().run(
        field, make_locations(("a",), (0,), (0,)), make_locations(("r1",), (8480,), (0,))
    )  # 5 m higher
    row = result.trix_table.loc[("a", "r1")]
    assert (row["trix"], row["A"], row["transferability"]) == pytest.approx((0.5, 8.5, 2))


@pytest.mark.parametrize("nan_policy, expected_rix", [("mask", 0.3), ("truncate", 0.5)])
def test_run_returns_the_rix_of_a_site_at_the_edge_by_the_nan_policy_of_the_specification(
    make_analyzer, product_spec, make_field, make_locations, nan_policy, expected_rix
):
    """The eastwards ray leaves DEM after 100m: 0.2 of it is steep with `mask`, all of the cut ray with `truncate`."""
    parameters = tuple(
        dataclasses.replace(p, value=nan_policy) if p.name == "nan_policy" else p for p in product_spec.parameters
    )
    spec = dataclasses.replace(product_spec, parameters=parameters)
    edge = make_locations(("edge",), (1900,), (0,))
    assert make_analyzer(spec).run(make_field(), edge).summary["rix"].tolist() == pytest.approx([expected_rix])


def test_run_returns_the_findings_of_the_references_with_their_collection(
    make_analyzer, make_spec, make_field, make_locations
):
    spec = make_spec(check_parameters={checks.NAN_COUNT: {"max_fraction": 0.0}})
    record = make_analyzer(spec).run(make_field(), make_locations(), make_locations(("edge",), (1900,), (0,))).record
    nan_findings = [item for item in record.findings if item.finding.check == checks.NAN_COUNT]
    assert {(item.collection, item.location_id) for item in nan_findings} == {("locations_reference", "edge")}


def test_run_returns_the_references_given_a_specification_that_requires_them(
    make_analyzer, product_spec, make_field, make_locations
):
    required = dataclasses.replace(product_spec.inputs["locations_reference"], required=True)
    spec = dataclasses.replace(product_spec, inputs={**product_spec.inputs, "locations_reference": required})
    result = make_analyzer(spec).run(make_field(), make_locations(), make_locations(**REFERENCES))
    assert result.trix_table is not None


@pytest.mark.parametrize("n_angles", [0, 361])
def test_run_raises_given_n_angles_outside_the_range_of_the_check(
    make_analyzer, make_spec, make_field, make_locations, n_angles
):
    analyzer = make_analyzer(make_spec(severities={checks.N_ANGLE: "abort"}), n_angles=n_angles)
    with pytest.raises(errors.CheckAborted, match=checks.N_ANGLE):
        analyzer.run(make_field(), make_locations())


def test_run_raises_before_sampling_given_duplicate_identifiers_and_the_severity_abort(
    make_analyzer, make_spec, make_field, make_locations, monkeypatch
):
    def no_sampler(*args, **kwargs):
        raise AssertionError("the sampler must not be created after an abort of the inputs")

    monkeypatch.setattr(analyzer_module, "RegularGridXYSampler", no_sampler)
    analyzer = make_analyzer(make_spec(severities={checks.UNIQUE_IDS: "abort"}))
    with pytest.raises(errors.CheckAborted, match=checks.UNIQUE_IDS):
        analyzer.run(make_field(), make_locations(ids=("a", "a", "a")))


@pytest.mark.parametrize("severity, aborts", [("abort", True), ("warn", False)])
def test_run_returns_or_raises_given_another_crs_of_sites_and_references_by_the_severity(
    make_analyzer, make_spec, make_field, make_locations, severity, aborts
):
    analyzer = make_analyzer(make_spec(severities={checks.CRS_SITE_VS_REFERENCE: severity}))
    sites, references = make_locations(crs="EPSG:32633"), make_locations(
        ids=("r1",), xs=(500,), ys=(500,), crs="EPSG:4326"
    )
    if aborts:
        with pytest.raises(errors.CheckAborted, match=checks.CRS_SITE_VS_REFERENCE):
            analyzer.run(make_field(), sites, references)
    else:
        assert (
            dict(analyzer.run(make_field(), sites, references).record.finding_counts)[checks.CRS_SITE_VS_REFERENCE] == 1
        )


@pytest.mark.parametrize("severity, aborts", [("abort", True), ("warn", False)])
def test_run_returns_or_raises_given_rays_leaving_the_field_by_the_severity(
    make_analyzer, make_spec, make_field, make_locations, severity, aborts
):
    spec = make_spec(
        severities={checks.NAN_COUNT: severity}, check_parameters={checks.NAN_COUNT: {"max_fraction": 0.0}}
    )
    edge = make_locations(("edge",), (1900,), (0,))  # the rays to the east leave the DEM after 100 m
    if aborts:
        with pytest.raises(errors.CheckAborted, match=checks.NAN_COUNT):
            make_analyzer(spec).run(make_field(), edge)
    else:
        record = make_analyzer(spec).run(make_field(), edge).record
        assert (
            record.finding_counts[checks.NAN_COUNT],
            record.collections["locations_site"].n_locations_with_nans,
        ) == (1, 1)
        assert record.collections["locations_site"].nan_count > 0


def test_run_raises_given_an_input_the_specification_requires_and_the_analyzer_was_not_given(
    make_analyzer, product_spec, make_field, make_locations
):
    required = dataclasses.replace(product_spec.inputs["locations_reference"], required=True)
    spec = dataclasses.replace(product_spec, inputs={**product_spec.inputs, "locations_reference": required})
    with pytest.raises(errors.InputRequirementError, match="missing: locations_reference"):
        make_analyzer(spec).run(make_field(), make_locations())


def test_analyzer_raises_given_a_specification_without_a_check_the_pipeline_raises(product_spec):
    spec_checks = {name: check for name, check in product_spec.checks.items() if name != checks.NAN_COUNT}
    with pytest.raises(product.SpecificationError, match=checks.NAN_COUNT):
        TRIXAnalyzer(dataclasses.replace(product_spec, checks=spec_checks), None, None)


def run_and_publish(analyzer, field, locations, references=None, profile="summary", **run_kwargs):
    result = analyzer.run(field, locations, references, **run_kwargs)
    return result, analyzer.publish(result, "target", profile)


def test_publish_raises_given_no_writer(make_analyzer, make_field, make_locations):
    analyzer = make_analyzer()
    with pytest.raises(RuntimeError, match="without a writer"):
        analyzer.publish(analyzer.run(make_field(), make_locations()), "target")


def test_publish_returns_what_was_written_and_writes_the_manifest_last(
    make_analyzer, make_writer, make_field, make_locations, product_spec
):
    writer = make_writer()
    _, written = run_and_publish(make_analyzer(writer=writer), make_field(), make_locations())
    assert isinstance(written, ports.WrittenArtifacts) and set(written.files) == {"rix_table"}
    assert writer.calls == [("artifacts", "summary"), ("manifest",)]
    assert writer.manifest["meta"]["version"] == product_spec.version
    assert writer.manifest["artifacts"] == {
        "profile": "summary",
        "files": {"rix_table": "rix_table.csv", "manifest": "summary.yaml"},
        "table_records": {"rix_table": 3},
        "layer_records": {},
    }
    assert writer.manifest["run"]["n_sites"] == 3


def test_publish_returns_the_trix_table_given_references_and_all_layers_given_the_full_profile(
    make_analyzer, make_writer, make_field, make_locations
):
    analyzer = make_analyzer(writer=make_writer())
    result = analyzer.run(make_field(), make_locations(), make_locations(**REFERENCES))
    assert set(analyzer.publish(result, "target", "summary").files) == {"rix_table", "trix_table"}
    full = analyzer.publish(result, "target", "full")
    assert set(full.files) == {"rix_table", "trix_table", "geopackage"}
    assert set(full.layer_columns) == {"locations_site", "locations_reference", "ruggedness", "trix_table"}


def test_publish_raises_before_writing_given_a_result_without_a_table_of_the_profile(
    make_analyzer, make_writer, make_field, make_locations
):
    writer = make_writer()
    analyzer = make_analyzer(writer=writer)
    result = analyzer.run(make_field(), make_locations(), make_locations(**REFERENCES), include_steep_segments=False)
    with pytest.raises(errors.ContractViolation, match="no table 'ruggedness'"):
        analyzer.publish(result, "target", "full")
    assert writer.calls == []


def test_publish_raises_without_a_manifest_given_a_writer_that_reports_a_deviation(
    make_analyzer, make_writer, make_field, make_locations
):
    def drop_the_table(written):
        return dataclasses.replace(written, files={})

    writer = make_writer(tamper=drop_the_table)
    with pytest.raises(errors.ContractViolation, match="artifact 'rix_table' was not written"):
        run_and_publish(make_analyzer(writer=writer), make_field(), make_locations())
    assert writer.calls == [("artifacts", "summary")] and writer.manifest is None


def test_publish_raises_without_a_manifest_given_a_writer_that_fails(
    make_analyzer, make_writer, make_field, make_locations
):
    writer = make_writer(fail=errors.PortError("disk full"))
    with pytest.raises(errors.PortError, match="disk full"):
        run_and_publish(make_analyzer(writer=writer), make_field(), make_locations())
    assert writer.calls == [("artifacts", "summary")] and writer.manifest is None


def test_publish_raises_given_an_unknown_profile(make_analyzer, make_writer, make_field, make_locations):
    with pytest.raises(KeyError, match="Unknown profile 'nope'"):
        run_and_publish(make_analyzer(writer=make_writer()), make_field(), make_locations(), profile="nope")
