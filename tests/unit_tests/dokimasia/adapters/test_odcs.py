import json
import pathlib

import pytest

from phoibe.dokimasia import adapters, wind

CONTRACT = adapters.parse_contract_yaml(
    """\
id: wind-scada-10min
version: 1.0.0
domain: wind
layers:
  raw:
    measures:
      required_signals:
        params: {signals: [power, wind_speed]}
      time_gaps:
        params: {expected_step_s: 600}
    checks:
      signals_present:
        measure: required_signals
        value: n_missing_signals
        severity: critical
        good: {max: 0}
        clause: "TR6 section 3.1"
      availability:
        measure: time_gaps
        value: share_missing_steps
        severity: warning
        good: {max: 0.001}
        acceptable: {max: 0.10}
  bronze:
    measures:
      wind_speed_limits:
        rule: ranges
        signal: wind_speed
        params: {min: 0, max: 60}
      power_limits:
        rule: ranges
        signal: power_kw
        params: {min: 0, max: {subject: rated_power_kw, factor: 1.05}}
    checks:
      no_wind_speed_outliers:
        measure: wind_speed_limits
        value: n_out_of_range
        severity: critical
        good: {max: 0}
        acceptable: {max: 10}
      power_within_rating:
        measure: power_limits
        value: n_out_of_range
        severity: info
        good: {max: 0}
      share_in_range:
        measure: wind_speed_limits
        value: share_out_of_range
        severity: info
        signal: wind_speed
        good: {min: 0, max: 0.01}
      enough_data:
        measure: wind_speed_limits
        value: n_checked
        severity: info
        good: {min: 100}
"""
)


@pytest.fixture(scope="module")
def odcs() -> dict[str, object]:
    return adapters.export_odcs(CONTRACT, catalog=wind.create_catalog())


def _layer(odcs, name):
    return next(item for item in odcs["schema"] if item["name"] == name)


def _rule(odcs, layer, name):
    return next(rule for rule in _layer(odcs, layer)["quality"] if rule["name"] == name)


def _custom(item, name):
    return next(prop["value"] for prop in item["customProperties"] if prop["property"] == f"dokimasia.{name}")


def test_the_envelope_names_the_contract_and_the_standard_version(odcs):
    assert [odcs[key] for key in ("apiVersion", "kind", "id", "name", "version", "status", "domain")] == [
        "v3.2.0", "DataContract", "wind-scada-10min", "wind-scada-10min", "1.0.0", "draft", "wind",
    ]  # fmt: skip
    assert list(odcs)[:3] == ["apiVersion", "kind", "id"]
    assert _custom(odcs, "export_version") == 1


def test_the_status_is_passed_through():
    assert adapters.export_odcs(CONTRACT, status="active")["status"] == "active"


def test_each_layer_becomes_a_schema_object(odcs):
    assert [item["name"] for item in odcs["schema"]] == ["raw", "bronze"]


def test_required_signals_become_required_properties(odcs):
    properties = {prop["name"]: prop for prop in _layer(odcs, "raw")["properties"]}

    assert list(properties) == ["power", "wind_speed"]
    assert all(prop["required"] is True and prop["logicalType"] == "number" for prop in properties.values())


def test_signals_named_by_other_measures_are_properties_but_not_required(odcs):
    properties = {prop["name"]: prop for prop in _layer(odcs, "bronze")["properties"]}

    assert list(properties) == ["wind_speed", "power_kw"]
    assert not any(prop["required"] for prop in properties.values())


def test_catalog_signals_carry_quantity_statistic_and_unit(odcs):
    properties = {prop["name"]: prop for prop in _layer(odcs, "bronze")["properties"]}

    assert _custom(properties["power_kw"], "quantity") == "power"
    assert _custom(properties["power_kw"], "statistic") == "mean"
    assert _custom(properties["power_kw"], "unit") == "kW"


def test_without_a_catalog_signals_carry_no_custom_properties(odcs):
    without_catalog = adapters.export_odcs(CONTRACT)

    assert all("customProperties" not in prop for prop in _layer(without_catalog, "bronze")["properties"])


def test_a_check_becomes_one_custom_rule_that_names_its_measure(odcs):
    rule = _rule(odcs, "raw", "availability")

    assert rule["type"] == "custom"
    assert rule["engine"] == "dokimasia"
    assert rule["implementation"] == "time_gaps"
    assert [r["name"] for r in _layer(odcs, "bronze")["quality"]] == [
        "no_wind_speed_outliers", "power_within_rating", "share_in_range", "enough_data",
    ]  # fmt: skip


def test_a_custom_rule_has_no_comparison_operator_because_the_standard_allows_none(odcs):
    for layer in odcs["schema"]:
        for rule in layer["quality"]:
            assert not any(key.startswith("mustBe") for key in rule)


@pytest.mark.parametrize(
    ("layer", "check", "text"),
    [
        ("raw", "availability", "share_missing_steps of time_gaps; good: max 0.001"),
        ("bronze", "enough_data", "n_checked of wind_speed_limits; good: min 100"),
        ("bronze", "share_in_range", "share_out_of_range of wind_speed_limits; good: min 0, max 0.01"),
    ],
)
def test_the_description_states_the_good_band_for_people(odcs, layer, check, text):
    assert _rule(odcs, layer, check)["description"] == text


@pytest.mark.parametrize(("critical", "error"), [("critical", "error"), ("warning", "warning"), ("info", "info")])
def test_severity_is_mapped_and_kept_under_its_own_name(odcs, critical, error):
    rule = next(rule for layer in odcs["schema"] for rule in layer["quality"] if _custom(rule, "severity") == critical)

    assert rule["severity"] == error


def test_bands_clause_and_signal_travel_as_custom_properties(odcs):
    availability = _rule(odcs, "raw", "availability")
    present = _rule(odcs, "raw", "signals_present")
    in_range = _rule(odcs, "bronze", "share_in_range")

    assert _custom(availability, "good") == {"max": 0.001}
    assert _custom(availability, "acceptable") == {"max": 0.10}
    assert _custom(availability, "value") == "share_missing_steps"
    assert _custom(availability, "measure") == "time_gaps"
    assert _custom(present, "clause") == "TR6 section 3.1"
    assert _custom(in_range, "signal") == "wind_speed"


def test_a_check_without_a_clause_or_acceptable_band_has_no_such_property(odcs):
    names = [prop["property"] for prop in _rule(odcs, "bronze", "enough_data")["customProperties"]]

    assert "dokimasia.clause" not in names
    assert "dokimasia.acceptable" not in names


def test_a_limit_that_depends_on_the_subject_gets_no_operator_but_keeps_its_band():
    contract = adapters.parse_contract_yaml(
        """\
id: c
version: 1.0.0
domain: wind
layers:
  bronze:
    measures:
      m:
        rule: ranges
        signal: power_kw
        params: {min: 0, max: 10}
    checks:
      within_rating:
        measure: m
        value: n_out_of_range
        severity: warning
        good: {max: {subject: rated_power_kw, factor: 0.01}}
"""
    )

    rule = _rule(adapters.export_odcs(contract), "bronze", "within_rating")

    assert rule["description"] == "n_out_of_range of m; good: max rated_power_kw * 0.01"
    assert _custom(rule, "good") == {"max": {"subject": "rated_power_kw", "factor": 0.01}}


def test_the_measures_travel_with_the_layer_so_nothing_of_the_contract_is_lost(odcs):
    measures = _custom(_layer(odcs, "bronze"), "measures")

    assert measures["power_limits"]["params"]["max"] == {"subject": "rated_power_kw", "factor": 1.05}
    assert measures["wind_speed_limits"]["rule"] == "ranges"
    assert _custom(_layer(odcs, "raw"), "measures")["required_signals"]["params"] == {
        "signals": ["power", "wind_speed"]
    }


def test_the_export_is_the_same_for_the_same_contract():
    assert adapters.export_odcs(CONTRACT) == adapters.export_odcs(CONTRACT)


@pytest.mark.parametrize("fmt", ["yaml", "json"])
def test_the_text_parses_back_to_the_exported_data(odcs, fmt):
    text = adapters.format_odcs(CONTRACT, catalog=wind.create_catalog(), fmt=fmt)

    assert adapters.parse_artifact(text, fmt) == odcs


def test_save_odcs_writes_yaml_or_json_by_suffix(tmp_path):
    adapters.save_odcs(CONTRACT, tmp_path / "c.yaml")
    adapters.save_odcs(CONTRACT, tmp_path / "c.json")

    assert (tmp_path / "c.yaml").read_text().startswith("apiVersion: v3.2.0\nkind: DataContract\n")
    assert json.loads((tmp_path / "c.json").read_text())["kind"] == "DataContract"


def test_the_contract_is_not_changed_by_exporting_it():
    before = adapters.contract_to_data(CONTRACT)

    adapters.export_odcs(CONTRACT)

    assert adapters.contract_to_data(CONTRACT) == before


class TestAgainstTheStandard:
    """These need the optional packages `jsonschema` and `open-data-contract-standard`; they skip without them."""

    def test_the_file_validates_against_the_json_schema_of_the_standard(self, odcs):
        jsonschema = pytest.importorskip("jsonschema")
        standard = pytest.importorskip("open_data_contract_standard")
        schema = json.loads((pathlib.Path(standard.__file__).parent / "schema.json").read_text(encoding="utf-8"))

        jsonschema.Draft201909Validator(schema).validate(json.loads(json.dumps(odcs)))

    def test_the_reference_model_loads_the_file_and_keeps_the_rules(self, odcs):
        model = pytest.importorskip("open_data_contract_standard.model")

        loaded = model.OpenDataContractStandard.from_string(
            adapters.format_odcs(CONTRACT, catalog=wind.create_catalog())
        )

        assert loaded.id == "wind-scada-10min"
        raw = loaded.schema_[0]
        assert [rule.name for rule in raw.quality] == ["signals_present", "availability"]
        assert raw.quality[1].implementation == "time_gaps"
        assert raw.quality[1].engine == "dokimasia"
