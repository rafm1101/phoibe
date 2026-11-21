import pathlib

import pytest

from phoibe.dokimasia import adapters, contract

TEXT = """\
id: wind-scada-10min
version: 1.0.0
domain: wind
layers:
  bronze:
    measures:
      wind_speed_limits:
        rule: ranges
        signal: wind_speed
        params: {min: 0, max: 60}
      power_limits:
        rule: ranges
        signal: power
        params:
          min: 0
          max: {subject: rated_power_kw, factor: 1.05}
      time_gaps:
    checks:
      no_wind_speed_outliers:
        measure: wind_speed_limits
        value: n_out_of_range
        severity: critical
        good: {max: 0}
        acceptable: {max: 10}
        clause: "TR6 section 3"
      availability:
        measure: time_gaps
        value: share_missing_steps
        severity: warning
        good: {max: 0.05}
"""


def test_parse_contract_yaml_reads_the_contract():
    parsed = adapters.parse_contract_yaml(TEXT)

    layer = parsed.get_layer("bronze")
    assert [measure.id for measure in layer.measures] == ["wind_speed_limits", "power_limits", "time_gaps"]
    assert layer.measures[1].params["max"] == contract.SubjectRef("rated_power_kw", 1.05)
    assert layer.checks[0].clause == "TR6 section 3"


def test_parse_contract_yaml_given_a_duplicate_key_raises_instead_of_keeping_the_last():
    text = TEXT.replace("      time_gaps:\n", "      time_gaps:\n      wind_speed_limits:\n")

    with pytest.raises(contract.ContractError, match="duplicate key 'wind_speed_limits'"):
        adapters.parse_contract_yaml(text)


@pytest.mark.parametrize("text", ["id: [unclosed", "just text", "- a\n- b\n"])
def test_parse_contract_yaml_given_invalid_input_raises_a_contract_error(text):
    with pytest.raises(contract.ContractError):
        adapters.parse_contract_yaml(text)


def test_parse_contract_yaml_does_not_construct_arbitrary_objects():
    with pytest.raises(contract.ContractError, match="yaml"):
        adapters.parse_contract_yaml("id: !!python/object/apply:os.getcwd []\n")


def test_dump_contract_yaml_round_trips_losslessly():
    parsed = adapters.parse_contract_yaml(TEXT)

    assert adapters.parse_contract_yaml(adapters.dump_contract_yaml(parsed)) == parsed


def test_contract_to_data_leaves_out_defaults():
    data = adapters.contract_to_data(adapters.parse_contract_yaml(TEXT))

    layer = data["layers"]["bronze"]
    assert layer["measures"]["time_gaps"] is None
    assert layer["measures"]["power_limits"]["params"]["max"] == {"subject": "rated_power_kw", "factor": 1.05}
    assert list(layer["checks"]["availability"]) == ["measure", "value", "severity", "good"]


def test_save_contract_and_load_contract_round_trip_through_a_file(tmp_path: pathlib.Path):
    parsed = adapters.parse_contract_yaml(TEXT)

    adapters.save_contract(parsed, tmp_path / "contract.yaml")

    assert adapters.load_contract(tmp_path / "contract.yaml") == parsed


def test_load_contract_given_a_missing_file_raises_os_error(tmp_path: pathlib.Path):
    with pytest.raises(OSError):
        adapters.load_contract(tmp_path / "absent.yaml")
