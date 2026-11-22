import pytest

from phoibe.dokimasia import core, rules, standardize, wind
from phoibe.synthetic_data import turbine

STATISTICS = ["", "_min", "_max"]
QUANTITIES = ["power_kw", "wind_speed", "rotor_speed", "pitch_angle"]
NOTEBOOK_COLUMNS = [
    "power", "power_min", "power_max", "wind_speed", "wind_speed_min", "wind_speed_max",
    "rotor_speed", "rotor_speed_min", "rotor_speed_max", "pitch_angle", "pitch_angle_min", "pitch_angle_max",
]  # fmt: skip


def test_the_catalog_has_the_twelve_signals_with_statistic_and_unit():
    catalog = wind.create_catalog()

    assert catalog.names == tuple(f"{q}{s}" for q in QUANTITIES for s in STATISTICS)
    assert catalog.get_signal("power_kw_max").statistic == core.Statistic.MAX
    assert catalog.get_signal("power_kw").unit == "kW"
    assert catalog.get_signal("wind_speed_min").unit == "m/s"
    assert catalog.get_signal("rotor_speed").quantity == "rotor_speed"


def test_the_catalog_has_no_time_signals():
    assert not {"date", "time", "datetime"} & set(wind.create_catalog().names)


def test_every_signal_has_patterns_and_every_pattern_has_a_signal():
    assert [entry.signal for entry in wind.PATTERNS] == list(wind.create_catalog().names)


def test_the_columns_of_the_notebook_data_are_detected_without_ambiguity():
    detection = wind.detect_wind_signals(NOTEBOOK_COLUMNS)

    assert detection.is_clear
    assert detection.missing == ()
    assert detection.mapping == {
        "power": "power_kw",
        "power_min": "power_kw_min",
        "power_max": "power_kw_max",
    }
    assert len(detection.detected) == 12


def test_power_min_is_no_candidate_for_power_which_the_old_patterns_could_not_tell():
    detection = wind.detect_wind_signals(["power", "power_min"])

    assert detection.is_clear
    assert detection.detected == {"power_kw": "power", "power_kw_min": "power_min"}


@pytest.mark.parametrize(
    ("column", "signal"),
    [
        ("Power (kW)", "power_kw"),
        ("Wind Speed [m/s]", "wind_speed"),
        ("WindSpeed_avg", "wind_speed"),
        ("production_max", "power_kw_max"),
        ("Rotor-Speed Min", "rotor_speed_min"),
        ("pitch_angle_avg", "pitch_angle"),
        ("max_wind_speed", "wind_speed_max"),
    ],
)
def test_common_column_names_are_detected(column, signal):
    assert wind.detect_wind_signals([column]).detected == {signal: column}


def test_standard_deviation_columns_are_unmatched_and_do_not_disturb_the_mean():
    detection = wind.detect_wind_signals(["wind_speed", "wind_speed_std"])

    assert detection.is_clear
    assert detection.unmatched == ("wind_speed_std",)


@pytest.mark.parametrize("columns", [["power", "reactive_power"], ["power", "production"]])
def test_two_columns_for_one_signal_are_reported_and_not_resolved_by_order(columns):
    detection = wind.detect_wind_signals(columns)

    assert not detection.is_clear
    assert detection.ambiguities[0].kind == "signal_has_several_columns"
    assert detection.ambiguities[0].subject == "power_kw"
    assert "power_kw" not in detection.detected
    assert detection.mapping == {}


def test_a_column_for_two_quantities_is_reported():
    detection = wind.detect_wind_signals(["power_vs_wind_speed"])

    assert {item.subject for item in detection.ambiguities} >= {"power_vs_wind_speed"}
    assert detection.mapping == {}


def test_time_columns_are_not_signals():
    detection = wind.detect_wind_signals(["datetime", "timestamp", "date", "time", "power"])

    assert detection.unmatched == ("datetime", "timestamp", "date", "time")
    assert detection.is_clear


def test_the_mapping_is_what_rename_to_catalog_takes():
    frame = turbine.generate_wtg_scada(time=turbine.Time("2026-02-14T04:00:00", "10min", 30), seed=1)
    dataset = core.Dataset.from_frame(frame)
    detection = wind.detect_wind_signals(dataset.signal_names)
    detection.raise_if_ambiguous()

    outcome = standardize.RenameToCatalog(detection.mapping).apply(dataset)

    assert isinstance(outcome, core.Transformed)
    assert set(outcome.frame.columns) == {"power_kw", "wind_speed", "rotor_speed", "pitch_angle"}
    assert set(outcome.frame.columns) <= set(wind.create_catalog().names)


def test_rules_and_wind_are_importable_together():
    assert rules.detect_signals is not None
