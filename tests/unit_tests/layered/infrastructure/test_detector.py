import re

import pandas as pd
import pytest

from phoibe.layered.infrastructure.detector import RegexVariableDetector


class TestRegexVariableDetectorEdgeCases:

    @pytest.mark.parametrize(
        "df, expected_timestamp, expected_power",
        [
            (pd.DataFrame(), None, None),
            (pd.DataFrame(columns=["timestamp", "power", "speed"]), "timestamp", "power"),
            (pd.DataFrame({"timestamp": [1, 2, 3]}), "timestamp", None),
        ],
    )
    def test_detects_in_simple_dataframe(self, df, expected_timestamp, expected_power):
        detector = RegexVariableDetector({"timestamp": [r"time"], "power": [r"power"]})
        result = detector.detect(df)
        assert result["timestamp"] == expected_timestamp
        assert result["power"] == expected_power

    def test_empty_pattern_list_returns_none(self):
        df = pd.DataFrame({"timestamp": [1, 2]})
        detector = RegexVariableDetector({"timestamp": []})
        result = detector.detect(df)
        assert result["timestamp"] is None

    def test_invalid_regex_pattern_raises(self):
        with pytest.raises(re.error):
            RegexVariableDetector({"signal": [r"[invalid(regex"]})

    def test_very_complex_regex_pattern(self):
        df = pd.DataFrame({"ws_nacelle_10min_avg": [1, 2], "power_output_kW": [3, 4]})
        detector = RegexVariableDetector({"wind_speed": [r"ws_\w+_\d+min_avg"], "power": [r"power_\w+_k[W|w]"]})
        result = detector.detect(df)
        assert result["wind_speed"] == "ws_nacelle_10min_avg"
        assert result["power"] == "power_output_kW"

    def test_pattern_with_special_characters(self):
        df = pd.DataFrame({"temperature[°C]": [1, 2], "power(kW)": [3, 4], "wind.speed": [5, 6]})
        detector = RegexVariableDetector(
            {"temp": [r"temperature\[°C\]"], "power": [r"power\(kW\)"], "wind": [r"wind\.speed"]}
        )
        result = detector.detect(df)
        assert result["temp"] == "temperature[°C]"
        assert result["power"] == "power(kW)"
        assert result["wind"] == "wind.speed"

    def test_pattern_with_unicode_characters(self):
        df = pd.DataFrame({"Außentemperatur": [1, 2], "Geschwindigkeit": [3, 4]})
        detector = RegexVariableDetector({"temp": [r"außentemperatur"], "speed": [r"geschwindigkeit"]})
        result = detector.detect(df)
        assert result["temp"] == "Außentemperatur"
        assert result["speed"] == "Geschwindigkeit"

    def test_case_insensitive_matching(self):
        df = pd.DataFrame({"TIMESTAMP": [1, 2], "Power": [3, 4], "wind_SPEED": [5, 6]})
        detector = RegexVariableDetector({"timestamp": [r"timestamp"], "power": [r"power"], "wind": [r"wind_speed"]})
        result = detector.detect(df)

        assert result["timestamp"] == "TIMESTAMP"
        assert result["power"] == "Power"
        assert result["wind"] == "wind_SPEED"

    def test_case_insensitive_with_mixed_case_pattern(self):
        df = pd.DataFrame({"timestamp": [1, 2]})
        detector = RegexVariableDetector({"signal": [r"TiMeStAmP"]})
        result = detector.detect(df)
        assert result["signal"] == "timestamp"

    def test_first_matching_pattern_wins(self):
        df = pd.DataFrame({"ws_nacelle": [1, 2], "wind_speed_gondel": [3, 4]})
        detector = RegexVariableDetector({"wind_speed": [r"ws_nacelle", r"wind.*speed", r"gondel"]})
        result = detector.detect(df)
        assert result["wind_speed"] == "ws_nacelle"

    def test_fallback_to_second_pattern(self):
        df = pd.DataFrame({"wind_speed": [1, 2]})
        detector = RegexVariableDetector({"wind_speed": [r"ws_nacelle", r"wind.*speed"]})
        result = detector.detect(df)
        assert result["wind_speed"] == "wind_speed"

    def test_all_patterns_fail_returns_none(self):
        df = pd.DataFrame({"temperature": [1, 2]})
        detector = RegexVariableDetector({"wind_speed": [r"ws", r"wind", r"gondel"]})
        result = detector.detect(df)
        assert result["wind_speed"] is None

    def test_column_with_spaces(self):
        df = pd.DataFrame({"Wind Speed": [1, 2], "Active Power": [3, 4]})
        detector = RegexVariableDetector({"wind": [r"wind\s+speed"], "power": [r"active\s+power"]})
        result = detector.detect(df)
        assert result["wind"] == "Wind Speed"
        assert result["power"] == "Active Power"

    def test_column_with_numbers(self):
        df = pd.DataFrame({"sensor_1": [1, 2], "sensor_2": [3, 4], "sensor_10": [5, 6]})
        detector = RegexVariableDetector({"sensor1": [r"sensor_1$"], "sensor10": [r"sensor_10"]})
        result = detector.detect(df)
        assert result["sensor1"] == "sensor_1"
        assert result["sensor10"] == "sensor_10"

    def test_column_with_underscores(self):
        df = pd.DataFrame({"wind_speed_nacelle_10min": [1, 2]})
        detector = RegexVariableDetector({"wind": [r"wind_speed_\w+_\d+min"]})
        result = detector.detect(df)
        assert result["wind"] == "wind_speed_nacelle_10min"

    def test_very_long_column_name(self):
        long_name = "wind_speed_nacelle_anemometer_10minute_average_corrected_calibrated"
        df = pd.DataFrame({long_name: [1, 2]})
        detector = RegexVariableDetector({"wind": [r"wind.*speed.*nacelle"]})
        result = detector.detect(df)
        assert result["wind"] == long_name

    def test_column_with_only_numbers(self):
        df = pd.DataFrame({"123": [1, 2], "456": [3, 4]})
        detector = RegexVariableDetector({"signal": [r"^\d+$"]})
        result = detector.detect(df)
        assert result["signal"] in ["123", "456"]

    def test_multiple_columns_match_first_wins(self):
        df = pd.DataFrame({"wind_speed_1": [1, 2], "wind_speed_2": [3, 4], "wind_speed_3": [5, 6]})
        detector = RegexVariableDetector({"wind": [r"wind_speed"]})
        result = detector.detect(df)
        assert result["wind"] == "wind_speed_1"

    def test_pattern_matches_substring(self):
        df = pd.DataFrame({"my_wind_speed_data": [1, 2]})
        detector = RegexVariableDetector({"wind": [r"wind"]})
        result = detector.detect(df)
        assert result["wind"] == "my_wind_speed_data"

    def test_exact_match_vs_partial_match(self):
        df = pd.DataFrame({"power": [1, 2], "power_avg": [3, 4]})
        detector = RegexVariableDetector({"exact": [r"^power$"], "partial": [r"power"]})
        result = detector.detect(df)
        assert result["exact"] == "power"
        assert result["partial"] == "power"

    def test_empty_patterns_dict(self):
        df = pd.DataFrame({"timestamp": [1, 2]})
        detector = RegexVariableDetector({})
        result = detector.detect(df)
        assert result == {}

    def test_detector_with_single_signal(self):
        df = pd.DataFrame({"timestamp": [1, 2], "power": [3, 4]})
        detector = RegexVariableDetector({"timestamp": [r"time"]})
        result = detector.detect(df)
        assert len(result) == 1
        assert result["timestamp"] == "timestamp"

    def test_detect_is_idempotent(self):
        df = pd.DataFrame({"timestamp": [1, 2], "power": [3, 4]})
        detector = RegexVariableDetector({"timestamp": [r"time"], "power": [r"power"]})
        result1 = detector.detect(df)
        result2 = detector.detect(df)
        result3 = detector.detect(df)
        assert result1 == result2 == result3

    def test_detect_different_dataframes(self):
        df1 = pd.DataFrame({"timestamp": [1, 2]})
        df2 = pd.DataFrame({"time": [3, 4]})
        df3 = pd.DataFrame({"datetime": [5, 6]})
        detector = RegexVariableDetector({"timestamp": [r"time"]})
        result1 = detector.detect(df1)
        result2 = detector.detect(df2)
        result3 = detector.detect(df3)

        assert result1["timestamp"] == "timestamp"
        assert result2["timestamp"] == "time"
        assert result3["timestamp"] == "datetime"

    def test_german_scada_columns(self):
        df = pd.DataFrame(
            {
                "Zeitstempel": [1],
                "Windgeschwindigkeit_Gondel": [2],
                "Leistung_Wirkleistung": [3],
                "Windrichtung_Gondel_Absolut": [4],
                "Außentemperatur_Gondel": [5],
                "Rotordrehzahl": [6],
                "Pitchwinkel_Mittelwert": [7],
            }
        )
        detector = RegexVariableDetector(
            {
                "timestamp": [r"zeitstempel", r"timestamp"],
                "wind_speed": [r"windgeschwindigkeit", r"wind.*speed"],
                "power": [r"leistung.*wirkleistung", r"active.*power"],
                "wind_dir": [r"windrichtung", r"wind.*dir"],
                "temp": [r"außentemperatur", r"temperature"],
                "rotor_speed": [r"rotordrehzahl", r"rotor.*speed"],
                "pitch": [r"pitchwinkel", r"pitch.*angle"],
            }
        )
        result = detector.detect(df)

        assert result["timestamp"] == "Zeitstempel"
        assert result["wind_speed"] == "Windgeschwindigkeit_Gondel"
        assert result["power"] == "Leistung_Wirkleistung"
        assert result["wind_dir"] == "Windrichtung_Gondel_Absolut"
        assert result["temp"] == "Außentemperatur_Gondel"
        assert result["rotor_speed"] == "Rotordrehzahl"
        assert result["pitch"] == "Pitchwinkel_Mittelwert"

    def test_english_scada_columns(self):
        df = pd.DataFrame(
            {
                "Time": [1],
                "WS_Nacelle_Avg": [2],
                "ActivePower_kW": [3],
                "WindDirection_Nacelle": [4],
                "AmbTemp_Nacelle": [5],
                "RotorSpeed_RPM": [6],
                "PitchAngle_Deg": [7],
            }
        )
        detector = RegexVariableDetector(
            {
                "timestamp": [r"time", r"timestamp", r"datetime"],
                "wind_speed": [r"ws.*nacelle", r"wind.*speed"],
                "power": [r"active.*power", r"power.*kw"],
                "wind_dir": [r"wind.*direction"],
                "temp": [r"amb.*temp", r"temperature"],
                "rotor_speed": [r"rotor.*speed", r"rpm"],
                "pitch": [r"pitch.*angle", r"pitch.*deg"],
            }
        )
        result = detector.detect(df)

        assert result["timestamp"] == "Time"
        assert result["wind_speed"] == "WS_Nacelle_Avg"
        assert result["power"] == "ActivePower_kW"
        assert result["wind_dir"] == "WindDirection_Nacelle"
        assert result["temp"] == "AmbTemp_Nacelle"
        assert result["rotor_speed"] == "RotorSpeed_RPM"
        assert result["pitch"] == "PitchAngle_Deg"
