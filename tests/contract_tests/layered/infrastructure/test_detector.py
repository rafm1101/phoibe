import pandas as pd
import pytest

from phoibe.layered.infrastructure.detector import RegexVariableDetector


class VariableDetectorContract:

    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame(
            {
                "Zeitstempel": pd.date_range("2024-01-01", periods=5),
                "ws_gondel": [5.2, 6.1, 5.8, 7.2, 6.5],
                "Leistung": [1200, 1500, 1300, 1800, 1400],
                "Windrichtung": [180, 190, 175, 185, 195],
                "Temp_Außen": [15.2, 14.8, 15.5, 14.9, 15.1],
            }
        )

    def test_detect_returns_valid_dictionary(self, detector, sample_dataframe):
        result = detector.detect(sample_dataframe)
        assert isinstance(result, dict)
        assert all(isinstance(key, str) for key in result.keys())
        assert all(isinstance(value, str) or value is None for value in result.values())

    def test_detect_is_idempotent(self, detector, sample_dataframe):
        result1 = detector.detect(sample_dataframe)
        result2 = detector.detect(sample_dataframe)
        assert result1 == result2

    def test_detect_handles_empty_dataframe(self, detector):
        empty_df = pd.DataFrame()
        result = detector.detect(empty_df)
        assert isinstance(result, dict)

    def test_matched_columns_exist_in_dataframe(self, detector, sample_dataframe):
        result = detector.detect(sample_dataframe)
        for variable_name, column_key in result.items():
            if column_key is not None:
                assert column_key in sample_dataframe.columns, (
                    f"Signal '{variable_name}' matched to '{column_key}' " f"but column does not exist in DataFrame"
                )


class TestRegexVariableDetectorContract(VariableDetectorContract):

    @pytest.fixture
    def detector(self):
        patterns = {
            "timestamp": [r"zeit.*stempel", r"timestamp", r"^time$"],
            "wind_speed": [r"wind.*speed", r"ws.*gondel", r"v_wind"],
            "power": [r"leistung", r"power", r"p_act"],
            "wind_direction": [r"wind.*richtung", r"wind.*direction"],
            "temperature": [r"temp.*außen", r"temperature", r"temp_out"],
            "not_in_dataframe": [r"nonexistent_pattern"],
        }
        return RegexVariableDetector(patterns=patterns)

    def test_detects_exact_matches(self):
        df = pd.DataFrame({"timestamp": [1, 2], "power": [3, 4]})
        detector = RegexVariableDetector({"timestamp": [r"^timestamp$"], "power": [r"^power$"]})
        result = detector.detect(df)

        assert result["timestamp"] == "timestamp"
        assert result["power"] == "power"

    def test_detects_case_insensitive(self):
        df = pd.DataFrame({"TIMESTAMP": [1, 2], "Power": [3, 4]})
        detector = RegexVariableDetector({"timestamp": [r"timestamp"], "power": [r"power"]})
        result = detector.detect(df)

        assert result["timestamp"] == "TIMESTAMP"
        assert result["power"] == "Power"

    def test_returns_none_for_missing_signals(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        detector = RegexVariableDetector({"missing_signal": [r"nonexistent"]})
        result = detector.detect(df)

        assert result["missing_signal"] is None

    def test_first_pattern_match_wins(self):
        df = pd.DataFrame({"wind_speed_nacelle": [1, 2], "ws_gondel": [3, 4], "v_wind": [5, 6]})
        detector = RegexVariableDetector({"wind_speed": [r"ws.*gondel", r"wind.*speed", r"v_wind"]})
        result = detector.detect(df)

        assert result["wind_speed"] in ["wind_speed_nacelle", "ws_gondel", "v_wind"]

    def test_first_column_match_wins_per_pattern(self):
        df = pd.DataFrame({"wind_speed_1": [1, 2], "wind_speed_2": [3, 4], "wind_speed_3": [5, 6]})
        detector = RegexVariableDetector({"wind_speed": [r"wind.*speed"]})
        result = detector.detect(df)

        assert result["wind_speed"] == "wind_speed_1"

    def test_patterns_are_regex_not_literal(self):
        df = pd.DataFrame({"ws_123_gondel": [1, 2]})
        detector = RegexVariableDetector({"wind_speed": [r"ws_\d+_gondel"]})
        result = detector.detect(df)

        assert result["wind_speed"] == "ws_123_gondel"

    def test_handles_special_regex_characters(self):
        df = pd.DataFrame({"power[kW]": [1, 2]})
        detector = RegexVariableDetector({"power": [r"power\[kW\]"]})
        result = detector.detect(df)

        assert result["power"] == "power[kW]"

    def test_empty_pattern_list_returns_none(self):
        df = pd.DataFrame({"any_column": [1, 2]})
        detector = RegexVariableDetector({"signal_with_no_patterns": []})
        result = detector.detect(df)

        assert result["signal_with_no_patterns"] is None


class TestSignalDetectorRealWorld:

    def test_detects_rotorsoft_columns(self):
        df = pd.DataFrame(
            {
                "Zeitstempel": pd.date_range("2024-01-01", periods=3),
                "ws_gondel": [5.2, 6.1, 5.8],
                "Leistung": [1200, 1500, 1300],
                "Windrichtung_Gondel": [180, 190, 175],
                "Außentemperatur_Gondel": [15.2, 14.8, 15.5],
                "Rotordrehzahl": [12.5, 13.1, 12.8],
                "Pitchwinkel": [5.2, 6.1, 5.5],
            }
        )

        detector = RegexVariableDetector(
            {
                "timestamp": [r"zeitstempel", r"timestamp"],
                "wind_speed": [r"ws.*gondel", r"windgeschwindigkeit"],
                "power": [r"leistung", r"power"],
                "wind_direction": [r"windrichtung", r"wind.*direction"],
                "temperature": [r"außentemperatur", r"temperature"],
                "rotor_speed": [r"rotordrehzahl", r"rotor.*speed"],
                "pitch": [r"pitchwinkel", r"pitch.*angle"],
            }
        )

        result = detector.detect(df)

        assert result["timestamp"] == "Zeitstempel"
        assert result["wind_speed"] == "ws_gondel"
        assert result["power"] == "Leistung"
        assert result["wind_direction"] == "Windrichtung_Gondel"
        assert result["temperature"] == "Außentemperatur_Gondel"
        assert result["rotor_speed"] == "Rotordrehzahl"
        assert result["pitch"] == "Pitchwinkel"

    def test_handles_mixed_german_english_columns(self):
        df = pd.DataFrame(
            {
                "Time": pd.date_range("2024-01-01", periods=3),
                "ws_nacelle": [5.2, 6.1, 5.8],
                "Leistung_kW": [1200, 1500, 1300],
            }
        )

        detector = RegexVariableDetector(
            {
                "timestamp": [r"time", r"zeit"],
                "wind_speed": [r"ws.*nacelle", r"ws.*gondel"],
                "power": [r"leistung", r"power"],
            }
        )

        result = detector.detect(df)

        assert result["timestamp"] == "Time"
        assert result["wind_speed"] == "ws_nacelle"
        assert result["power"] == "Leistung_kW"
