import pytest

from phoibe.dokimasia import rules

PATTERNS = (
    rules.SignalPattern("power", (r"power",)),
    rules.SignalPattern("speed", (r"speed", r"velocity")),
    rules.SignalPattern("angle", (r"^angle$",)),
)


def test_detect_signals_given_one_column_per_signal_maps_them():
    detection = rules.detect_signals(["Power (kW)", "wind_speed", "angle", "note"], PATTERNS)

    assert detection.mapping == {"Power (kW)": "power", "wind_speed": "speed"}
    assert dict(detection.detected) == {"power": "Power (kW)", "speed": "wind_speed", "angle": "angle"}
    assert detection.unmatched == ("note",)
    assert detection.missing == ()
    assert detection.is_clear


def test_a_column_that_already_has_the_signal_name_is_detected_but_not_in_the_mapping():
    detection = rules.detect_signals(["power", "velocity"], PATTERNS)

    assert dict(detection.mapping) == {"velocity": "speed"}
    assert detection.detected["power"] == "power"
    assert detection.missing == ("angle",)


def test_a_signal_without_a_column_is_missing_and_that_is_not_ambiguous():
    detection = rules.detect_signals(["power"], PATTERNS)

    assert detection.missing == ("speed", "angle")
    assert detection.is_clear


def test_a_signal_with_two_candidate_columns_is_ambiguous_and_neither_is_mapped():
    detection = rules.detect_signals(["power_a", "power_b", "speed"], PATTERNS)

    assert [item.kind for item in detection.ambiguities] == ["signal_has_several_columns"]
    assert detection.ambiguities[0].subject == "power"
    assert detection.ambiguities[0].candidates == ("power_a", "power_b")
    assert "power" not in detection.detected
    assert "power_a" not in detection.mapping
    assert "power_b" not in detection.mapping
    assert not detection.is_clear


def test_a_column_claimed_by_two_signals_is_ambiguous_for_both_signals():
    detection = rules.detect_signals(["power_speed"], PATTERNS)

    kinds = {item.kind for item in detection.ambiguities}
    assert "column_matches_several_signals" in kinds
    assert detection.mapping == {}
    assert detection.detected == {}


def test_a_column_name_that_appears_twice_is_ambiguous_and_not_mapped():
    detection = rules.detect_signals(["power", "power", "speed"], PATTERNS)

    assert [item.kind for item in detection.ambiguities] == ["column_repeated"]
    assert "power" not in detection.detected
    assert detection.detected["speed"] == "speed"


def test_matching_ignores_case_and_uses_search_semantics():
    detection = rules.detect_signals(["MY_POWER_COL"], PATTERNS)

    assert detection.detected == {"power": "MY_POWER_COL"}


def test_detect_signals_given_no_columns_reports_everything_missing():
    detection = rules.detect_signals([], PATTERNS)

    assert detection.missing == ("power", "speed", "angle")
    assert detection.mapping == {}
    assert detection.is_clear


def test_detect_signals_given_duplicate_patterns_raises():
    with pytest.raises(ValueError, match="power"):
        rules.detect_signals(["a"], (*PATTERNS, rules.SignalPattern("power", ("x",))))


def test_a_pattern_that_does_not_compile_is_refused_naming_signal_and_pattern():
    with pytest.raises(ValueError, match=r"signal 'power'.*'\(unclosed'"):
        rules.SignalPattern("power", ("(unclosed",))


def test_raise_if_ambiguous_lists_every_ambiguity():
    detection = rules.detect_signals(["power_a", "power_b"], PATTERNS)

    with pytest.raises(rules.DetectionError, match="signal 'power' matches several columns: 'power_a', 'power_b'"):
        detection.raise_if_ambiguous()


def test_raise_if_ambiguous_does_nothing_for_a_clear_result():
    rules.detect_signals(["power"], PATTERNS).raise_if_ambiguous()


def test_the_mappings_are_read_only():
    detection = rules.detect_signals(["pow_er"], PATTERNS)

    with pytest.raises(TypeError):
        detection.mapping["x"] = "y"  # type: ignore[index]


def test_build_table_has_one_row_per_fact():
    table = rules.detect_signals(["power_a", "power_b", "speed", "note"], PATTERNS).build_table()

    assert table["status"].tolist() == ["detected", "missing", "ambiguous", "unmatched"]
    assert table["column"].tolist()[0] == "speed"
    assert "several columns" in table["detail"].iloc[2]
