import pandas as pd
import pytest

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.core.entities import Status
from phoibe.layered.infrastructure.detector import RegexVariableDetector
from phoibe.layered.rules.rules_index import DataGaps

PATTERNS = {"datetime": [r"zeitstempel", r"timestamp", r"datetime"]}


def _frame(index_name: str | None) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=60, freq="10min", name=index_name)
    return pd.DataFrame({"Leistung": range(60)}, index=index)


@pytest.mark.characterization
@pytest.mark.xfail(strict=True, reason="defect 11: the detector reads columns only")
@pytest.mark.parametrize("index_name", ["Zeitstempel", None], ids=["named index", "unnamed index"])
def test_detector_finds_datetime_index(index_name):
    detected = RegexVariableDetector(PATTERNS).detect(_frame(index_name))

    assert detected["datetime"] is not None


@pytest.mark.characterization
@pytest.mark.xfail(strict=True, reason="defect 11: temporal rules return NOT_CHECKED for a datetime index")
def test_temporal_rule_is_evaluated_for_datetime_index():
    df = _frame("Zeitstempel")
    detected = RegexVariableDetector(PATTERNS).detect(df)
    context = ValidationContext(layer_name="raw", detected_variables=detected, turbine_id="WEA_01")

    result = DataGaps().execute(df, context)

    assert result.status != Status.NOT_CHECKED
