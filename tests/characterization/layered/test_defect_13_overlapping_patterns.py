import pandas as pd
import pytest

from phoibe.layered.infrastructure.detector import RegexVariableDetector

# The user's configuration defines "date" and "time" next to "datetime", and both patterns match "Datetime".
PATTERNS = {
    "datetime": [r"datetime", r"zeitstempel"],
    "date": [r"date"],
    "time": [r"time"],
    "power_kw": [r"leistung", r"power"],
}


@pytest.mark.characterization
@pytest.mark.xfail(strict=True, reason="defect 13: one column is detected for several variables")
def test_a_column_is_detected_for_at_most_one_variable():
    df = pd.DataFrame({"Datetime": ["2024-01-01 00:00"], "Leistung": [1.0]})

    detected = RegexVariableDetector(PATTERNS).detect(df)

    columns = [column for column in detected.values() if column is not None]
    assert len(columns) == len(set(columns)), f"column detected more than once: {detected}"
