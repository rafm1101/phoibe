import pandas as pd
import pytest

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.rules.rules_index import AvailabilityRule

ENGLISH_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
GERMAN_DAYS = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
ENGLISH_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
GERMAN_MONTHS = [
    "Januar",
    "Februar",
    "März",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember",
]


@pytest.fixture
def german_names(monkeypatch):
    """Make `de_DE` names available independently of the locales installed on the machine."""
    day_name, month_name = pd.DatetimeIndex.day_name, pd.DatetimeIndex.month_name

    def localised_day_name(self, locale=None):
        names = day_name(self)
        return pd.Index([GERMAN_DAYS[ENGLISH_DAYS.index(n)] for n in names]) if locale == "de_DE" else names

    def localised_month_name(self, locale=None):
        names = month_name(self)
        return pd.Index([GERMAN_MONTHS[ENGLISH_MONTHS.index(n)] for n in names]) if locale == "de_DE" else names

    monkeypatch.setattr(pd.DatetimeIndex, "day_name", localised_day_name)
    monkeypatch.setattr(pd.DatetimeIndex, "month_name", localised_month_name)


@pytest.mark.characterization
@pytest.mark.xfail(strict=True, reason="defect 3: the locale option of availability is ignored")
def test_availability_uses_requested_locale(german_names):
    context = ValidationContext(layer_name="test", detected_variables={"datetime": "Zeitstempel"}, turbine_id="WEA_01")
    df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=24 * 6 * 10, freq="10min")})

    result = AvailabilityRule(locale="de_DE").execute(df, context)

    assert list(result.details["availability_days"]) == GERMAN_DAYS
    assert list(result.details["availability_months"]) == GERMAN_MONTHS


def test_availability_without_locale_uses_english_names(german_names):
    """Guard: the default stays English once defect 3 is fixed."""
    context = ValidationContext(layer_name="test", detected_variables={"datetime": "Zeitstempel"}, turbine_id="WEA_01")
    df = pd.DataFrame({"Zeitstempel": pd.date_range("2024-01-01", periods=24 * 6 * 10, freq="10min")})

    result = AvailabilityRule().execute(df, context)

    assert list(result.details["availability_days"]) == ENGLISH_DAYS
    assert list(result.details["availability_months"]) == ENGLISH_MONTHS
