import pytest

from phoibe.dokimasia import core


def _signal(name: str) -> core.Signal:
    return core.Signal(name=name, quantity="wind_speed", statistic=core.Statistic.MEAN, unit="m/s")


class TestCatalog:
    def test_finds_signals_by_name(self):
        catalog = core.Catalog((_signal("wind_speed"), _signal("wind_speed_max")))

        assert catalog.names == ("wind_speed", "wind_speed_max")
        assert catalog.find_signal("wind_speed_max") == _signal("wind_speed_max")
        assert catalog.find_signal("power_kw") is None

    def test_get_signal_raises_for_an_unknown_name(self):
        with pytest.raises(KeyError):
            core.Catalog((_signal("wind_speed"),)).get_signal("power_kw")

    def test_rejects_duplicate_names_and_names_them_all(self):
        with pytest.raises(ValueError, match="a, b"):
            core.Catalog((_signal("a"), _signal("b"), _signal("a"), _signal("b"), _signal("c")))
