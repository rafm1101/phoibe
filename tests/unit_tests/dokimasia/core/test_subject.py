import pytest

from phoibe.dokimasia import core


class TestSubject:
    def test_specification_is_a_read_only_copy(self):
        specification = {"rated_power_kw": 5600}
        subject = core.Subject(id="t1", device_type="wind_turbine", specification=specification)
        specification["rated_power_kw"] = 1

        assert subject.specification["rated_power_kw"] == 5600
        with pytest.raises(TypeError):
            subject.specification["rated_power_kw"] = 2  # type: ignore[index]

    def test_specification_defaults_to_empty(self):
        assert dict(core.Subject(id="t1", device_type="wind_turbine").specification) == {}


def test_specification_is_frozen_deeply():
    subject = core.Subject("t", "d", {"curve": [1, 2], "site": {"tz": "UTC", "tags": ["a"]}})

    assert subject.specification["curve"] == (1, 2)
    assert subject.specification["site"]["tags"] == ("a",)
    with pytest.raises(TypeError):
        subject.specification["site"]["tz"] = "CET"  # type: ignore[index]
    assert subject.specification == {"curve": (1, 2), "site": {"tz": "UTC", "tags": ("a",)}}
