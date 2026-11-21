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
