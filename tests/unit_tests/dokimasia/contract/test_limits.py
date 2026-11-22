import pytest

from phoibe.dokimasia import contract, core

SPECIFICATION = {"rated_power_kw": 5600, "site": "WP Moptal"}


class TestResolveLimit:
    def test_a_number_resolves_to_itself(self):
        resolved = contract.resolve_limit(3.0, SPECIFICATION)

        assert (resolved.value, resolved.text) == (3.0, "3")

    def test_a_subject_reference_says_where_the_number_came_from(self):
        resolved = contract.resolve_limit(contract.SubjectRef("rated_power_kw", 1.05), SPECIFICATION)

        assert resolved.value == pytest.approx(5880.0)
        assert resolved.text == "5880 = 1.05 × rated_power_kw 5600"

    def test_an_offset_is_applied_and_shown(self):
        resolved = contract.resolve_limit(contract.SubjectRef("rated_power_kw", 1.0, -100.0), SPECIFICATION)

        assert resolved.value == 5500.0
        assert resolved.text.endswith("+ -100")

    def test_raises_for_a_missing_attribute(self):
        with pytest.raises(KeyError):
            contract.resolve_limit(contract.SubjectRef("hub_height"), SPECIFICATION)

    @pytest.mark.parametrize("attribute", ["site", "flag"])
    def test_raises_for_an_attribute_that_is_not_a_number(self, attribute):
        with pytest.raises(TypeError, match=attribute):
            contract.resolve_limit(contract.SubjectRef(attribute), SPECIFICATION | {"flag": True})


def _contract() -> contract.Contract:
    return contract.build_contract(
        {
            "id": "c",
            "version": "1",
            "domain": "wind",
            "layers": {
                "bronze": {
                    "measures": {"ranges": {"params": {"max": {"subject": "rated_power_kw", "factor": 1.05}}}},
                    "checks": {
                        "c": {
                            "measure": "ranges",
                            "value": "n",
                            "severity": "info",
                            "good": {"max": {"subject": "rotor_diameter"}},
                            "acceptable": {"max": {"subject": "rotor_diameter", "factor": 2}},
                        }
                    },
                }
            },
        }
    )


def test_list_subject_attributes_finds_references_in_parameters_and_bands():
    assert contract.list_subject_attributes(_contract()) == ("rated_power_kw", "rotor_diameter")


def test_validate_subject_references_given_a_complete_subject_passes():
    subject = core.Subject("t1", "wind_turbine", {"rated_power_kw": 5600, "rotor_diameter": 150})

    contract.validate_subject_references(_contract(), subject)


def test_validate_subject_references_names_every_missing_or_invalid_attribute():
    subject = core.Subject("t1", "wind_turbine", {"rated_power_kw": "5600"})

    with pytest.raises(contract.ContractError) as raised:
        contract.validate_subject_references(_contract(), subject)

    assert len(raised.value.problems) == 2
    assert "not a number" in raised.value.problems[0]
    assert "no attribute 'rotor_diameter'" in raised.value.problems[1]
