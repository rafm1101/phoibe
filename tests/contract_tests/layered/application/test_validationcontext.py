import dataclasses

import pytest

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.core.entities import ValidationMode


class TestValidationContextContract:

    @pytest.fixture
    def context(self):
        return ValidationContext(
            detected_variables={"timestamp": "Zeitstempel", "wind_speed": "ws_gondel", "power": None},
            turbine_id="WEA 01",
            layer_name="raw",
            metadata={"custom_field": "value"},
        )

    def test_context_is_frozen(self, context):
        with pytest.raises(dataclasses.FrozenInstanceError):
            context.turbine_id = "WEA 02"

    def test_detected_variables_are_frozen(self, context):
        with pytest.raises(dataclasses.FrozenInstanceError):
            context.detected_variables = {}

    def test_context_has_attributes(self, context):
        assert hasattr(context, "detected_variables")
        assert isinstance(context.layer_name, str)
        assert isinstance(context.detected_variables, dict)
        assert isinstance(context.turbine_id, str)
        assert isinstance(context.validation_mode, str)
        assert isinstance(context.metadata, dict)

    def test_context_get_column_key_returns_column_name(self, context):
        result = context.get_column_key("timestamp")
        assert result == "Zeitstempel"

    def test_context_get_column_key_returns_none_for_missing(self, context):
        result = context.get_column_key("power")
        assert result is None

    def test_context_get_column_key_returns_none_for_unknown(self, context):
        result = context.get_column_key("nonexistent_variable")
        assert result is None

    def test_context_has_variable_true_for_detected(self, context):
        assert context.has_variable("timestamp") is True
        assert context.has_variable("wind_speed") is True

    def test_context_has_variable_false_for_not_detected(self, context):
        assert context.has_variable("power") is False

    def test_context_has_variable_false_for_unknown(self, context):
        assert context.has_variable("unknown") is False

    # def test_context_layer_name_has_default(self):
    #     context = ValidationContext(detected_variables={}, turbine_id="WEA_01")
    #     assert context.layer_name == "raw"

    def test_context_metadata_has_default(self):
        context = ValidationContext(layer_name="platinum", detected_variables={}, turbine_id="WEA 01")
        assert context.metadata == {}

    def test_context_accepts_empty_detected_variables(self):
        context = ValidationContext(layer_name="platinum", detected_variables={}, turbine_id="WEA 01")
        assert context.detected_variables == {}

    def test_context_accepts_all_none_variables(self):
        context = ValidationContext(
            layer_name="platinum",
            detected_variables={"variable1": None, "variable2": None, "variable3": None},
            turbine_id="WEA_01",
        )
        assert all(v is None for v in context.detected_variables.values())

    def test_context_metadata_can_contain_any_type(self):
        context = ValidationContext(
            layer_name="platinum",
            detected_variables={},
            turbine_id="WEA_01",
            metadata={
                "string": "value",
                "int": 42,
                "float": 3.14,
                "bool": True,
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
            },
        )
        assert context.metadata["string"] == "value"
        assert context.metadata["int"] == 42

    @pytest.mark.parametrize(
        "validation_mode, expected_profiling, expected_contract",
        [(ValidationMode.PROFILING, True, False), (ValidationMode.CONTRACT, False, True)],
    )
    def test_context_is_mode_returns_correct_bools(self, validation_mode, expected_profiling, expected_contract):
        context = ValidationContext(
            layer_name="platinum",
            detected_variables={},
            turbine_id="WEA_01",
            validation_mode=validation_mode,
            metadata={"custom_field": "value"},
        )
        assert context.is_profiling_mode is expected_profiling
        assert context.is_contract_mode is expected_contract
