import json
import logging

import pytest

from phoibe.layered.logging.formatter import JSONFormatter


class TestJSONFormatterUnitTests:

    @pytest.fixture
    def formatter(self):
        return JSONFormatter()

    def test_json_formatter_handles_non_serializable_extra_fields(self, formatter):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py", lineno=42, msg="Test message", args=(), exc_info=None
        )
        record.non_serializable_obj = object()
        record.lambda_func = lambda x: x

        try:
            output = formatter.format(record)
            parsed = json.loads(output)
            assert parsed["message"] == "Test message"
            assert parsed["level"] == "INFO"
        except (TypeError, json.JSONDecodeError) as exception:
            pytest.fail(f"Formatter crashed on non-serializable object: {exception}")

    def test_json_formatter_handles_circular_reference_in_extra(self, formatter):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py", lineno=42, msg="Test message", args=(), exc_info=None
        )
        circular_dict = {"key": "value"}
        circular_dict["self"] = circular_dict
        record.circular = circular_dict
        try:
            output = formatter.format(record)
            parsed = json.loads(output)
            assert parsed["message"] == "Test message"
        except (TypeError, json.JSONDecodeError, RecursionError) as e:
            pytest.fail(f"Formatter crashed on circular reference: {e}")
