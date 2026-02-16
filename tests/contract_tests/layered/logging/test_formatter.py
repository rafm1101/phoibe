import json
import logging
import sys

import pytest

from phoibe.layered.logging.formatter import JSONFormatter


@pytest.fixture
def logging_record():
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="test.py", lineno=42, msg="Test message", args=(), exc_info=None
    )
    return record


class TestJSONFormatterContract:

    @pytest.fixture
    def formatter(self):
        return JSONFormatter()

    def test_formatter_produces_valid_json(self, formatter, logging_record):
        output = formatter.format(logging_record)
        parsed_output = json.loads(output)
        assert isinstance(parsed_output, dict)

    def test_parsed_output_contains_required_fields(self, formatter, logging_record):
        output = formatter.format(logging_record)
        parsed_output = json.loads(output)
        assert "timestamp" in parsed_output
        assert "level" in parsed_output
        assert "logger" in parsed_output
        assert "message" in parsed_output

    def test_parsed_output_preserves_extra_fields(self, formatter, logging_record):
        logging_record.turbine_id = "WEA 01"
        logging_record.rule_name = "test_rule"
        output = formatter.format(logging_record)
        parsed_output = json.loads(output)

        assert parsed_output["turbine_id"] == "WEA 01"
        assert parsed_output["rule_name"] == "test_rule"

    def test_formatter_handles_exception_info(self, formatter, logging_record):
        try:
            raise ValueError("Test error")
        except ValueError:
            exc_info = sys.exc_info()
            logging_record.exc_info = exc_info
            logging_record.level = logging.ERROR

        output = formatter.format(logging_record)
        parsed_output = json.loads(output)

        assert "exception" in parsed_output
        assert "ValueError" in parsed_output["exception"]
