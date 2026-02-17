import json

import pytest

from phoibe.layered.logging.logging import LoggerFactory
from phoibe.layered.logging.logging import LoggingConfig
from phoibe.layered.logging.logging import RuleExecutionTracker


class TestLoggingIntegrationContract:

    @pytest.fixture
    def logger(self, tmp_path):
        logging_config = LoggingConfig(log_dir=tmp_path)
        logger = LoggerFactory(logging_config).create_logger("test")
        return logger

    def test_logger_writes_message_to_file(self, logger, tmp_path):
        test_message = "Test log message"
        logger.info(test_message)

        for handler in logger.handlers:
            handler.flush()

        log_file = tmp_path / "validation.log"
        assert log_file.exists()

        content = log_file.read_text()
        assert test_message in content

    def test_logger_writes_message_to_json(self, logger, tmp_path):
        test_message = "JSON test message"
        logger.info(test_message, extra={"turbine_id": "WEA 01"})

        for handler in logger.handlers:
            handler.flush()

        json_file = tmp_path / "validation_audit.jsonl"
        assert json_file.exists()

        content = json_file.read_text()
        assert test_message in content

        parsed = json.loads(content.strip())
        assert parsed["message"] == test_message
        assert parsed["turbine_id"] == "WEA 01"

    def test_tracker_logs_timing(self, logger, tmp_path):
        with RuleExecutionTracker(logger, "test_rule", "WEA_01"):
            pass

        for handler in logger.handlers:
            handler.flush()

        json_file = tmp_path / "validation_audit.jsonl"
        content = json_file.read_text()

        assert "completed" in content.lower()
        lines = content.strip().split("\n")

        for line in lines:
            parsed = json.loads(line)
            if "duration_ms" in parsed:
                assert isinstance(parsed["duration_ms"], (int, float))
                assert parsed["duration_ms"] >= 0
                break
            else:
                pytest.fail("No duration_ms found in logs")
