import json

import pytest

from phoibe.layered.logging.logging import ContextualLogger
from phoibe.layered.logging.logging import LoggerFactory
from phoibe.layered.logging.logging import LoggingConfig


class TestContextualLoggerUnitTests:

    @pytest.fixture
    def logger(self, tmp_path):
        logging_config = LoggingConfig(log_dir=tmp_path)
        return LoggerFactory(config=logging_config).create_logger("test")

    def test_contextual_logger_preserves_context_dict(self, logger):
        context = {"turbine_id": "WEA 01", "layer": "raw"}
        context_copy = context.copy()
        ctx = ContextualLogger(logger, context)
        ctx.info("Test message", extra={"status": "ok", "new_field": "value"})

        assert context == context_copy
        assert "status" not in context
        assert "new_field" not in context

    def test_contextual_logger_extra_beats_context_on_key_collision(self, logger, tmp_path):
        context = {"run_id": "context_value", "turbine_id": "WEA 01"}
        ctx = ContextualLogger(logger, context)
        ctx.info("Test", extra={"run_id": "extra_value"})

        for handler in logger.handlers:
            handler.flush()

        json_log = tmp_path / "validation_audit.jsonl"
        content = json_log.read_text()
        parsed = json.loads(content.strip())

        assert parsed["run_id"] == "extra_value"
        assert parsed["turbine_id"] == "WEA 01"

    def test_contextual_logger_preserves_context_given_empty_extra(self, logger, tmp_path):
        context = {"turbine_id": "WEA 01", "layer": "raw", "status": "processing"}
        ctx = ContextualLogger(logger, context)
        ctx.info("Test message")

        for handler in logger.handlers:
            handler.flush()

        json_log = tmp_path / "validation_audit.jsonl"
        content = json_log.read_text()
        parsed = json.loads(content.strip())

        assert parsed["turbine_id"] == "WEA 01"
        assert parsed["layer"] == "raw"
        assert parsed["status"] == "processing"
