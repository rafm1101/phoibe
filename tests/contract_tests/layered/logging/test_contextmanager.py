import json
import sys

import pytest

from phoibe.layered.logging.logging import ContextualLogger
from phoibe.layered.logging.logging import LoggerFactory
from phoibe.layered.logging.logging import LoggingConfig
from phoibe.layered.logging.logging import RuleExecutionTracker


class TestRuleExecutionTrackerContract:

    @pytest.fixture
    def logger(self, tmp_path):
        logging_config = LoggingConfig(log_dir=tmp_path)
        logger = LoggerFactory(logging_config).create_logger("test")
        return logger

    @pytest.fixture
    def tracker(self, logger):
        tracker = RuleExecutionTracker(logger, "test_rule", "WEA 01")
        return tracker

    def test_tracker_implements_context_manager_protocol(self, tracker):
        assert hasattr(tracker, "__enter__")
        assert hasattr(tracker, "__exit__")
        assert callable(tracker.__enter__)
        assert callable(tracker.__exit__)

    def test_tracker_is_usable_as_context_manager(self, tracker):
        with tracker:
            pass

    def test_tracker_enter_returns_self(self, tracker):
        result = tracker.__enter__()
        assert result is tracker

    def test_tracker_exit_handles_no_exception(self, tracker, tmp_path):
        tracker.__enter__()
        result = tracker.__exit__(None, None, None)
        assert not result

        log_file = tmp_path / "validation_audit.jsonl"
        assert log_file.exists()

        content = log_file.read_text()
        assert "completed" in content.lower()
        assert "duration_ms" in content

    def test_tracker_exit_handles_exception(self, tracker, tmp_path):
        tracker.__enter__()
        try:
            raise ValueError("Test error")
        except ValueError:
            exc_type, exc_val, exc_tb = sys.exc_info()
            result = tracker.__exit__(exc_type, exc_val, exc_tb)

        assert not result

        log_file = tmp_path / "validation_audit.jsonl"
        assert log_file.exists()

        content = log_file.read_text()
        assert "failed" in content.lower() or "error" in content.lower()
        assert "ValueError" in content
        assert "Test error" in content


class TestContextualLoggerContract:

    @pytest.fixture
    def logger(self, tmp_path):
        logging_config = LoggingConfig(log_dir=tmp_path)
        logger = LoggerFactory(logging_config).create_logger("test")
        return logger

    @pytest.fixture
    def contextual_logger(self, logger):
        context = {"turbine_id": "WEA 01", "layer": "raw"}
        return ContextualLogger(logger, context)

    def test_contextual_logger_implements_logger_interface(self, contextual_logger):
        assert hasattr(contextual_logger, "debug")
        assert hasattr(contextual_logger, "info")
        assert hasattr(contextual_logger, "warning")
        assert hasattr(contextual_logger, "error")
        assert hasattr(contextual_logger, "critical")

        assert callable(contextual_logger.debug)
        assert callable(contextual_logger.info)
        assert callable(contextual_logger.warning)
        assert callable(contextual_logger.error)
        assert callable(contextual_logger.critical)

    def test_contextual_logger_includes_context_in_logs(self, contextual_logger, tmp_path):
        test_message = "Test contextual message."
        contextual_logger.info(test_message)

        for handler in contextual_logger.logger.handlers:
            handler.flush()

        json_log = tmp_path / "validation_audit.jsonl"
        assert json_log.exists()

        content = json_log.read_text()
        parsed = json.loads(content.strip())

        assert parsed["turbine_id"] == "WEA 01"
        assert parsed["layer"] == "raw"
        assert parsed["message"] == test_message

    def test_contextual_logger_preserves_log_levels(self, contextual_logger, tmp_path):
        contextual_logger.debug("Debug message")
        contextual_logger.info("Info message")
        contextual_logger.warning("Warning message")
        contextual_logger.error("Error message")

        for handler in contextual_logger.logger.handlers:
            handler.flush()

        json_log = tmp_path / "validation_audit.jsonl"
        lines = json_log.read_text().strip().split("\n")

        for line in lines:
            if line:
                parsed = json.loads(line)
                assert parsed["turbine_id"] == "WEA 01"
                assert parsed["layer"] == "raw"

    def test_contextual_logger_runs_as_context_manager(self, logger):
        context = {"turbine_id": "WEA 02"}

        with ContextualLogger(logger, context) as ctx_logger:
            assert ctx_logger is not None
            ctx_logger.info("Test")

    def test_contextual_logger_enter_returns_self(self, logger):
        context = {"turbine_id": "WEA 03"}
        ctx_logger = ContextualLogger(logger, context)
        result = ctx_logger.__enter__()

        assert result is ctx_logger

    def test_contextual_logger_accepts_extra_fields(self, contextual_logger, tmp_path):
        contextual_logger.info("Test with extra", extra={"rule_name": "test_rule", "status": "passed"})

        for handler in contextual_logger.logger.handlers:
            handler.flush()

        json_log = tmp_path / "validation_audit.jsonl"
        content = json_log.read_text()
        parsed = json.loads(content.strip())

        assert parsed["turbine_id"] == "WEA 01"
        assert parsed["layer"] == "raw"
        assert parsed["rule_name"] == "test_rule"
        assert parsed["status"] == "passed"

    def test_contextual_logger_accepts_empty_context(self, logger, tmp_path):
        ctx_logger = ContextualLogger(logger, {})
        ctx_logger.info("Message without context.")

        for handler in ctx_logger.logger.handlers:
            handler.flush()

        json_log = tmp_path / "validation_audit.jsonl"
        content = json_log.read_text()
        parsed = json.loads(content.strip())

        assert parsed["message"] == "Message without context."
