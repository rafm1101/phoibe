import logging

import pytest

from phoibe.layered.logging.logging import LoggerFactory
from phoibe.layered.logging.logging import LoggingConfig


@pytest.fixture
def logging_config(tmp_path):
    return LoggingConfig(log_dir=tmp_path)


class TestLoggerFactoryContract:

    @pytest.fixture
    def logger(self, logging_config, request):
        logger = LoggerFactory(logging_config).create_logger(request.param)
        return logger

    @pytest.mark.parametrize("logger", ["test.logger"], indirect=["logger"])
    def test_logger_factory_returns_logger_instance(self, logger):
        assert isinstance(logger, logging.Logger)

    @pytest.mark.parametrize("logger, expected_name", [("test.module.name", "test.module.name")], indirect=["logger"])
    def test_created_logger_has_properties(self, logger, expected_name):
        assert logger.name == expected_name
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) > 0

    def test_created_logger_creates_log_directory(self, tmp_path):
        log_dir = tmp_path / "new_logs"
        assert not log_dir.exists()
        logging_config = LoggingConfig(log_dir=log_dir)
        LoggerFactory(logging_config).create_logger("test")
        assert log_dir.exists()
        assert log_dir.is_dir()

    def test_logger_creation_is_idempotent(self, logging_config):
        logger1 = LoggerFactory(logging_config).create_logger("test")
        logger2 = LoggerFactory(logging_config).create_logger("test")
        assert logger1.name == logger2.name
