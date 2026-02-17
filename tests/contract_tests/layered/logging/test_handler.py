import logging

import pytest

from phoibe.layered.logging.handler import ConsoleHandler
from phoibe.layered.logging.handler import FileHandler
from phoibe.layered.logging.handler import JSONHandler


class HandlerContracts:

    def test_handler_is_handler_instance(self, handler):
        assert isinstance(handler, logging.Handler)


class TestConsoleHandlerContracts(HandlerContracts):

    @pytest.fixture
    def handler(self):
        return ConsoleHandler.create()

    def test_console_handler_has_formatter(self, handler):
        assert handler.formatter is not None


class TestFileHandlerContracts(HandlerContracts):

    @pytest.fixture
    def handler(self, tmp_path):
        handler = FileHandler.create(log_dir=tmp_path)
        yield handler
        handler.close()

    @pytest.fixture
    def logger(self, handler):
        logger = logging.getLogger("test_file_handler")
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        logger.propagate = False
        logger.addHandler(handler)
        yield logger
        logger.handlers.clear()

    def test_file_handler_writes_to_file(self, logger, handler, tmp_path):
        log_file = tmp_path / "validation.log"
        test_message = "test message"
        logger.info(test_message)
        handler.flush()

        assert log_file.exists()
        content = log_file.read_text()
        assert test_message in content
        assert handler is not None


class TestJSONHandlerContracts(HandlerContracts):

    @pytest.fixture
    def handler(self, tmp_path):
        return JSONHandler.create(log_dir=tmp_path)

    def test_json_handler_has_json_formatter(self, handler):
        """Contract: JSON handler uses JSONFormatter"""
