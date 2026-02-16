import logging
import pathlib
import time
from typing import Any

import yaml

from .handler import ConsoleHandler
from .handler import FileHandler
from .handler import JSONHandler


class LoggingConfig:
    """Configuration of logging setup."""

    def __init__(
        self,
        log_dir: pathlib.Path | str = "logs",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        json_level: int = logging.INFO,
        max_file_size_mb: int = 10,
        backup_count: int = 5,
    ):
        self.log_dir = pathlib.Path(log_dir)
        self.console_level = console_level
        self.file_level = file_level
        self.json_level = json_level
        self.max_file_size_mb = max_file_size_mb
        self.backup_count = backup_count

    @classmethod
    def from_yaml(cls, config_path: pathlib.Path | str):
        with open(config_path) as filestream:
            data = yaml.safe_load(filestream)
            return cls(**data.get("logging", {}))


class LoggerFactory:
    """Stateless factory for creating configured loggers."""

    def __init__(self, config: LoggingConfig = LoggingConfig()):
        """Initialize factory with configuration."""
        self.config = config

    def create_logger(self, name: str) -> logging.Logger:
        """Create a fully configured logger.

        Parameters
        ----------
        name
            Logger name.

        Returns
        -------
        logger
            Configured logger instance.
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        logger.handlers.clear()

        self._add_handlers(logger)

        return logger

    def _add_handlers(self, logger: logging.Logger) -> None:
        console_handler = ConsoleHandler.create(level=self.config.console_level)
        logger.addHandler(console_handler)

        file_handler = FileHandler.create(
            log_dir=self.config.log_dir,
            level=self.config.file_level,
            max_bytes=self.config.max_file_size_mb * 1024 * 1024,
            backup_count=self.config.backup_count,
        )
        if file_handler:
            logger.addHandler(file_handler)

        json_handler = JSONHandler.create(log_dir=self.config.log_dir, level=self.config.json_level)
        if json_handler:
            logger.addHandler(json_handler)


class RuleExecutionTracker:
    """Context manager for tracking rule execution results.

    Examples
    --------
    1. Run check within the `RuleExecutionTracker`:
       > with RuleExecutionTracker(logger, "temporal_resolution", "WEA_01"):
       >     result = rule.execute(df, context)

    Notes
    -----
    1. The `RuleExecutionTracker` automatically tracks:
       - The start of execution on DEBUG level.
       - The completion with timing on DEBUG level.
       - Errors with stack trace on ERROR level.
    """

    def __init__(self, logger: logging.Logger, rule_name: str, turbine_id: str):
        self.logger = logger
        self.rule_name = rule_name
        self.turbine_id = turbine_id
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.debug(
            f"Executing rule '{self.rule_name}' for {self.turbine_id}",
            extra={"rule_name": self.rule_name, "turbine_id": self.turbine_id},
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000

        if exc_type is None:
            self.logger.info(
                f"Rule '{self.rule_name}' completed for {self.turbine_id} in {duration_ms:.2f}ms",
                extra={
                    "rule_name": self.rule_name,
                    "turbine_id": self.turbine_id,
                    "duration_ms": round(duration_ms, 2),
                },
            )
        else:
            self.logger.error(
                f"Rule '{self.rule_name}' failed for {self.turbine_id}: {exc_val}",
                extra={
                    "rule_name": self.rule_name,
                    "turbine_id": self.turbine_id,
                    "duration_ms": round(duration_ms, 2),
                },
                exc_info=True,
            )
        return False


class ContextualLogger:
    """A contextual logger that is tread-safe and includes a `context` dictionary automatically.

    Examples
    --------
    1. Use instead of the standard logger:
       > logger = get_logger(__name__)
       > ctx_logger = ContextualLogger(logger, {'turbine_id': 'WEA_01', 'run_id': '123'})
       > ctx_logger.info("Processing")
    2. Use as a context manager:
       > with ContextualLogger(logger, {'turbine_id': 'WEA_01', 'run_id': '123'}) as ctx:
       >     ctx.info("Processing...")
    """

    def __init__(self, logger: logging.Logger, context: dict[str, Any]):
        self.logger = logger
        self.context = context

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        extra = kwargs.get("extra", {})
        kwargs["extra"] = {**self.context, **extra}
        self.logger.log(level, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        kwargs.setdefault("exc_info", True)
        self.error(msg, *args, **kwargs)


def get_logger(name: str, logger_config: LoggingConfig = LoggingConfig()) -> logging.Logger:
    """Retrieve a fully configured logger for a module."""
    return LoggerFactory(config=logger_config).create_logger(name=name)
