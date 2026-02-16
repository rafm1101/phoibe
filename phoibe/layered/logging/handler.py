import logging
import logging.handlers
import pathlib
import sys

from .formatter import JSONFormatter


class ConsoleHandler:
    """Console handler for human-readable output."""

    @staticmethod
    def create(level: int = logging.INFO) -> logging.Handler:
        """Create a console handler with colored output.

        Parameters
        ----------
        level
            Minimum log level. Defaults to INFO.

        Returns
        -------
        handler
            Configured StreamHandler.
        """
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)

        return handler


class FileHandler:
    """Rotating file handler for detailed logs."""

    @staticmethod
    def create(
        log_dir: pathlib.Path | str,
        level: int = logging.DEBUG,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ) -> logging.Handler | None:
        """Create rotating file handler.

        Parameters
        ----------
        log_dir
            Directory for log files.
        level
            Minimum log level. Defaults to DEBUG.
        max_bytes
            Maximal file size before rotation. Defaults to 10MB.
        backup_count
            Number of backup files to keep. Defaults to 5.

        Returns
        -------
        handler
            Configured RotatingFileHandler or None if setup fails.
        """
        try:
            log_dir = pathlib.Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "validation.log"

            handler = logging.handlers.RotatingFileHandler(
                filename=log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            handler.setLevel(level)

            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)

            return handler

        except (OSError, PermissionError) as exception:
            print(f"WARNING: Could not setup file logging: {exception}", file=sys.stderr)
            return None


class JSONHandler:
    """JSON handler for structured audit logs."""

    @staticmethod
    def create(log_dir: pathlib.Path | str, level: int = logging.DEBUG) -> logging.Handler | None:
        """Create JSON file handler.

        Parameters
        ----------
        log_dir
            Directory for log files.
        level
            Minimum log level. Defaults to DEBUG.

        Returns
        -------
        handler
            Configured FileHandler with JSONFormatter or None if setup fails.
        """
        try:
            log_dir = pathlib.Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            json_log_file = log_dir / "validation_audit.jsonl"

            handler = logging.FileHandler(filename=json_log_file, encoding="utf-8")
            handler.setLevel(level)
            handler.setFormatter(JSONFormatter())

            return handler

        except (OSError, PermissionError) as exception:
            print(f"WARNING: Could not setup JSON logging: {exception}", file=sys.stderr)
            return None
