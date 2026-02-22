import datetime
import json
import logging


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured log record logging."""

    EXTRA_FIELDS = ["turbine_id", "rule_name", "status", "duration_ms", "run_id", "layer", "file_path"]
    """Accepted optional fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as a single-line JSON.

        Parameters
        ----------
        record
            Logging record to be formatted.

        Returns
        -------
        str
            Record formatted to JSON.
        """
        log_data = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        for field in self.EXTRA_FIELDS:
            if hasattr(record, field):
                log_data[field] = getattr(record, field)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)
