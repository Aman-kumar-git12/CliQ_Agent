"""Logging setup shared by the server and the command-line scripts."""

import json
import logging
from datetime import UTC, datetime

from .request_id import RequestIdFilter

TEXT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(request_id)s]: %(message)s"


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for log aggregators (LOG_FORMAT=json)."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", None),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO", log_format: str = "text") -> None:
    handler = logging.StreamHandler()
    is_json = log_format.strip().lower() == "json"
    handler.setFormatter(JsonFormatter() if is_json else logging.Formatter(TEXT_FORMAT, datefmt="%H:%M:%S"))
    # On the handler (not a logger) so records from every logger get a request_id
    handler.addFilter(RequestIdFilter())
    logging.basicConfig(level=level.upper(), handlers=[handler])
    # These libraries log every HTTP request at INFO, which drowns out our own lines
    for noisy in ("httpx", "pymongo", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
