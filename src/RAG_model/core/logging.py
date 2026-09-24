import json
import logging
import sys
from datetime import UTC, datetime

_LOG_FIELDS = (
    "request_id",
    "total_latency_ms",
    "total_tokens",
    "citations_resolved",
    "pipeline_version",
    "status_code",
    "error_type",
)


class JsonFormatter(logging.Formatter):
    """Serialize only explicitly approved request metadata."""

    def format(self, record: logging.LogRecord) -> str:
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        for field in _LOG_FIELDS:
            if hasattr(record, field):
                event[field] = getattr(record, field)

        return json.dumps(event, ensure_ascii=True)


def configure_logging(level: str) -> None:
    """Configure application logs without replacing Uvicorn's handlers."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    app_logger = logging.getLogger("RAG_model")
    app_logger.handlers.clear()
    app_logger.addHandler(handler)
    app_logger.setLevel(level.upper())
    app_logger.propagate = False
