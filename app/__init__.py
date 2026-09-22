import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime
from logging import LogRecord
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import override

from app.configs import get_settings

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class ColorFormatter(logging.Formatter):
    _colors = {  # noqa
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # White on Red
    }
    _reset = "\033[0m"

    @override
    def format(self, record: LogRecord) -> str:
        color = self._colors.get(record.levelname, "")
        prefix = f"{color}{record.levelname:<8}{self._reset}"
        base = super().format(record)
        return f"{prefix} {base}"


class JSONFormatter(logging.Formatter):
    @override
    def format(self, record: LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "request_id": request_id_var.get(),
        }
        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, ensure_ascii=False)


def setup_logging(level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()

    settings = get_settings()
    is_debug = settings.ENV == "dev"

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / ("dev.log.jsonl" if is_debug else "app.log.jsonl")

    console_handler = logging.StreamHandler()

    file_handler = RotatingFileHandler(
        logfile,
        maxBytes=50 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    console_handler.setFormatter(ColorFormatter(fmt="%(message)s"))
    file_handler.setFormatter(JSONFormatter())

    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
