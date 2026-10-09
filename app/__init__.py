import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog
from structlog.typing import Processor

from app.configs import get_settings

REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")
FRAMEWORK_LOGGERS = ("uvicorn", "uvicorn.error", "fastapi")
SHARED_PROCESSORS: list[Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
]
_LOG_FILE_PATH: str | None = get_settings().LOG_FILE_PATH
DEFAULT_LOG_FILE_PATH: Path | None = (
    None if _LOG_FILE_PATH is None else Path(_LOG_FILE_PATH)
)


def _make_formatter(*renderers: Processor) -> structlog.stdlib.ProcessorFormatter:
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=SHARED_PROCESSORS,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *renderers,
        ],
    )


def _json_formatter() -> structlog.stdlib.ProcessorFormatter:
    return _make_formatter(
        structlog.processors.dict_tracebacks,
        structlog.processors.JSONRenderer(),
    )


def _console_formatter() -> structlog.stdlib.ProcessorFormatter:
    return _make_formatter(structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty()))


def _build_handlers(json_logs: bool, log_file: Path | None) -> list[logging.Handler]:
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(
        _json_formatter() if json_logs else _console_formatter()
    )
    handlers: list[logging.Handler] = [stdout_handler]

    if log_file is None:
        return handlers

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=get_settings().LOG_MAX_BYTES,
        backupCount=get_settings().LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(_json_formatter())
    handlers.append(file_handler)
    return handlers


def _route_framework_logs_to_root() -> None:
    for name in FRAMEWORK_LOGGERS:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True
    access_logger = logging.getLogger("uvicorn.access")
    access_logger.handlers.clear()
    access_logger.propagate = False


def configure_logging(
    level: int = logging.INFO,
    json_logs: bool = True,
    log_file: Path | None = DEFAULT_LOG_FILE_PATH,
) -> None:
    structlog.configure(
        processors=[
            *SHARED_PROCESSORS,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    for handler in _build_handlers(json_logs, log_file):
        root_logger.addHandler(handler)

    _route_framework_logs_to_root()
