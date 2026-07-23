import logging
from logging import LogRecord
from pathlib import Path
from typing import override


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # White on Red
    }
    RESET = "\033[0m"

    @override
    def format(self, record: LogRecord):
        color = self.COLORS.get(record.levelname, "")
        prefix = f"{color}{record.levelname:<8}{self.RESET}"
        base = super().format(record)
        return f"{prefix} {base}"


def setup_logging(
    logfile: str = "logs/log.log",
    level: int = logging.INFO,
):
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)

    console_fmt = "%(asctime)s | %(filename)s:%(lineno)d | %(message)s"
    file_fmt = "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
    datefmt = "%H:%M:%S"

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(logfile, mode="w", encoding="utf-8")

    console_handler.setFormatter(ColorFormatter(fmt=console_fmt, datefmt=datefmt))
    file_handler.setFormatter(logging.Formatter(fmt=file_fmt, datefmt=datefmt))

    logging.basicConfig(
        level=level,
        handlers=[console_handler, file_handler],
        force=True,
    )
