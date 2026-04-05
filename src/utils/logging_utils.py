from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(log_file: Path, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("anime_gmm")
    logger.handlers.clear()
    logger.setLevel(level.upper())
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console_handler = RichHandler(rich_tracebacks=True, markup=False, show_path=False)
    console_handler.setLevel(level.upper())
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level.upper())
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger
