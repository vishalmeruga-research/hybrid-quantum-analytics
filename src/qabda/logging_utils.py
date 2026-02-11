from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler


def setup_logging(level: str = "INFO", *, logger_name: Optional[str] = None) -> logging.Logger:
    """Configure a nice console logger.

    We intentionally keep logging minimal and dependency-light:
    - Rich handler for readable traces
    - No external logging backend assumptions

    Parameters
    ----------
    level:
        Logging level string (e.g., 'INFO', 'DEBUG').
    logger_name:
        If provided, return a named logger; else returns root logger.

    Returns
    -------
    logging.Logger
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger(logger_name)
