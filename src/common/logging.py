"""Logging setup shared by every entry point.

Kept deliberately small (research brief section 38: prefer simple, transparent implementations).
Its one non-obvious job is to tee console output to a per-run file under ``logs/``, so that a
run's stdout survives alongside its manifest and can be inspected months later when someone
asks why a particular sweep behaved oddly.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.common.paths import LOGS_DIR, ensure_dir

__all__ = ["setup_logging", "get_logger"]

_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def setup_logging(
    *,
    level: int | str = logging.INFO,
    log_file: Path | str | None = None,
    quiet_libraries: bool = True,
) -> logging.Logger:
    """Configure the root logger for a command-line entry point.

    Args:
        level: Threshold for the console handler.
        log_file: Optional path, absolute or relative to ``logs/``, receiving DEBUG and above.
        quiet_libraries: Suppress the routine chatter from matplotlib and numexpr, which
            otherwise buries the project's own messages during long sweeps.

    Returns:
        The configured root logger.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        if not path.is_absolute():
            path = LOGS_DIR / path
        ensure_dir(path.parent)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    if quiet_libraries:
        for noisy in ("matplotlib", "matplotlib.font_manager", "numexpr", "PIL"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    return root


def get_logger(name: str) -> logging.Logger:
    """Return a module logger. Use ``get_logger(__name__)`` at module scope."""
    return logging.getLogger(name)
