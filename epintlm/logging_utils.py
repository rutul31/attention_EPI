"""Structured logging with run-id injection. Console + per-run file handlers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_RUN_ID = "init"
_FILE_HANDLER: Optional[logging.Handler] = None


class _RunIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _RUN_ID
        return True


def set_run_id(run_id: str) -> None:
    """Update the run_id stamped on every subsequent log record."""
    global _RUN_ID
    _RUN_ID = run_id


def configure_logging(
    level: str = "INFO",
    fmt: str = "%(asctime)s [%(run_id)s] %(levelname)s %(name)s: %(message)s",
    console: bool = True,
    log_file: Optional[Path] = None,
) -> None:
    """Configure root logger. Idempotent: clears prior handlers."""
    global _FILE_HANDLER

    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers (idempotent)
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(fmt)
    run_filter = _RunIdFilter()

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.addFilter(run_filter)
        root.addHandler(ch)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(formatter)
        fh.addFilter(run_filter)
        root.addHandler(fh)
        _FILE_HANDLER = fh


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
