"""
src/utils/logging_utils.py

Consistent logging configuration across the training CLI (main.py) and
the API service (src/api/main.py). Idempotent -- safe to call from both
without duplicating handlers if a process ends up importing both.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)

    for noisy in ("PIL", "matplotlib", "urllib3", "ultralytics"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
