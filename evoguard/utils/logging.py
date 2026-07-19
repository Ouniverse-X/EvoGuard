"""Centralized logging configuration for EvoGuard.

Provides a single :func:`get_logger` entry point so all modules share a
consistent, timestamped format. Verbosity is controlled by the ``EVOGUARD_LOG``
environment variable (defaults to ``INFO``).
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.environ.get("EVOGUARD_LOG", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root = logging.getLogger("evoguard")
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the ``evoguard`` root logger."""

    _configure_root()
    if not name.startswith("evoguard"):
        name = f"evoguard.{name}"
    return logging.getLogger(name)
