"""Structured logging via Loguru with file + console sinks."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

# Remove default handler
logger.remove()

# ── Console sink (rich-friendly) ─────────────────────────────
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> — {message}",
    level="INFO",
    colorize=True,
)

# ── File sink (rotated) ─────────────────────────────────────
_log_dir = Path("outputs/logs")
_log_dir.mkdir(parents=True, exist_ok=True)
logger.add(
    _log_dir / "turb_detr_{time:YYYY-MM-DD}.log",
    rotation="10 MB",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
)


def get_logger(name: str) -> logger.__class__:
    """Return a contextual logger bound to the given module name."""
    return logger.bind(name=name)
