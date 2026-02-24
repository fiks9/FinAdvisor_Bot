"""
app/logger.py — Unified logging configuration
==============================================
Call setup_logging() ONCE at the very top of main.py before anything else.
Every other module gets a logger with:
    import logging
    logger = logging.getLogger(__name__)

Log format example:
    2026-02-24 21:55:01 | INFO     | app.bot.handlers | User 123456789 sent: "Привіт"
"""

import logging
import sys
from pathlib import Path


def setup_logging(log_level: str = "INFO", log_to_file: bool = False) -> None:
    """
    Configure root logger with console (and optionally file) output.

    Args:
        log_level:   Minimum severity to capture. One of DEBUG / INFO / WARNING / ERROR.
        log_to_file: If True, also write logs to logs/bot.log (rotated daily).
                     Disabled by default — on Railway, stdout is captured automatically.
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # ── Root logger ────────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # ── Console handler (stdout so Railway captures it) ─────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # ── Optional file handler ────────────────────────────────────────────
    if log_to_file:
        from logging.handlers import TimedRotatingFileHandler

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            filename=log_dir / "bot.log",
            when="midnight",
            backupCount=7,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # ── Silence noisy third-party loggers ───────────────────────────────
    # httpx is used by python-telegram-bot internally — very chatty at DEBUG
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # ChromaDB telemetry spam
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    # Sentence-transformers download progress
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    logging.getLogger(__name__).info("Logging initialised at level %s", log_level)
