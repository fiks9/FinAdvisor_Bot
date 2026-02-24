"""
app/db/database.py — Async SQLite initialisation via aiosqlite
===============================================================
Call init_db() exactly ONCE at startup (inside post_init in main.py).
It creates the tables if they don't exist — safe to call on every restart.

Why aiosqlite instead of sqlite3?
  The standard sqlite3 module is synchronous. Any DB call (even a fast SELECT)
  blocks the entire asyncio event loop, which means the bot stops processing
  ALL other users' messages until the query finishes. aiosqlite wraps sqlite3
  in a background thread, keeping the event loop free.
"""

import logging
from pathlib import Path
from typing import AsyncGenerator
from contextlib import asynccontextmanager

import aiosqlite

logger = logging.getLogger(__name__)

# Module-level reference so get_db() can reuse the same path
_DB_PATH: Path | None = None


async def init_db(db_path: Path) -> None:
    """
    Initialise the database: create parent dir, create tables.
    Safe to call multiple times (uses IF NOT EXISTS).

    Args:
        db_path: Path to the .db file, e.g. Path("data/memory.db")
    """
    global _DB_PATH
    _DB_PATH = db_path

    # Ensure the parent directory exists (validator in config does this too,
    # but we guard here as well for safety when path comes from Railway Volume)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(db_path) as db:
        # Enable WAL mode — better concurrent read performance
        await db.execute("PRAGMA journal_mode=WAL")
        # Enforce foreign key constraints (off by default in SQLite!)
        await db.execute("PRAGMA foreign_keys=ON")

        # ── users table ────────────────────────────────────────────────
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id    INTEGER PRIMARY KEY,
                username   TEXT,
                first_name TEXT    NOT NULL DEFAULT 'Unknown',
                is_active  INTEGER NOT NULL DEFAULT 1,
                created_at TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)

        # ── messages table ─────────────────────────────────────────────
        # role CHECK ensures only "user" or "assistant" can be stored
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                role       TEXT    NOT NULL CHECK(role IN ('user', 'assistant')),
                content    TEXT    NOT NULL,
                created_at TEXT    NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
        """)

        # ── index for fast history queries ─────────────────────────────
        # Most common query: SELECT ... WHERE user_id = ? ORDER BY created_at
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_user_created
            ON messages(user_id, created_at)
        """)

        await db.commit()

    logger.info("Database initialised at %s", db_path)


@asynccontextmanager
async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """
    Async context manager that yields an open DB connection.
    Always closes the connection after the block — no leaked handles.

    Usage:
        async with get_db() as db:
            await db.execute("SELECT ...")

    Why context manager instead of a global connection?
        aiosqlite connections are NOT thread-safe across coroutines.
        Short-lived connections per operation are the safest pattern
        and are fast because SQLite has no network overhead.
    """
    if _DB_PATH is None:
        raise RuntimeError(
            "Database not initialised. Call init_db() first inside post_init()."
        )

    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row   # rows behave like dicts: row["column"]
        await db.execute("PRAGMA foreign_keys=ON")
        yield db
