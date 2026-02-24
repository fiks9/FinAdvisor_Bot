"""
app/db/crud.py — Async CRUD operations for users and conversation history
=========================================================================
All functions are async and use get_db() context manager.

Public API:
  upsert_user(user_id, username, first_name)  → None
  save_message(user_id, role, content)        → Message
  get_history(user_id, limit)                 → list[Message]
  clear_history(user_id)                      → int  (rows deleted)
"""

import logging
from datetime import datetime

from app.db.database import get_db
from app.db.models import Message

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Users
# ──────────────────────────────────────────────────────────────────────────────

async def upsert_user(
    user_id:    int,
    username:   str | None,
    first_name: str,
) -> None:
    """
    Insert the user if new, or update username/first_name if they changed.
    Called at the start of every message handler so the users table stays fresh.

    INSERT OR REPLACE would reset created_at — we use INSERT OR IGNORE + UPDATE
    to preserve the original registration timestamp.
    """
    async with get_db() as db:
        # Create row if it doesn't exist yet
        await db.execute(
            """
            INSERT OR IGNORE INTO users (user_id, username, first_name, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, username, first_name, datetime.utcnow().isoformat()),
        )
        # Update mutable fields in case user changed their Telegram name
        await db.execute(
            """
            UPDATE users
            SET username = ?, first_name = ?
            WHERE user_id = ?
            """,
            (username, first_name, user_id),
        )
        await db.commit()
    logger.debug("Upserted user %s (%s)", user_id, username)


# ──────────────────────────────────────────────────────────────────────────────
#  Messages
# ──────────────────────────────────────────────────────────────────────────────

async def save_message(
    user_id: int,
    role:    str,   # "user" | "assistant"
    content: str,
) -> Message:
    """
    Persist a single message and return it as a Message model.

    Order of operations in the handler:
      1. save_message(user_id, "user", text)       ← first, so chronology is correct
      2. response = await chain.invoke(...)
      3. save_message(user_id, "assistant", response)

    IMPORTANT: Always save the user message BEFORE the assistant reply.
    If you swap the order, the LLM will see its own reply before the question.
    """
    now = datetime.utcnow().isoformat()
    async with get_db() as db:
        cursor = await db.execute(
            """
            INSERT INTO messages (user_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, role, content, now),
        )
        await db.commit()
        row_id = cursor.lastrowid

    msg = Message(
        id=row_id,
        user_id=user_id,
        role=role,       # type: ignore[arg-type]  (Literal enforced by DB CHECK)
        content=content,
        created_at=datetime.fromisoformat(now),
    )
    logger.debug("Saved message id=%s role=%s user=%s", row_id, role, user_id)
    return msg


async def get_history(user_id: int, limit: int = 10) -> list[Message]:
    """
    Return the last `limit` messages for a user in chronological order
    (oldest first), ready to pass directly into the LLM prompt.

    The subquery trick:
      We want the LAST N rows, but in ASCENDING order for the LLM.
      Inner query: take the last N desc → outer query: reverse to asc.

    Args:
        user_id: Telegram user ID
        limit:   Max number of messages to return. Controlled by HISTORY_LIMIT
                 in config.py (default 10 = 5 exchanges).
    """
    async with get_db() as db:
        cursor = await db.execute(
            """
            SELECT id, user_id, role, content, created_at
            FROM (
                SELECT id, user_id, role, content, created_at
                FROM messages
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            )
            ORDER BY created_at ASC
            """,
            (user_id, limit),
        )
        rows = await cursor.fetchall()

    history = [
        Message(
            id=row["id"],
            user_id=row["user_id"],
            role=row["role"],            # type: ignore[arg-type]
            content=row["content"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
        for row in rows
    ]
    logger.debug("Fetched %d messages for user %s", len(history), user_id)
    return history


async def clear_history(user_id: int) -> int:
    """
    Delete ALL messages for the given user.
    Returns the number of rows deleted.

    Called by /clear command handler.
    """
    async with get_db() as db:
        cursor = await db.execute(
            "DELETE FROM messages WHERE user_id = ?",
            (user_id,),
        )
        await db.commit()
        deleted = cursor.rowcount

    logger.info("Cleared %d messages for user %s", deleted, user_id)
    return deleted
