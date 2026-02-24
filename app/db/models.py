"""
app/db/models.py — Pydantic v2 data models for DB entities
============================================================
These models serve as the bridge between raw SQLite rows (tuples/dicts)
and typed Python objects used throughout the application.

Two entities:
  - User    — tracks Telegram user metadata and status
  - Message — one turn in a conversation (role: "user" | "assistant")
"""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ──────────────────────────────────────────────────────────────────────────────
#  Enums
# ──────────────────────────────────────────────────────────────────────────────

class MessageRole(str, Enum):
    """Who authored the message — used when feeding history to the LLM."""
    USER      = "user"
    ASSISTANT = "assistant"


# ──────────────────────────────────────────────────────────────────────────────
#  User
# ──────────────────────────────────────────────────────────────────────────────

class User(BaseModel):
    """
    Represents a Telegram user stored in the `users` table.

    Fields:
      user_id    — Telegram's unique integer ID (never changes for a user)
      username   — @handle, may be None if the user has no username set
      first_name — Telegram first name
      is_active  — False if user has blocked the bot
      created_at — UTC timestamp of first interaction
    """

    model_config = ConfigDict(from_attributes=True)

    user_id:    int
    username:   str | None = None
    first_name: str        = Field(default="Unknown")
    is_active:  bool       = True
    created_at: datetime   = Field(default_factory=datetime.utcnow)

    def display_name(self) -> str:
        """Returns @username if available, else first_name."""
        if self.username:
            return f"@{self.username}"
        return self.first_name


# ──────────────────────────────────────────────────────────────────────────────
#  Message
# ──────────────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    """
    Represents one turn in a conversation, stored in `messages` table.

    Fields:
      id         — auto-increment PK (None before INSERT)
      user_id    — FK → users.user_id
      role       — "user" or "assistant" — CRITICAL for LLM context assembly
      content    — the actual text of the message
      created_at — UTC timestamp (used to maintain chronological order)

    Why we store both roles:
      LLaMA / Groq expects a list of {"role": ..., "content": ...} dicts.
      If we only stored user messages we couldn't reconstruct the dialogue.
    """

    model_config = ConfigDict(from_attributes=True)

    id:         int | None            = None   # None until written to DB
    user_id:    int
    role:       Literal["user", "assistant"]
    content:    str                   = Field(min_length=1)
    created_at: datetime              = Field(default_factory=datetime.utcnow)

    def to_llm_dict(self) -> dict:
        """
        Converts to the format expected by LangChain / Groq chat history.
        Example: {"role": "user", "content": "Що таке IBAN?"}
        """
        return {"role": self.role, "content": self.content}
