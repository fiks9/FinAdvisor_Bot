"""
app/config.py — Centralised configuration via Pydantic v2 Settings
====================================================================
All values are read from the .env file (or real environment variables).
If any REQUIRED variable is missing or has the wrong type, the application
will REFUSE to start and print a clear validation error.

Usage anywhere in the project:
    from app.config import get_settings
    settings = get_settings()
    print(settings.TELEGRAM_BOT_TOKEN)
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Single source of truth for all runtime configuration.

    Fields marked with no default value are REQUIRED — the app
    will not start if they are absent from .env or the environment.
    """

    model_config = SettingsConfigDict(
        # Look for .env in the project root (one level above app/)
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        # Ignore extra variables that may exist in .env but aren't declared here
        extra="ignore",
        # Allow reading from real OS environment variables too (important for Railway)
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ #
    #  Telegram                                                            #
    # ------------------------------------------------------------------ #
    TELEGRAM_BOT_TOKEN: str = Field(
        ...,  # "..." means REQUIRED — no default
        description="Bot token from @BotFather on Telegram.",
    )

    # ------------------------------------------------------------------ #
    #  Groq API                                                            #
    # ------------------------------------------------------------------ #
    GROQ_API_KEY: str = Field(
        ...,
        description="API key from console.groq.com.",
    )

    # ------------------------------------------------------------------ #
    #  LLM Parameters                                                      #
    # ------------------------------------------------------------------ #
    LLM_MODEL: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model identifier.",
    )
    LLM_TEMPERATURE: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Lower = more factual, higher = more creative.",
    )

    # ------------------------------------------------------------------ #
    #  Storage Paths                                                        #
    # ------------------------------------------------------------------ #
    DB_PATH: Path = Field(
        default=Path("data/memory.db"),
        description=(
            "Path for the SQLite database. "
            "On Railway, point this to your Volume mount, e.g. /app/data/memory.db"
        ),
    )
    CHROMA_PATH: Path = Field(
        default=Path("data/chroma"),
        description=(
            "Directory where ChromaDB persists vector data. "
            "On Railway, point this to your Volume mount, e.g. /app/data/chroma"
        ),
    )

    # ------------------------------------------------------------------ #
    #  RAG Parameters                                                       #
    # ------------------------------------------------------------------ #
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model name (runs locally, no API cost).",
    )
    RETRIEVER_TOP_K: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of relevant document chunks to inject into the LLM prompt.",
    )
    CHUNK_SIZE: int = Field(
        default=800,
        ge=100,
        le=4000,
        description="Max characters per text chunk when splitting documents.",
    )
    CHUNK_OVERLAP: int = Field(
        default=100,
        ge=0,
        le=500,
        description="Overlap between consecutive chunks so context is never cut in half.",
    )

    # ------------------------------------------------------------------ #
    #  Conversation Memory                                                  #
    # ------------------------------------------------------------------ #
    HISTORY_LIMIT: int = Field(
        default=10,
        ge=1,
        le=50,
        description="How many past messages to feed into the LLM for context.",
    )

    # ------------------------------------------------------------------ #
    #  Validators                                                           #
    # ------------------------------------------------------------------ #
    @field_validator("DB_PATH", "CHROMA_PATH", mode="after")
    @classmethod
    def ensure_parent_exists(cls, v: Path) -> Path:
        """Automatically create parent directories so the app never crashes
        with 'FileNotFoundError' on first run."""
        v.parent.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("CHUNK_OVERLAP", mode="after")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("CHUNK_SIZE", 800)
        if v >= chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({v}) must be strictly less than CHUNK_SIZE ({chunk_size})."
            )
        return v


# ------------------------------------------------------------------ #
#  Singleton accessor (cached — instantiated only once per process)    #
# ------------------------------------------------------------------ #
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.

    Using @lru_cache means .env is parsed exactly once when the app starts,
    not on every function call. This is both efficient and predictable.

    Example:
        from app.config import get_settings
        cfg = get_settings()
    """
    return Settings()
