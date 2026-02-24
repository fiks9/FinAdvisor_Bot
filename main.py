"""
FinAdvisor Bot — Application Entry Point
=========================================
IMPORTANT: python-telegram-bot v20+ manages its OWN event loop.
  - app.run_polling() is SYNCHRONOUS — do NOT await it inside asyncio.run()
  - Async pre-init (DB, ChromaDB) is handled via Application.post_init hook

Boot sequence:
  1. Setup logging                    (Step 5)  ✅
  2. Validate config / load .env      (Step 3)  ✅
  3. Init SQLite database             (Step 8  — TODO)
  4. Init ChromaDB vector store       (Step 12 — TODO)
  5. Build LangChain RAG chain        (Step 16 — TODO)
  6. Register Telegram handlers       (Step 4)  ✅
  7. Start polling loop               (Step 4)  ✅
"""

import logging

from telegram.ext import Application, CommandHandler, MessageHandler, filters

from app.config import get_settings
from app.logger import setup_logging
from app.bot.handlers import (
    start_handler,
    help_handler,
    clear_handler,
    message_handler,
    error_handler,
)

# ── 1. Logging (must be FIRST — before any other import side-effects) ──────────
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


async def post_init(app: Application) -> None:
    """
    Called by PTB AFTER the event loop is running and BEFORE polling starts.
    This is the correct place for async initialization (DB, ChromaDB, etc.)

    PTB guarantees this runs inside the same event loop as the bot.
    """
    cfg = get_settings()

    # ── 3. Database init ──────────────────────────────────────────────────
    # TODO (Step 8): from app.db.database import init_db
    #                await init_db(cfg.DB_PATH)
    logger.info("DB init skipped (Step 8 pending)")

    # ── 4. ChromaDB init ──────────────────────────────────────────────────
    # TODO (Step 12): from app.rag.vectorstore import init_vectorstore
    #                 await init_vectorstore(cfg.CHROMA_PATH)
    logger.info("ChromaDB init skipped (Step 12 pending)")

    # ── 5. Build RAG chain ────────────────────────────────────────────────
    # TODO (Step 16): from app.llm.chain import build_chain
    #                 app.bot_data["chain"] = build_chain()
    logger.info("RAG chain skipped (Step 16 pending)")

    logger.info("post_init complete — bot is ready to receive messages")


def main() -> None:
    """
    Entry point. Synchronous — PTB creates and owns the event loop.
    """
    logger.info("FinAdvisor Bot starting…")

    cfg = get_settings()
    logger.info("Config loaded | model=%s | temp=%s", cfg.LLM_MODEL, cfg.LLM_TEMPERATURE)

    # ── Build Application ─────────────────────────────────────────────────
    app = (
        Application.builder()
        .token(cfg.TELEGRAM_BOT_TOKEN)
        .post_init(post_init)          # async init hook ← runs after loop starts
        .build()
    )

    # ── Register handlers ─────────────────────────────────────────────────
    app.add_handler(CommandHandler("start", start_handler))
    app.add_handler(CommandHandler("help",  help_handler))
    app.add_handler(CommandHandler("clear", clear_handler))
    # TEXT & ~COMMAND catches all plain text, ignores /commands
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))

    # Global error handler — catches exceptions from ALL handlers above
    app.add_error_handler(error_handler)

    logger.info("Handlers registered. Starting polling…")

    # ── Start polling ─────────────────────────────────────────────────────
    # run_polling() is BLOCKING and manages its own event loop internally.
    # It calls post_init() before the first update is processed.
    app.run_polling(
        allowed_updates=["message"],
        drop_pending_updates=True,     # ignore messages sent while bot was offline
    )

    logger.info("Bot stopped gracefully.")


if __name__ == "__main__":
    main()
