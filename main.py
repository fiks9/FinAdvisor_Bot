"""
FinAdvisor Bot — Application Entry Point
=========================================
Bootstraps logging, validates config, initialises the database,
pre-loads the ChromaDB collection, and starts the Telegram bot.

TODO (Крок 4): replace echo handler with real RAG chain
"""
import asyncio
import logging

logger = logging.getLogger(__name__)


async def main() -> None:
    logger.info("FinAdvisor Bot starting…")
    # TODO: init DB          (Крок 8)
    # TODO: init ChromaDB    (Крок 12)
    # TODO: start bot        (Крок 4)
    logger.info("Bot stopped.")


if __name__ == "__main__":
    asyncio.run(main())
