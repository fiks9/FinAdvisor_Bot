"""
app/bot/handlers.py — Telegram message & command handlers
==========================================================
Registered on the Application instance in main.py.

Current state (Steps 4, 17, 18 — Fully integrated):
  /start  → greeting + upsert user in DB
  /help   → list of available commands
  /clear  → clears SQLite conversation history
  text    → RAG chain call (retrieve → history → LLM) + memory writeback

IMPORTANT: Every handler is an async function.
  Never use time.sleep() — always use await asyncio.sleep() if needed.
  Blocking calls will freeze the bot for ALL users simultaneously.
"""

import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from app.db import crud
from app.llm.chain import ask

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
#  /start
# ──────────────────────────────────────────────────────────────────────────────

WELCOME_TEXT = """
👋 *Вітаю! Я FinAdvisor Bot* — твій персональний фінансовий радник.

Ось що я вмію:
• 💬 Відповідати на фінансові питання на основі документів НБУ та законів
• 🧠 Запам'ятовувати контекст нашої розмови
• 📚 Шукати актуальну інформацію у базі знань

*Команди:*
/start — показати це повідомлення
/help  — детальна довідка
/clear — очистити історію розмови

Просто напиши своє питання і я відповім! 🚀
"""


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start command. Greets the user."""
    user = update.effective_user
    logger.info("User %s (%s) started the bot", user.id, user.username or "no_username")

    # Register / refresh user in the DB
    await crud.upsert_user(user.id, user.username, user.first_name)

    await update.message.reply_text(
        WELCOME_TEXT,
        parse_mode=ParseMode.MARKDOWN,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  /help
# ──────────────────────────────────────────────────────────────────────────────

HELP_TEXT = """
📖 *Довідка FinAdvisor Bot*

*Як користуватись:*
Просто пиши своє питання звичайним текстом.

*Приклади запитів:*
• Який ліміт на перекази без верифікації?
• Що таке IBAN?
• Як захистити картку від шахраїв?
• Поясни різницю між дебетовою та кредитною карткою

*Команди:*
/start — головне меню
/help  — ця довідка
/clear — очистити пам'ять розмови (почати з нуля)

*Примітка:* Я надаю загальну інформацію. Для персональних фінансових рішень звертайся до ліцензованого фінансового консультанта.
"""


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /help command."""
    user = update.effective_user
    logger.info("User %s requested /help", user.id)

    await update.message.reply_text(
        HELP_TEXT,
        parse_mode=ParseMode.MARKDOWN,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  /clear
# ──────────────────────────────────────────────────────────────────────────────

async def clear_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles the /clear command.
    Deletes the user's full conversation history from SQLite.
    """
    user = update.effective_user
    logger.info("User %s requested /clear", user.id)

    deleted = await crud.clear_history(user.id)
    await update.message.reply_text(
        f"🗑 Історію очищено ({deleted} повідомлень). Починаємо з нуля 👍",
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Text message handler — Echo (temporary)
# ──────────────────────────────────────────────────────────────────────────────

def _split_long_message(text: str, max_length: int = 4096) -> list[str]:
    """Splits a long text into chunks, trying to break at paragraph boundaries."""
    chunks = []
    current_chunk = ""
    paragraphs = text.split('\n\n') # Split by double newline for paragraphs

    for paragraph in paragraphs:
        # If adding the next paragraph (plus separator) exceeds max_length
        # and current_chunk is not empty, then save current_chunk and start new one.
        # Add 2 for the '\n\n' separator if it's not the first paragraph
        if current_chunk and len(current_chunk) + len(paragraph) + 2 > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

        # If a single paragraph is too long, split it further
        while len(current_chunk) > max_length:
            # Find a suitable split point within the current_chunk
            # Try to split by newline first, then by sentence, then just by length
            split_point = -1
            # Look for last newline before max_length
            if '\n' in current_chunk[:max_length]:
                split_point = current_chunk.rfind('\n', 0, max_length)
            # If no newline, look for last space
            if split_point == -1 and ' ' in current_chunk[:max_length]:
                split_point = current_chunk.rfind(' ', 0, max_length)
            # If still no good split point, just cut at max_length
            if split_point == -1:
                split_point = max_length

            chunks.append(current_chunk[:split_point].strip())
            current_chunk = current_chunk[split_point:].strip()

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles any plain text message.

    Full flow (Steps 17 + 18):
      1. Upsert user to DB
      2. Save user message  (← BEFORE chain call, order matters for history!)
      3. Show typing indicator
      4. Call RAG chain: retrieve → history → LLM → response
      5. Save assistant response
      6. Send response to user
    """
    user = update.effective_user
    text = update.message.text

    logger.info("User %s sent: %r", user.id, text[:80])

    # ── 1. Register user (no-op if already exists) ────────────────────────────
    await crud.upsert_user(user.id, user.username, user.first_name)

    # ── 2. Save user message BEFORE chain call (chronology!) ────────────────
    await crud.save_message(user.id, "user", text)

    # ── 3. Show "typing..." indicator ────────────────────────────────
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing",
    )

    # ── 4. Run the RAG chain (may take 2–5 seconds) ──────────────────────
    # ask() is async — it awaits DB calls + wraps the sync Groq call in
    # asyncio.to_thread() internally. Just await it normally here.
    try:
        response: str = await ask(user.id, text)
    except Exception as e:
        logger.error("Chain error for user %s: %s", user.id, e, exc_info=True)
        await update.message.reply_text(
            "⚠️ Не вдалось отримати відповідь. Спробуй ще раз через кілька секунд."
        )
        return

    # ── 5. Save assistant reply ───────────────────────────────────────────
    await crud.save_message(user.id, "assistant", response)

    # ── 6. Send response to user ───────────────────────────────────────
    # Telegram has a 4096-char message limit. Split if needed.
    if len(response) <= 4096:
        await update.message.reply_text(response)
    else:
        # Split at paragraph boundary to avoid cutting mid-sentence
        for chunk in _split_long_message(response):
            await update.message.reply_text(chunk)


# ──────────────────────────────────────────────────────────────────────────────
#  Global error handler
# ──────────────────────────────────────────────────────────────────────────────

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Catches ALL unhandled exceptions inside handlers.
    Logs the full traceback and sends a user-friendly message.
    """
    logger.error("Unhandled exception", exc_info=context.error)

    # Try to notify the user if the update came from a chat
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(
            "⚠️ Виникла технічна помилка. Спробуй ще раз або напиши /start."
        )
