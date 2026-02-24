"""
app/bot/handlers.py — Telegram message & command handlers
==========================================================
Registered on the Application instance in main.py.

Current state (Step 4 — Echo Bot):
  /start  → greeting message with feature overview
  /help   → list of available commands
  /clear  → placeholder (implemented in Step 9 with real DB)
  text    → echo back user's message (replaced by RAG chain in Step 17)

IMPORTANT: Every handler is an async function.
  Never use time.sleep() — always use await asyncio.sleep() if needed.
  Blocking calls will freeze the bot for ALL users simultaneously.
"""

import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from app.db import crud

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

async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handles any plain text message.

    Step 4  — echoes the message back (proves transport layer works).
    Step 17 — replace echo with: response = await ask(user.id, text)
    """
    user = update.effective_user
    text = update.message.text

    logger.info("User %s sent: %r", user.id, text[:80])  # log only first 80 chars

    # ── Typing indicator — shows "FinAdvisor Bot is typing..." ──────────
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing",
    )

    # TODO (Step 17): replace the line below with the RAG chain call
    echo_reply = f"🔁 *Ехо (тест):* {text}\n\n_RAG-ланцюжок буде підключено на Кроці 17_"

    await update.message.reply_text(echo_reply, parse_mode=ParseMode.MARKDOWN)


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
