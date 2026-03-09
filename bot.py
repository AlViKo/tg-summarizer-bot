import csv
import io
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler

from telegram import BotCommand, Update
from telegram.error import TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

import config
import reader
import summarizer
from summarizer import SummarizerError

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler()
stdout_handler.setFormatter(logging.Formatter(LOG_FORMAT))
root_logger.addHandler(stdout_handler)

file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "bot.log"), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

# --- analytics ---

ANALYTICS_PATH = os.path.join(LOG_DIR, "analytics.csv")
_ANALYTICS_FIELDS = ["timestamp", "chat_id", "command", "messages", "duration_sec", "status"]


def _ensure_analytics_header():
    if not os.path.exists(ANALYTICS_PATH):
        with open(ANALYTICS_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_ANALYTICS_FIELDS)


_ensure_analytics_header()


def log_analytics(chat_id: int, command: str, messages: int, duration: float, status: str) -> None:
    row = [
        datetime.now(timezone.utc).isoformat(),
        chat_id,
        command,
        messages,
        round(duration, 2),
        status,
    ]
    with open(ANALYTICS_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)

MAX_MESSAGES = 500
DEFAULT_MESSAGES = 100

USAGE_HINT = (
    "Формат команды:\n"
    "/summarize — последние 100 сообщений\n"
    "/summarize 50 — последние 50 сообщений\n"
    "/summarize 2h — за последние 2 часа\n"
    "/summarize 1d — за последний день\n\n"
    "Допустимые суффиксы: m (минуты), h (часы), d (дни)"
)

_TIME_RE = re.compile(r"^(\d+)([mhd])$", re.IGNORECASE)
_TIME_MULTIPLIERS = {"m": "minutes", "h": "hours", "d": "days"}

RATE_LIMIT_SECONDS = 60
_last_request: dict[int, datetime] = {}


def _check_rate_limit(chat_id: int) -> int | None:
    """Return remaining seconds if rate-limited, None otherwise."""
    now = datetime.now(timezone.utc)
    last = _last_request.get(chat_id)
    if last:
        elapsed = (now - last).total_seconds()
        if elapsed < RATE_LIMIT_SECONDS:
            return int(RATE_LIMIT_SECONDS - elapsed) + 1
    _last_request[chat_id] = now
    return None


async def _safe_reply(update: Update, text: str) -> None:
    try:
        await update.message.reply_text(text)
    except TelegramError:
        logger.exception("Не удалось отправить сообщение в чат %s", update.effective_chat.id)


async def _safe_edit(message, text: str, parse_mode: str | None = None) -> None:
    try:
        await message.edit_text(text, parse_mode=parse_mode)
    except TelegramError:
        logger.exception("Не удалось отредактировать сообщение")


_START_PRIVATE = {
    "ru": (
        "Привет! Я бот-суммаризатор групповых чатов — "
        "потому что жизнь слишком коротка, чтобы читать 200 сообщений о том, где заказать обед.\n\n"
        "Добавь меня в групповой чат и сделай админом (чтобы я мог читать сообщения), "
        "а дальше просто вызывай команды.\n\n"
        "Команды в группе:\n"
        "/summarize — последние 100 сообщений\n"
        "/summarize 50 — последние 50 сообщений\n"
        "/summarize 2h — за последние 2 часа\n"
        "/summarize 1d — за последний день\n"
        "/tldr — супер-короткое саммари в 3-5 предложений\n"
        "/help — справка"
    ),
    "en": (
        "Hey there! I'm your group chat summarizer — "
        "because life's too short to read 200 messages about where to order lunch.\n\n"
        "Add me to a group chat, make me an admin (so I can read messages), "
        "and I'll be ready to summarize on demand.\n\n"
        "Once I'm in the group, just use:\n"
        "/summarize — last 100 messages\n"
        "/summarize 50 — last 50 messages\n"
        "/summarize 2h — last 2 hours\n"
        "/summarize 1d — last day\n"
        "/tldr — super short 3-5 sentence summary\n"
        "/help — show help"
    ),
}

_START_GROUP = {
    "ru": (
        "Привет! Я здесь, чтобы вы не листали чат до посинения. "
        "Буду тихо собирать сообщения и делать саммари по запросу.\n\n"
        "/summarize — последние 100 сообщений\n"
        "/summarize N — последние N сообщений (макс 500)\n"
        "/summarize Nh — за последние N часов (напр. 2h, 12h)\n"
        "/summarize 1d — за последний день\n"
        "/tldr — короткое саммари в 3-5 предложений\n"
        "/help — справка"
    ),
    "en": (
        "Hi! I'm here to save you from scrolling. "
        "I'll quietly collect messages and summarize them when you ask.\n\n"
        "/summarize — last 100 messages\n"
        "/summarize N — last N messages (max 500)\n"
        "/summarize Nh — last N hours (e.g. 2h, 12h)\n"
        "/summarize 1d — last day\n"
        "/tldr — quick 3-5 sentence summary\n"
        "/help — show help"
    ),
}


def _get_user_lang(update: Update) -> str:
    user = update.effective_user
    if user and user.language_code:
        lang = user.language_code.split("-")[0].lower()
        if lang in _START_PRIVATE:
            return lang
    return "en"


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lang = _get_user_lang(update)
    chat = update.effective_chat
    if chat.type == "private":
        await _safe_reply(update, _START_PRIVATE[lang])
    else:
        await _safe_reply(update, _START_GROUP[lang])


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Chat summarizer bot.\n\n"
        f"{USAGE_HINT}\n\n"
        "/tldr — quick 3-5 sentence summary\n"
        "/help — this message\n\n"
        "I collect messages from the moment I'm added to the chat."
    )
    await _safe_reply(update, text)


async def collect_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        reader.store_message(update.message)


async def summarize_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    user = update.effective_user
    logger.info("/summarize от user_id=%s в chat_id=%s, args=%s", user.id if user else "?", chat.id, context.args)

    if chat.type == "private":
        await _safe_reply(update, "Эта команда работает только в групповых чатах.")
        return

    wait = _check_rate_limit(chat.id)
    if wait:
        await _safe_reply(update, f"⏳ Подождите ещё {wait} секунд")
        return

    messages = None
    truncated = False

    if not context.args:
        messages = reader.get_messages(chat.id, limit=DEFAULT_MESSAGES)
    else:
        arg = context.args[0]

        # Try number
        if arg.isdigit():
            limit = int(arg)
            if limit > MAX_MESSAGES:
                limit = MAX_MESSAGES
                truncated = True
            messages = reader.get_messages(chat.id, limit=max(1, limit))

        else:
            # Try time format (2h, 1d, 30m)
            match = _TIME_RE.match(arg)
            if match:
                value = int(match.group(1))
                unit = _TIME_MULTIPLIERS[match.group(2).lower()]
                since = datetime.now(timezone.utc) - timedelta(**{unit: value})
                messages = reader.get_messages_since(chat.id, since, limit=MAX_MESSAGES)
            else:
                await _safe_reply(update, USAGE_HINT)
                return

    if not messages:
        await _safe_reply(update, "Не удалось получить сообщения. Бот накапливает историю с момента добавления в чат.")
        return

    msg_count = len(messages)
    status_text = f"Собрано {msg_count} сообщений. Генерирую саммари..."
    if truncated:
        status_text = f"Ограничено до {MAX_MESSAGES} сообщений. Генерирую саммари..."

    try:
        status = await update.message.reply_text(status_text)
    except TelegramError:
        logger.exception("Не удалось отправить статус в чат %s", chat.id)
        return

    t0 = time.monotonic()
    try:
        summary = await summarizer.summarize(messages)
        elapsed = time.monotonic() - t0
        logger.info("/summarize завершён: chat_id=%s, сообщений=%d, время=%.1fс", chat.id, msg_count, elapsed)
        log_analytics(chat.id, "/summarize", msg_count, elapsed, "ok")
        await _safe_edit(status, summary, parse_mode="Markdown")
    except SummarizerError as e:
        elapsed = time.monotonic() - t0
        logger.error("/summarize ошибка LLM: chat_id=%s — %s", chat.id, e)
        log_analytics(chat.id, "/summarize", msg_count, elapsed, f"error: {e}")
        await _safe_edit(status, str(e))
    except Exception:
        elapsed = time.monotonic() - t0
        logger.exception("Непредвиденная ошибка при суммаризации, chat_id=%s", chat.id)
        log_analytics(chat.id, "/summarize", msg_count, elapsed, "error: unexpected")
        await _safe_edit(status, "Произошла непредвиденная ошибка. Попробуйте позже.")


async def tldr_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    user = update.effective_user
    logger.info("/tldr от user_id=%s в chat_id=%s", user.id if user else "?", chat.id)

    if chat.type == "private":
        await _safe_reply(update, "Эта команда работает только в групповых чатах.")
        return

    wait = _check_rate_limit(chat.id)
    if wait:
        await _safe_reply(update, f"⏳ Подождите ещё {wait} секунд")
        return

    messages = reader.get_messages(chat.id, limit=DEFAULT_MESSAGES)
    if not messages:
        await _safe_reply(update, "Не удалось получить сообщения. Бот накапливает историю с момента добавления в чат.")
        return

    msg_count = len(messages)
    try:
        status = await update.message.reply_text(
            f"Собрано {msg_count} сообщений. Генерирую TL;DR..."
        )
    except TelegramError:
        logger.exception("Не удалось отправить статус в чат %s", chat.id)
        return

    t0 = time.monotonic()
    try:
        summary = await summarizer.tldr(messages)
        elapsed = time.monotonic() - t0
        logger.info("/tldr завершён: chat_id=%s, сообщений=%d, время=%.1fс", chat.id, msg_count, elapsed)
        log_analytics(chat.id, "/tldr", msg_count, elapsed, "ok")
        await _safe_edit(status, summary, parse_mode="Markdown")
    except SummarizerError as e:
        elapsed = time.monotonic() - t0
        logger.error("/tldr ошибка LLM: chat_id=%s — %s", chat.id, e)
        log_analytics(chat.id, "/tldr", msg_count, elapsed, f"error: {e}")
        await _safe_edit(status, str(e))
    except Exception:
        elapsed = time.monotonic() - t0
        logger.exception("Непредвиденная ошибка при суммаризации, chat_id=%s", chat.id)
        log_analytics(chat.id, "/tldr", msg_count, elapsed, "error: unexpected")
        await _safe_edit(status, "Произошла непредвиденная ошибка. Попробуйте позже.")


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _safe_reply(update,
        "Неизвестная команда.\n\n"
        "/summarize — суммаризация чата\n"
        "/tldr — короткое саммари в 3-5 предложений\n"
        "/help — справка"
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Необработанное исключение:", exc_info=context.error)


BOT_COMMANDS = [
    BotCommand("start", "Start the bot"),
    BotCommand("summarize", "Summarize recent messages"),
    BotCommand("tldr", "Quick 3-5 sentence summary"),
    BotCommand("help", "Show help"),
]


async def post_init(application: Application) -> None:
    await application.bot.set_my_commands(BOT_COMMANDS)
    logger.info("Bot commands registered")


def main() -> None:
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("summarize", summarize_command))
    app.add_handler(CommandHandler("tldr", tldr_command))
    app.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, collect_message))
    app.add_error_handler(error_handler)
    logger.info("Бот запущен")
    app.run_polling()


if __name__ == "__main__":
    main()
