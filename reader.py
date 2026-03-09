from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from telegram import Message

MAX_HISTORY = 500

_history: dict[int, deque[dict]] = defaultdict(lambda: deque(maxlen=MAX_HISTORY))


def _describe_media(msg: Message) -> str | None:
    if msg.photo:
        return "[фото]"
    if msg.voice:
        return "[голосовое]"
    if msg.video_note:
        return "[видеосообщение]"
    if msg.video:
        return "[видео]"
    if msg.sticker:
        return "[стикер]"
    if msg.audio:
        return "[аудио]"
    if msg.document:
        return "[файл]"
    return None


def store_message(msg: Message) -> None:
    if not msg.chat or msg.chat.type == "private":
        return

    user = msg.from_user
    name = "Unknown"
    if user:
        name = user.first_name or ""
        if user.last_name:
            name = f"{name} {user.last_name}"

    text = msg.text or msg.caption or ""
    media_tag = _describe_media(msg)
    if media_tag:
        text = f"{media_tag} {text}".strip() if text else media_tag

    if not text:
        return

    _history[msg.chat.id].append({
        "sender": name,
        "text": text,
        "timestamp": msg.date or datetime.now(timezone.utc),
    })


def get_messages(chat_id: int, limit: int = 100) -> list[dict]:
    history = _history.get(chat_id)
    if not history:
        return []
    items = list(history)
    return items[-limit:]


def get_messages_since(chat_id: int, since: datetime, limit: int = 500) -> list[dict]:
    history = _history.get(chat_id)
    if not history:
        return []
    items = [m for m in history if m["timestamp"] >= since]
    return items[-limit:]
