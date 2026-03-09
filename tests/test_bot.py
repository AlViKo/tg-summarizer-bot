import csv
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import bot
import reader
from summarizer import SummarizerError


@pytest.fixture(autouse=True)
def reset_state(tmp_path):
    reader._history.clear()
    bot._last_request.clear()
    # Redirect analytics to a temp file for each test
    bot.ANALYTICS_PATH = str(tmp_path / "analytics.csv")
    bot._ensure_analytics_header()
    yield
    reader._history.clear()
    bot._last_request.clear()


def _make_update(chat_id=-100, chat_type="supergroup", user_id=1, args=None, language_code="en"):
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_chat.type = chat_type
    update.effective_user.id = user_id
    update.effective_user.language_code = language_code
    update.message.reply_text = AsyncMock(return_value=MagicMock(edit_text=AsyncMock()))
    context = MagicMock()
    context.args = args or []
    return update, context


def _seed_messages(chat_id=-100, n=20):
    for i in range(n):
        msg = MagicMock()
        msg.chat.id = chat_id
        msg.chat.type = "supergroup"
        msg.text = f"Test message {i}"
        msg.caption = None
        msg.from_user.first_name = f"User{i}"
        msg.from_user.last_name = None
        msg.date = datetime.now(timezone.utc)
        msg.photo = None
        msg.voice = None
        msg.video = None
        msg.video_note = None
        msg.sticker = None
        msg.audio = None
        msg.document = None
        reader.store_message(msg)


# --- rate limiting ---

class TestRateLimit:
    def test_first_request_allowed(self):
        assert bot._check_rate_limit(-100) is None

    def test_second_request_blocked(self):
        bot._check_rate_limit(-100)
        wait = bot._check_rate_limit(-100)
        assert wait is not None
        assert wait > 0

    def test_different_chats_independent(self):
        bot._check_rate_limit(-100)
        assert bot._check_rate_limit(-200) is None

    def test_expired_rate_limit(self):
        bot._last_request[-100] = datetime.now(timezone.utc) - timedelta(seconds=120)
        assert bot._check_rate_limit(-100) is None


# --- /start ---

class TestStartCommand:
    @pytest.mark.asyncio
    async def test_private_en(self):
        update, context = _make_update(chat_type="private", language_code="en")
        await bot.start_command(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "Add me" in text
        assert "/summarize" in text
        assert "/tldr" in text

    @pytest.mark.asyncio
    async def test_private_ru(self):
        update, context = _make_update(chat_type="private", language_code="ru")
        await bot.start_command(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "Добавь" in text
        assert "/summarize" in text
        assert "админ" in text.lower()

    @pytest.mark.asyncio
    async def test_group_en(self):
        update, context = _make_update(chat_type="supergroup", language_code="en")
        await bot.start_command(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "/summarize" in text
        assert "/tldr" in text
        assert "Add me" not in text

    @pytest.mark.asyncio
    async def test_group_ru(self):
        update, context = _make_update(chat_type="supergroup", language_code="ru")
        await bot.start_command(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "/summarize" in text
        assert "Добавь" not in text

    @pytest.mark.asyncio
    async def test_unknown_lang_falls_back_to_en(self):
        update, context = _make_update(chat_type="private", language_code="ja")
        await bot.start_command(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "Add me" in text

    @pytest.mark.asyncio
    async def test_no_language_code_falls_back_to_en(self):
        update, context = _make_update(chat_type="private", language_code=None)
        await bot.start_command(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "Add me" in text

    @pytest.mark.asyncio
    async def test_regional_lang_code(self):
        update, context = _make_update(chat_type="private", language_code="ru-RU")
        await bot.start_command(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "Добавь" in text


# --- /help ---

class TestHelpCommand:
    @pytest.mark.asyncio
    async def test_sends_help_text(self):
        update, context = _make_update()
        await bot.help_command(update, context)
        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "/summarize" in text
        assert "/tldr" in text
        assert "/help" in text


# --- /summarize ---

class TestSummarizeCommand:
    @pytest.mark.asyncio
    async def test_private_chat_rejected(self):
        update, context = _make_update(chat_type="private")
        await bot.summarize_command(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "групповых" in text

    @pytest.mark.asyncio
    async def test_no_messages(self):
        update, context = _make_update()
        await bot.summarize_command(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "Не удалось получить сообщения" in text

    @pytest.mark.asyncio
    async def test_rate_limited(self):
        _seed_messages()
        update1, context1 = _make_update()
        update2, context2 = _make_update()

        with patch("summarizer.summarize", new_callable=AsyncMock, return_value="Summary"):
            await bot.summarize_command(update1, context1)
        await bot.summarize_command(update2, context2)
        text = update2.message.reply_text.call_args[0][0]
        assert "Подождите" in text

    @pytest.mark.asyncio
    @patch("summarizer.summarize", new_callable=AsyncMock, return_value="Summary result")
    async def test_default_limit(self, mock_sum):
        _seed_messages(n=150)
        update, context = _make_update()
        await bot.summarize_command(update, context)
        # Should have called summarize with <= DEFAULT_MESSAGES
        called_msgs = mock_sum.call_args[0][0]
        assert len(called_msgs) <= bot.DEFAULT_MESSAGES

    @pytest.mark.asyncio
    @patch("summarizer.summarize", new_callable=AsyncMock, return_value="Summary result")
    async def test_numeric_arg(self, mock_sum):
        _seed_messages(n=100)
        update, context = _make_update(args=["30"])
        await bot.summarize_command(update, context)
        called_msgs = mock_sum.call_args[0][0]
        assert len(called_msgs) == 30

    @pytest.mark.asyncio
    async def test_invalid_arg_shows_hint(self):
        update, context = _make_update(args=["blah"])
        await bot.summarize_command(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "Формат команды" in text

    @pytest.mark.asyncio
    @patch("summarizer.summarize", new_callable=AsyncMock, return_value="Summary result")
    async def test_time_arg_hours(self, mock_sum):
        _seed_messages(n=20)
        update, context = _make_update(args=["2h"])
        await bot.summarize_command(update, context)
        mock_sum.assert_called_once()

    @pytest.mark.asyncio
    @patch("summarizer.summarize", new_callable=AsyncMock, return_value="Summary result")
    async def test_time_arg_days(self, mock_sum):
        _seed_messages(n=20)
        update, context = _make_update(args=["1d"])
        await bot.summarize_command(update, context)
        mock_sum.assert_called_once()

    @pytest.mark.asyncio
    @patch("summarizer.summarize", new_callable=AsyncMock, return_value="Summary result")
    async def test_time_arg_minutes(self, mock_sum):
        _seed_messages(n=20)
        update, context = _make_update(args=["30m"])
        await bot.summarize_command(update, context)
        mock_sum.assert_called_once()

    @pytest.mark.asyncio
    @patch("summarizer.summarize", new_callable=AsyncMock, return_value="OK")
    async def test_truncation_warning(self, mock_sum):
        _seed_messages(n=bot.MAX_MESSAGES)
        update, context = _make_update(args=["9999"])
        await bot.summarize_command(update, context)
        status_text = update.message.reply_text.call_args[0][0]
        assert "Ограничено" in status_text

    @pytest.mark.asyncio
    async def test_summarizer_error_shown(self):
        _seed_messages()
        update, context = _make_update()
        with patch("summarizer.summarize", new_callable=AsyncMock, side_effect=SummarizerError("LLM broke")):
            await bot.summarize_command(update, context)
        status_msg = update.message.reply_text.return_value
        status_msg.edit_text.assert_called()
        edit_text = status_msg.edit_text.call_args[0][0]
        assert "LLM broke" in edit_text

    @pytest.mark.asyncio
    async def test_unexpected_error_handled(self):
        _seed_messages()
        update, context = _make_update()
        with patch("summarizer.summarize", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
            await bot.summarize_command(update, context)
        status_msg = update.message.reply_text.return_value
        status_msg.edit_text.assert_called()
        edit_text = status_msg.edit_text.call_args[0][0]
        assert "непредвиденная" in edit_text.lower()


# --- /tldr ---

class TestTldrCommand:
    @pytest.mark.asyncio
    async def test_private_chat_rejected(self):
        update, context = _make_update(chat_type="private")
        await bot.tldr_command(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "групповых" in text

    @pytest.mark.asyncio
    async def test_no_messages(self):
        update, context = _make_update()
        await bot.tldr_command(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "Не удалось получить сообщения" in text

    @pytest.mark.asyncio
    @patch("summarizer.tldr", new_callable=AsyncMock, return_value="TL;DR result")
    async def test_success(self, mock_tldr):
        _seed_messages()
        update, context = _make_update()
        await bot.tldr_command(update, context)
        mock_tldr.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarizer_error_shown(self):
        _seed_messages()
        update, context = _make_update()
        with patch("summarizer.tldr", new_callable=AsyncMock, side_effect=SummarizerError("error")):
            await bot.tldr_command(update, context)
        status_msg = update.message.reply_text.return_value
        edit_text = status_msg.edit_text.call_args[0][0]
        assert "error" in edit_text


# --- unknown command ---

class TestUnknownCommand:
    @pytest.mark.asyncio
    async def test_sends_hint(self):
        update, context = _make_update()
        await bot.unknown_command(update, context)
        text = update.message.reply_text.call_args[0][0]
        assert "Неизвестная команда" in text
        assert "/summarize" in text
        assert "/tldr" in text


# --- collect_message ---

class TestCollectMessage:
    @pytest.mark.asyncio
    async def test_stores_message(self):
        msg = MagicMock()
        msg.chat.id = -100
        msg.chat.type = "supergroup"
        msg.text = "hello"
        msg.caption = None
        msg.from_user.first_name = "Test"
        msg.from_user.last_name = None
        msg.date = datetime.now(timezone.utc)
        msg.photo = None
        msg.voice = None
        msg.video = None
        msg.video_note = None
        msg.sticker = None
        msg.audio = None
        msg.document = None

        update = MagicMock()
        update.message = msg
        context = MagicMock()

        await bot.collect_message(update, context)
        assert len(reader.get_messages(-100)) == 1

    @pytest.mark.asyncio
    async def test_no_message_in_update(self):
        update = MagicMock()
        update.message = None
        context = MagicMock()
        await bot.collect_message(update, context)
        # Should not crash


# --- error_handler ---

class TestErrorHandler:
    @pytest.mark.asyncio
    async def test_does_not_crash(self):
        update = MagicMock()
        context = MagicMock()
        context.error = RuntimeError("test")
        await bot.error_handler(update, context)


# --- analytics ---

def _read_analytics_rows() -> list[dict]:
    with open(bot.ANALYTICS_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestAnalytics:
    def test_log_analytics_writes_row(self):
        bot.log_analytics(-100, "/summarize", 50, 2.5, "ok")
        rows = _read_analytics_rows()
        assert len(rows) == 1
        assert rows[0]["chat_id"] == "-100"
        assert rows[0]["command"] == "/summarize"
        assert rows[0]["messages"] == "50"
        assert rows[0]["duration_sec"] == "2.5"
        assert rows[0]["status"] == "ok"
        assert rows[0]["timestamp"]  # not empty

    def test_multiple_rows(self):
        bot.log_analytics(-100, "/summarize", 50, 2.5, "ok")
        bot.log_analytics(-200, "/tldr", 30, 1.1, "ok")
        rows = _read_analytics_rows()
        assert len(rows) == 2

    def test_error_status_logged(self):
        bot.log_analytics(-100, "/summarize", 20, 0.5, "error: timeout")
        rows = _read_analytics_rows()
        assert rows[0]["status"] == "error: timeout"

    def test_header_created(self, tmp_path):
        path = str(tmp_path / "new_analytics.csv")
        bot.ANALYTICS_PATH = path
        bot._ensure_analytics_header()
        with open(path, encoding="utf-8") as f:
            header = f.readline().strip()
        assert "timestamp" in header
        assert "chat_id" in header

    @pytest.mark.asyncio
    @patch("summarizer.summarize", new_callable=AsyncMock, return_value="Summary")
    async def test_summarize_logs_analytics_ok(self, mock_sum):
        _seed_messages()
        update, context = _make_update()
        await bot.summarize_command(update, context)
        rows = _read_analytics_rows()
        assert len(rows) == 1
        assert rows[0]["command"] == "/summarize"
        assert rows[0]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_summarize_logs_analytics_on_error(self):
        _seed_messages()
        update, context = _make_update()
        with patch("summarizer.summarize", new_callable=AsyncMock, side_effect=SummarizerError("broke")):
            await bot.summarize_command(update, context)
        rows = _read_analytics_rows()
        assert len(rows) == 1
        assert "error" in rows[0]["status"]

    @pytest.mark.asyncio
    @patch("summarizer.tldr", new_callable=AsyncMock, return_value="TL;DR")
    async def test_tldr_logs_analytics_ok(self, mock_tldr):
        _seed_messages()
        update, context = _make_update()
        await bot.tldr_command(update, context)
        rows = _read_analytics_rows()
        assert len(rows) == 1
        assert rows[0]["command"] == "/tldr"
        assert rows[0]["status"] == "ok"

    @pytest.mark.asyncio
    async def test_unexpected_error_logs_analytics(self):
        _seed_messages()
        update, context = _make_update()
        with patch("summarizer.summarize", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
            await bot.summarize_command(update, context)
        rows = _read_analytics_rows()
        assert len(rows) == 1
        assert rows[0]["status"] == "error: unexpected"
