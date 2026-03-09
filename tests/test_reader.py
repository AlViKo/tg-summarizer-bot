from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
import reader


@pytest.fixture(autouse=True)
def clear_history():
    reader._history.clear()
    yield
    reader._history.clear()


def _make_msg(
    chat_id=-100,
    chat_type="supergroup",
    text="hello",
    first_name="Ivan",
    last_name=None,
    date=None,
    photo=None,
    voice=None,
    video=None,
    video_note=None,
    sticker=None,
    audio=None,
    document=None,
    caption=None,
):
    msg = MagicMock()
    msg.chat.id = chat_id
    msg.chat.type = chat_type
    msg.text = text
    msg.caption = caption
    msg.from_user.first_name = first_name
    msg.from_user.last_name = last_name
    msg.date = date or datetime.now(timezone.utc)
    msg.photo = photo
    msg.voice = voice
    msg.video = video
    msg.video_note = video_note
    msg.sticker = sticker
    msg.audio = audio
    msg.document = document
    return msg


# --- store_message ---

class TestStoreMessage:
    def test_stores_text_message(self):
        reader.store_message(_make_msg(text="привет"))
        msgs = reader.get_messages(-100)
        assert len(msgs) == 1
        assert msgs[0]["text"] == "привет"
        assert msgs[0]["sender"] == "Ivan"

    def test_full_name(self):
        reader.store_message(_make_msg(first_name="Ivan", last_name="Petrov"))
        msgs = reader.get_messages(-100)
        assert msgs[0]["sender"] == "Ivan Petrov"

    def test_ignores_private_chat(self):
        reader.store_message(_make_msg(chat_type="private"))
        assert reader.get_messages(-100) == []

    def test_ignores_empty_message(self):
        reader.store_message(_make_msg(text=None, caption=None))
        assert reader.get_messages(-100) == []

    def test_ignores_no_chat(self):
        msg = MagicMock()
        msg.chat = None
        reader.store_message(msg)
        assert reader.get_messages(-100) == []

    def test_stores_caption(self):
        reader.store_message(_make_msg(text=None, caption="подпись к фото", photo=True))
        msgs = reader.get_messages(-100)
        assert msgs[0]["text"] == "[фото] подпись к фото"

    def test_unknown_user(self):
        msg = _make_msg(text="hi")
        msg.from_user = None
        reader.store_message(msg)
        msgs = reader.get_messages(-100)
        assert msgs[0]["sender"] == "Unknown"

    def test_stores_timestamp(self):
        now = datetime.now(timezone.utc)
        reader.store_message(_make_msg(date=now))
        msgs = reader.get_messages(-100)
        assert msgs[0]["timestamp"] == now


# --- _describe_media ---

class TestDescribeMedia:
    def test_photo(self):
        assert reader._describe_media(_make_msg(photo=True)) == "[фото]"

    def test_voice(self):
        assert reader._describe_media(_make_msg(voice=True)) == "[голосовое]"

    def test_video(self):
        assert reader._describe_media(_make_msg(video=True)) == "[видео]"

    def test_video_note(self):
        assert reader._describe_media(_make_msg(video_note=True)) == "[видеосообщение]"

    def test_sticker(self):
        assert reader._describe_media(_make_msg(sticker=True)) == "[стикер]"

    def test_audio(self):
        assert reader._describe_media(_make_msg(audio=True)) == "[аудио]"

    def test_document(self):
        assert reader._describe_media(_make_msg(document=True)) == "[файл]"

    def test_no_media(self):
        msg = _make_msg()
        msg.photo = None
        msg.voice = None
        msg.video = None
        msg.video_note = None
        msg.sticker = None
        msg.audio = None
        msg.document = None
        assert reader._describe_media(msg) is None

    def test_media_with_text(self):
        reader.store_message(_make_msg(text="описание", photo=True))
        msgs = reader.get_messages(-100)
        assert msgs[0]["text"] == "[фото] описание"

    def test_media_without_text(self):
        reader.store_message(_make_msg(text=None, caption=None, photo=True))
        msgs = reader.get_messages(-100)
        assert msgs[0]["text"] == "[фото]"


# --- get_messages ---

class TestGetMessages:
    def test_returns_empty_for_unknown_chat(self):
        assert reader.get_messages(999) == []

    def test_respects_limit(self):
        for i in range(20):
            reader.store_message(_make_msg(text=f"msg {i}"))
        msgs = reader.get_messages(-100, limit=5)
        assert len(msgs) == 5
        assert msgs[0]["text"] == "msg 15"
        assert msgs[-1]["text"] == "msg 19"

    def test_returns_all_if_fewer_than_limit(self):
        for i in range(3):
            reader.store_message(_make_msg(text=f"msg {i}"))
        msgs = reader.get_messages(-100, limit=100)
        assert len(msgs) == 3

    def test_max_history_cap(self):
        for i in range(reader.MAX_HISTORY + 50):
            reader.store_message(_make_msg(text=f"msg {i}"))
        all_msgs = reader.get_messages(-100, limit=9999)
        assert len(all_msgs) == reader.MAX_HISTORY


# --- get_messages_since ---

class TestGetMessagesSince:
    def test_returns_empty_for_unknown_chat(self):
        assert reader.get_messages_since(999, datetime.now(timezone.utc)) == []

    def test_filters_by_time(self):
        now = datetime.now(timezone.utc)
        old = now - timedelta(hours=3)
        recent = now - timedelta(minutes=30)

        reader.store_message(_make_msg(text="old", date=old))
        reader.store_message(_make_msg(text="recent", date=recent))
        reader.store_message(_make_msg(text="now", date=now))

        since = now - timedelta(hours=1)
        msgs = reader.get_messages_since(-100, since)
        assert len(msgs) == 2
        assert msgs[0]["text"] == "recent"
        assert msgs[1]["text"] == "now"

    def test_respects_limit(self):
        now = datetime.now(timezone.utc)
        for i in range(10):
            reader.store_message(_make_msg(text=f"msg {i}", date=now))
        msgs = reader.get_messages_since(-100, now - timedelta(hours=1), limit=3)
        assert len(msgs) == 3
