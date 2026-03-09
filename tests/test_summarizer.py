from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

import pytest
import summarizer
from summarizer import SummarizerError


def _make_messages(n):
    return [
        {"sender": f"User{i}", "text": f"Message number {i}", "timestamp": datetime.now(timezone.utc)}
        for i in range(n)
    ]


# --- helpers ---

class TestEstimateTokens:
    def test_basic(self):
        assert summarizer._estimate_tokens("abcd") == 1
        assert summarizer._estimate_tokens("a" * 100) == 25

    def test_empty(self):
        assert summarizer._estimate_tokens("") == 0


class TestBuildLog:
    def test_format(self):
        msgs = [{"sender": "A", "text": "hi"}, {"sender": "B", "text": "bye"}]
        log = summarizer._build_log(msgs)
        assert log == "A: hi\nB: bye"


# --- summarize ---

class TestSummarize:
    @pytest.mark.asyncio
    async def test_too_few_messages(self):
        result = await summarizer.summarize(_make_messages(5))
        assert "хотя бы 10" in result

    @pytest.mark.asyncio
    @patch("summarizer._call_llm", return_value="Summary text")
    async def test_small_log_single_call(self, mock_llm):
        result = await summarizer.summarize(_make_messages(15))
        assert result == "Summary text"
        assert mock_llm.call_count == 1
        # Should use SYSTEM_PROMPT
        assert mock_llm.call_args[0][0] == summarizer.SYSTEM_PROMPT

    @pytest.mark.asyncio
    @patch("summarizer._call_llm", return_value="Chunk summary")
    async def test_large_log_chunking(self, mock_llm):
        # Create messages that exceed TOKEN_LIMIT
        msgs = [
            {"sender": f"User{i}", "text": "x" * 200, "timestamp": datetime.now(timezone.utc)}
            for i in range(200)
        ]
        result = await summarizer.summarize(msgs)
        # Should have multiple calls: chunks + merge
        assert mock_llm.call_count > 1
        # Last call should use MERGE_PROMPT
        last_call_system = mock_llm.call_args_list[-1][0][0]
        assert last_call_system == summarizer.MERGE_PROMPT


# --- tldr ---

class TestTldr:
    @pytest.mark.asyncio
    async def test_too_few_messages(self):
        result = await summarizer.tldr(_make_messages(3))
        assert "хотя бы 10" in result

    @pytest.mark.asyncio
    @patch("summarizer._call_llm", return_value="TL;DR text")
    async def test_small_log(self, mock_llm):
        result = await summarizer.tldr(_make_messages(15))
        assert result == "TL;DR text"
        assert mock_llm.call_args[0][0] == summarizer.TLDR_PROMPT


# --- _call_llm error handling ---

class TestCallLlmErrors:
    @patch("summarizer._client")
    def test_timeout(self, mock_client):
        from openai import APITimeoutError
        mock_client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())
        with pytest.raises(SummarizerError, match="не ответил вовремя"):
            summarizer._call_llm("sys", "usr")

    @patch("summarizer._client")
    def test_rate_limit(self, mock_client):
        from openai import RateLimitError
        resp = MagicMock()
        resp.status_code = 429
        resp.headers = {}
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="rate limited", response=resp, body={}
        )
        with pytest.raises(SummarizerError, match="лимит запросов"):
            summarizer._call_llm("sys", "usr")

    @patch("summarizer._client")
    def test_api_status_error(self, mock_client):
        from openai import APIStatusError
        resp = MagicMock()
        resp.status_code = 500
        resp.headers = {}
        mock_client.chat.completions.create.side_effect = APIStatusError(
            message="server error", response=resp, body={}
        )
        with pytest.raises(SummarizerError, match="код 500"):
            summarizer._call_llm("sys", "usr")

    @patch("summarizer._client")
    def test_context_length_exceeded(self, mock_client):
        from openai import APIStatusError
        resp = MagicMock()
        resp.status_code = 400
        resp.headers = {}
        mock_client.chat.completions.create.side_effect = APIStatusError(
            message="context_length_exceeded", response=resp, body={}
        )
        with pytest.raises(SummarizerError, match="не помещается"):
            summarizer._call_llm("sys", "usr")
