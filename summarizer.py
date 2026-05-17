import logging

from openai import OpenAI, APITimeoutError, RateLimitError, APIStatusError
import config

logger = logging.getLogger(__name__)

_client = OpenAI(api_key=config.OPENAI_API_KEY)

SYSTEM_PROMPT = """\
You're summarizing a casual group chat for someone who missed it.
Write in third person — "ребята обсуждали", "кто-то скинул", "Аня пожаловалась".
Never write in first person plural ("мы", "у нас") — you're not part of the chat.
Never invent details, feelings, or context that wasn't in the messages.

Tone: neutral but alive. Not a news report, not a roleplay. 
Like a calm observer telling someone what happened. 
Match the language of the chat.

Format: 2-4 short paragraphs, one per topic. No headers, no emoji, no bullets.
Skip stickers, reactions, "+1", one-word messages.

If the chat referenced something specific (a meme, a video, a person) — 
mention it briefly without overexplaining.

Stay grounded in what actually happened. Don't add philosophical 
endings or vibe commentary.
"""

MERGE_PROMPT = """\
You'll get a few partial summaries of the same chat. \
Merge them into one smooth recap — like you're telling a friend what happened. \
No duplicates, no robotic lists, just a natural retelling.

Rules:
- Write in the same language as the summaries
- Keep it conversational and flowing
- If plans, decisions, or links come up in multiple parts — mention them once
- Stay honest and factual\
"""

TLDR_PROMPT = """\
Give a super quick recap in 3-5 sentences. Hit the main points only — \
like a friend giving you the 10-second version of what happened.

- Write in the same language as the messages
- Skip noise, keep it natural
- No bullet points — just a short paragraph\
"""

TOKEN_LIMIT = 3000
CHUNK_SIZE = 100


class SummarizerError(Exception):
    """User-facing error from the summarizer."""


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _build_log(messages: list[dict]) -> str:
    return "\n".join(f"{m['sender']}: {m['text']}" for m in messages)


def _call_llm(system: str, user: str) -> str:
    try:
        response = _client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content
    except APITimeoutError:
        logger.exception("OpenAI timeout")
        raise SummarizerError("Сервер OpenAI не ответил вовремя. Попробуйте позже.")
    except RateLimitError:
        logger.exception("OpenAI rate limit")
        raise SummarizerError("Превышен лимит запросов к OpenAI. Попробуйте через минуту.")
    except APIStatusError as e:
        logger.exception("OpenAI API error: %s", e.status_code)
        if "context_length_exceeded" in str(e):
            raise SummarizerError("Слишком много сообщений — текст не помещается в контекст модели. Попробуйте меньшее количество.")
        raise SummarizerError(f"Ошибка OpenAI API (код {e.status_code}). Попробуйте позже.")


async def summarize(messages: list[dict]) -> str:
    if len(messages) < 10:
        return "Недостаточно сообщений для саммари (нужно хотя бы 10)."

    log = _build_log(messages)

    if _estimate_tokens(log) <= TOKEN_LIMIT:
        return _call_llm(SYSTEM_PROMPT, f"Chat log ({len(messages)} messages):\n\n{log}")

    # Hierarchical chunking
    chunks = [messages[i:i + CHUNK_SIZE] for i in range(0, len(messages), CHUNK_SIZE)]
    partial_summaries = []
    for idx, chunk in enumerate(chunks, 1):
        chunk_log = _build_log(chunk)
        summary = _call_llm(
            SYSTEM_PROMPT,
            f"Chat log (part {idx}/{len(chunks)}, {len(chunk)} messages):\n\n{chunk_log}",
        )
        partial_summaries.append(summary)

    merged = "\n\n---\n\n".join(
        f"Part {i}: {s}" for i, s in enumerate(partial_summaries, 1)
    )
    return _call_llm(MERGE_PROMPT, f"Partial summaries to merge:\n\n{merged}")


async def tldr(messages: list[dict]) -> str:
    if len(messages) < 10:
        return "Недостаточно сообщений для саммари (нужно хотя бы 10)."

    log = _build_log(messages)

    if _estimate_tokens(log) <= TOKEN_LIMIT:
        return _call_llm(TLDR_PROMPT, f"Chat log ({len(messages)} messages):\n\n{log}")

    # For large logs, first do a full summarize, then condense
    full_summary = await summarize(messages)
    return _call_llm(TLDR_PROMPT, f"Condense this summary:\n\n{full_summary}")
